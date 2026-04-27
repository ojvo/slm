package slm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

// -----------------------------------------------------------------------------
// OpenAI Chat Completions Driver
// -----------------------------------------------------------------------------

// OpenAIEngine OpenAI 协议驱动。
// 只负责 OpenAI 协议的编解码，HTTP 通信和认证由 Transport 实现。

type OpenAIEngine struct {
	adapter *genericAdapter[*Request, *Response, StreamIterator]
}

// NewOpenAIProtocol 创建 OpenAI 协议引擎（使用标准 HTTP 传输）。
func NewOpenAIProtocol(baseURL, apiKey, defaultModel string) Engine {
	base := protocolBase{transport: NewHTTPTransport(baseURL, apiKey), defaultModel: defaultModel}
	codec := NewOpenAICodec()
	return &OpenAIEngine{adapter: newChatAdapter(base, codec)}
}

func NewOpenAIWithTransport(transport Transport, defaultModel string) Engine {
	base := protocolBase{transport: transport, defaultModel: defaultModel}
	codec := NewOpenAICodec()
	return &OpenAIEngine{adapter: newChatAdapter(base, codec)}
}

func (e *OpenAIEngine) Generate(ctx context.Context, req *Request) (*Response, error) {
	return e.adapter.generate(ctx, req)
}

func (e *OpenAIEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	return e.adapter.stream(ctx, req)
}

// -----------------------------------------------------------------------------
// OpenAI Chat Wire Types
// -----------------------------------------------------------------------------

type oaiTool struct {
	Type     string      `json:"type"`
	Function oaiFunction `json:"function"`
}

type oaiFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters"`
}

type oaiMessage struct {
	Role             string        `json:"role"`
	Name             string        `json:"name,omitempty"`
	Content          any           `json:"content"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
	ToolCalls        []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string        `json:"tool_call_id,omitempty"`
}

func (m oaiMessage) MarshalJSON() ([]byte, error) {
	type alias oaiMessage
	a := alias(m)
	if a.Content == nil && len(a.ToolCalls) == 0 {
		a.Content = ""
	}
	return json.Marshal(a)
}

type oaiToolCall struct {
	Index    int    `json:"index,omitempty"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaiChoice struct {
	Index        int        `json:"index"`
	Message      oaiMessage `json:"message,omitempty"`
	Delta        oaiDelta   `json:"delta,omitempty"`
	FinishReason *string    `json:"finish_reason"`
	Usage        *Usage     `json:"usage,omitempty"`
}

type oaiDelta struct {
	Content          *string      `json:"content,omitempty"`
	ReasoningContent *string      `json:"reasoning_content,omitempty"`
	ToolCalls        []oaiDeltaTC `json:"tool_calls,omitempty"`
}

type oaiDeltaTC struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function,omitempty"`
}

type oaiResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []oaiChoice `json:"choices"`
	Usage   *Usage      `json:"usage,omitempty"`
}

type oaiStreamChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []oaiStreamChoice `json:"choices"`
	Usage   *Usage            `json:"usage,omitempty"`
}

type oaiStreamChoice struct {
	Index        int      `json:"index"`
	Delta        oaiDelta `json:"delta"`
	FinishReason *string  `json:"finish_reason"`
}

// -----------------------------------------------------------------------------
// OpenAI Responses Driver
// -----------------------------------------------------------------------------

type OpenAIResponsesEngine struct {
	adapter *genericAdapter[*ResponseRequest, *ResponseObject, ResponseStream]
	wrapped *responsesMiddlewareEngine
}

func NewOpenAIResponsesProtocol(baseURL, apiKey, defaultModel string) *OpenAIResponsesEngine {
	base := protocolBase{transport: NewHTTPTransport(baseURL, apiKey), defaultModel: defaultModel}
	codec := NewOpenAICodec()
	return &OpenAIResponsesEngine{adapter: newResponsesAdapter(base, codec)}
}

func NewOpenAIResponsesWithTransport(transport Transport, defaultModel string) *OpenAIResponsesEngine {
	base := protocolBase{transport: transport, defaultModel: defaultModel}
	codec := NewOpenAICodec()
	return &OpenAIResponsesEngine{adapter: newResponsesAdapter(base, codec)}
}

func (e *OpenAIResponsesEngine) Create(ctx context.Context, req *ResponseRequest) (*ResponseObject, error) {
	if e.wrapped != nil {
		return e.wrapped.create(ctx, req)
	}
	return e.adapter.generate(ctx, req)
}

func (e *OpenAIResponsesEngine) Stream(ctx context.Context, req *ResponseRequest) (ResponseStream, error) {
	if e.wrapped != nil {
		return e.wrapped.stream(ctx, req)
	}
	return e.adapter.stream(ctx, req)
}

func (e *OpenAIResponsesEngine) Close() {
	if e.wrapped != nil {
		e.wrapped.close()
	}
}

// -----------------------------------------------------------------------------
// OpenAI Responses Stream + Wire Types
// -----------------------------------------------------------------------------

type openAIResponseStream struct {
	resp    *http.Response
	framer  *sseFrameReader
	codec   *OpenAICodec
	current ResponseEvent
	err     error
	done    bool
}

func newOpenAIResponseStream(resp *http.Response, codec *OpenAICodec) *openAIResponseStream {
	return &openAIResponseStream{resp: resp, framer: newSSEFrameReader(resp.Body), codec: codec}
}

func (s *openAIResponseStream) Next() bool {
	if s.done || s.err != nil {
		return false
	}

	for {
		result := consumeSSEFrame(s.framer)
		if result.Done {
			s.done = true
			return false
		}
		if result.Err != nil {
			wrapped := WrapOperationalError("response stream read error", result.Err)
			var llmErr *LLMError
			if errors.As(wrapped, &llmErr) && (llmErr.Code == ErrCodeTimeout || llmErr.Code == ErrCodeCancelled || llmErr.Code == ErrCodeNetwork) {
				s.err = wrapped
			} else {
				s.err = NewLLMError(ErrCodeParse, result.Err.Error(), nil)
			}
			return false
		}
		if s.dispatch(result.Frame) {
			return true
		}
		if s.done || s.err != nil {
			return false
		}
	}
}

func (s *openAIResponseStream) dispatch(frame sseFrame) bool {
	var decoded oaiResponseEvent
	if err := json.Unmarshal(frame.Data, &decoded); err != nil {
		s.err = NewLLMError(ErrCodeParse, "parse response stream event", err)
		return false
	}
	if decoded.Type == "" {
		decoded.Type = frame.Event
	}
	s.current = s.codec.ConvertResponseEvent(decoded)
	return true
}

func (s *openAIResponseStream) Current() ResponseEvent {
	return s.current
}

func (s *openAIResponseStream) Err() error {
	return s.err
}

func (s *openAIResponseStream) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}

type oaiResponseObject struct {
	ID          string           `json:"id"`
	Object      string           `json:"object"`
	Status      string           `json:"status"`
	Model       string           `json:"model"`
	Output      []ResponseOutput `json:"output,omitempty"`
	Usage       *ResponseUsage   `json:"usage,omitempty"`
	CreatedAt   float64          `json:"created_at,omitempty"`
	CompletedAt float64          `json:"completed_at,omitempty"`
}

type oaiResponseEvent struct {
	Type     string             `json:"type"`
	Delta    string             `json:"delta,omitempty"`
	Item     *ResponseOutput    `json:"item,omitempty"`
	Response *oaiResponseObject `json:"response,omitempty"`
}

// codec

type OpenAICodec struct{}

func NewOpenAICodec() *OpenAICodec {
	return &OpenAICodec{}
}

func (c *OpenAICodec) BuildChatRequestBody(req *Request, model string, stream bool) ([]byte, error) {
	reqMap := map[string]any{
		"model":    model,
		"messages": convertMessages(req.Messages),
		"stream":   stream,
	}

	if req.Temperature != nil {
		reqMap["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		reqMap["top_p"] = *req.TopP
	}
	if req.PresencePenalty != nil {
		reqMap["presence_penalty"] = *req.PresencePenalty
	}
	if req.FrequencyPenalty != nil {
		reqMap["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.MaxTokens > 0 {
		reqMap["max_tokens"] = req.MaxTokens
	}
	if len(req.Stop) > 0 {
		reqMap["stop"] = req.Stop
	}
	if stream {
		reqMap["stream_options"] = map[string]any{"include_usage": true}
	}
	if req.JSONMode {
		reqMap["response_format"] = map[string]any{"type": "json_object"}
	}
	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		reqMap["reasoning_effort"] = req.Reasoning.Effort
	}
	if len(req.Tools) > 0 {
		reqMap["tools"] = convertTools(req.Tools)
	}
	for k, v := range req.ExtraBody {
		reqMap[k] = v
	}

	return json.Marshal(reqMap)
}

func (c *OpenAICodec) ParseChatSSEChunk(event string, data []byte) (*Response, bool, error) {
	var chunk oaiStreamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil, false, err
	}

	if len(chunk.Choices) == 0 {
		resp := &Response{}
		if chunk.Usage != nil {
			resp.Usage = *chunk.Usage
		}
		return resp, false, nil
	}

	choice := chunk.Choices[0]
	content := ""
	reasoningContent := ""
	var toolCalls []APIToolCall

	if choice.Delta.Content != nil {
		content = *choice.Delta.Content
	}
	if choice.Delta.ReasoningContent != nil {
		reasoningContent = *choice.Delta.ReasoningContent
	}
	if len(choice.Delta.ToolCalls) > 0 {
		for _, tc := range choice.Delta.ToolCalls {
			toolCalls = append(toolCalls, APIToolCall{
				Index:     tc.Index,
				ID:        tc.ID,
				Type:      tc.Type,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	}

	response := &Response{
		Content:          content,
		ReasoningContent: reasoningContent,
		ToolCalls:        toolCalls,
	}

	isDone := choice.FinishReason != nil && *choice.FinishReason != ""
	if isDone {
		response.FinishReason = *choice.FinishReason
	}

	if chunk.Usage != nil {
		response.Usage = *chunk.Usage
	}

	return response, isDone, nil
}

func (c *OpenAICodec) ConvertChatResponse(oaiResp *oaiResponse) *Response {
	if len(oaiResp.Choices) == 0 {
		return &Response{}
	}

	choice := oaiResp.Choices[0]
	content := extractResponseText(choice.Message.Content)

	var toolCalls []APIToolCall
	for _, tc := range choice.Message.ToolCalls {
		toolCalls = append(toolCalls, APIToolCall{
			ID:        tc.ID,
			Type:      tc.Type,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	finishReason := ""
	if choice.FinishReason != nil {
		finishReason = *choice.FinishReason
	}

	usage := Usage{}
	if oaiResp.Usage != nil {
		usage = *oaiResp.Usage
	}

	return &Response{
		Content:          content,
		ReasoningContent: choice.Message.ReasoningContent,
		FinishReason:     finishReason,
		ToolCalls:        toolCalls,
		Usage:            usage,
	}
}

func (c *OpenAICodec) BuildResponsesRequestBody(req *ResponseRequest, model string, stream bool) ([]byte, error) {
	reqMap := map[string]any{
		"model":  model,
		"input":  convertResponseInputItems(req.Input),
		"stream": stream,
		"store":  req.Store,
	}
	if req.MaxOutputTokens > 0 {
		reqMap["max_output_tokens"] = req.MaxOutputTokens
	}
	if req.Reasoning != nil {
		reasoning := map[string]any{}
		if req.Reasoning.Effort != "" {
			reasoning["effort"] = req.Reasoning.Effort
		}
		if req.Reasoning.Summary != "" {
			reasoning["summary"] = req.Reasoning.Summary
		}
		if len(reasoning) > 0 {
			reqMap["reasoning"] = reasoning
		}
	}
	if len(req.Tools) > 0 {
		reqMap["tools"] = req.Tools
	}
	for key, value := range req.ExtraBody {
		reqMap[key] = value
	}
	return json.Marshal(reqMap)
}

func (c *OpenAICodec) ConvertResponseObject(response *oaiResponseObject) *ResponseObject {
	if response == nil {
		return nil
	}
	output := make([]ResponseOutput, len(response.Output))
	for i, o := range response.Output {
		o.Type = defaultString(o.Type, "message")
		output[i] = o
	}
	return normalizeCompletedResponseObject(&ResponseObject{
		ID:          response.ID,
		Object:      defaultString(response.Object, "response"),
		Status:      response.Status,
		Model:       response.Model,
		Output:      output,
		Usage:       response.Usage,
		CreatedAt:   unixFloatToTime(response.CreatedAt),
		CompletedAt: unixFloatToTime(response.CompletedAt),
	})
}

func (c *OpenAICodec) ConvertResponseEvent(event oaiResponseEvent) ResponseEvent {
	result := ResponseEvent{Type: event.Type, Delta: event.Delta}
	if event.Item != nil {
		event.Item.Type = defaultString(event.Item.Type, "message")
		result.Item = event.Item
	}
	if event.Response != nil {
		result.Response = c.ConvertResponseObject(event.Response)
	}
	return result
}

func isReasoningEffortUnsupported(body []byte) bool {
	msg := strings.ToLower(string(body))
	return strings.Contains(msg, "unrecognized request argument") && strings.Contains(msg, "reasoning_effort")
}
