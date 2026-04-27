package slm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

var openAIChatCapabilitiesTemplate = ProtocolCapabilities{
	SupportedParameters: map[string]ParameterRange{
		"reasoning_effort": {
			Values: []string{"low", "medium", "high"},
		},
	},
	ParameterMapping: map[string]string{
		"reasoning_effort": "reasoning_effort",
	},
	ConflictingParameters: [][]string{},
	Description:           "OpenAI Chat Completions API (supports gpt-4, gpt-4o, o1 models)",
}

var openAIResponsesCapabilitiesTemplate = ProtocolCapabilities{
	SupportedParameters:   map[string]ParameterRange{},
	ParameterMapping:      map[string]string{},
	ConflictingParameters: [][]string{},
	Description:           "OpenAI Responses API (for reasoning-focused completions)",
}

// -------------------------------------------------------------------------
// OpenAI Chat Completions Driver
// -----------------------------------------------------------------------------

// OpenAIEngine OpenAI 协议驱动。
// 只负责 OpenAI 协议的编解码，HTTP 通信和认证由 Transport 实现。

type OpenAIEngine struct {
	adapter *genericAdapter[*Request, *Response, StreamIterator]
}

// NewOpenAIProtocol 创建 OpenAI 协议引擎（使用标准 HTTP 传输）。
func NewOpenAIProtocol(baseURL, apiKey, defaultModel string) Engine {
	return NewOpenAIEngine(NewHTTPTransport(baseURL, apiKey), defaultModel)
}

// NewOpenAIEngine 创建 OpenAI 协议引擎（使用自定义 Transport）。
func NewOpenAIEngine(transport Transport, defaultModel string) Engine {
	base := protocolBase{transport: transport, defaultModel: defaultModel}
	return &OpenAIEngine{adapter: newChatAdapter(base, openaiCodec)}
}

// 已弃用：使用 NewOpenAIEngine 代替。
func NewOpenAIWithTransport(transport Transport, defaultModel string) Engine {
	return NewOpenAIEngine(transport, defaultModel)
}

func (e *OpenAIEngine) Generate(ctx context.Context, req *Request) (*Response, error) {
	return e.adapter.generate(ctx, req)
}

func (e *OpenAIEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	return e.adapter.stream(ctx, req)
}

// Capabilities returns the protocol capabilities supported by OpenAI Chat Completions API.
func (e *OpenAIEngine) Capabilities() *ProtocolCapabilities {
	return cloneProtocolCapabilities(openAIChatCapabilitiesTemplate)
}

// -----------------------------------------------------------------------------
// OpenAI Responses Driver
// -----------------------------------------------------------------------------

type OpenAIResponsesEngine struct {
	adapter *genericAdapter[*ResponseRequest, *ResponseObject, ResponseStream]
	wrapped *responsesMiddlewareEngine
}

func NewOpenAIResponsesProtocol(baseURL, apiKey, defaultModel string) *OpenAIResponsesEngine {
	return NewOpenAIResponsesEngine(NewHTTPTransport(baseURL, apiKey), defaultModel)
}

// NewOpenAIResponsesEngine 创建 Responses API 引擎（使用自定义 Transport）。
func NewOpenAIResponsesEngine(transport Transport, defaultModel string) *OpenAIResponsesEngine {
	base := protocolBase{transport: transport, defaultModel: defaultModel}
	return &OpenAIResponsesEngine{adapter: newResponsesAdapter(base, openaiCodec)}
}

// 已弃用：使用 NewOpenAIResponsesEngine 代替。
func NewOpenAIResponsesWithTransport(transport Transport, defaultModel string) *OpenAIResponsesEngine {
	return NewOpenAIResponsesEngine(transport, defaultModel)
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

// Capabilities returns the protocol capabilities supported by OpenAI Responses API.
func (e *OpenAIResponsesEngine) Capabilities() *ProtocolCapabilities {
	return cloneProtocolCapabilities(openAIResponsesCapabilitiesTemplate)
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

// request

// openaiCodec 是单个共享实例，因为 OpenAICodec 是 stateless。
var openaiCodec = &OpenAICodec{}

// OpenAICodec implements request encoding for OpenAI-compatible APIs.
type OpenAICodec struct{}

// NewOpenAICodec 返回共享的 OpenAI codec 实例。
// 已弃用：直接使用 openaiCodec 全局变量。
func NewOpenAICodec() *OpenAICodec {
	return openaiCodec
}

func buildResponsesReasoning(reasoning *ResponseReasoning) map[string]any {
	if reasoning == nil {
		return nil
	}
	result := map[string]any{}
	putStringField(result, "effort", reasoning.Effort)
	putStringField(result, "summary", reasoning.Summary)
	if len(result) == 0 {
		return nil
	}
	return result
}

func (c *OpenAICodec) BuildChatRequestBody(req *Request, model string, stream bool) ([]byte, error) {
	reqMap := map[string]any{
		"model":    model,
		"messages": convertMessages(req.Messages),
		"stream":   stream,
	}

	putFloat64PtrField(reqMap, "temperature", req.Temperature)
	putFloat64PtrField(reqMap, "top_p", req.TopP)
	putFloat64PtrField(reqMap, "presence_penalty", req.PresencePenalty)
	putFloat64PtrField(reqMap, "frequency_penalty", req.FrequencyPenalty)
	putPositiveIntField(reqMap, "max_tokens", req.MaxTokens)
	putSliceField(reqMap, "stop", req.Stop)
	if stream {
		reqMap["stream_options"] = map[string]any{"include_usage": true}
	}
	if req.JSONMode {
		reqMap["response_format"] = map[string]any{"type": "json_object"}
	}
	if req.Reasoning != nil {
		putStringField(reqMap, "reasoning_effort", req.Reasoning.Effort)
	}
	putAnyField(reqMap, "tools", convertTools(req.Tools))
	mergeFields(reqMap, req.Capabilities)

	return json.Marshal(reqMap)
}

func (c *OpenAICodec) BuildResponsesRequestBody(req *ResponseRequest, model string, stream bool) ([]byte, error) {
	reqMap := map[string]any{
		"model":  model,
		"input":  convertResponseInputItems(req.Input),
		"stream": stream,
		"store":  req.Store,
	}
	putPositiveIntField(reqMap, "max_output_tokens", req.MaxOutputTokens)
	putAnyField(reqMap, "reasoning", buildResponsesReasoning(req.Reasoning))
	putAnyField(reqMap, "tools", req.Tools)
	mergeFields(reqMap, req.Capabilities)
	return json.Marshal(reqMap)
}

// response

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
