package slm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
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

type openAIEngine struct {
	adapter *genericAdapter[*Request, *Response, StreamIterator]
}

func newOpenAIEngine(transport Transport, defaultModel string) Engine {
	base := protocolBase{transport: transport, defaultModel: defaultModel}
	return &openAIEngine{adapter: newChatAdapter(base, openaiCodec)}
}

func (e *openAIEngine) Generate(ctx context.Context, req *Request) (*Response, error) {
	return e.adapter.generate(ctx, req)
}

func (e *openAIEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	return e.adapter.stream(ctx, req)
}

func (e *openAIEngine) Capabilities() *ProtocolCapabilities {
	return cloneProtocolCapabilities(openAIChatCapabilitiesTemplate)
}

// -----------------------------------------------------------------------------
// OpenAI Responses Driver
// -----------------------------------------------------------------------------

type openAIResponsesEngine struct {
	adapter *genericAdapter[*ResponseRequest, *ResponseObject, ResponseStream]
}

func newOpenAIResponsesEngine(transport Transport, defaultModel string) ResponsesEngine {
	base := protocolBase{transport: transport, defaultModel: defaultModel}
	return &openAIResponsesEngine{adapter: newResponsesAdapter(base, openaiCodec)}
}

func (e *openAIResponsesEngine) Create(ctx context.Context, req *ResponseRequest) (*ResponseObject, error) {
	return e.adapter.generate(ctx, req)
}

func (e *openAIResponsesEngine) Stream(ctx context.Context, req *ResponseRequest) (ResponseStream, error) {
	return e.adapter.stream(ctx, req)
}

func (e *openAIResponsesEngine) Close() error {
	return nil
}

func (e *openAIResponsesEngine) Capabilities() *ProtocolCapabilities {
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
	core      sseIteratorCore
	resp      *http.Response
	codec     *openAICodec
	current   ResponseEvent
	closeOnce sync.Once
}

func newOpenAIResponseStream(resp *http.Response, codec *openAICodec) *openAIResponseStream {
	return &openAIResponseStream{
		resp: resp,
		core: sseIteratorCore{
			framer: newSSEFrameReader(resp.Body),
			wrapError: func(err error) error {
				wrapped := WrapOperationalError("response stream read error", err)
				var llmErr *LLMError
				if errors.As(wrapped, &llmErr) && (llmErr.Code == ErrCodeTimeout || llmErr.Code == ErrCodeCancelled || llmErr.Code == ErrCodeNetwork) {
					return wrapped
				}
				return NewLLMError(ErrCodeParse, err.Error(), nil)
			},
		},
		codec: codec,
	}
}

func (s *openAIResponseStream) Next() bool {
	return s.core.Next(s.dispatch)
}

func (s *openAIResponseStream) dispatch(frame sseFrame) sseDispatchResult {
	var decoded oaiResponseEvent
	if err := json.Unmarshal(frame.Data, &decoded); err != nil {
		return sseDispatchResult{Err: NewLLMError(ErrCodeParse, "parse response stream event", err)}
	}
	if decoded.Type == "" {
		decoded.Type = frame.Event
	}
	s.current = s.codec.ConvertResponseEvent(decoded)
	return sseDispatchResult{Yield: true}
}

func (s *openAIResponseStream) Current() ResponseEvent {
	return s.current
}

func (s *openAIResponseStream) Err() error {
	return s.core.Err()
}

func (s *openAIResponseStream) Close() error {
	var err error
	s.closeOnce.Do(func() {
		if s.resp != nil && s.resp.Body != nil {
			err = s.resp.Body.Close()
		}
	})
	return err
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

var openaiCodec = &openAICodec{}

type openAICodec struct{}

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

func (c *openAICodec) BuildChatRequestBody(req *Request, model string, stream bool) ([]byte, error) {
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

func (c *openAICodec) BuildResponsesRequestBody(req *ResponseRequest, model string, stream bool) ([]byte, error) {
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

func (c *openAICodec) ParseChatSSEChunk(event string, data []byte) (*Response, bool, error) {
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

func (c *openAICodec) ConvertChatResponse(oaiResp *oaiResponse) *Response {
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

func (c *openAICodec) ConvertResponseObject(response *oaiResponseObject) *ResponseObject {
	if response == nil {
		return nil
	}
	output := make([]ResponseOutput, len(response.Output))
	for i, o := range response.Output {
		o.Type = DefaultString(o.Type, "message")
		output[i] = o
	}
	return normalizeCompletedResponseObject(&ResponseObject{
		ID:          response.ID,
		Object:      DefaultString(response.Object, "response"),
		Status:      response.Status,
		Model:       response.Model,
		Output:      output,
		Usage:       response.Usage,
		CreatedAt:   unixFloatToTime(response.CreatedAt),
		CompletedAt: unixFloatToTime(response.CompletedAt),
	})
}

func (c *openAICodec) ConvertResponseEvent(event oaiResponseEvent) ResponseEvent {
	result := ResponseEvent{Type: event.Type, Delta: event.Delta}
	if event.Item != nil {
		event.Item.Type = DefaultString(event.Item.Type, "message")
		result.Item = event.Item
	}
	if event.Response != nil {
		result.Response = c.ConvertResponseObject(event.Response)
	}
	return result
}

func convertMessages(messages []Message) []oaiMessage {
	result := make([]oaiMessage, len(messages))
	for i, msg := range messages {
		oaiMsg := oaiMessage{Role: string(msg.Role), Name: msg.Name, ToolCallID: msg.ToolCallID}

		switch len(msg.Content) {
		case 0:
			oaiMsg.Content = nil
		case 1:
			switch p := msg.Content[0].(type) {
			case TextPart:
				oaiMsg.Content = string(p)
			case ImagePart:
				if img := buildImageContent(p); img != nil {
					oaiMsg.Content = []map[string]any{img}
				}
			}
		default:
			var parts []map[string]any
			for _, part := range msg.Content {
				switch p := part.(type) {
				case TextPart:
					parts = append(parts, map[string]any{"type": "text", "text": string(p)})
				case ImagePart:
					if img := buildImageContent(p); img != nil {
						parts = append(parts, img)
					}
				}
			}
			if len(parts) > 0 {
				oaiMsg.Content = parts
			}
		}

		if len(msg.ToolCalls) > 0 {
			oaiMsg.ToolCalls = make([]oaiToolCall, len(msg.ToolCalls))
			for k, tc := range msg.ToolCalls {
				oaiMsg.ToolCalls[k] = oaiToolCall{
					Index: k,
					ID:    tc.ID,
					Type:  tc.Type,
					Function: struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					}{Name: tc.Name, Arguments: tc.Arguments},
				}
			}
		}

		result[i] = oaiMsg
	}
	return result
}

func convertTools(tools []Tool) []oaiTool {
	result := make([]oaiTool, len(tools))
	for i, t := range tools {
		result[i] = oaiTool{
			Type: "function",
			Function: oaiFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		}
	}
	return result
}
