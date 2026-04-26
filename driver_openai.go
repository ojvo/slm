package slm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// -----------------------------------------------------------------------------
// OpenAI Chat Completions Driver
// -----------------------------------------------------------------------------

// OpenAIEngine OpenAI 协议驱动。
// 只负责 OpenAI 协议的编解码，HTTP 通信和认证由 Transport 实现。
type OpenAIEngine struct {
	transport    Transport
	defaultModel string
	capabilities *CapabilityNegotiationOptions
}

type OpenAIOptions struct {
	DefaultModel string
	// Capabilities enables protocol-level negotiation.
	//
	// Preferred default path: configure capability negotiation through
	// ApplyStandardMiddleware (or Config builders) so negotiation has a
	// single middleware entry point.
	//
	// Compatibility path: use this field when constructing a protocol engine
	// directly without the standard middleware chain.
	Capabilities *CapabilityNegotiationOptions
}

// NewOpenAIProtocol 创建 OpenAI 协议引擎（使用标准 HTTP 传输）。
func NewOpenAIProtocol(baseURL, apiKey, defaultModel string) Engine {
	return &OpenAIEngine{
		transport:    NewHTTPTransport(baseURL, apiKey),
		defaultModel: defaultModel,
	}
}

// NewOpenAIProtocolWithOptions creates an OpenAI protocol engine with optional
// protocol-level capability negotiation.
//
// For most applications, prefer middleware-first negotiation via
// ApplyStandardMiddleware or Config builders.
func NewOpenAIProtocolWithOptions(baseURL, apiKey string, opts OpenAIOptions) Engine {
	return NewOpenAIWithTransportAndOptions(NewHTTPTransport(baseURL, apiKey), opts)
}

// NewOpenAIWithTransport 创建使用自定义 Transport 的 OpenAI 协议引擎。
// 这使得 OpenAIEngine 可以搭配不同的传输层，例如 CopilotTransport。
//
// 使用示例：
//
//	// HTTP 直连
//	engine := slm.NewOpenAIWithTransport(slm.NewHTTPTransport(url, key), "gpt-4o")
//
//	// Copilot 传输
//	client := copilot.NewClient()
//	client.Auth(ctx)
//	engine := slm.NewOpenAIWithTransport(copilot.NewTransport(client), "gpt-4o")
func NewOpenAIWithTransport(transport Transport, defaultModel string) Engine {
	return &OpenAIEngine{
		transport:    transport,
		defaultModel: defaultModel,
	}
}

// NewOpenAIWithTransportAndOptions creates an OpenAI protocol engine with
// optional protocol-level capability negotiation.
//
// For most applications, prefer middleware-first negotiation via
// ApplyStandardMiddleware or Config builders.
func NewOpenAIWithTransportAndOptions(transport Transport, opts OpenAIOptions) Engine {
	engine := &OpenAIEngine{transport: transport, defaultModel: opts.DefaultModel}
	if opts.Capabilities != nil {
		clone := *opts.Capabilities
		if clone.DefaultModel == "" {
			clone.DefaultModel = opts.DefaultModel
		}
		engine.capabilities = &clone
	}
	return engine
}

const maxResponseSize = 50 * 1024 * 1024 // 50MB

// Generate 生成完整响应
func (e *OpenAIEngine) Generate(ctx context.Context, req *Request) (*Response, error) {
	effectiveReq := normalizeRequestForOperation(req, e.defaultModel, false)
	resp, err := e.doRequest(ctx, effectiveReq, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, llmErrorFromResponse(resp)
	}

	var oaiResp oaiResponse
	if err := decodeJSONResponse(resp, &oaiResp); err != nil {
		return nil, err
	}

	return e.convertResponse(&oaiResp), nil
}

// Stream 流式生成
func (e *OpenAIEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	effectiveReq := normalizeRequestForOperation(req, e.defaultModel, true)
	resp, err := e.doRequest(ctx, effectiveReq, true)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		return nil, llmErrorFromResponse(resp)
	}

	return NewSSEReader(resp.Body, e.parseSSEChunk), nil
}

func (e *OpenAIEngine) doRequest(ctx context.Context, req *Request, stream bool) (*http.Response, error) {
	if req == nil {
		return nil, NewLLMError(ErrCodeInvalidConfig, "request is nil", nil)
	}

	if len(req.Messages) == 0 {
		return nil, NewLLMError(ErrCodeInvalidConfig, "messages is required", nil)
	}
	if e.capabilities != nil && e.capabilities.Resolver != nil {
		var err error
		ctx, req, err = NegotiateRequestCapabilities(ctx, req, *e.capabilities)
		if err != nil {
			return nil, err
		}
	}

	model := resolveRequestedModel(req.Model, e.defaultModel)
	if model == "" {
		return nil, NewLLMError(ErrCodeInvalidModel, "model is required", nil)
	}

	body, err := e.buildRequestBody(req, model, stream)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	headers := streamRequestHeaders(stream)

	resp, err := e.transport.Do(ctx, http.MethodPost, "/chat/completions", headers, body)
	if err != nil {
		return nil, err
	}

	if req.Reasoning != nil && resp.StatusCode == http.StatusBadRequest {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()

		if isReasoningEffortUnsupported(bodyBytes) {
			fallbackReq := cloneRequestShallow(req)
			fallbackReq.Reasoning = nil

			fallbackBody, err := e.buildRequestBody(fallbackReq, model, stream)
			if err != nil {
				return nil, fmt.Errorf("marshal fallback request: %w", err)
			}

			return e.transport.Do(ctx, http.MethodPost, "/chat/completions", headers, fallbackBody)
		}

		resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		resp.ContentLength = int64(len(bodyBytes))
	}

	return resp, nil
}

func isReasoningEffortUnsupported(body []byte) bool {
	msg := strings.ToLower(string(body))
	return strings.Contains(msg, "unrecognized request argument") && strings.Contains(msg, "reasoning_effort")
}

func (e *OpenAIEngine) buildRequestBody(req *Request, model string, stream bool) ([]byte, error) {
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

func (e *OpenAIEngine) parseSSEChunk(event string, data []byte) (*Response, bool, error) {
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

func (e *OpenAIEngine) convertResponse(oaiResp *oaiResponse) *Response {
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

func extractResponseText(content any) string {
	switch value := content.(type) {
	case nil:
		return ""
	case string:
		return value
	case []any:
		var builder strings.Builder
		for _, item := range value {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			partType, _ := part["type"].(string)
			switch partType {
			case "", "text", "output_text":
				if text, ok := part["text"].(string); ok {
					builder.WriteString(text)
				}
			}
		}
		return builder.String()
	default:
		return ""
	}
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

// 辅助函数
func buildImageContent(p ImagePart) map[string]any {
	if p.URL == "" && p.Base64 == "" {
		return nil
	}
	img := map[string]any{"type": "image_url"}
	imageURL := map[string]any{}
	if p.URL != "" {
		imageURL["url"] = p.URL
	} else {
		mime := p.MIME
		if mime == "" {
			mime = "image/png"
		}
		imageURL["url"] = "data:" + mime + ";base64," + p.Base64
	}
	if p.Detail != "" {
		imageURL["detail"] = p.Detail
	}
	img["image_url"] = imageURL
	return img
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

// -----------------------------------------------------------------------------
// OpenAI Responses Driver
// -----------------------------------------------------------------------------

type OpenAIResponsesEngine struct {
	transport    Transport
	defaultModel string
	capabilities *CapabilityNegotiationOptions
}

type OpenAIResponsesOptions struct {
	DefaultModel string
	// Capabilities enables protocol-level negotiation.
	//
	// Preferred default path: configure capability negotiation through
	// ApplyStandardMiddleware (or Config builders) so negotiation has a
	// single middleware entry point.
	//
	// Compatibility path: use this field when constructing a responses engine
	// directly without the standard middleware chain.
	Capabilities *CapabilityNegotiationOptions
}

func NewOpenAIResponsesProtocol(baseURL, apiKey, defaultModel string) *OpenAIResponsesEngine {
	return &OpenAIResponsesEngine{
		transport:    NewHTTPTransport(baseURL, apiKey),
		defaultModel: defaultModel,
	}
}

func NewOpenAIResponsesProtocolWithOptions(baseURL, apiKey string, opts OpenAIResponsesOptions) *OpenAIResponsesEngine {
	return NewOpenAIResponsesWithTransportAndOptions(NewHTTPTransport(baseURL, apiKey), opts)
}

func NewOpenAIResponsesWithTransport(transport Transport, defaultModel string) *OpenAIResponsesEngine {
	return &OpenAIResponsesEngine{transport: transport, defaultModel: defaultModel}
}

func NewOpenAIResponsesWithTransportAndOptions(transport Transport, opts OpenAIResponsesOptions) *OpenAIResponsesEngine {
	engine := &OpenAIResponsesEngine{transport: transport, defaultModel: opts.DefaultModel}
	if opts.Capabilities != nil {
		clone := *opts.Capabilities
		if clone.DefaultModel == "" {
			clone.DefaultModel = opts.DefaultModel
		}
		engine.capabilities = &clone
	}
	return engine
}

func (e *OpenAIResponsesEngine) Create(ctx context.Context, req *ResponseRequest) (*ResponseObject, error) {
	effectiveReq := normalizeResponseRequestForOperation(req, e.defaultModel, false)
	resp, err := e.doRequest(ctx, effectiveReq, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, llmErrorFromResponse(resp)
	}

	var decoded oaiResponseObject
	if err := decodeJSONResponse(resp, &decoded); err != nil {
		return nil, err
	}
	return convertResponseObject(&decoded), nil
}

func (e *OpenAIResponsesEngine) Stream(ctx context.Context, req *ResponseRequest) (ResponseStream, error) {
	effectiveReq := normalizeResponseRequestForOperation(req, e.defaultModel, true)
	resp, err := e.doRequest(ctx, effectiveReq, true)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		return nil, llmErrorFromResponse(resp)
	}
	return newOpenAIResponseStream(resp), nil
}

func (e *OpenAIResponsesEngine) doRequest(ctx context.Context, req *ResponseRequest, stream bool) (*http.Response, error) {
	if req == nil {
		return nil, NewLLMError(ErrCodeInvalidConfig, "request is nil", nil)
	}
	if len(req.Input) == 0 {
		return nil, NewLLMError(ErrCodeInvalidConfig, "input is required", nil)
	}
	if e.capabilities != nil && e.capabilities.Resolver != nil {
		var err error
		ctx, req, err = NegotiateResponseCapabilities(ctx, req, *e.capabilities)
		if err != nil {
			return nil, err
		}
	}
	model := resolveRequestedModel(req.Model, e.defaultModel)
	if model == "" {
		return nil, NewLLMError(ErrCodeInvalidModel, "model is required", nil)
	}

	body, err := e.buildRequestBody(req, model, stream)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	headers := streamRequestHeaders(stream)
	return e.transport.Do(ctx, http.MethodPost, "/responses", headers, body)
}

func (e *OpenAIResponsesEngine) buildRequestBody(req *ResponseRequest, model string, stream bool) ([]byte, error) {
	input := make([]map[string]any, 0, len(req.Input))
	for _, item := range req.Input {
		input = append(input, map[string]any{
			"role":    item.Role,
			"content": item.Content,
		})
	}

	reqMap := map[string]any{
		"model":  model,
		"input":  input,
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
		tools := make([]any, 0, len(req.Tools))
		for index, tool := range req.Tools {
			var value any
			if err := json.Unmarshal(tool, &value); err != nil {
				return nil, fmt.Errorf("tools[%d]: %w", index, err)
			}
			tools = append(tools, value)
		}
		reqMap["tools"] = tools
	}
	for key, value := range req.ExtraBody {
		reqMap[key] = value
	}
	return json.Marshal(reqMap)
}

// -----------------------------------------------------------------------------
// OpenAI Responses Stream + Wire Types
// -----------------------------------------------------------------------------

type openAIResponseStream struct {
	resp    *http.Response
	framer  *sseFrameReader
	current ResponseEvent
	err     error
	done    bool
}

func newOpenAIResponseStream(resp *http.Response) *openAIResponseStream {
	return &openAIResponseStream{resp: resp, framer: newSSEFrameReader(resp.Body)}
}

func (s *openAIResponseStream) Next() bool {
	if s.done || s.err != nil {
		return false
	}

	for {
		frame, ok, err := s.framer.Next()
		if err != nil {
			if err == io.EOF {
				s.done = true
				return false
			}
			wrapped := WrapOperationalError("response stream read error", err)
			var llmErr *LLMError
			if errors.As(wrapped, &llmErr) && (llmErr.Code == ErrCodeTimeout || llmErr.Code == ErrCodeCancelled || llmErr.Code == ErrCodeNetwork) {
				s.err = wrapped
			} else {
				s.err = NewLLMError(ErrCodeParse, err.Error(), nil)
			}
			return false
		}
		if !ok {
			s.done = true
			return false
		}
		if frame.Done {
			s.done = true
			return false
		}
		if s.dispatch(frame) {
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
	s.current = convertResponseEvent(decoded)
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
	ID          string              `json:"id"`
	Object      string              `json:"object"`
	Status      string              `json:"status"`
	Model       string              `json:"model"`
	Output      []oaiResponseOutput `json:"output,omitempty"`
	Usage       *ResponseUsage      `json:"usage,omitempty"`
	CreatedAt   float64             `json:"created_at,omitempty"`
	CompletedAt float64             `json:"completed_at,omitempty"`
}

type oaiResponseOutput struct {
	Type    string                  `json:"type"`
	ID      string                  `json:"id,omitempty"`
	Role    string                  `json:"role,omitempty"`
	Status  string                  `json:"status,omitempty"`
	Content []ResponseOutputContent `json:"content,omitempty"`
	Summary []ResponseOutputContent `json:"summary,omitempty"`
}

type oaiResponseEvent struct {
	Type     string             `json:"type"`
	Delta    string             `json:"delta,omitempty"`
	Item     *oaiResponseOutput `json:"item,omitempty"`
	Response *oaiResponseObject `json:"response,omitempty"`
}

func convertResponseObject(response *oaiResponseObject) *ResponseObject {
	if response == nil {
		return nil
	}
	result := &ResponseObject{
		ID:          response.ID,
		Object:      defaultResponseString(response.Object, "response"),
		Status:      response.Status,
		Model:       response.Model,
		Output:      make([]ResponseOutput, 0, len(response.Output)),
		Usage:       response.Usage,
		CreatedAt:   unixFloatToTime(response.CreatedAt),
		CompletedAt: unixFloatToTime(response.CompletedAt),
	}
	for _, output := range response.Output {
		result.Output = append(result.Output, ResponseOutput{
			Type:    defaultResponseString(output.Type, "message"),
			ID:      output.ID,
			Role:    output.Role,
			Status:  output.Status,
			Content: append([]ResponseOutputContent(nil), output.Content...),
			Summary: append([]ResponseOutputContent(nil), output.Summary...),
		})
	}
	return result
}

func convertResponseEvent(event oaiResponseEvent) ResponseEvent {
	result := ResponseEvent{Type: event.Type, Delta: event.Delta}
	if event.Item != nil {
		item := ResponseOutput{
			Type:    defaultResponseString(event.Item.Type, "message"),
			ID:      event.Item.ID,
			Role:    event.Item.Role,
			Status:  event.Item.Status,
			Content: append([]ResponseOutputContent(nil), event.Item.Content...),
			Summary: append([]ResponseOutputContent(nil), event.Item.Summary...),
		}
		result.Item = &item
	}
	if event.Response != nil {
		result.Response = convertResponseObject(event.Response)
	}
	return result
}
