package slm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// OpenAIEngine OpenAI 协议驱动。
// 只负责 OpenAI 协议的编解码，HTTP 通信和认证由 Transport 实现。
type OpenAIEngine struct {
	transport    Transport
	defaultModel string
	capabilities *CapabilityNegotiationOptions
	logger       Logger
}

type OpenAIOptions struct {
	DefaultModel string
	Capabilities *CapabilityNegotiationOptions
	Logger       Logger
}

// NewOpenAIProtocol 创建 OpenAI 协议引擎（使用标准 HTTP 传输）。
func NewOpenAIProtocol(baseURL, apiKey, defaultModel string) Engine {
	return &OpenAIEngine{
		transport:    NewHTTPTransport(baseURL, apiKey),
		defaultModel: defaultModel,
	}
}

// NewOpenAIProtocolWithOptions creates an OpenAI protocol engine with optional capability negotiation.
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

// NewOpenAIWithTransportAndOptions creates an OpenAI protocol engine with optional capability negotiation.
func NewOpenAIWithTransportAndOptions(transport Transport, opts OpenAIOptions) Engine {
	engine := &OpenAIEngine{transport: transport, defaultModel: opts.DefaultModel, logger: opts.Logger}
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
	requestID := GetRequestID(ctx)
	start := time.Now()
	e.logDebug("LLM request start", append([]any{"request_id", requestID}, RequestDiagnosticFields(effectiveReq)...)...)
	resp, err := e.doRequest(ctx, effectiveReq, false)
	if err != nil {
		e.logError("LLM request failed", start, requestID, effectiveReq, err)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		err := NewLLMError(classifyHTTPError(resp.StatusCode, bodyBytes), fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes)), nil)
		e.logError("LLM request failed", start, requestID, effectiveReq, err)
		return nil, err
	}

	var oaiResp oaiResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxResponseSize)).Decode(&oaiResp); err != nil {
		wrapped := fmt.Errorf("decode response: %w", err)
		e.logError("LLM request failed", start, requestID, effectiveReq, wrapped)
		return nil, wrapped
	}

	result := e.convertResponse(&oaiResp)
	e.logDebug("LLM request completed", "duration", time.Since(start), "request_id", requestID, "model", effectiveReq.Model, "finish_reason", result.FinishReason, "tokens", result.Usage.TotalTokens)
	return result, nil
}

// Stream 流式生成
func (e *OpenAIEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	effectiveReq := normalizeRequestForOperation(req, e.defaultModel, true)
	requestID := GetRequestID(ctx)
	start := time.Now()
	e.logDebug("LLM stream start", append([]any{"request_id", requestID}, RequestDiagnosticFields(effectiveReq)...)...)
	resp, err := e.doRequest(ctx, effectiveReq, true)
	if err != nil {
		e.logError("LLM stream failed", start, requestID, effectiveReq, err)
		return nil, err
	}

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		resp.Body.Close()
		err := NewLLMError(classifyHTTPError(resp.StatusCode, bodyBytes), fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes)), nil)
		e.logError("LLM stream failed", start, requestID, effectiveReq, err)
		return nil, err
	}

	e.logDebug("LLM stream connected", "duration", time.Since(start), "request_id", requestID, "model", effectiveReq.Model)
	return &loggingOpenAIStreamIterator{inner: NewSSEReader(resp.Body, e.parseSSEChunk), logger: e.logger, start: start, requestID: requestID, request: cloneRequest(effectiveReq)}, nil
}

func (e *OpenAIEngine) logDebug(msg string, args ...any) {
	if e.logger == nil {
		return
	}
	e.logger.Debug(msg, args...)
}

func (e *OpenAIEngine) logError(msg string, start time.Time, requestID string, req *Request, err error) {
	if e.logger == nil {
		return
	}
	args := append([]any{"duration", time.Since(start), "request_id", requestID}, RequestDiagnosticFields(req)...)
	args = append(args, ErrorDiagnosticFields(err)...)
	e.logger.Error(msg, args...)
}

type loggingOpenAIStreamIterator struct {
	inner     StreamIterator
	logger    Logger
	start     time.Time
	requestID string
	request   *Request
	closeOnce sync.Once
	closeErr  error
}

func (l *loggingOpenAIStreamIterator) Next() bool {
	ok := l.inner.Next()
	if !ok {
		_ = l.Close()
	}
	return ok
}

func (l *loggingOpenAIStreamIterator) Chunk() []byte       { return l.inner.Chunk() }
func (l *loggingOpenAIStreamIterator) Text() string        { return l.inner.Text() }
func (l *loggingOpenAIStreamIterator) FullText() string    { return l.inner.FullText() }
func (l *loggingOpenAIStreamIterator) Err() error          { return l.inner.Err() }
func (l *loggingOpenAIStreamIterator) Usage() *Usage       { return l.inner.Usage() }
func (l *loggingOpenAIStreamIterator) Response() *Response { return l.inner.Response() }

func (l *loggingOpenAIStreamIterator) Close() error {
	l.closeOnce.Do(func() {
		l.closeErr = l.inner.Close()
		if l.logger == nil {
			return
		}
		duration := time.Since(l.start)
		if l.closeErr != nil {
			args := append([]any{"duration", duration, "request_id", l.requestID}, RequestDiagnosticFields(l.request)...)
			args = append(args, ErrorDiagnosticFields(l.closeErr)...)
			l.logger.Error("LLM stream closed with error", args...)
			return
		}
		if err := l.inner.Err(); err != nil {
			args := append([]any{"duration", duration, "request_id", l.requestID}, RequestDiagnosticFields(l.request)...)
			args = append(args, ErrorDiagnosticFields(err)...)
			l.logger.Error("LLM stream closed with error", args...)
			return
		}
		args := []any{"duration", duration, "request_id", l.requestID}
		if usage := l.inner.Usage(); usage != nil {
			args = append(args, "tokens", usage.TotalTokens)
		}
		l.logger.Debug("LLM stream completed", args...)
	})
	return l.closeErr
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

	model := e.resolveModel(req.Model)
	if model == "" {
		return nil, NewLLMError(ErrCodeInvalidModel, "model is required", nil)
	}

	body, err := e.buildRequestBody(req, model, stream)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	headers := map[string]string{}
	if stream {
		headers["Accept"] = "text/event-stream"
	}

	return e.transport.Do(ctx, http.MethodPost, "/chat/completions", headers, body)
}

func (e *OpenAIEngine) resolveModel(model string) string {
	if model != "" {
		return model
	}
	return e.defaultModel
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

// 内部请求/响应类型
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

// APIToolCall API 工具调用结构
type APIToolCall struct {
	Index     int    `json:"index,omitempty"`
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool 工具定义
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

func classifyHTTPError(statusCode int, body []byte) ErrorCode {
	bodyStr := strings.ToLower(string(body))

	switch statusCode {
	case 401, 403:
		return ErrCodeAuth
	case 408:
		return ErrCodeTimeout
	case 429:
		return ErrCodeRateLimit
	case 503:
		return ErrCodeOverloaded
	case 400:
		if strings.Contains(bodyStr, "context_length") || strings.Contains(bodyStr, "too long") {
			return ErrCodeContextTooLong
		}
		if strings.Contains(bodyStr, "content_filter") || strings.Contains(bodyStr, "content policy") {
			return ErrCodeContentFilter
		}
		if strings.Contains(bodyStr, "unsupported_api_for_model") || strings.Contains(bodyStr, "not supported via responses api") || strings.Contains(bodyStr, "not supported via chat completions api") {
			return ErrCodeUnsupportedCapability
		}
		if strings.Contains(bodyStr, "unknown_model") || strings.Contains(bodyStr, "model_not_found") {
			return ErrCodeInvalidModel
		}
		return ErrCodeInvalidConfig
	case 404:
		return ErrCodeInvalidConfig
	default:
		if statusCode >= 500 {
			return ErrCodeServer
		}
		return ErrCodeInternal
	}
}
