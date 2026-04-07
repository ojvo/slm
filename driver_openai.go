package slm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAIEngine OpenAI 协议驱动
type OpenAIEngine struct {
	BaseEngine
}

// NewOpenAIProtocol 创建 OpenAI 协议引擎
func NewOpenAIProtocol(baseURL, apiKey, defaultModel string) Engine {
	baseURL = strings.TrimRight(baseURL, "/")
	return &OpenAIEngine{
		BaseEngine: BaseEngine{
			Client: &http.Client{
				Timeout: 120 * time.Second,
			},
			BaseURL:      baseURL,
			APIKey:       apiKey,
			AuthHeader:   "Authorization",
			AuthPrefix:   "Bearer ",
			DefaultModel: defaultModel,
			ExtraHeader:  make(map[string]string),
		},
	}
}

// Generate 生成完整响应
func (e *OpenAIEngine) Generate(ctx context.Context, req *Request) (*Response, error) {
	resp, err := e.doRequest(ctx, req, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, NewLLMError(classifyHTTPError(resp.StatusCode, bodyBytes), fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes)), nil)
	}

	var oaiResp oaiResponse
	if err := json.NewDecoder(resp.Body).Decode(&oaiResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return e.convertResponse(&oaiResp), nil
}

// Stream 流式生成
func (e *OpenAIEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	resp, err := e.doRequest(ctx, req, true)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		resp.Body.Close()
		return nil, NewLLMError(classifyHTTPError(resp.StatusCode, bodyBytes), fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes)), nil)
	}

	return NewSSEReader(resp.Body, e.parseSSEChunk), nil
}

func (e *OpenAIEngine) doRequest(ctx context.Context, req *Request, stream bool) (*http.Response, error) {
	if req == nil {
		return nil, NewLLMError(ErrCodeInvalidConfig, "request is nil", nil)
	}

	model := e.resolveModel(req.Model)
	if model == "" {
		return nil, NewLLMError(ErrCodeInvalidModel, "model is required", nil)
	}

	oaiReq := e.buildRequest(req, model, stream)

	body, err := e.marshalRequest(oaiReq, req.ExtraBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, e.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	e.setHeaders(httpReq)

	resp, err := e.Client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	return resp, nil
}

func (e *OpenAIEngine) resolveModel(model string) string {
	if model != "" {
		return model
	}
	return e.DefaultModel
}

func (e *OpenAIEngine) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set(e.AuthHeader, e.AuthPrefix+e.APIKey)
	}
	for k, v := range e.ExtraHeader {
		req.Header.Set(k, v)
	}
}

func (e *OpenAIEngine) buildRequest(req *Request, model string, stream bool) oaiRequest {
	messages := convertMessages(req.Messages)

	oaiReq := oaiRequest{
		Model:            model,
		Messages:         messages,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		MaxTokens:        req.MaxTokens,
		Stop:             req.Stop,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
		Stream:           stream,
	}

	if stream {
		oaiReq.StreamOptions = &oaiStreamOptions{IncludeUsage: true}
	}

	if req.JSONMode {
		oaiReq.ResponseFormat = &struct {
			Type string `json:"type"`
		}{Type: "json_object"}
	}

	if len(req.Tools) > 0 {
		oaiReq.Tools = convertTools(req.Tools)
	}

	return oaiReq
}

func (e *OpenAIEngine) marshalRequest(oaiReq oaiRequest, extraBody map[string]any) ([]byte, error) {
	reqMap := map[string]any{
		"model":             oaiReq.Model,
		"messages":          oaiReq.Messages,
		"stream":            oaiReq.Stream,
		"temperature":       oaiReq.Temperature,
		"top_p":             oaiReq.TopP,
		"presence_penalty":  oaiReq.PresencePenalty,
		"frequency_penalty": oaiReq.FrequencyPenalty,
	}

	if oaiReq.MaxTokens > 0 {
		reqMap["max_tokens"] = oaiReq.MaxTokens
	}

	if len(oaiReq.Stop) > 0 {
		reqMap["stop"] = oaiReq.Stop
	}
	if oaiReq.StreamOptions != nil {
		reqMap["stream_options"] = oaiReq.StreamOptions
	}
	if oaiReq.ResponseFormat != nil {
		reqMap["response_format"] = oaiReq.ResponseFormat
	}
	if len(oaiReq.Tools) > 0 {
		reqMap["tools"] = oaiReq.Tools
	}

	for k, v := range extraBody {
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
		return &Response{}, false, nil
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
	content := ""
	if choice.Message.Content != nil {
		if s, ok := choice.Message.Content.(string); ok {
			content = s
		}
	}

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

// 内部请求/响应类型
type oaiRequest struct {
	Model            string
	Messages         []oaiMessage
	Temperature      float64
	TopP             float64
	MaxTokens        int
	Stop             []string
	PresencePenalty  float64
	FrequencyPenalty float64
	Stream           bool
	StreamOptions    *oaiStreamOptions
	ResponseFormat   *struct {
		Type string `json:"type"`
	}
	Tools []oaiTool
}

type oaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

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
	Content          any           `json:"content"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
	ToolCalls        []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string        `json:"tool_call_id,omitempty"`
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
	if p.URL != "" {
		img["image_url"] = map[string]any{"url": p.URL}
	} else {
		mime := p.MIME
		if mime == "" {
			mime = "image/png"
		}
		img["image_url"] = map[string]any{"url": "data:" + mime + ";base64," + p.Base64}
	}
	if p.Detail != "" {
		img["detail"] = p.Detail
	}
	return img
}

func convertMessages(messages []Message) []oaiMessage {
	result := make([]oaiMessage, len(messages))
	for i, msg := range messages {
		oaiMsg := oaiMessage{Role: string(msg.Role), ToolCallID: msg.ToolCallID}

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
	bodyStr := string(body)

	switch statusCode {
	case 401, 403:
		return ErrCodeAuth
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
		if strings.Contains(bodyStr, "unknown_model") || strings.Contains(bodyStr, "model_not_found") {
			return ErrCodeInvalidModel
		}
		return ErrCodeInvalidConfig
	default:
		if statusCode >= 500 {
			return ErrCodeNetwork
		}
		return ErrCodeInternal
	}
}
