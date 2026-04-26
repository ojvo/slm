package slm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const maxHTTPErrorBodySize = 2048

// NewTextMessage 创建文本消息
func NewTextMessage(role Role, text string) Message {
	return Message{
		Role:    role,
		Content: []ContentPart{TextPart(text)},
	}
}

// NewToolMessage 创建工具响应消息
func NewToolMessage(toolCallID, content string) Message {
	return Message{
		Role:       RoleTool,
		Content:    []ContentPart{TextPart(content)},
		ToolCallID: toolCallID,
	}
}

// NewAssistantMessage 创建助手消息（支持工具调用）
func NewAssistantMessage(content string, toolCalls ...APIToolCall) Message {
	msg := Message{
		Role:    RoleAssistant,
		Content: []ContentPart{TextPart(content)},
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	return msg
}

// NewImageMessage 创建图片消息
func NewImageMessage(imageURL string) Message {
	return Message{
		Role:    RoleUser,
		Content: []ContentPart{ImagePart{URL: imageURL}},
	}
}

// Float64 returns a pointer to the given float64 value.
// Use this to set optional Request fields like Temperature:
//
//	req.Temperature = slm.Float64(0.0)  // explicitly set to 0
//	req.TopP = slm.Float64(0.9)
func Float64(v float64) *float64 { return &v }

func cloneResponseRequest(req *ResponseRequest) *ResponseRequest {
	if req == nil {
		return nil
	}
	clone := *req
	clone.Input = append([]ResponseInputItem(nil), req.Input...)
	clone.Tools = append([]json.RawMessage(nil), req.Tools...)
	if req.ExtraBody != nil {
		clone.ExtraBody = make(map[string]any, len(req.ExtraBody))
		for key, value := range req.ExtraBody {
			clone.ExtraBody[key] = value
		}
	}
	if req.Reasoning != nil {
		reasoning := *req.Reasoning
		clone.Reasoning = &reasoning
	}
	return &clone
}

func normalizeResponseRequestForOperation(req *ResponseRequest, defaultModel string, stream bool) *ResponseRequest {
	if req == nil {
		return nil
	}

	model := strings.TrimSpace(req.Model)
	resolvedModel := model
	if resolvedModel == "" {
		resolvedModel = strings.TrimSpace(defaultModel)
	}

	if req.Stream == stream && model == resolvedModel {
		return req
	}

	clone := cloneResponseRequest(req)
	clone.Stream = stream
	if model == "" {
		clone.Model = resolvedModel
	}
	return clone
}

func unixFloatToTime(value float64) time.Time {
	if value <= 0 {
		return time.Time{}
	}
	seconds := int64(value)
	nanos := int64((value - float64(seconds)) * float64(time.Second))
	return time.Unix(seconds, nanos).UTC()
}

// streamRequestHeaders returns protocol-agnostic headers for streaming transport.
func streamRequestHeaders(stream bool) map[string]string {
	if !stream {
		return nil
	}
	return map[string]string{"Accept": "text/event-stream"}
}

// resolveRequestedModel returns the explicit model when provided, else defaultModel.
func resolveRequestedModel(requestedModel, defaultModel string) string {
	requestedModel = strings.TrimSpace(requestedModel)
	if requestedModel != "" {
		return requestedModel
	}
	return strings.TrimSpace(defaultModel)
}

// llmErrorFromResponse normalizes non-success HTTP responses to LLMError.
func llmErrorFromResponse(resp *http.Response) error {
	if resp == nil {
		return NewLLMError(ErrCodeInternal, "http response is nil", nil)
	}
	bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, maxHTTPErrorBodySize))
	return NewLLMError(
		classifyHTTPError(resp.StatusCode, bodyBytes),
		fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes)),
		nil,
	)
}

// decodeJSONResponse decodes a JSON body with a bounded reader to avoid oversized payload issues.
func decodeJSONResponse(resp *http.Response, out any) error {
	if resp == nil {
		return fmt.Errorf("decode response: nil http response")
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxResponseSize)).Decode(out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

func defaultResponseString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

// -----------------------------------------------------------------------------
// HTTP Error Classification
// -----------------------------------------------------------------------------

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
