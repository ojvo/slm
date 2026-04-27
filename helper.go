package slm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const maxResponseSize = 50 * 1024 * 1024

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

func normalizeRequestForOperation(req *Request, defaultModel string, stream bool) *Request {
	return normalizeForOperation(req, defaultModel, stream,
		func(r *Request) string { return r.Model },
		func(r *Request) bool { return r.Stream },
		func(r *Request, s bool, m string) *Request {
			clone := cloneRequestShallow(r)
			clone.Stream = s
			if r.Model == "" {
				clone.Model = m
			}
			return clone
		},
	)
}

func normalizeResponseRequestForOperation(req *ResponseRequest, defaultModel string, stream bool) *ResponseRequest {
	return normalizeForOperation(req, defaultModel, stream,
		func(r *ResponseRequest) string { return r.Model },
		func(r *ResponseRequest) bool { return r.Stream },
		func(r *ResponseRequest, s bool, m string) *ResponseRequest {
			clone := cloneResponseRequest(r)
			clone.Stream = s
			if r.Model == "" {
				clone.Model = m
			}
			return clone
		},
	)
}

func normalizeCompletedResponseObject(resp *ResponseObject) *ResponseObject {
	if resp == nil {
		return nil
	}
	if resp.Status == "completed" && resp.Output == nil {
		resp.Output = []ResponseOutput{}
	}
	return resp
}

func normalizeForOperation[T any](req T, defaultModel string, stream bool,
	getModel func(T) string,
	getStream func(T) bool,
	cloneAndAssign func(T, bool, string) T,
) T {
	var zero T
	if any(req) == nil {
		return zero
	}
	resolved := resolveRequestedModel(getModel(req), defaultModel)
	if getStream(req) == stream && getModel(req) == resolved {
		return req
	}
	return cloneAndAssign(req, stream, resolved)
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

func defaultString(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
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

type cloneDepth int

const (
	cloneDepthShallow cloneDepth = iota
	cloneDepthMetadata
	cloneDepthDeep
)

func cloneRequest(req *Request) *Request {
	return cloneRequestWithDepth(req, cloneDepthDeep)
}

func cloneRequestShallow(req *Request) *Request {
	return cloneRequestWithDepth(req, cloneDepthShallow)
}

func cloneRequestForMetadata(req *Request) *Request {
	return cloneRequestWithDepth(req, cloneDepthMetadata)
}

func cloneRequestWithDepth(req *Request, depth cloneDepth) *Request {
	if req == nil {
		return nil
	}

	clone := *req
	if depth == cloneDepthShallow {
		return &clone
	}

	if depth == cloneDepthMetadata {
		clone.Meta = cloneMap(clone.Meta)
		return &clone
	}

	clone.Messages = cloneMessages(req.Messages)
	clone.Tools = cloneTools(req.Tools)
	clone.Stop = cloneStringSlice(req.Stop)
	clone.Meta = cloneMap(req.Meta)
	clone.Capabilities = cloneMap(req.Capabilities)
	clone.Temperature = cloneFloat64(req.Temperature)
	clone.TopP = cloneFloat64(req.TopP)
	clone.PresencePenalty = cloneFloat64(req.PresencePenalty)
	clone.FrequencyPenalty = cloneFloat64(req.FrequencyPenalty)
	clone.Reasoning = cloneReasoningOptions(req.Reasoning)

	return &clone
}

func cloneResponseRequest(req *ResponseRequest) *ResponseRequest {
	if req == nil {
		return nil
	}

	clone := *req
	clone.Input = cloneResponseInputItems(req.Input)
	clone.Tools = cloneResponseTools(req.Tools)
	clone.Capabilities = cloneMap(req.Capabilities)
	clone.Reasoning = cloneReasoningOptions(req.Reasoning)

	return &clone
}

func cloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}

	cloned := make([]Message, len(messages))
	for i, msg := range messages {
		cloned[i] = Message{
			Role:       msg.Role,
			Content:    append([]ContentPart(nil), msg.Content...),
			Name:       msg.Name,
			ToolCalls:  append([]APIToolCall(nil), msg.ToolCalls...),
			ToolCallID: msg.ToolCallID,
		}
	}

	return cloned
}

func cloneTools(tools []Tool) []Tool {
	if tools == nil {
		return nil
	}

	cloned := make([]Tool, len(tools))
	copy(cloned, tools)
	return cloned
}

func cloneResponseInputItems(items []ResponseInputItem) []ResponseInputItem {
	if items == nil {
		return nil
	}

	cloned := make([]ResponseInputItem, len(items))
	for i, item := range items {
		cloned[i] = ResponseInputItem{
			Role:    item.Role,
			Content: cloneResponseInputContent(item.Content),
		}
	}
	return cloned
}

func cloneResponseInputContent(content any) any {
	switch c := content.(type) {
	case nil:
		return nil
	case string:
		return c
	case []ResponseInputContentPart:
		parts := make([]ResponseInputContentPart, len(c))
		copy(parts, c)
		return parts
	default:
		return c
	}
}

func cloneResponseTools(tools []ResponseTool) []ResponseTool {
	if tools == nil {
		return nil
	}
	out := make([]ResponseTool, len(tools))
	copy(out, tools)
	return out
}

// cloneReasoningOptions clones a ReasoningOptions (and by alias ResponseReasoning) value.
func cloneReasoningOptions(reasoning *ReasoningOptions) *ReasoningOptions {
	if reasoning == nil {
		return nil
	}
	clone := *reasoning
	return &clone
}

func cloneFloat64(p *float64) *float64 {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func cloneMap(m map[string]any) map[string]any {
	if len(m) == 0 {
		return nil
	}
	cp := make(map[string]any, len(m))
	for k, v := range m {
		cp[k] = v
	}
	return cp
}

func cloneStringSlice(s []string) []string {
	if s == nil {
		return nil
	}
	cp := make([]string, len(s))
	copy(cp, s)
	return cp
}

//

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

func convertResponseInputItems(items []ResponseInputItem) []map[string]any {
	input := make([]map[string]any, 0, len(items))
	for _, item := range items {
		entry := map[string]any{"role": item.Role}
		switch v := item.Content.(type) {
		case string:
			entry["content"] = v
		case []ResponseInputContentPart:
			parts := make([]any, 0, len(v))
			for _, p := range v {
				parts = append(parts, p)
			}
			entry["content"] = parts
		default:
			entry["content"] = v
		}
		input = append(input, entry)
	}
	return input
}

// ============================================================================
// Codec Helper Functions
// ============================================================================
// These functions are used by all protocol codecs to build request bodies.
// They provide a consistent pattern for conditionally adding fields to
// protocol-specific request maps.

// putFloat64PtrField adds a float64 pointer field to the request map if not nil.
func putFloat64PtrField(dst map[string]any, key string, value *float64) {
	if value != nil {
		dst[key] = *value
	}
}

// putStringField adds a string field to the request map if not empty.
func putStringField(dst map[string]any, key, value string) {
	if value != "" {
		dst[key] = value
	}
}

// putPositiveIntField adds an int field to the request map if greater than 0.
func putPositiveIntField(dst map[string]any, key string, value int) {
	if value > 0 {
		dst[key] = value
	}
}

// putSliceField adds a slice field to the request map if not empty.
func putSliceField[T any](dst map[string]any, key string, value []T) {
	if len(value) > 0 {
		dst[key] = value
	}
}

// putAnyField adds any field to the request map if not nil.
func putAnyField(dst map[string]any, key string, value any) {
	if value != nil {
		dst[key] = value
	}
}

// mergeFields merges extra fields into the destination map.
func mergeFields(dst map[string]any, extra map[string]any) {
	for key, value := range extra {
		dst[key] = value
	}
}

// toMap converts a value to a map[string]interface{}, returning empty map if conversion fails.
func toMap(v any) map[string]interface{} {
	m, ok := v.(map[string]interface{})
	if ok {
		return m
	}
	return make(map[string]interface{})
}
