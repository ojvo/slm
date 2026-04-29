package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"ojv/slm"
)

type ChatCompletionRequest struct {
	Model               string            `json:"model"`
	Messages            []ChatMessage     `json:"messages"`
	Stream              bool              `json:"stream,omitempty"`
	Temperature         *float64          `json:"temperature,omitempty"`
	TopP                *float64          `json:"top_p,omitempty"`
	MaxTokens           int               `json:"max_tokens,omitempty"`
	MaxCompletionTokens int               `json:"max_completion_tokens,omitempty"`
	User                string            `json:"user,omitempty"`
	Tools               []json.RawMessage `json:"tools,omitempty"`
	ResponseFormat      json.RawMessage   `json:"response_format,omitempty"`
	Reasoning           *WireReasoning    `json:"reasoning,omitempty"`
}

func (r ChatCompletionRequest) Normalize(defaultModel string) (*slm.Request, error) {
	model := ResolveModelName(r.Model, defaultModel)
	if model == "" {
		return nil, fmt.Errorf("model is required")
	}
	messages := NormalizeWireChatMessages(r.Messages)
	tools := normalizeWireChatTools(r.Tools)
	capabilities := map[string]any{}
	responseFormatType := ParseResponseFormatType(r.ResponseFormat)
	if responseFormatType != "" {
		capabilities["response_format"] = rawJSONValue(r.ResponseFormat)
	}
	req := &slm.Request{
		Model:       model,
		Messages:    messages,
		Stream:      r.Stream,
		Temperature: r.Temperature,
		TopP:        r.TopP,
		MaxTokens:   r.MaxTokens,
		JSONMode:    responseFormatType != "",
		Tools:       tools,
	}
	if r.MaxCompletionTokens > 0 {
		capabilities["max_completion_tokens"] = r.MaxCompletionTokens
		req.MaxTokens = 0
	}
	if r.Reasoning != nil && r.Reasoning.Effort != "" {
		req.Reasoning = &slm.ReasoningOptions{Effort: r.Reasoning.Effort}
	}
	if len(capabilities) > 0 {
		req.Capabilities = capabilities
	}
	return req, nil
}

type ChatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

type WireReasoning struct {
	Effort string `json:"effort,omitempty"`
}

type WireContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *WireImageURL `json:"image_url,omitempty"`
	Image    *WireImageURL `json:"image,omitempty"`
}

type WireImageURL struct {
	URL    string `json:"url,omitempty"`
	Detail string `json:"detail,omitempty"`
	Base64 string `json:"base64,omitempty"`
	MIME   string `json:"mime_type,omitempty"`
}

type WireToolCall struct {
	Index    int              `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function WireToolCallFunc `json:"function,omitempty"`
}

type WireToolCallFunc struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type WireTool struct {
	Type        string           `json:"type,omitempty"`
	Function    WireToolFunction `json:"function,omitempty"`
	Name        string           `json:"name,omitempty"`
	Description string           `json:"description,omitempty"`
	Parameters  any              `json:"parameters,omitempty"`
}

type WireToolFunction struct {
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type WireResponseFormat struct {
	Type string `json:"type,omitempty"`
}

type WireResponseRequest struct {
	Model           string                 `json:"model"`
	Input           json.RawMessage        `json:"input"`
	Stream          bool                   `json:"stream,omitempty"`
	Store           bool                   `json:"store,omitempty"`
	MaxOutputTokens int                    `json:"max_output_tokens,omitempty"`
	Reasoning       *slm.ResponseReasoning `json:"reasoning,omitempty"`
	Tools           []json.RawMessage      `json:"tools,omitempty"`
	Text            *WireTextConfig        `json:"text,omitempty"`
	ResponseFormat  json.RawMessage        `json:"response_format,omitempty"`
	User            string                 `json:"user,omitempty"`
}

func (r WireResponseRequest) Normalize(defaultModel string) (*slm.ResponseRequest, error) {
	model := ResolveModelName(r.Model, defaultModel)
	if model == "" {
		return nil, fmt.Errorf("model is required")
	}
	input, _ := NormalizeWireResponsesInput(r.Input)
	if len(input) == 0 {
		input = []slm.ResponseInputItem{{Role: "user", Content: ""}}
	}
	capabilities := map[string]any{}
	if r.Text != nil {
		if ParseResponseFormatType(r.Text.Format) != "" {
			format := rawJSONValue(r.Text.Format)
			capabilities["text"] = map[string]any{"format": format}
		}
	} else if ParseResponseFormatType(r.ResponseFormat) != "" {
		format := rawJSONValue(r.ResponseFormat)
		capabilities["text"] = map[string]any{"format": format}
	}
	tools := normalizeWireResponseTools(r.Tools)
	req := &slm.ResponseRequest{
		Model:           model,
		Input:           input,
		Stream:          r.Stream,
		Store:           r.Store,
		MaxOutputTokens: r.MaxOutputTokens,
		Reasoning:       r.Reasoning,
		Tools:           tools,
	}
	if len(capabilities) > 0 {
		req.Capabilities = capabilities
	}
	return req, nil
}

type WireTextConfig struct {
	Format json.RawMessage `json:"format,omitempty"`
}

type WireResponseInputItem struct {
	Role    string          `json:"role,omitempty"`
	Type    string          `json:"type,omitempty"`
	Content json.RawMessage `json:"content,omitempty"`
	Text    string          `json:"text,omitempty"`
}

type ChatMessageStats = slm.MessageStats

func ResolveModelName(model, fallback string) string {
	if strings.EqualFold(strings.TrimSpace(model), "ghc-0") {
		model = ""
	}
	return slm.ResolveRequestedModel(model, fallback)
}

func ResponseInputDisplayText(item slm.ResponseInputItem) string {
	switch v := item.Content.(type) {
	case string:
		return v
	case []slm.ResponseInputContentPart:
		var texts []string
		for _, part := range v {
			if tp, ok := part.(slm.ResponseInputTextPart); ok {
				texts = append(texts, tp.Text)
			}
		}
		return strings.Join(texts, "")
	default:
		data, _ := json.Marshal(v)
		return string(data)
	}
}

func ResponseInputWireContent(item slm.ResponseInputItem) any {
	return item.Content
}

func ResponseWireTools(tools []slm.ResponseTool) []any {
	if len(tools) == 0 {
		return nil
	}
	result := make([]any, len(tools))
	for i, tool := range tools {
		result[i] = tool
	}
	return result
}

func NormalizeWireChatMessages(messages []ChatMessage) []slm.Message {
	result := make([]slm.Message, 0, len(messages))
	for _, msg := range messages {
		normalized, _ := NormalizeWireChatMessage(msg)
		result = append(result, normalized)
	}
	return result
}

func CountMessageStats(messages []slm.Message) ChatMessageStats {
	return slm.ScanMessages(messages)
}

func CountRawChatMessageStats(messages []ChatMessage) ChatMessageStats {
	return CountMessageStats(NormalizeWireChatMessages(messages))
}

func NormalizeWireResponsesInput(raw json.RawMessage) ([]slm.ResponseInputItem, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []slm.ResponseInputItem{{Role: "user", Content: text}}, nil
	}
	var items []WireResponseInputItem
	if err := json.Unmarshal(raw, &items); err == nil {
		result := make([]slm.ResponseInputItem, 0, len(items))
		for _, item := range items {
			content, _ := normalizeWireInputContent(item.Content)
			role := strings.TrimSpace(item.Role)
			if role == "" {
				role = "user"
			}
			result = append(result, slm.ResponseInputItem{Role: role, Content: content})
		}
		return result, nil
	}
	var item WireResponseInputItem
	if err := json.Unmarshal(raw, &item); err == nil && (item.Role != "" || len(item.Content) > 0 || item.Text != "") {
		content, _ := normalizeWireInputContent(item.Content)
		if isEmptyStringContent(content) && item.Text != "" {
			content = item.Text
		}
		role := strings.TrimSpace(item.Role)
		if role == "" {
			role = "user"
		}
		return []slm.ResponseInputItem{{Role: role, Content: content}}, nil
	}
	return nil, nil
}

func normalizeWireInputContent(raw json.RawMessage) (any, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	var parts []WireContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		result := make([]slm.ResponseInputContentPart, 0, len(parts))
		for _, part := range parts {
			switch strings.TrimSpace(part.Type) {
			case "", "text", "input_text":
				result = append(result, slm.ResponseInputTextPart{Type: defaultResponseInputPartType(part.Type, "input_text"), Text: part.Text})
			case "image", "image_url", "input_image":
				if image := wireImagePart(part); image.URL != "" {
					result = append(result, slm.ResponseInputImagePart{Type: "input_image", ImageURL: image.URL})
				}
			}
		}
		return result, nil
	}
	return string(raw), nil
}

func NormalizeWireChatMessage(msg ChatMessage) (slm.Message, error) {
	role := msg.Role
	content, _ := normalizeWireMessageContent(msg.Content)
	return slm.Message{Role: slm.Role(role), Content: content, Name: msg.Name, ToolCalls: normalizeWireToolCalls(msg.ToolCalls), ToolCallID: msg.ToolCallID}, nil
}

func normalizeWireMessageContent(raw json.RawMessage) ([]slm.ContentPart, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []slm.ContentPart{slm.TextPart(text)}, nil
	}
	var parts []WireContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		result := make([]slm.ContentPart, 0, len(parts))
		for _, part := range parts {
			switch strings.TrimSpace(part.Type) {
			case "", "text", "input_text":
				result = append(result, slm.TextPart(part.Text))
			case "image", "image_url", "input_image":
				image := wireImagePart(part)
				if image.URL != "" || image.Base64 != "" {
					result = append(result, image)
				}
			}
		}
		return result, nil
	}
	return nil, nil
}

func normalizeWireChatTools(rawTools []json.RawMessage) []slm.Tool {
	if len(rawTools) == 0 {
		return nil
	}
	tools := make([]slm.Tool, 0, len(rawTools))
	for _, raw := range rawTools {
		var tool WireTool
		if err := json.Unmarshal(raw, &tool); err != nil {
			continue
		}
		function := tool.Function
		if strings.TrimSpace(function.Name) == "" {
			function = WireToolFunction{Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters}
		}
		name := strings.TrimSpace(function.Name)
		if name == "" {
			continue
		}
		tools = append(tools, slm.Tool{Name: name, Description: function.Description, Parameters: function.Parameters})
	}
	return tools
}

func normalizeWireResponseTools(rawTools []json.RawMessage) []slm.ResponseTool {
	if len(rawTools) == 0 {
		return nil
	}
	tools := make([]slm.ResponseTool, 0, len(rawTools))
	for _, raw := range rawTools {
		var tool WireTool
		if err := json.Unmarshal(raw, &tool); err != nil {
			continue
		}
		name := strings.TrimSpace(tool.Name)
		description := tool.Description
		parameters := tool.Parameters
		if name == "" {
			name = strings.TrimSpace(tool.Function.Name)
			description = tool.Function.Description
			parameters = tool.Function.Parameters
		}
		if name == "" {
			continue
		}
		toolType := strings.TrimSpace(tool.Type)
		if toolType == "" {
			toolType = "function"
		}
		tools = append(tools, slm.ResponseTool{Type: toolType, Name: name, Description: description, Parameters: parameters})
	}
	return tools
}

func normalizeWireToolCalls(raw json.RawMessage) []slm.APIToolCall {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var calls []WireToolCall
	if err := json.Unmarshal(raw, &calls); err != nil {
		return nil
	}
	result := make([]slm.APIToolCall, 0, len(calls))
	for _, call := range calls {
		callType := strings.TrimSpace(call.Type)
		if callType == "" {
			callType = "function"
		}
		result = append(result, slm.APIToolCall{Index: call.Index, ID: call.ID, Type: callType, Name: call.Function.Name, Arguments: call.Function.Arguments})
	}
	return result
}

func wireImagePart(part WireContentPart) slm.ImagePart {
	image := part.ImageURL
	if image == nil {
		image = part.Image
	}
	if image == nil {
		return slm.ImagePart{}
	}
	return slm.ImagePart{URL: image.URL, Base64: image.Base64, Detail: image.Detail, MIME: image.MIME}
}

func defaultResponseInputPartType(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" || value == "text" {
		return fallback
	}
	return value
}

func rawJSONValue(raw json.RawMessage) any {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil
	}
	return value
}

func isEmptyStringContent(content any) bool {
	text, ok := content.(string)
	return ok && text == ""
}

func ParseResponseFormatType(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}
	var format WireResponseFormat
	if err := json.Unmarshal(raw, &format); err != nil {
		return ""
	}
	formatType := strings.TrimSpace(format.Type)
	if strings.EqualFold(formatType, "text") {
		return ""
	}
	if strings.EqualFold(formatType, "json_object") || strings.EqualFold(formatType, "json_schema") {
		return formatType
	}
	return ""
}

func ResponseInputKind(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return "empty"
	}
	switch trimmed[0] {
	case '"':
		return "string"
	case '{':
		return "object"
	case '[':
		return "array"
	default:
		return "other"
	}
}
