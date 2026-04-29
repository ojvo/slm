package openai

import (
	"time"

	"ojv/slm"
)

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   *slm.Usage             `json:"usage,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      WireMessage `json:"message,omitempty"`
	Delta        WireMessage `json:"delta,omitempty"`
	FinishReason *string     `json:"finish_reason"`
}

type WireMessage struct {
	Role             string         `json:"role,omitempty"`
	Content          any            `json:"content,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []WireToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
}

type WireResponse struct {
	ID          string               `json:"id"`
	Object      string               `json:"object"`
	Status      string               `json:"status"`
	Model       string               `json:"model"`
	Output      []WireResponseOutput `json:"output,omitempty"`
	Usage       *slm.ResponseUsage   `json:"usage,omitempty"`
	CreatedAt   int64                `json:"created_at,omitempty"`
	CompletedAt int64                `json:"completed_at,omitempty"`
}

type WireResponseOutput struct {
	Type    string             `json:"type"`
	ID      string             `json:"id,omitempty"`
	Role    string             `json:"role,omitempty"`
	Status  string             `json:"status,omitempty"`
	Content []WireResponsePart `json:"content,omitempty"`
	Summary []WireResponsePart `json:"summary,omitempty"`
}

type WireResponsePart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type WireResponseEvent struct {
	Type     string              `json:"type"`
	Delta    string              `json:"delta,omitempty"`
	Item     *WireResponseOutput `json:"item,omitempty"`
	Response *WireResponse       `json:"response,omitempty"`
}

func NewAssistantResponseOutput(text string) WireResponseOutput {
	return WireResponseOutput{
		Type: "message",
		Role: "assistant",
		Content: []WireResponsePart{
			{Type: "output_text", Text: text},
		},
	}
}

func ToChatResponse(now time.Time, model string, response *slm.Response) ChatCompletionResponse {
	finishReason := response.FinishReason
	if finishReason == "" {
		finishReason = "stop"
	}
	message := WireMessage{Role: "assistant", ReasoningContent: response.ReasoningContent}
	if response.Content != "" {
		message.Content = response.Content
	}
	if len(response.ToolCalls) > 0 {
		message.ToolCalls = toWireToolCalls(response.ToolCalls)
	}
	return ChatCompletionResponse{
		ID:      formatChatID(now),
		Object:  "chat.completion",
		Created: now.Unix(),
		Model:   model,
		Choices: []ChatCompletionChoice{{Index: 0, Message: message, FinishReason: &finishReason}},
		Usage:   wireUsagePtr(response.Usage),
	}
}

func ToChatChunk(id string, created int64, model string, response *slm.Response, includeRole bool) ChatCompletionResponse {
	if response == nil {
		response = &slm.Response{}
	}
	choice := ChatCompletionChoice{Index: 0}
	if includeRole || response.Content != "" || response.ReasoningContent != "" || len(response.ToolCalls) > 0 || response.FinishReason != "" {
		if includeRole {
			choice.Delta.Role = "assistant"
		}
		if response.Content != "" {
			choice.Delta.Content = response.Content
		}
		if response.ReasoningContent != "" {
			choice.Delta.ReasoningContent = response.ReasoningContent
		}
		if len(response.ToolCalls) > 0 {
			choice.Delta.ToolCalls = toWireToolCalls(response.ToolCalls)
		}
		if response.FinishReason != "" {
			finishReason := response.FinishReason
			choice.FinishReason = &finishReason
		}
	}
	chunk := ChatCompletionResponse{ID: id, Object: "chat.completion.chunk", Created: created, Model: model, Usage: wireUsagePtr(response.Usage)}
	if choice.Delta.Role != "" || choice.Delta.Content != nil || choice.Delta.ReasoningContent != "" || len(choice.Delta.ToolCalls) > 0 || choice.FinishReason != nil {
		chunk.Choices = []ChatCompletionChoice{choice}
	}
	return chunk
}

func ToWireResponse(response *slm.ResponseObject) *WireResponse {
	if response == nil {
		return nil
	}
	result := &WireResponse{
		ID:     response.ID,
		Object: slm.DefaultString(response.Object, "response"),
		Status: response.Status,
		Model:  response.Model,
		Output: make([]WireResponseOutput, 0, len(response.Output)),
		Usage:  response.Usage,
	}
	if !response.CreatedAt.IsZero() {
		result.CreatedAt = response.CreatedAt.Unix()
	}
	if !response.CompletedAt.IsZero() {
		result.CompletedAt = response.CompletedAt.Unix()
	}
	for _, output := range response.Output {
		result.Output = append(result.Output, toWireResponseOutput(output))
	}
	return result
}

func ToWireResponseEvent(event slm.ResponseEvent) WireResponseEvent {
	result := WireResponseEvent{Type: event.Type, Delta: event.Delta}
	if event.Item != nil {
		item := toWireResponseOutput(*event.Item)
		result.Item = &item
	}
	if event.Response != nil {
		result.Response = ToWireResponse(event.Response)
	}
	return result
}

func toWireToolCalls(calls []slm.APIToolCall) []WireToolCall {
	result := make([]WireToolCall, 0, len(calls))
	for index, call := range calls {
		callIndex := call.Index
		if callIndex == 0 && index > 0 {
			callIndex = index
		}
		callType := call.Type
		if callType == "" {
			callType = "function"
		}
		result = append(result, WireToolCall{
			Index: callIndex,
			ID:    call.ID,
			Type:  callType,
			Function: WireToolCallFunc{
				Name:      call.Name,
				Arguments: call.Arguments,
			},
		})
	}
	return result
}

func toWireResponseOutput(output slm.ResponseOutput) WireResponseOutput {
	result := WireResponseOutput{
		Type:    output.Type,
		ID:      output.ID,
		Role:    output.Role,
		Status:  output.Status,
		Content: make([]WireResponsePart, 0, len(output.Content)),
		Summary: make([]WireResponsePart, 0, len(output.Summary)),
	}
	for _, content := range output.Content {
		result.Content = append(result.Content, WireResponsePart{Type: content.Type, Text: content.Text})
	}
	for _, content := range output.Summary {
		result.Summary = append(result.Summary, WireResponsePart{Type: content.Type, Text: content.Text})
	}
	return result
}

func wireUsagePtr(usage slm.Usage) *slm.Usage {
	if usage.PromptTokens == 0 && usage.CompletionTokens == 0 && usage.TotalTokens == 0 {
		return nil
	}
	copy := usage
	return &copy
}

func formatChatID(now time.Time) string {
	return formatChatIDFromInt(now.UnixNano())
}

func formatChatIDFromInt(nano int64) string {
	return "chatcmpl-" + numberToString(nano)
}

func numberToString(n int64) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}
