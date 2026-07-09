package openai

import (
	"testing"
	"time"

	"ojv/slm"
)

func TestToChatResponse_MapsContentAndToolCalls(t *testing.T) {
	now := time.Unix(1714200000, 123)
	response := &slm.Response{
		Content:          "ok",
		ReasoningContent: "think",
		ToolCalls:        []slm.APIToolCall{{ID: "call_1", Type: "function", Name: "lookup", Arguments: "{}"}},
		Usage:            slm.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12},
	}

	wire := ToChatResponse(now, "gpt-4.1", response)
	if wire.Object != "chat.completion" || wire.Model != "gpt-4.1" || len(wire.Choices) != 1 {
		t.Fatalf("unexpected wire response header: %#v", wire)
	}
	if wire.ID == "" || wire.Choices[0].Message.Content != "ok" || wire.Choices[0].Message.ReasoningContent != "think" {
		t.Fatalf("unexpected wire message: %#v", wire.Choices[0].Message)
	}
	if len(wire.Choices[0].Message.ToolCalls) != 1 || wire.Choices[0].Message.ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("unexpected wire tool calls: %#v", wire.Choices[0].Message.ToolCalls)
	}
	if wire.Usage == nil || wire.Usage.TotalTokens != 12 {
		t.Fatalf("unexpected usage: %#v", wire.Usage)
	}
}

func TestToWireResponseEvent_MapsNestedResponse(t *testing.T) {
	started := time.Unix(1714200000, 0)
	event := slm.ResponseEvent{
		Type:  "response.output_text.delta",
		Delta: "hel",
		Item:  &slm.ResponseOutput{Type: "message", ID: "msg_1", Role: "assistant", Content: []slm.ResponseOutputContent{{Type: "output_text", Text: "hel"}}},
		Response: &slm.ResponseObject{
			ID:        "resp_1",
			Object:    "response",
			Status:    "in_progress",
			Model:     "gpt-5-mini",
			CreatedAt: started,
			Output:    []slm.ResponseOutput{{Type: "message", ID: "msg_1", Role: "assistant"}},
		},
	}

	wire := ToWireResponseEvent(event)
	if wire.Type != "response.output_text.delta" || wire.Delta != "hel" {
		t.Fatalf("unexpected wire event basics: %#v", wire)
	}
	if wire.Item == nil || wire.Item.ID != "msg_1" || len(wire.Item.Content) != 1 {
		t.Fatalf("unexpected wire item: %#v", wire.Item)
	}
	if wire.Response == nil || wire.Response.ID != "resp_1" || wire.Response.CreatedAt != started.Unix() {
		t.Fatalf("unexpected nested wire response: %#v", wire.Response)
	}
}
