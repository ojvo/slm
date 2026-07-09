package openai

import (
	"encoding/json"
	"testing"

	"ojv/slm"
)

func TestNormalizeChatCompletionRequest_PreservesOpenAICapabilities(t *testing.T) {
	request := ChatCompletionRequest{
		Model: "gpt-4.1",
		Messages: []ChatMessage{
			{Role: "user", Name: "alice", Content: json.RawMessage(`[
				{"type":"text","text":"look"},
				{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"high"}}
			]`)},
			{Role: "assistant", Content: json.RawMessage(`"done"`), ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]`)},
			{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage(`"result"`)},
		},
		Tools:               []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"lookup","description":"Find data","parameters":{"type":"object"}}}`)},
		ResponseFormat:      json.RawMessage(`{"type":"json_schema","json_schema":{"name":"answer"}}`),
		Reasoning:           &WireReasoning{Effort: "medium"},
		MaxTokens:           4096,
		MaxCompletionTokens: 16000,
	}

	normalized, err := request.Normalize("")
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if normalized.Model != "gpt-4.1" || normalized.MaxTokens != 0 || !normalized.JSONMode {
		t.Fatalf("unexpected normalized request basics: %#v", normalized)
	}
	if len(normalized.Tools) != 1 || normalized.Tools[0].Name != "lookup" || normalized.Tools[0].Description != "Find data" {
		t.Fatalf("unexpected tools: %#v", normalized.Tools)
	}
	if normalized.Reasoning == nil || normalized.Reasoning.Effort != "medium" {
		t.Fatalf("unexpected reasoning: %#v", normalized.Reasoning)
	}
	if got := normalized.Capabilities["max_completion_tokens"]; got != 16000 {
		t.Fatalf("expected max_completion_tokens capability, got %#v", normalized.Capabilities)
	}
	if normalized.Capabilities["response_format"] == nil {
		t.Fatalf("expected response_format capability, got %#v", normalized.Capabilities)
	}
	if len(normalized.Messages) != 3 || normalized.Messages[0].Name != "alice" || normalized.Messages[2].ToolCallID != "call_1" {
		t.Fatalf("unexpected normalized messages: %#v", normalized.Messages)
	}
	if _, ok := normalized.Messages[0].Content[1].(slm.ImagePart); !ok {
		t.Fatalf("expected second content part to be image, got %#v", normalized.Messages[0].Content)
	}
	if len(normalized.Messages[1].ToolCalls) != 1 || normalized.Messages[1].ToolCalls[0].Name != "lookup" {
		t.Fatalf("unexpected tool calls: %#v", normalized.Messages[1].ToolCalls)
	}
}

func TestNormalizeResponsesRequest_PreservesToolsAndStructuredOutput(t *testing.T) {
	request := WireResponseRequest{
		Model: "gpt-5-mini",
		Input: json.RawMessage(`[{"role":"user","content":[
			{"type":"input_text","text":"look"},
			{"type":"input_image","image_url":{"url":"https://example.com/image.png"}}
		]}]`),
		Tools:           []json.RawMessage{json.RawMessage(`{"type":"function","name":"lookup","description":"Find data","parameters":{"type":"object"}}`)},
		Text:            &WireTextConfig{Format: json.RawMessage(`{"type":"json_schema","json_schema":{"name":"answer"}}`)},
		Reasoning:       &slm.ResponseReasoning{Effort: "medium"},
		MaxOutputTokens: 256,
		Store:           true,
	}

	normalized, err := request.Normalize("")
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if normalized.Model != "gpt-5-mini" || normalized.MaxOutputTokens != 256 || !normalized.Store {
		t.Fatalf("unexpected normalized response request: %#v", normalized)
	}
	if len(normalized.Tools) != 1 || normalized.Tools[0].Name != "lookup" || normalized.Tools[0].Description != "Find data" {
		t.Fatalf("unexpected response tools: %#v", normalized.Tools)
	}
	if normalized.Capabilities["text"] == nil {
		t.Fatalf("expected text.format capability, got %#v", normalized.Capabilities)
	}
	if len(normalized.Input) != 1 || normalized.Input[0].Role != "user" {
		t.Fatalf("unexpected input items: %#v", normalized.Input)
	}
	parts, ok := normalized.Input[0].Content.([]slm.ResponseInputContentPart)
	if !ok || len(parts) != 2 {
		t.Fatalf("expected multipart response input, got %#v", normalized.Input[0].Content)
	}
}

func TestNormalizeResponsesRequest_TextFormatDoesNotImplyStructuredOutput(t *testing.T) {
	request := WireResponseRequest{
		Model: "gpt-5-mini",
		Input: json.RawMessage(`"hi"`),
		Text:  &WireTextConfig{Format: json.RawMessage(`{"type":"text"}`)},
	}

	normalized, err := request.Normalize("")
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if len(normalized.Capabilities) != 0 {
		t.Fatalf("expected no structured output capabilities for text format, got %#v", normalized.Capabilities)
	}
}

func TestNormalizeChatCompletionRequest_GHCZeroFallsBackToDefaultModel(t *testing.T) {
	request := ChatCompletionRequest{
		Model: "ghc-0",
		Messages: []ChatMessage{
			{Role: "user", Content: json.RawMessage(`"hi"`)},
		},
	}

	normalized, err := request.Normalize("gpt-5-mini")
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if normalized.Model != "gpt-5-mini" {
		t.Fatalf("expected fallback model gpt-5-mini, got %q", normalized.Model)
	}
}

func TestNormalizeResponsesRequest_GHCZeroFallsBackToDefaultModel(t *testing.T) {
	request := WireResponseRequest{
		Model: "ghc-0",
		Input: json.RawMessage(`"hi"`),
	}

	normalized, err := request.Normalize("gpt-5-mini")
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if normalized.Model != "gpt-5-mini" {
		t.Fatalf("expected fallback model gpt-5-mini, got %q", normalized.Model)
	}
}

func TestResponseInputWireContent_ReturnsUnderlyingContent(t *testing.T) {
	item := slm.ResponseInputItem{Role: "user", Content: "hello"}
	if got := ResponseInputWireContent(item); got != "hello" {
		t.Fatalf("expected string content, got %#v", got)
	}
}

func TestResponseWireTools_MapsToAnySlice(t *testing.T) {
	if got := ResponseWireTools(nil); got != nil {
		t.Fatalf("expected nil for empty tools, got %#v", got)
	}
	tools := []slm.ResponseTool{{Type: "function", Name: "lookup"}}
	got := ResponseWireTools(tools)
	if len(got) != 1 {
		t.Fatalf("expected one tool, got %#v", got)
	}
	mapped, ok := got[0].(slm.ResponseTool)
	if !ok || mapped.Name != "lookup" {
		t.Fatalf("unexpected mapped tool: %#v", got[0])
	}
}

func TestNormalizeWireChatMessages_AndCountStats(t *testing.T) {
	messages := []ChatMessage{
		{
			Role: "user",
			Content: json.RawMessage(`[
				{"type":"text","text":"hello"},
				{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}
			]`),
		},
		{
			Role:      "assistant",
			Content:   json.RawMessage(`"done"`),
			ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]`),
		},
	}

	normalized := NormalizeWireChatMessages(messages)
	if len(normalized) != 2 {
		t.Fatalf("expected 2 normalized messages, got %#v", normalized)
	}
	stats := CountMessageStats(normalized)
	if stats.MultipartMessages != 1 || stats.VisionParts != 1 || stats.ToolMessages != 1 || stats.ToolCalls != 1 {
		t.Fatalf("unexpected stats from normalized messages: %#v", stats)
	}

	rawStats := CountRawChatMessageStats(messages)
	if rawStats != stats {
		t.Fatalf("expected raw stats to match normalized stats, raw=%#v normalized=%#v", rawStats, stats)
	}
}
