package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"ojv/slm"
)

func TestSummarizeRawChatRequest_UsesNormalizedSLMMessages(t *testing.T) {
	summary := SummarizeRawChatRequest(ChatCompletionRequest{
		Model: "gpt-4.1",
		Messages: []ChatMessage{
			{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"high"}}]`)},
			{Role: "assistant", Content: json.RawMessage(`"done"`), ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]`)},
		},
	})
	if !strings.Contains(summary, `messages=2`) {
		t.Fatalf("expected summary to include messages=2, got %s", summary)
	}
}

func TestSummarizeRawResponsesRequest_IncludesResponseFormatFromText(t *testing.T) {
	summary := SummarizeRawResponsesRequest(WireResponseRequest{
		Model: "gpt-5-mini",
		Text:  &WireTextConfig{Format: []byte(`{"type":"json_schema","json_schema":{"name":"answer"}}`)},
		Input: []byte(`"hi"`),
	})
	if !strings.Contains(summary, `response_format="json_schema"`) {
		t.Fatalf("expected summary to include json_schema response_format, got %s", summary)
	}
}

func TestSummarizeNormalizedChatRequest_IncludesToolCallsAndVision(t *testing.T) {
	summary := SummarizeNormalizedChatRequest(&slm.Request{
		Model: "gpt-4.1",
		Messages: []slm.Message{{
			Role:      slm.RoleUser,
			Content:   []slm.ContentPart{slm.TextPart("look"), slm.ImagePart{URL: "https://example.com/image.png", Detail: "high"}},
			ToolCalls: []slm.APIToolCall{{ID: "call_1", Type: "function", Name: "lookup", Arguments: `{}`}},
		}},
		Tools:    []slm.Tool{{Name: "lookup"}},
		JSONMode: true,
	})
	if !strings.Contains(summary, `vision_parts=1`) || !strings.Contains(summary, `tool_calls=1`) || !strings.Contains(summary, `json_mode=true`) {
		t.Fatalf("expected summary to include key normalized fields, got %s", summary)
	}
}
