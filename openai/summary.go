package openai

import (
	"fmt"
	"strings"

	"ojv/slm"
)

// SummarizeRawChatRequest returns a compact diagnostics summary for raw chat requests.
func SummarizeRawChatRequest(request ChatCompletionRequest) string {
	stats := CountRawChatMessageStats(request.Messages)
	responseFormatType := ParseResponseFormatType(request.ResponseFormat)
	reasoning := ""
	if request.Reasoning != nil {
		reasoning = strings.TrimSpace(request.Reasoning.Effort)
	}
	return fmt.Sprintf("chat raw_model=%q stream=%t messages=%d multipart_messages=%d vision_parts=%d tool_messages=%d tools=%d response_format=%q reasoning=%q max_tokens=%d max_completion_tokens=%d", request.Model, request.Stream, len(request.Messages), stats.MultipartMessages, stats.VisionParts, stats.ToolMessages, len(request.Tools), responseFormatType, reasoning, request.MaxTokens, request.MaxCompletionTokens)
}

// SummarizeNormalizedChatRequest returns a compact diagnostics summary for normalized chat requests.
func SummarizeNormalizedChatRequest(request *slm.Request) string {
	if request == nil {
		return "chat normalized=nil"
	}
	reasoning := ""
	if request.Reasoning != nil {
		reasoning = strings.TrimSpace(request.Reasoning.Effort)
	}
	stats := CountMessageStats(request.Messages)
	return fmt.Sprintf("chat model=%q stream=%t messages=%d multipart_messages=%d vision_parts=%d tool_calls=%d tools=%d json_mode=%t reasoning=%q max_tokens=%d stop_count=%d capabilities_keys=%d", request.Model, request.Stream, len(request.Messages), stats.MultipartMessages, stats.VisionParts, stats.ToolCalls, len(request.Tools), request.JSONMode, reasoning, request.MaxTokens, len(request.Stop), len(request.Capabilities))
}

// SummarizeRawResponsesRequest returns a compact diagnostics summary for raw responses requests.
func SummarizeRawResponsesRequest(request WireResponseRequest) string {
	reasoning := ""
	if request.Reasoning != nil {
		reasoning = strings.TrimSpace(request.Reasoning.Effort)
	}
	responseFormat := ParseResponseFormatType(request.ResponseFormat)
	if request.Text != nil {
		if format := ParseResponseFormatType(request.Text.Format); format != "" {
			responseFormat = format
		}
	}
	return fmt.Sprintf("responses raw_model=%q stream=%t input_kind=%s tools=%d response_format=%q reasoning=%q max_output_tokens=%d store=%t", request.Model, request.Stream, ResponseInputKind(request.Input), len(request.Tools), responseFormat, reasoning, request.MaxOutputTokens, request.Store)
}

// SummarizeNormalizedResponsesRequest returns a compact diagnostics summary for normalized responses requests.
func SummarizeNormalizedResponsesRequest(request *slm.ResponseRequest) string {
	if request == nil {
		return "responses normalized=nil"
	}
	reasoning := ""
	if request.Reasoning != nil {
		reasoning = strings.TrimSpace(request.Reasoning.Effort)
	}
	return fmt.Sprintf("responses model=%q stream=%t input_items=%d tools=%d reasoning=%q max_output_tokens=%d store=%t capabilities_keys=%d", request.Model, request.Stream, len(request.Input), len(request.Tools), reasoning, request.MaxOutputTokens, request.Store, len(request.Capabilities))
}
