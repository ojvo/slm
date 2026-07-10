package slm

import (
	"errors"
	"fmt"
)

// ErrorDiagnosticFields returns reusable key/value pairs describing an error.
func ErrorDiagnosticFields(err error) []any {
	if err == nil {
		return nil
	}
	fields := []any{"error", err}
	var llmErr *LLMError
	if errors.As(err, &llmErr) {
		fields = append(fields,
			"error_code", llmErr.Code.String(),
			"retryable", llmErr.Code.IsRetryable(),
		)
		if llmErr.Cause != nil {
			fields = append(fields, "cause", llmErr.Cause)
			if source := timeoutSource(llmErr.Cause); source != "" {
				fields = append(fields, "timeout_source", source)
			}
		}
	} else if source := timeoutSource(err); source != "" {
		fields = append(fields, "timeout_source", source)
	}
	return fields
}

// RequestDiagnosticFields returns reusable key/value pairs describing a chat request shape.
func RequestDiagnosticFields(req *Request) []any {
	if req == nil {
		return []any{"request_nil", true}
	}
	stats := ScanMessages(req.Messages)
	reasoning := ""
	if req.Reasoning != nil {
		reasoning = req.Reasoning.Effort
	}
	return []any{
		"model", req.Model,
		"stream", req.Stream,
		"messages", len(req.Messages),
		"multipart_messages", stats.MultipartMessages,
		"vision_parts", stats.VisionParts,
		"tool_calls", stats.ToolCalls,
		"tools", len(req.Tools),
		"json_mode", req.JSONMode,
		"reasoning", reasoning,
		"max_tokens", req.MaxTokens,
		"stop_count", len(req.Stop),
		"meta_keys", len(req.Meta),
		"capabilities_keys", len(req.Capabilities),
	}
}

// ResponseRequestDiagnosticFields returns reusable key/value pairs describing a /responses request shape.
func ResponseRequestDiagnosticFields(req *ResponseRequest) []any {
	if req == nil {
		return []any{"request_nil", true}
	}
	stats := ScanResponseInput(req.Input)
	reasoning := ""
	reasoningSummary := ""
	if req.Reasoning != nil {
		reasoning = req.Reasoning.Effort
		reasoningSummary = req.Reasoning.Summary
	}
	return []any{
		"model", req.Model,
		"stream", req.Stream,
		"input_items", stats.InputItems,
		"vision_parts", stats.VisionParts,
		"tools", len(req.Tools),
		"reasoning", reasoning,
		"reasoning_summary", reasoningSummary,
		"max_output_tokens", req.MaxOutputTokens,
		"store", req.Store,
		"capabilities_keys", len(req.Capabilities),
	}
}

// eventDiagnosticFields returns diagnostic fields for a LifecycleEvent,
// automatically selecting the correct diagnostic function based on which
// request type is populated.
func eventDiagnosticFields(event LifecycleEvent) []any {
	if event.ResponseRequest != nil {
		return ResponseRequestDiagnosticFields(event.ResponseRequest)
	}
	return RequestDiagnosticFields(event.Request)
}

// SummarizeNormalizedChatRequest returns a compact diagnostics summary for
// canonical normalized chat requests.
func SummarizeNormalizedChatRequest(request *Request) string {
	if request == nil {
		return "chat normalized=nil"
	}
	reasoning := ""
	if request.Reasoning != nil {
		reasoning = request.Reasoning.Effort
	}
	stats := ScanMessages(request.Messages)
	return fmt.Sprintf("chat model=%q stream=%t messages=%d multipart_messages=%d vision_parts=%d tool_calls=%d tools=%d json_mode=%t reasoning=%q max_tokens=%d stop_count=%d capabilities_keys=%d", request.Model, request.Stream, len(request.Messages), stats.MultipartMessages, stats.VisionParts, stats.ToolCalls, len(request.Tools), request.JSONMode, reasoning, request.MaxTokens, len(request.Stop), len(request.Capabilities))
}

// SummarizeNormalizedResponsesRequest returns a compact diagnostics summary for
// canonical normalized responses requests.
func SummarizeNormalizedResponsesRequest(request *ResponseRequest) string {
	if request == nil {
		return "responses normalized=nil"
	}
	reasoning := ""
	if request.Reasoning != nil {
		reasoning = request.Reasoning.Effort
	}
	return fmt.Sprintf("responses model=%q stream=%t input_items=%d tools=%d reasoning=%q max_output_tokens=%d store=%t capabilities_keys=%d", request.Model, request.Stream, len(request.Input), len(request.Tools), reasoning, request.MaxOutputTokens, request.Store, len(request.Capabilities))
}
