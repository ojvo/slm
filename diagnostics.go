package slm

import "errors"

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
	multipartMessages := 0
	visionParts := 0
	toolCalls := 0
	for _, message := range req.Messages {
		if len(message.Content) > 1 {
			multipartMessages++
		}
		toolCalls += len(message.ToolCalls)
		for _, part := range message.Content {
			if _, ok := part.(ImagePart); ok {
				visionParts++
			}
		}
	}
	reasoning := ""
	if req.Reasoning != nil {
		reasoning = req.Reasoning.Effort
	}
	return []any{
		"model", req.Model,
		"stream", req.Stream,
		"messages", len(req.Messages),
		"multipart_messages", multipartMessages,
		"vision_parts", visionParts,
		"tool_calls", toolCalls,
		"tools", len(req.Tools),
		"json_mode", req.JSONMode,
		"reasoning", reasoning,
		"max_tokens", req.MaxTokens,
		"stop_count", len(req.Stop),
		"meta_keys", len(req.Meta),
		"extra_body_keys", len(req.ExtraBody),
	}
}

// ResponseRequestDiagnosticFields returns reusable key/value pairs describing a /responses request shape.
func ResponseRequestDiagnosticFields(req *ResponseRequest) []any {
	if req == nil {
		return []any{"request_nil", true}
	}
	visionParts := 0
	for _, item := range req.Input {
		if parts, ok := item.Content.([]ResponseInputContentPart); ok {
			for _, p := range parts {
				if _, isImg := p.(ResponseInputImagePart); isImg {
					visionParts++
				}
			}
		}
	}
	reasoning := ""
	reasoningSummary := ""
	if req.Reasoning != nil {
		reasoning = req.Reasoning.Effort
		reasoningSummary = req.Reasoning.Summary
	}
	return []any{
		"model", req.Model,
		"stream", req.Stream,
		"input_items", len(req.Input),
		"vision_parts", visionParts,
		"tools", len(req.Tools),
		"reasoning", reasoning,
		"reasoning_summary", reasoningSummary,
		"max_output_tokens", req.MaxOutputTokens,
		"store", req.Store,
		"extra_body_keys", len(req.ExtraBody),
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
