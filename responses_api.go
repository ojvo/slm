package slm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

type ResponseReasoning struct {
	Effort  string
	Summary string
}

type ResponseInputItem struct {
	Role    string
	Content string
}

type ResponseRequest struct {
	Model           string
	Input           []ResponseInputItem
	Stream          bool
	Store           bool
	MaxOutputTokens int
	Reasoning       *ResponseReasoning
	Tools           []json.RawMessage
	ExtraBody       map[string]any
}

type ResponseOutputContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type ResponseOutput struct {
	Type    string                  `json:"type"`
	ID      string                  `json:"id,omitempty"`
	Role    string                  `json:"role,omitempty"`
	Status  string                  `json:"status,omitempty"`
	Content []ResponseOutputContent `json:"content,omitempty"`
	Summary []ResponseOutputContent `json:"summary,omitempty"`
}

type ResponseUsage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
}

type ResponseObject struct {
	ID          string
	Object      string
	Status      string
	Model       string
	Output      []ResponseOutput
	Usage       *ResponseUsage
	CreatedAt   time.Time
	CompletedAt time.Time
}

type ResponseEvent struct {
	Type     string
	Delta    string
	Response *ResponseObject
	Item     *ResponseOutput
	Err      error
}

type ResponseStream interface {
	Next() bool
	Current() ResponseEvent
	Err() error
	Close() error
}

type OpenAIResponsesEngine struct {
	transport    Transport
	defaultModel string
	capabilities *CapabilityNegotiationOptions
	logger       Logger
}

type OpenAIResponsesOptions struct {
	DefaultModel string
	Capabilities *CapabilityNegotiationOptions
	Logger       Logger
}

func NewOpenAIResponsesProtocol(baseURL, apiKey, defaultModel string) *OpenAIResponsesEngine {
	return &OpenAIResponsesEngine{
		transport:    NewHTTPTransport(baseURL, apiKey),
		defaultModel: defaultModel,
	}
}

func NewOpenAIResponsesProtocolWithOptions(baseURL, apiKey string, opts OpenAIResponsesOptions) *OpenAIResponsesEngine {
	return NewOpenAIResponsesWithTransportAndOptions(NewHTTPTransport(baseURL, apiKey), opts)
}

func NewOpenAIResponsesWithTransport(transport Transport, defaultModel string) *OpenAIResponsesEngine {
	return &OpenAIResponsesEngine{transport: transport, defaultModel: defaultModel}
}

func NewOpenAIResponsesWithTransportAndOptions(transport Transport, opts OpenAIResponsesOptions) *OpenAIResponsesEngine {
	engine := &OpenAIResponsesEngine{transport: transport, defaultModel: opts.DefaultModel, logger: opts.Logger}
	if opts.Capabilities != nil {
		clone := *opts.Capabilities
		if clone.DefaultModel == "" {
			clone.DefaultModel = opts.DefaultModel
		}
		engine.capabilities = &clone
	}
	return engine
}

func (e *OpenAIResponsesEngine) Create(ctx context.Context, req *ResponseRequest) (*ResponseObject, error) {
	effectiveReq := normalizeResponseRequestForOperation(req, e.defaultModel, false)
	requestID := GetRequestID(ctx)
	start := time.Now()
	e.logDebug("LLM responses request start", append([]any{"request_id", requestID}, ResponseRequestDiagnosticFields(effectiveReq)...)...)
	resp, err := e.doRequest(ctx, effectiveReq, false)
	if err != nil {
		e.logError("LLM responses request failed", start, requestID, effectiveReq, err)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		err := NewLLMError(classifyHTTPError(resp.StatusCode, bodyBytes), fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes)), nil)
		e.logError("LLM responses request failed", start, requestID, effectiveReq, err)
		return nil, err
	}

	var decoded oaiResponseObject
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxResponseSize)).Decode(&decoded); err != nil {
		wrapped := fmt.Errorf("decode response: %w", err)
		e.logError("LLM responses request failed", start, requestID, effectiveReq, wrapped)
		return nil, wrapped
	}
	result := convertResponseObject(&decoded)
	if result != nil && result.Usage != nil {
		e.logDebug("LLM responses request completed", "duration", time.Since(start), "request_id", requestID, "model", effectiveReq.Model, "status", result.Status, "tokens", result.Usage.TotalTokens)
	} else if result != nil {
		e.logDebug("LLM responses request completed", "duration", time.Since(start), "request_id", requestID, "model", effectiveReq.Model, "status", result.Status)
	}
	return result, nil
}

func (e *OpenAIResponsesEngine) Stream(ctx context.Context, req *ResponseRequest) (ResponseStream, error) {
	effectiveReq := normalizeResponseRequestForOperation(req, e.defaultModel, true)
	requestID := GetRequestID(ctx)
	start := time.Now()
	e.logDebug("LLM responses stream start", append([]any{"request_id", requestID}, ResponseRequestDiagnosticFields(effectiveReq)...)...)
	resp, err := e.doRequest(ctx, effectiveReq, true)
	if err != nil {
		e.logError("LLM responses stream failed", start, requestID, effectiveReq, err)
		return nil, err
	}
	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		resp.Body.Close()
		err := NewLLMError(classifyHTTPError(resp.StatusCode, bodyBytes), fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes)), nil)
		e.logError("LLM responses stream failed", start, requestID, effectiveReq, err)
		return nil, err
	}
	e.logDebug("LLM responses stream connected", "duration", time.Since(start), "request_id", requestID, "model", effectiveReq.Model)
	return &loggingResponseStream{inner: newOpenAIResponseStream(resp), logger: e.logger, start: start, requestID: requestID, request: cloneResponseRequest(effectiveReq)}, nil
}

func (e *OpenAIResponsesEngine) logDebug(msg string, args ...any) {
	if e.logger == nil {
		return
	}
	e.logger.Debug(msg, args...)
}

func (e *OpenAIResponsesEngine) logError(msg string, start time.Time, requestID string, req *ResponseRequest, err error) {
	if e.logger == nil {
		return
	}
	args := append([]any{"duration", time.Since(start), "request_id", requestID}, ResponseRequestDiagnosticFields(req)...)
	args = append(args, ErrorDiagnosticFields(err)...)
	e.logger.Error(msg, args...)
}

func (e *OpenAIResponsesEngine) doRequest(ctx context.Context, req *ResponseRequest, stream bool) (*http.Response, error) {
	if req == nil {
		return nil, NewLLMError(ErrCodeInvalidConfig, "request is nil", nil)
	}
	if len(req.Input) == 0 {
		return nil, NewLLMError(ErrCodeInvalidConfig, "input is required", nil)
	}
	if e.capabilities != nil && e.capabilities.Resolver != nil {
		var err error
		ctx, req, err = NegotiateResponseCapabilities(ctx, req, *e.capabilities)
		if err != nil {
			return nil, err
		}
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = strings.TrimSpace(e.defaultModel)
	}
	if model == "" {
		return nil, NewLLMError(ErrCodeInvalidModel, "model is required", nil)
	}

	body, err := e.buildRequestBody(req, model, stream)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	headers := map[string]string{}
	if stream {
		headers["Accept"] = "text/event-stream"
	}
	return e.transport.Do(ctx, http.MethodPost, "/responses", headers, body)
}

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
	clone := cloneResponseRequest(req)
	clone.Stream = stream
	if strings.TrimSpace(clone.Model) == "" {
		clone.Model = strings.TrimSpace(defaultModel)
	}
	return clone
}

func (e *OpenAIResponsesEngine) buildRequestBody(req *ResponseRequest, model string, stream bool) ([]byte, error) {
	input := make([]map[string]any, 0, len(req.Input))
	for _, item := range req.Input {
		input = append(input, map[string]any{
			"role":    item.Role,
			"content": item.Content,
		})
	}

	reqMap := map[string]any{
		"model":  model,
		"input":  input,
		"stream": stream,
		"store":  req.Store,
	}
	if req.MaxOutputTokens > 0 {
		reqMap["max_output_tokens"] = req.MaxOutputTokens
	}
	if req.Reasoning != nil {
		reasoning := map[string]any{}
		if req.Reasoning.Effort != "" {
			reasoning["effort"] = req.Reasoning.Effort
		}
		if req.Reasoning.Summary != "" {
			reasoning["summary"] = req.Reasoning.Summary
		}
		if len(reasoning) > 0 {
			reqMap["reasoning"] = reasoning
		}
	}
	if len(req.Tools) > 0 {
		tools := make([]any, 0, len(req.Tools))
		for index, tool := range req.Tools {
			var value any
			if err := json.Unmarshal(tool, &value); err != nil {
				return nil, fmt.Errorf("tools[%d]: %w", index, err)
			}
			tools = append(tools, value)
		}
		reqMap["tools"] = tools
	}
	for key, value := range req.ExtraBody {
		reqMap[key] = value
	}
	return json.Marshal(reqMap)
}

type openAIResponseStream struct {
	resp    *http.Response
	framer  *sseFrameReader
	current ResponseEvent
	err     error
	done    bool
}

type loggingResponseStream struct {
	inner     ResponseStream
	logger    Logger
	start     time.Time
	requestID string
	request   *ResponseRequest
	closeOnce sync.Once
	closeErr  error
}

func (l *loggingResponseStream) Next() bool {
	ok := l.inner.Next()
	if !ok {
		_ = l.Close()
	}
	return ok
}

func (l *loggingResponseStream) Current() ResponseEvent { return l.inner.Current() }
func (l *loggingResponseStream) Err() error             { return l.inner.Err() }
func (l *loggingResponseStream) Close() error {
	l.closeOnce.Do(func() {
		l.closeErr = l.inner.Close()
		if l.logger == nil {
			return
		}
		duration := time.Since(l.start)
		if l.closeErr != nil {
			args := append([]any{"duration", duration, "request_id", l.requestID}, ResponseRequestDiagnosticFields(l.request)...)
			args = append(args, ErrorDiagnosticFields(l.closeErr)...)
			l.logger.Error("LLM responses stream closed with error", args...)
			return
		}
		if err := l.inner.Err(); err != nil {
			args := append([]any{"duration", duration, "request_id", l.requestID}, ResponseRequestDiagnosticFields(l.request)...)
			args = append(args, ErrorDiagnosticFields(err)...)
			l.logger.Error("LLM responses stream closed with error", args...)
			return
		}
		l.logger.Debug("LLM responses stream completed", "duration", duration, "request_id", l.requestID, "model", l.request.Model)
	})
	return l.closeErr
}

func newOpenAIResponseStream(resp *http.Response) *openAIResponseStream {
	return &openAIResponseStream{resp: resp, framer: newSSEFrameReader(resp.Body)}
}

func (s *openAIResponseStream) Next() bool {
	if s.done || s.err != nil {
		return false
	}

	for {
		frame, ok, err := s.framer.Next()
		if err != nil {
			if err == io.EOF {
				s.done = true
				return false
			}
			wrapped := WrapOperationalError("response stream read error", err)
			var llmErr *LLMError
			if errors.As(wrapped, &llmErr) && (llmErr.Code == ErrCodeTimeout || llmErr.Code == ErrCodeCancelled || llmErr.Code == ErrCodeNetwork) {
				s.err = wrapped
			} else {
				s.err = NewLLMError(ErrCodeParse, err.Error(), nil)
			}
			return false
		}
		if !ok {
			s.done = true
			return false
		}
		if frame.Done {
			s.done = true
			return false
		}
		if s.dispatch(frame) {
			return true
		}
		if s.done || s.err != nil {
			return false
		}
	}
}

func (s *openAIResponseStream) dispatch(frame sseFrame) bool {
	var decoded oaiResponseEvent
	if err := json.Unmarshal(frame.Data, &decoded); err != nil {
		s.err = NewLLMError(ErrCodeParse, "parse response stream event", err)
		return false
	}
	if decoded.Type == "" {
		decoded.Type = frame.Event
	}
	s.current = convertResponseEvent(decoded)
	return true
}

func (s *openAIResponseStream) Current() ResponseEvent {
	return s.current
}

func (s *openAIResponseStream) Err() error {
	return s.err
}

func (s *openAIResponseStream) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}

type oaiResponseObject struct {
	ID          string              `json:"id"`
	Object      string              `json:"object"`
	Status      string              `json:"status"`
	Model       string              `json:"model"`
	Output      []oaiResponseOutput `json:"output,omitempty"`
	Usage       *ResponseUsage      `json:"usage,omitempty"`
	CreatedAt   float64             `json:"created_at,omitempty"`
	CompletedAt float64             `json:"completed_at,omitempty"`
}

type oaiResponseOutput struct {
	Type    string                  `json:"type"`
	ID      string                  `json:"id,omitempty"`
	Role    string                  `json:"role,omitempty"`
	Status  string                  `json:"status,omitempty"`
	Content []ResponseOutputContent `json:"content,omitempty"`
	Summary []ResponseOutputContent `json:"summary,omitempty"`
}

type oaiResponseEvent struct {
	Type     string             `json:"type"`
	Delta    string             `json:"delta,omitempty"`
	Item     *oaiResponseOutput `json:"item,omitempty"`
	Response *oaiResponseObject `json:"response,omitempty"`
}

func convertResponseObject(response *oaiResponseObject) *ResponseObject {
	if response == nil {
		return nil
	}
	result := &ResponseObject{
		ID:          response.ID,
		Object:      defaultResponseString(response.Object, "response"),
		Status:      response.Status,
		Model:       response.Model,
		Output:      make([]ResponseOutput, 0, len(response.Output)),
		Usage:       response.Usage,
		CreatedAt:   unixFloatToTime(response.CreatedAt),
		CompletedAt: unixFloatToTime(response.CompletedAt),
	}
	for _, output := range response.Output {
		result.Output = append(result.Output, ResponseOutput{
			Type:    defaultResponseString(output.Type, "message"),
			ID:      output.ID,
			Role:    output.Role,
			Status:  output.Status,
			Content: append([]ResponseOutputContent(nil), output.Content...),
			Summary: append([]ResponseOutputContent(nil), output.Summary...),
		})
	}
	return result
}

func convertResponseEvent(event oaiResponseEvent) ResponseEvent {
	result := ResponseEvent{Type: event.Type, Delta: event.Delta}
	if event.Item != nil {
		item := ResponseOutput{
			Type:    defaultResponseString(event.Item.Type, "message"),
			ID:      event.Item.ID,
			Role:    event.Item.Role,
			Status:  event.Item.Status,
			Content: append([]ResponseOutputContent(nil), event.Item.Content...),
			Summary: append([]ResponseOutputContent(nil), event.Item.Summary...),
		}
		result.Item = &item
	}
	if event.Response != nil {
		result.Response = convertResponseObject(event.Response)
	}
	return result
}

func defaultResponseString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func unixFloatToTime(value float64) time.Time {
	if value <= 0 {
		return time.Time{}
	}
	seconds := int64(value)
	nanos := int64((value - float64(seconds)) * float64(time.Second))
	return time.Unix(seconds, nanos).UTC()
}
