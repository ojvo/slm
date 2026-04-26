package slm

import (
	"context"
	"sync"
)

// Attribute is a backend-agnostic observability attribute.
type Attribute struct {
	Key   string
	Value any
}

// Int64Counter records additive integer metrics.
type Int64Counter interface {
	Add(context.Context, int64, ...Attribute)
}

// Float64Histogram records floating-point observations.
type Float64Histogram interface {
	Record(context.Context, float64, ...Attribute)
}

// Meter creates metric instruments used by MetricsObserver.
type Meter interface {
	Int64Counter(name, description, unit string) Int64Counter
	Float64Histogram(name, description, unit string) Float64Histogram
}

// Span represents an in-flight trace span.
type Span interface {
	SetAttributes(...Attribute)
	RecordError(error)
	End()
}

// Tracer starts spans for request and stream lifecycles.
type Tracer interface {
	Start(context.Context, string, ...Attribute) (context.Context, Span)
}

// MetricsObserverOptions configures metric naming.
type MetricsObserverOptions struct {
	Namespace string
}

// TraceObserverOptions configures span naming.
type TraceObserverOptions struct {
	SpanPrefix string
}

// NewLogObserver returns a lifecycle observer that emits structured logs.
func NewLogObserver(logger Logger) LifecycleObserver {
	return &logObserver{logger: resolvedLogger(logger)}
}

// NewMetricsObserver returns an official metrics adapter built on LifecycleObserver.
func NewMetricsObserver(meter Meter, opts MetricsObserverOptions) LifecycleObserver {
	if meter == nil {
		return nil
	}
	prefix := stringsOrDefault(opts.Namespace, "slm")
	return &metricsObserver{
		started:          meter.Int64Counter(prefix+".requests_started", "Number of LLM requests started", "1"),
		finished:         meter.Int64Counter(prefix+".requests_finished", "Number of LLM requests finished", "1"),
		streamConnected:  meter.Int64Counter(prefix+".streams_connected", "Number of LLM streams connected", "1"),
		errors:           meter.Int64Counter(prefix+".request_errors", "Number of LLM request errors", "1"),
		duration:         meter.Float64Histogram(prefix+".request_duration_seconds", "LLM request duration", "s"),
		promptTokens:     meter.Float64Histogram(prefix+".prompt_tokens", "Prompt token count", "1"),
		completionTokens: meter.Float64Histogram(prefix+".completion_tokens", "Completion token count", "1"),
		totalTokens:      meter.Float64Histogram(prefix+".total_tokens", "Total token count", "1"),
	}
}

// NewTraceObserver returns an official trace adapter built on LifecycleObserver.
func NewTraceObserver(tracer Tracer, opts TraceObserverOptions) LifecycleObserver {
	if tracer == nil {
		return nil
	}
	return &traceObserver{
		tracer:     tracer,
		spanPrefix: stringsOrDefault(opts.SpanPrefix, "slm"),
		spans:      make(map[string]Span),
	}
}

type metricsObserver struct {
	started          Int64Counter
	finished         Int64Counter
	streamConnected  Int64Counter
	errors           Int64Counter
	duration         Float64Histogram
	promptTokens     Float64Histogram
	completionTokens Float64Histogram
	totalTokens      Float64Histogram
}

type logObserver struct {
	logger Logger
}

func (o *logObserver) OnRequestStart(ctx context.Context, event LifecycleEvent) {
	logRequestStart(o.logger, "LLM request start", GetRequestID(ctx), RequestDiagnosticFields(event.Request))
}

func (o *logObserver) OnRequestFinish(ctx context.Context, event LifecycleEvent) {
	requestID := GetRequestID(ctx)
	requestFields := RequestDiagnosticFields(event.Request)
	if event.Err != nil {
		logRequestFailure(o.logger, "LLM request failed", event.Duration, requestID, requestFields, event.Err)
		return
	}
	if event.Response != nil {
		logRequestCompleted(o.logger, "LLM request completed", event.Duration, requestID,
			"tokens", event.Response.Usage.TotalTokens,
			"finish_reason", event.Response.FinishReason,
		)
		return
	}
	logRequestCompleted(o.logger, "LLM request completed", event.Duration, requestID)
}

func (o *logObserver) OnStreamStart(ctx context.Context, event LifecycleEvent) {
	logRequestStart(o.logger, "LLM stream start", GetRequestID(ctx), RequestDiagnosticFields(event.Request))
}

func (o *logObserver) OnStreamConnected(ctx context.Context, _ LifecycleEvent) {
	logRequestCompleted(o.logger, "LLM stream connected", 0, GetRequestID(ctx))
}

func (o *logObserver) OnStreamFinish(ctx context.Context, event LifecycleEvent) {
	requestID := GetRequestID(ctx)
	requestFields := RequestDiagnosticFields(event.Request)
	if event.Err != nil {
		logRequestFailure(o.logger, "LLM stream closed with error", event.Duration, requestID, requestFields, event.Err)
		return
	}
	tokens := 0
	if event.Response != nil {
		tokens = event.Response.Usage.TotalTokens
	}
	logRequestCompleted(o.logger, "LLM stream completed", event.Duration, requestID,
		"tokens", tokens,
		"model", requestModel(event.Request),
	)
}

func (m *metricsObserver) OnRequestStart(ctx context.Context, event LifecycleEvent) {
	m.started.Add(ctx, 1, metricAttributes(event)...)
}

func (m *metricsObserver) OnRequestFinish(ctx context.Context, event LifecycleEvent) {
	m.recordFinish(ctx, event)
}

func (m *metricsObserver) OnStreamStart(ctx context.Context, event LifecycleEvent) {
	m.started.Add(ctx, 1, metricAttributes(event)...)
}

func (m *metricsObserver) OnStreamConnected(ctx context.Context, event LifecycleEvent) {
	m.streamConnected.Add(ctx, 1, metricAttributes(event)...)
}

func (m *metricsObserver) OnStreamFinish(ctx context.Context, event LifecycleEvent) {
	m.recordFinish(ctx, event)
}

func (m *metricsObserver) recordFinish(ctx context.Context, event LifecycleEvent) {
	attrs := metricAttributes(event)
	m.finished.Add(ctx, 1, attrs...)
	m.duration.Record(ctx, event.Duration.Seconds(), attrs...)
	if event.Err != nil {
		m.errors.Add(ctx, 1, attrs...)
	}
	if event.Response != nil {
		m.promptTokens.Record(ctx, float64(event.Response.Usage.PromptTokens), attrs...)
		m.completionTokens.Record(ctx, float64(event.Response.Usage.CompletionTokens), attrs...)
		m.totalTokens.Record(ctx, float64(event.Response.Usage.TotalTokens), attrs...)
	}
}

type traceObserver struct {
	tracer     Tracer
	spanPrefix string
	mu         sync.Mutex
	spans      map[string]Span
}

func (t *traceObserver) OnRequestStart(ctx context.Context, event LifecycleEvent) {
	t.start(ctx, event, t.spanPrefix+".request")
}

func (t *traceObserver) OnRequestFinish(ctx context.Context, event LifecycleEvent) {
	t.finish(event)
}

func (t *traceObserver) OnStreamStart(ctx context.Context, event LifecycleEvent) {
	t.start(ctx, event, t.spanPrefix+".stream")
}

func (t *traceObserver) OnStreamConnected(_ context.Context, event LifecycleEvent) {
	if span, ok := t.lookup(event.OperationID); ok {
		span.SetAttributes(Attribute{Key: "stream.connected", Value: true})
	}
}

func (t *traceObserver) OnStreamFinish(_ context.Context, event LifecycleEvent) {
	t.finish(event)
}

func (t *traceObserver) start(ctx context.Context, event LifecycleEvent, name string) {
	_, span := t.tracer.Start(ctx, name, metricAttributes(event)...)
	t.mu.Lock()
	t.spans[event.OperationID] = span
	t.mu.Unlock()
}

func (t *traceObserver) finish(event LifecycleEvent) {
	span, ok := t.take(event.OperationID)
	if !ok {
		return
	}
	attrs := metricAttributes(event)
	attrs = append(attrs, Attribute{Key: "duration_ms", Value: float64(event.Duration.Milliseconds())})
	if event.Response != nil {
		attrs = append(attrs,
			Attribute{Key: "finish_reason", Value: event.Response.FinishReason},
			Attribute{Key: "usage.prompt_tokens", Value: event.Response.Usage.PromptTokens},
			Attribute{Key: "usage.completion_tokens", Value: event.Response.Usage.CompletionTokens},
			Attribute{Key: "usage.total_tokens", Value: event.Response.Usage.TotalTokens},
		)
	}
	span.SetAttributes(attrs...)
	if event.Err != nil {
		span.RecordError(event.Err)
	}
	span.End()
}

func (t *traceObserver) lookup(key string) (Span, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	span, ok := t.spans[key]
	return span, ok
}

func (t *traceObserver) take(key string) (Span, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	span, ok := t.spans[key]
	if ok {
		delete(t.spans, key)
	}
	return span, ok
}

func metricAttributes(event LifecycleEvent) []Attribute {
	attrs := []Attribute{
		{Key: "operation_id", Value: event.OperationID},
		{Key: "request_id", Value: event.RequestID},
		{Key: "model", Value: event.Model},
		{Key: "stream", Value: event.Stream},
	}
	if negotiated, ok := GetNegotiatedCapabilities(event.Context); ok {
		if negotiated.State.Source != "" {
			attrs = append(attrs, Attribute{Key: "capabilities.source", Value: negotiated.State.Source})
		}
		attrs = append(attrs,
			Attribute{Key: "capabilities.known", Value: negotiated.Known},
			Attribute{Key: "capabilities.stale", Value: negotiated.Stale},
		)
		attrs = append(attrs, negotiated.Requested.attributes()...)
	}
	return attrs
}

func stringsOrDefault(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}
