package slm

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// LifecycleEvent describes a unary or streaming lifecycle transition.
type LifecycleEvent struct {
	OperationID     string
	RequestID       string
	Model           string
	Context         context.Context
	Request         *Request
	Response        *Response
	ResponseRequest *ResponseRequest
	ResponseObject  *ResponseObject
	Duration        time.Duration
	Err             error
	Stream          bool
}

func (e LifecycleEvent) TokenUsage() (prompt, completion, total int) {
	if e.Response != nil {
		return e.Response.Usage.PromptTokens, e.Response.Usage.CompletionTokens, e.Response.Usage.TotalTokens
	}
	if e.ResponseObject != nil && e.ResponseObject.Usage != nil {
		return e.ResponseObject.Usage.InputTokens, e.ResponseObject.Usage.OutputTokens, e.ResponseObject.Usage.TotalTokens
	}
	return 0, 0, 0
}

func (e LifecycleEvent) FinishStatus() string {
	if e.Response != nil {
		return e.Response.FinishReason
	}
	if e.ResponseObject != nil {
		return e.ResponseObject.Status
	}
	return ""
}

// LifecycleObserver receives request and stream lifecycle callbacks.
//
// Implementations can export metrics, tracing spans, audit records, or custom
// diagnostics without duplicating middleware logic.
type LifecycleObserver interface {
	OnRequestStart(context.Context, LifecycleEvent)
	OnRequestFinish(context.Context, LifecycleEvent)
	OnStreamStart(context.Context, LifecycleEvent)
	OnStreamConnected(context.Context, LifecycleEvent)
	OnStreamFinish(context.Context, LifecycleEvent)
}

type compositeLifecycleObserver struct {
	observers []LifecycleObserver
}

func (o compositeLifecycleObserver) OnRequestStart(ctx context.Context, event LifecycleEvent) {
	for _, observer := range o.observers {
		observer.OnRequestStart(ctx, event)
	}
}

func (o compositeLifecycleObserver) OnRequestFinish(ctx context.Context, event LifecycleEvent) {
	for _, observer := range o.observers {
		observer.OnRequestFinish(ctx, event)
	}
}

func (o compositeLifecycleObserver) OnStreamStart(ctx context.Context, event LifecycleEvent) {
	for _, observer := range o.observers {
		observer.OnStreamStart(ctx, event)
	}
}

func (o compositeLifecycleObserver) OnStreamConnected(ctx context.Context, event LifecycleEvent) {
	for _, observer := range o.observers {
		observer.OnStreamConnected(ctx, event)
	}
}

func (o compositeLifecycleObserver) OnStreamFinish(ctx context.Context, event LifecycleEvent) {
	for _, observer := range o.observers {
		observer.OnStreamFinish(ctx, event)
	}
}

// LifecycleObserverMiddleware creates unary and stream middlewares that emit
// structured lifecycle events to the supplied observer.
type eventBuilder[Req any, Resp any, StreamResp any] struct {
	model         func(Req) string
	buildEvent    func(opID, reqID, model string, ctx context.Context, req Req, stream bool) LifecycleEvent
	setResponse   func(event *LifecycleEvent, resp Resp)
	setStreamResp func(event *LifecycleEvent, stream StreamResp)
}

var chatEventBuilder = eventBuilder[*Request, *Response, StreamIterator]{
	model: requestModel,
	buildEvent: func(opID, reqID, model string, ctx context.Context, req *Request, stream bool) LifecycleEvent {
		return LifecycleEvent{OperationID: opID, RequestID: reqID, Model: model, Context: ctx, Request: req, Stream: stream}
	},
	setResponse:   func(event *LifecycleEvent, resp *Response) { event.Response = resp },
	setStreamResp: func(event *LifecycleEvent, stream StreamIterator) { event.Response = stream.Response() },
}

var responsesEventBuilder = eventBuilder[*ResponseRequest, *ResponseObject, ResponseStream]{
	model: responseRequestModel,
	buildEvent: func(opID, reqID, model string, ctx context.Context, req *ResponseRequest, stream bool) LifecycleEvent {
		return LifecycleEvent{OperationID: opID, RequestID: reqID, Model: model, Context: ctx, ResponseRequest: req, Stream: stream}
	},
	setResponse:   func(event *LifecycleEvent, resp *ResponseObject) { event.ResponseObject = resp },
	setStreamResp: func(event *LifecycleEvent, stream ResponseStream) {},
}

func lifecycleObserverMiddleware[Req any, Resp any, StreamResp any](
	observer LifecycleObserver,
	builder eventBuilder[Req, Resp, StreamResp],
) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
	passUnary := func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] { return next }
	passStream := func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] { return next }
	if observer == nil {
		return passUnary, passStream
	}

	unary := func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] {
		return func(ctx context.Context, req Req) (Resp, error) {
			event := builder.buildEvent(generateOperationID(), GetRequestID(ctx), builder.model(req), ctx, req, false)
			observer.OnRequestStart(ctx, event)
			start := time.Now()
			resp, err := next(ctx, req)
			builder.setResponse(&event, resp)
			event.Err = err
			event.Duration = time.Since(start)
			observer.OnRequestFinish(ctx, event)
			return resp, err
		}
	}

	stream := func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] {
		return func(ctx context.Context, req Req) (StreamResp, error) {
			var zero StreamResp
			event := builder.buildEvent(generateOperationID(), GetRequestID(ctx), builder.model(req), ctx, req, true)
			observer.OnStreamStart(ctx, event)
			start := time.Now()
			result, err := next(ctx, req)
			if err != nil {
				event.Err = err
				event.Duration = time.Since(start)
				observer.OnStreamFinish(ctx, event)
				return zero, err
			}
			event.Duration = time.Since(start)
			observer.OnStreamConnected(ctx, event)
			return wrapObservedStream(observer, ctx, start, event, result, builder), nil
		}
	}

	return unary, stream
}

type observedStreamCore struct {
	observer  LifecycleObserver
	ctx       context.Context
	start     time.Time
	event     LifecycleEvent
	onFinish  func(*LifecycleEvent)
	closeOnce syncOnceErr
}

func (c *observedStreamCore) onClose(inner BaseStream) error {
	return c.closeOnce.Do(func() error {
		closeErr := inner.Close()
		c.event.Duration = time.Since(c.start)
		if c.onFinish != nil {
			c.onFinish(&c.event)
		}
		if closeErr != nil {
			c.event.Err = closeErr
		} else {
			c.event.Err = inner.Err()
		}
		c.observer.OnStreamFinish(c.ctx, c.event)
		return closeErr
	})
}

type observedChatStream struct {
	streamIteratorWrapper
	core observedStreamCore
}

func (o *observedChatStream) Next() bool {
	ok := o.inner.Next()
	if !ok {
		_ = o.core.onClose(o.inner)
	}
	return ok
}
func (o *observedChatStream) Interrupt(err error) { interruptStreamIterator(o.inner, err) }
func (o *observedChatStream) Close() error {
	return o.core.onClose(o.inner)
}

type observedResponseStreamWrapper struct {
	responseStreamWrapper
	core observedStreamCore
}

func (o *observedResponseStreamWrapper) Next() bool {
	ok := o.inner.Next()
	if !ok {
		_ = o.core.onClose(o.inner)
	}
	return ok
}
func (o *observedResponseStreamWrapper) Close() error {
	return o.core.onClose(o.inner)
}

func wrapObservedStream[Req any, Resp any, StreamResp any](
	observer LifecycleObserver,
	ctx context.Context,
	start time.Time,
	event LifecycleEvent,
	inner StreamResp,
	builder eventBuilder[Req, Resp, StreamResp],
) StreamResp {
	core := observedStreamCore{observer: observer, ctx: ctx, start: start, event: event}
	builder.setStreamResp(&event, inner)
	switch s := any(inner).(type) {
	case StreamIterator:
		core.onFinish = func(e *LifecycleEvent) {
			e.Response = s.Response()
		}
		w := &observedChatStream{streamIteratorWrapper: streamIteratorWrapper{inner: s}, core: core}
		if result, ok := any(w).(StreamResp); ok {
			return result
		}
	case ResponseStream:
		core.onFinish = func(e *LifecycleEvent) {
			if current := s.Current(); current.Response != nil {
				e.ResponseObject = current.Response
			}
		}
		w := &observedResponseStreamWrapper{responseStreamWrapper: responseStreamWrapper{inner: s}, core: core}
		if result, ok := any(w).(StreamResp); ok {
			return result
		}
	}
	return inner
}

func LifecycleObserverMiddleware(observer LifecycleObserver) (Middleware, StreamMiddleware) {
	return lifecycleObserverMiddleware(observer, chatEventBuilder)
}

func ResponseLifecycleObserverMiddleware(observer LifecycleObserver) (ResponseMiddleware, ResponseStreamMiddleware) {
	return lifecycleObserverMiddleware(observer, responsesEventBuilder)
}

type syncOnceErr struct {
	once sync.Once
	err  error
}

func (s *syncOnceErr) Do(fn func() error) error {
	s.once.Do(func() {
		s.err = fn()
	})
	return s.err
}

func generateOperationID() string {
	return fmt.Sprintf("op_%d_%d", time.Now().UnixNano(), operationCounter.Add(1))
}

var operationCounter atomic.Int64

func responseRequestModel(req *ResponseRequest) string {
	if req == nil {
		return ""
	}
	return req.Model
}
