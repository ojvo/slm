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
	OperationID string
	RequestID   string
	Model       string
	Context     context.Context
	Request     *Request
	Response    *Response
	Duration    time.Duration
	Err         error
	Stream      bool
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
func LifecycleObserverMiddleware(observer LifecycleObserver) (Middleware, StreamMiddleware) {
	return lifecycleObserverMiddlewareWithOwner(observer, telemetryOwnerObserver)
}

func lifecycleObserverMiddlewareWithOwner(observer LifecycleObserver, owner telemetryOwner) (Middleware, StreamMiddleware) {
	unary := func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			ctx, owns := claimTelemetryOwnership(ctx, owner)
			if !owns {
				return next(ctx, req)
			}

			event := LifecycleEvent{OperationID: generateOperationID(), RequestID: GetRequestID(ctx), Model: requestModel(req), Context: ctx, Request: req}
			observer.OnRequestStart(ctx, event)
			start := time.Now()

			resp, err := next(ctx, req)
			event.Response = resp
			event.Err = err
			event.Duration = time.Since(start)
			observer.OnRequestFinish(ctx, event)
			return resp, err
		}
	}

	stream := func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			ctx, owns := claimTelemetryOwnership(ctx, owner)
			if !owns {
				return next(ctx, req)
			}

			event := LifecycleEvent{OperationID: generateOperationID(), RequestID: GetRequestID(ctx), Model: requestModel(req), Context: ctx, Request: req, Stream: true}
			observer.OnStreamStart(ctx, event)
			start := time.Now()

			iter, err := next(ctx, req)
			if err != nil {
				event.Err = err
				event.Duration = time.Since(start)
				observer.OnStreamFinish(ctx, event)
				return nil, err
			}

			event.Duration = time.Since(start)
			observer.OnStreamConnected(ctx, event)
			return &observedStreamIterator{inner: iter, observer: observer, ctx: ctx, start: start, event: event}, nil
		}
	}

	return unary, stream
}

type observedStreamIterator struct {
	inner     StreamIterator
	observer  LifecycleObserver
	ctx       context.Context
	start     time.Time
	event     LifecycleEvent
	closeOnce syncOnceErr
}

func (o *observedStreamIterator) Next() bool {
	ok := o.inner.Next()
	if !ok {
		_ = o.Close()
	}
	return ok
}
func (o *observedStreamIterator) Chunk() []byte       { return o.inner.Chunk() }
func (o *observedStreamIterator) Text() string        { return o.inner.Text() }
func (o *observedStreamIterator) FullText() string    { return o.inner.FullText() }
func (o *observedStreamIterator) Err() error          { return o.inner.Err() }
func (o *observedStreamIterator) Usage() *Usage       { return o.inner.Usage() }
func (o *observedStreamIterator) Response() *Response { return o.inner.Response() }
func (o *observedStreamIterator) Interrupt(err error) { interruptStreamIterator(o.inner, err) }

func (o *observedStreamIterator) Close() error {
	return o.closeOnce.Do(func() error {
		closeErr := o.inner.Close()
		o.event.Duration = time.Since(o.start)
		o.event.Response = o.inner.Response()
		if closeErr != nil {
			o.event.Err = closeErr
		} else {
			o.event.Err = o.inner.Err()
		}
		observer := o.observer
		observer.OnStreamFinish(o.ctx, o.event)
		return closeErr
	})
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
