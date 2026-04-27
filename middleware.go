package slm

import (
	"context"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// -----------------------------------------------------------------------------
// Pipeline Core Types
// -----------------------------------------------------------------------------

type PipelineHandler[Req any, Resp any] func(ctx context.Context, req Req) (Resp, error)
type PipelineStreamHandler[Req any, StreamResp any] func(ctx context.Context, req Req) (StreamResp, error)
type PipelineMiddleware[Req any, Resp any] func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp]
type PipelineStreamMiddleware[Req any, StreamResp any] func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp]

type PipelineEngine[Req any, Resp any, StreamResp any] interface {
	Generate(ctx context.Context, req Req) (Resp, error)
	Stream(ctx context.Context, req Req) (StreamResp, error)
}

type PipelineCloser interface{ Close() }

type genericPipelineEngine[Req any, Resp any, StreamResp any] struct {
	inner             PipelineEngine[Req, Resp, StreamResp]
	middlewares       []PipelineMiddleware[Req, Resp]
	streamMiddlewares []PipelineStreamMiddleware[Req, StreamResp]
	closers           []func()
	closeOnce         sync.Once
}

func ChainPipelineWithStreamAndClosers[Req any, Resp any, StreamResp any](engine PipelineEngine[Req, Resp, StreamResp], middlewares []PipelineMiddleware[Req, Resp], streamMiddlewares []PipelineStreamMiddleware[Req, StreamResp], closers []func()) PipelineEngine[Req, Resp, StreamResp] {
	return &genericPipelineEngine[Req, Resp, StreamResp]{inner: engine, middlewares: middlewares, streamMiddlewares: streamMiddlewares, closers: closers}
}

func (m *genericPipelineEngine[Req, Resp, StreamResp]) Generate(ctx context.Context, req Req) (Resp, error) {
	handler := m.inner.Generate
	for i := len(m.middlewares) - 1; i >= 0; i-- {
		handler = m.middlewares[i](handler)
	}
	return handler(ctx, req)
}

func (m *genericPipelineEngine[Req, Resp, StreamResp]) Stream(ctx context.Context, req Req) (StreamResp, error) {
	handler := m.inner.Stream
	for i := len(m.streamMiddlewares) - 1; i >= 0; i-- {
		handler = m.streamMiddlewares[i](handler)
	}
	return handler(ctx, req)
}

func (m *genericPipelineEngine[Req, Resp, StreamResp]) Close() {
	m.closeOnce.Do(func() {
		for _, closer := range m.closers {
			closer()
		}
	})
}

// -----------------------------------------------------------------------------
// Config Types
// -----------------------------------------------------------------------------

type RateLimitConfig struct {
	Limit float64
	Burst int
}

type CrossCuttingMiddlewareOptions struct {
	Timeout   time.Duration
	Retry     *RetryConfig
	RateLimit *RateLimitConfig
}

type StandardMiddlewareOptions struct {
	DefaultModel       string
	Capabilities       *CapabilityNegotiationOptions
	Observers          []LifecycleObserver
	EnableRequestID    bool
	RequestIDGenerator func() string
	CrossCutting       CrossCuttingMiddlewareOptions
}

type Handler = PipelineHandler[*Request, *Response]
type StreamHandler = PipelineStreamHandler[*Request, StreamIterator]
type Middleware = PipelineMiddleware[*Request, *Response]
type StreamMiddleware = PipelineStreamMiddleware[*Request, StreamIterator]

type ResponseHandler = PipelineHandler[*ResponseRequest, *ResponseObject]
type ResponseStreamHandler = PipelineStreamHandler[*ResponseRequest, ResponseStream]
type ResponseMiddleware = PipelineMiddleware[*ResponseRequest, *ResponseObject]
type ResponseStreamMiddleware = PipelineStreamMiddleware[*ResponseRequest, ResponseStream]

type MiddlewareSuite[Req any, Resp any, StreamResp any] struct {
	Normalize  func(defaultModel string) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp])
	RequestID  func(generator func() string) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp])
	Capability func(opts CapabilityNegotiationOptions) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp])
	Observer   func(observer LifecycleObserver) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp])
	Timeout    func(timeout time.Duration) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp])
	Retry      func(cfg RetryConfig) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp])
	RateLimit  func(limit float64, burst int) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp], func())
}

type middlewareSuiteCallbacks[Req any, Resp any, StreamResp any] struct {
	normalize            func(Req, string, bool) Req
	setRequestIDMeta     func(Req, string) Req
	negotiate            func(context.Context, Req, CapabilityNegotiationOptions) (context.Context, Req, error)
	observerBuilder      eventBuilder[Req, Resp, StreamResp]
	timeoutErr           string
	streamTimeoutErr     string
	streamReadTimeoutErr string
	streamCanInterrupt   bool
	cloneForRetry        func(Req) Req
}

func buildMiddlewareSuite[Req any, Resp any, StreamResp any](cb middlewareSuiteCallbacks[Req, Resp, StreamResp]) MiddlewareSuite[Req, Resp, StreamResp] {
	return MiddlewareSuite[Req, Resp, StreamResp]{
		Normalize: func(defaultModel string) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
			return normalizeMiddleware[Req, Resp, StreamResp](defaultModel, cb.normalize)
		},
		RequestID: func(generator func() string) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
			return requestIDMiddleware[Req, Resp](generator, cb.setRequestIDMeta),
				requestIDStreamMiddleware[Req, StreamResp](generator, cb.setRequestIDMeta)
		},
		Capability: func(opts CapabilityNegotiationOptions) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
			return capabilityNegotiationMiddleware[Req, Resp, StreamResp](opts, cb.negotiate)
		},
		Observer: func(observer LifecycleObserver) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
			return lifecycleObserverMiddleware(observer, cb.observerBuilder)
		},
		Timeout: func(timeout time.Duration) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
			return timeoutMiddleware[Req, Resp](timeout, cb.timeoutErr),
				timeoutStreamMiddleware[Req, Resp, StreamResp](timeout, cb.streamTimeoutErr, cb.streamReadTimeoutErr, cb.streamCanInterrupt)
		},
		Retry: func(cfg RetryConfig) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
			return retryMiddlewareWithConfig[Req, Resp, StreamResp](cfg, cb.cloneForRetry)
		},
		RateLimit: func(limit float64, burst int) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp], func()) {
			return rateLimitMiddlewares[Req, Resp, StreamResp](limit, burst)
		},
	}
}

func setRequestMetaForID(req *Request, id string) *Request {
	cloned := cloneRequestForMetadata(req)
	if cloned.Meta == nil {
		cloned.Meta = make(map[string]any)
	}
	cloned.Meta["request_id"] = id
	return cloned
}

var chatSuite = buildMiddlewareSuite[*Request, *Response, StreamIterator](middlewareSuiteCallbacks[*Request, *Response, StreamIterator]{
	normalize:            normalizeRequestForOperation,
	setRequestIDMeta:     setRequestMetaForID,
	negotiate:            NegotiateRequestCapabilities,
	observerBuilder:      chatEventBuilder,
	timeoutErr:           "request timeout",
	streamTimeoutErr:     "stream request timeout",
	streamReadTimeoutErr: "stream read timeout",
	streamCanInterrupt:   true,
	cloneForRetry: func(req *Request) *Request {
		return cloneRequestForMetadata(req)
	},
})

var responsesSuite = buildMiddlewareSuite[*ResponseRequest, *ResponseObject, ResponseStream](middlewareSuiteCallbacks[*ResponseRequest, *ResponseObject, ResponseStream]{
	normalize:            normalizeResponseRequestForOperation,
	setRequestIDMeta:     nil,
	negotiate:            NegotiateResponseCapabilities,
	observerBuilder:      responsesEventBuilder,
	timeoutErr:           "response request timeout",
	streamTimeoutErr:     "response stream request timeout",
	streamReadTimeoutErr: "response stream read timeout",
	streamCanInterrupt:   false,
	cloneForRetry: func(req *ResponseRequest) *ResponseRequest {
		return cloneResponseRequest(req)
	},
})

func applyStandardMiddleware[Req any, Resp any, StreamResp any](
	suite MiddlewareSuite[Req, Resp, StreamResp],
	engine PipelineEngine[Req, Resp, StreamResp],
	opts StandardMiddlewareOptions,
) PipelineEngine[Req, Resp, StreamResp] {
	var middlewares []PipelineMiddleware[Req, Resp]
	var streamMiddlewares []PipelineStreamMiddleware[Req, StreamResp]
	var closers []func()

	if opts.EnableRequestID && suite.RequestID != nil {
		unary, stream := suite.RequestID(opts.RequestIDGenerator)
		middlewares = append(middlewares, unary)
		streamMiddlewares = append(streamMiddlewares, stream)
	}

	if suite.Normalize != nil {
		unary, stream := suite.Normalize(opts.DefaultModel)
		middlewares = append(middlewares, unary)
		streamMiddlewares = append(streamMiddlewares, stream)
	}

	if opts.Capabilities != nil && opts.Capabilities.Resolver != nil && suite.Capability != nil {
		unary, stream := suite.Capability(*opts.Capabilities)
		middlewares = append(middlewares, unary)
		streamMiddlewares = append(streamMiddlewares, stream)
	}

	if len(opts.Observers) > 0 && suite.Observer != nil {
		observer := compositeLifecycleObserver{observers: opts.Observers}
		unary, stream := suite.Observer(observer)
		middlewares = append(middlewares, unary)
		streamMiddlewares = append(streamMiddlewares, stream)
	}

	if opts.CrossCutting.Timeout > 0 && suite.Timeout != nil {
		unary, stream := suite.Timeout(opts.CrossCutting.Timeout)
		middlewares = append(middlewares, unary)
		streamMiddlewares = append(streamMiddlewares, stream)
	}

	if opts.CrossCutting.Retry != nil && opts.CrossCutting.Retry.MaxAttempts > 1 && suite.Retry != nil {
		unary, stream := suite.Retry(*opts.CrossCutting.Retry)
		middlewares = append(middlewares, unary)
		streamMiddlewares = append(streamMiddlewares, stream)
	}

	if opts.CrossCutting.RateLimit != nil && suite.RateLimit != nil {
		unary, stream, closer := suite.RateLimit(opts.CrossCutting.RateLimit.Limit, opts.CrossCutting.RateLimit.Burst)
		middlewares = append(middlewares, unary)
		streamMiddlewares = append(streamMiddlewares, stream)
		closers = append(closers, closer)
	}

	if len(middlewares) == 0 && len(streamMiddlewares) == 0 {
		return engine
	}

	return ChainPipelineWithStreamAndClosers[Req, Resp, StreamResp](engine, middlewares, streamMiddlewares, closers)
}

// -----------------------------------------------------------------------------
// Generic Middleware Factories (unify Chat + Responses)
// -----------------------------------------------------------------------------

func timeoutMiddleware[Req any, Resp any](timeout time.Duration, errMsg string) PipelineMiddleware[Req, Resp] {
	return func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] {
		return func(ctx context.Context, req Req) (Resp, error) {
			var zero Resp
			ctx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()

			resp, err := next(ctx, req)
			if err != nil {
				if ctx.Err() == context.DeadlineExceeded {
					return zero, NewLLMError(ErrCodeTimeout, errMsg, ctx.Err())
				}
				return zero, err
			}
			return resp, nil
		}
	}
}

func rateLimitMiddlewares[Req any, Resp any, StreamResp any](limit float64, burst int) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp], func()) {
	limiter, closer := newLimiterWithCleanup(limit, burst)
	unary := func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] {
		return func(ctx context.Context, req Req) (Resp, error) {
			var zero Resp
			if err := limiter.Wait(ctx); err != nil {
				return zero, fmt.Errorf("rate limit exceeded: %w", err)
			}
			return next(ctx, req)
		}
	}
	stream := func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] {
		return func(ctx context.Context, req Req) (StreamResp, error) {
			var zero StreamResp
			if err := limiter.Wait(ctx); err != nil {
				return zero, fmt.Errorf("rate limit exceeded: %w", err)
			}
			return next(ctx, req)
		}
	}
	return unary, stream, closer
}

func requestIDMiddleware[Req any, Resp any](generator func() string, setMeta func(Req, string) Req) PipelineMiddleware[Req, Resp] {
	if generator == nil {
		generator = func() string { return generateRequestID() }
	}
	return func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] {
		return func(ctx context.Context, req Req) (Resp, error) {
			reqID := GetRequestID(ctx)
			if reqID == "" {
				reqID = generator()
				ctx = context.WithValue(ctx, ctxKeyRequestID{}, reqID)
			}
			if setMeta != nil {
				req = setMeta(req, reqID)
			}
			return next(ctx, req)
		}
	}
}

func requestIDStreamMiddleware[Req any, StreamResp any](generator func() string, setMeta func(Req, string) Req) PipelineStreamMiddleware[Req, StreamResp] {
	if generator == nil {
		generator = func() string { return generateRequestID() }
	}
	return func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] {
		return func(ctx context.Context, req Req) (StreamResp, error) {
			reqID := GetRequestID(ctx)
			if reqID == "" {
				reqID = generator()
				ctx = context.WithValue(ctx, ctxKeyRequestID{}, reqID)
			}
			if setMeta != nil {
				req = setMeta(req, reqID)
			}
			return next(ctx, req)
		}
	}
}

func capabilityNegotiationMiddleware[Req any, Resp any, StreamResp any](
	opts CapabilityNegotiationOptions,
	negotiate func(ctx context.Context, req Req, opts CapabilityNegotiationOptions) (context.Context, Req, error),
) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
	passUnary := func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] { return next }
	passStream := func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] { return next }
	if opts.Resolver == nil || negotiate == nil {
		return passUnary, passStream
	}
	unary := func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] {
		return func(ctx context.Context, req Req) (Resp, error) {
			var zero Resp
			ctx, req, err := negotiate(ctx, req, opts)
			if err != nil {
				return zero, err
			}
			return next(ctx, req)
		}
	}
	stream := func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] {
		return func(ctx context.Context, req Req) (StreamResp, error) {
			var zero StreamResp
			ctx, req, err := negotiate(ctx, req, opts)
			if err != nil {
				return zero, err
			}
			return next(ctx, req)
		}
	}
	return unary, stream
}

// -----------------------------------------------------------------------------
// Concrete middleware constructors (public API, delegate to generics)
// -----------------------------------------------------------------------------

func TimeoutMiddleware(timeout time.Duration) Middleware {
	return timeoutMiddleware[*Request, *Response](timeout, "request timeout")
}

func TimeoutStreamMiddleware(timeout time.Duration) StreamMiddleware {
	return timeoutStreamMiddleware[*Request, *Response, StreamIterator](timeout, "stream request timeout", "stream read timeout", true)
}

func RateLimitMiddlewares(limit float64, burst int) (Middleware, StreamMiddleware, func()) {
	return rateLimitMiddlewares[*Request, *Response, StreamIterator](limit, burst)
}

func RequestIDMiddleware(generator func() string) Middleware {
	return requestIDMiddleware[*Request, *Response](generator, func(req *Request, id string) *Request {
		cloned := cloneRequestForMetadata(req)
		if cloned.Meta == nil {
			cloned.Meta = make(map[string]any)
		}
		cloned.Meta["request_id"] = id
		return cloned
	})
}

func RequestIDStreamMiddleware(generator func() string) StreamMiddleware {
	inner := requestIDStreamMiddleware[*Request, StreamIterator](generator, func(req *Request, id string) *Request {
		cloned := cloneRequestForMetadata(req)
		if cloned.Meta == nil {
			cloned.Meta = make(map[string]any)
		}
		cloned.Meta["request_id"] = id
		return cloned
	})
	return func(next StreamHandler) StreamHandler {
		wrapped := inner(next)
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			iter, err := wrapped(ctx, req)
			if err != nil {
				return nil, err
			}
			return &requestIDStreamIterator{streamIteratorWrapper: streamIteratorWrapper{inner: iter}, requestID: GetRequestID(ctx)}, nil
		}
	}
}

func ResponseTimeoutMiddleware(timeout time.Duration) ResponseMiddleware {
	return timeoutMiddleware[*ResponseRequest, *ResponseObject](timeout, "response request timeout")
}

func ResponseStreamTimeoutMiddleware(timeout time.Duration) ResponseStreamMiddleware {
	return timeoutStreamMiddleware[*ResponseRequest, *ResponseObject, ResponseStream](timeout, "response stream request timeout", "response stream read timeout", false)
}

func ResponseRateLimitMiddlewares(limit float64, burst int) (ResponseMiddleware, ResponseStreamMiddleware, func()) {
	return rateLimitMiddlewares[*ResponseRequest, *ResponseObject, ResponseStream](limit, burst)
}

func ResponseRequestIDMiddleware(generator func() string) ResponseMiddleware {
	return requestIDMiddleware[*ResponseRequest, *ResponseObject](generator, nil)
}

func ResponseRequestIDStreamMiddleware(generator func() string) ResponseStreamMiddleware {
	return requestIDStreamMiddleware[*ResponseRequest, ResponseStream](generator, nil)
}

func ResponseCapabilityNegotiationMiddleware(opts CapabilityNegotiationOptions) (ResponseMiddleware, ResponseStreamMiddleware) {
	return capabilityNegotiationMiddleware[*ResponseRequest, *ResponseObject, ResponseStream](opts, func(ctx context.Context, req *ResponseRequest, o CapabilityNegotiationOptions) (context.Context, *ResponseRequest, error) {
		return NegotiateResponseCapabilities(ctx, req, o)
	})
}

// -----------------------------------------------------------------------------
// Stream Timeout Middleware (Generic)
// -----------------------------------------------------------------------------

func timeoutStreamMiddleware[Req any, Resp any, StreamResp any](timeout time.Duration, reqErrMsg, readErrMsg string, canInterrupt bool) PipelineStreamMiddleware[Req, StreamResp] {
	return func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] {
		return func(ctx context.Context, req Req) (StreamResp, error) {
			var zero StreamResp
			ctx, cancel := context.WithTimeout(ctx, timeout)
			result, err := next(ctx, req)
			if err != nil {
				cancel()
				if ctx.Err() == context.DeadlineExceeded {
					return zero, NewLLMError(ErrCodeTimeout, reqErrMsg, ctx.Err())
				}
				return zero, err
			}
			var onCtxDone func()
			if canInterrupt {
				if si, ok := any(result).(InterruptibleStreamIterator); ok {
					onCtxDone = func() { si.Interrupt(NewLLMError(ErrCodeTimeout, readErrMsg, ctx.Err())) }
				}
			}
			w := wrapTimeoutStream(result, cancel, ctx, readErrMsg, onCtxDone)
			if wrapped, ok := any(w).(StreamResp); ok {
				return wrapped, nil
			}
			return result, nil
		}
	}
}

func wrapTimeoutStream(inner any, cancel context.CancelFunc, ctx context.Context, errMsg string, onCtxDone func()) any {
	base := timeoutBaseStream{
		inner:     asBaseStream(inner),
		cancel:    cancel,
		ctx:       ctx,
		errMsg:    errMsg,
		onCtxDone: onCtxDone,
	}
	switch s := inner.(type) {
	case StreamIterator:
		return &timeoutStreamIterator{streamIteratorWrapper: streamIteratorWrapper{inner: s}, base: base}
	case ResponseStream:
		return &timeoutResponseStream{responseStreamWrapper: responseStreamWrapper{inner: s}, base: base}
	}
	return nil
}

// -----------------------------------------------------------------------------
// Timeout Stream Internals
// -----------------------------------------------------------------------------

type timeoutBaseStream struct {
	inner      BaseStream
	cancel     context.CancelFunc
	ctx        context.Context
	ctxStopped bool
	closeOnce  sync.Once
	closeErr   error
	errMsg     string
	onCtxDone  func()
}

func (t *timeoutBaseStream) Next() bool {
	if t.ctxStopped {
		return false
	}
	type nextResult struct{ ok bool }
	resultCh := make(chan nextResult, 1)
	go func() { resultCh <- nextResult{ok: t.inner.Next()} }()
	select {
	case result := <-resultCh:
		if !result.ok {
			_ = t.Close()
		}
		return result.ok
	case <-t.ctx.Done():
		t.ctxStopped = true
		if t.onCtxDone != nil {
			t.onCtxDone()
		}
		_ = t.Close()
		return false
	}
}

func (t *timeoutBaseStream) Err() error {
	err := t.inner.Err()
	if t.ctxStopped {
		if err != nil {
			return NewLLMError(ErrCodeTimeout, t.errMsg, err)
		}
		return NewLLMError(ErrCodeTimeout, t.errMsg, nil)
	}
	if err != nil && t.ctx.Err() == context.DeadlineExceeded {
		return NewLLMError(ErrCodeTimeout, t.errMsg, err)
	}
	return err
}

func (t *timeoutBaseStream) Close() error {
	t.closeOnce.Do(func() {
		t.closeErr = t.inner.Close()
		t.cancel()
	})
	return t.closeErr
}

type timeoutStreamIterator struct {
	streamIteratorWrapper
	base timeoutBaseStream
}

func (t *timeoutStreamIterator) Next() bool   { return t.base.Next() }
func (t *timeoutStreamIterator) Err() error   { return t.base.Err() }
func (t *timeoutStreamIterator) Close() error { return t.base.Close() }
func (t *timeoutStreamIterator) Interrupt(err error) {
	t.base.ctxStopped = true
	interruptStreamIterator(t.inner, err)
	_ = t.base.Close()
}

type timeoutResponseStream struct {
	responseStreamWrapper
	base timeoutBaseStream
}

func (t *timeoutResponseStream) Next() bool   { return t.base.Next() }
func (t *timeoutResponseStream) Err() error   { return t.base.Err() }
func (t *timeoutResponseStream) Close() error { return t.base.Close() }

// -----------------------------------------------------------------------------
// RequestID Stream Iterator
// -----------------------------------------------------------------------------

type requestIDStreamIterator struct {
	streamIteratorWrapper
	requestID string
	closeOnce syncOnceErr
}

func (r *requestIDStreamIterator) Next() bool {
	ok := r.inner.Next()
	if !ok {
		_ = r.Close()
	}
	return ok
}
func (r *requestIDStreamIterator) Close() error { return r.closeOnce.Do(r.inner.Close) }
func (r *requestIDStreamIterator) Interrupt(err error) {
	interruptStreamIterator(r.inner, err)
}
func (r *requestIDStreamIterator) Response() *Response {
	resp := r.inner.Response()
	if resp == nil {
		return nil
	}
	if resp.Meta == nil {
		resp.Meta = make(map[string]any)
	}
	resp.Meta["request_id"] = r.requestID
	return resp
}

// -----------------------------------------------------------------------------
// Rate Limiter
// -----------------------------------------------------------------------------

func newLimiterWithCleanup(limit float64, burst int) (*simpleLimiter, func()) {
	limiter := newSimpleLimiter(limit, burst)
	runtime.SetFinalizer(limiter, func(l *simpleLimiter) { l.Close() })
	return limiter, func() { limiter.Close() }
}

type simpleLimiter struct {
	mu     sync.Mutex
	tokens float64
	max    float64
	rate   float64
	last   time.Time
	ch     chan struct{}
	stop   chan struct{}
	once   sync.Once
}

func newSimpleLimiter(rate float64, burst int) *simpleLimiter {
	if rate <= 0 {
		rate = 1
	}
	if burst < 1 {
		burst = 1
	}
	s := &simpleLimiter{
		tokens: float64(burst),
		max:    float64(burst),
		rate:   rate,
		last:   time.Now(),
		ch:     make(chan struct{}, 1),
		stop:   make(chan struct{}),
	}
	go s.refill()
	return s
}

func (l *simpleLimiter) Close() { l.once.Do(func() { close(l.stop) }) }

func (l *simpleLimiter) refill() {
	interval := time.Second
	if l.rate >= 1 {
		interval = time.Duration(float64(time.Second) / l.rate)
		if interval < time.Millisecond*10 {
			interval = time.Millisecond * 10
		}
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-l.stop:
			return
		case <-ticker.C:
			l.mu.Lock()
			l.tokens += l.rate * interval.Seconds()
			if l.tokens > l.max {
				l.tokens = l.max
			}
			select {
			case l.ch <- struct{}{}:
			default:
			}
			l.mu.Unlock()
		}
	}
}

func (l *simpleLimiter) Wait(ctx context.Context) error {
	for {
		l.mu.Lock()
		if l.tokens >= 1 {
			l.tokens--
			l.mu.Unlock()
			return nil
		}
		l.mu.Unlock()
		select {
		case <-l.ch:
			continue
		case <-l.stop:
			return fmt.Errorf("limiter closed")
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// -----------------------------------------------------------------------------
// Normalize Middleware
// -----------------------------------------------------------------------------

func normalizeMiddleware[Req any, Resp any, StreamResp any](defaultModel string, normalize func(Req, string, bool) Req) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
	unary := func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] {
		return func(ctx context.Context, req Req) (Resp, error) {
			return next(ctx, normalize(req, defaultModel, false))
		}
	}
	stream := func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] {
		return func(ctx context.Context, req Req) (StreamResp, error) {
			return next(ctx, normalize(req, defaultModel, true))
		}
	}
	return unary, stream
}

func NormalizeRequestMiddleware(defaultModel string) Middleware {
	unary, _ := normalizeMiddleware[*Request, *Response, StreamIterator](defaultModel, normalizeRequestForOperation)
	return unary
}

func NormalizeRequestStreamMiddleware(defaultModel string) StreamMiddleware {
	_, stream := normalizeMiddleware[*Request, *Response, StreamIterator](defaultModel, normalizeRequestForOperation)
	return stream
}

func NormalizeResponseRequestMiddleware(defaultModel string) ResponseMiddleware {
	unary, _ := normalizeMiddleware[*ResponseRequest, *ResponseObject, ResponseStream](defaultModel, normalizeResponseRequestForOperation)
	return unary
}

func NormalizeResponseRequestStreamMiddleware(defaultModel string) ResponseStreamMiddleware {
	_, stream := normalizeMiddleware[*ResponseRequest, *ResponseObject, ResponseStream](defaultModel, normalizeResponseRequestForOperation)
	return stream
}

// -----------------------------------------------------------------------------
// Standard Middleware Application
// -----------------------------------------------------------------------------

func ApplyStandardMiddleware(engine Engine, opts StandardMiddlewareOptions) Engine {
	pipeline := applyStandardMiddleware(chatSuite, enginePipelineAdapter{engine}, opts)
	return &enginePipelineBridge{pipeline: pipeline, engine: engine}
}

func ApplyStandardResponseMiddleware(engine *OpenAIResponsesEngine, opts StandardMiddlewareOptions) *OpenAIResponsesEngine {
	pipeline := applyStandardMiddleware(responsesSuite, responseEnginePipelineAdapter{engine}, opts)
	return &OpenAIResponsesEngine{
		adapter: engine.adapter,
		wrapped: &responsesMiddlewareEngine{pipeline: pipeline, inner: engine},
	}
}

// -----------------------------------------------------------------------------
// Pipeline Chaining
// -----------------------------------------------------------------------------

func ChainWithStreamAndClosers(engine Engine, middlewares []Middleware, streamMiddlewares []StreamMiddleware, closers []func()) Engine {
	pipeline := ChainPipelineWithStreamAndClosers[*Request, *Response, StreamIterator](
		enginePipelineAdapter{engine}, middlewares, streamMiddlewares, closers,
	)
	return &enginePipelineBridge{pipeline: pipeline, engine: engine}
}

type enginePipelineAdapter struct{ Engine }

func (a enginePipelineAdapter) Generate(ctx context.Context, req *Request) (*Response, error) {
	return a.Engine.Generate(ctx, req)
}
func (a enginePipelineAdapter) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	return a.Engine.Stream(ctx, req)
}

type enginePipelineBridge struct {
	pipeline PipelineEngine[*Request, *Response, StreamIterator]
	engine   Engine
}

func (b *enginePipelineBridge) Generate(ctx context.Context, req *Request) (*Response, error) {
	return b.pipeline.Generate(ctx, req)
}
func (b *enginePipelineBridge) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	return b.pipeline.Stream(ctx, req)
}
func (b *enginePipelineBridge) Capabilities() *ProtocolCapabilities {
	if b.engine != nil {
		return b.engine.Capabilities()
	}
	return nil
}
func (b *enginePipelineBridge) Close() {
	if closer, ok := b.pipeline.(PipelineCloser); ok {
		closer.Close()
	}
}

type responsesMiddlewareEngine struct {
	pipeline PipelineEngine[*ResponseRequest, *ResponseObject, ResponseStream]
	inner    *OpenAIResponsesEngine
}

type responseEnginePipelineAdapter struct{ engine *OpenAIResponsesEngine }

func (a responseEnginePipelineAdapter) Generate(ctx context.Context, req *ResponseRequest) (*ResponseObject, error) {
	return a.engine.Create(ctx, req)
}
func (a responseEnginePipelineAdapter) Stream(ctx context.Context, req *ResponseRequest) (ResponseStream, error) {
	return a.engine.Stream(ctx, req)
}

func (m *responsesMiddlewareEngine) create(ctx context.Context, req *ResponseRequest) (*ResponseObject, error) {
	return m.pipeline.Generate(ctx, req)
}
func (m *responsesMiddlewareEngine) stream(ctx context.Context, req *ResponseRequest) (ResponseStream, error) {
	return m.pipeline.Stream(ctx, req)
}
func (m *responsesMiddlewareEngine) close() {
	if closer, ok := m.pipeline.(PipelineCloser); ok {
		closer.Close()
	}
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

type ctxKeyRequestID struct{}

func GetRequestID(ctx context.Context) string {
	if id, ok := ctx.Value(ctxKeyRequestID{}).(string); ok {
		return id
	}
	return ""
}

func WithRequestID(ctx context.Context, requestID string) context.Context {
	requestID = strings.TrimSpace(requestID)
	if requestID == "" {
		return ctx
	}
	return context.WithValue(ctx, ctxKeyRequestID{}, requestID)
}

func requestModel(req *Request) string {
	if req == nil {
		return ""
	}
	return req.Model
}

func interruptStreamIterator(iter StreamIterator, err error) {
	if interruptible, ok := iter.(InterruptibleStreamIterator); ok {
		interruptible.Interrupt(err)
	}
}

func generateRequestID() string {
	var buf [64]byte
	b := buf[:0]
	b = append(b, "req_"...)
	b = strconv.AppendInt(b, time.Now().UnixNano(), 10)
	b = append(b, '_')
	b = strconv.AppendInt(b, requestCounter.Add(1), 10)
	return string(b)
}

var requestCounter atomic.Int64
