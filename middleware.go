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

// RateLimitConfig defines rate limit settings for standard middleware chains.
type RateLimitConfig struct {
	Limit float64
	Burst int
}

// StandardMiddlewareOptions defines the recommended default middleware stack.
//
// Middleware order is fixed as:
// request id -> request normalization -> capability negotiation -> lifecycle observers -> timeout -> retry -> rate limit -> engine
//
// This bounds total request time, applies retries within the timeout window,
// and charges rate limits per
// underlying attempt rather than per logical request.
type StandardMiddlewareOptions struct {
	DefaultModel       string
	Capabilities       *CapabilityNegotiationOptions
	Observers          []LifecycleObserver
	Retry              *RetryConfig
	Timeout            time.Duration
	EnableRequestID    bool
	RequestIDGenerator func() string
	RateLimit          *RateLimitConfig
}

// Middleware 普通中间件定义
type Middleware func(next Handler) Handler

// StreamHandler 流式处理函数定义
type StreamHandler func(ctx context.Context, req *Request) (StreamIterator, error)

// StreamMiddleware 流式中间件定义
type StreamMiddleware func(next StreamHandler) StreamHandler

// Chain 创建带中间件的引擎链
func Chain(engine Engine, middlewares ...Middleware) Engine {
	return &middlewareEngine{
		inner:       engine,
		middlewares: middlewares,
	}
}

// ChainWithStream 创建同时支持普通和流式中间件的引擎链
func ChainWithStream(engine Engine, middlewares []Middleware, streamMiddlewares []StreamMiddleware) Engine {
	return &middlewareEngine{
		inner:             engine,
		middlewares:       middlewares,
		streamMiddlewares: streamMiddlewares,
	}
}

// ChainWithStreamAndClosers 创建同时支持普通和流式中间件的引擎链，并追踪清理函数
func ChainWithStreamAndClosers(engine Engine, middlewares []Middleware, streamMiddlewares []StreamMiddleware, closers []func()) Engine {
	return &middlewareEngine{
		inner:             engine,
		middlewares:       middlewares,
		streamMiddlewares: streamMiddlewares,
		closers:           closers,
	}
}

// ApplyStandardMiddleware wraps an engine with the recommended middleware stack.
func ApplyStandardMiddleware(engine Engine, opts StandardMiddlewareOptions) Engine {
	var middlewares []Middleware
	var streamMiddlewares []StreamMiddleware
	var closers []func()

	if opts.EnableRequestID {
		middlewares = append(middlewares, RequestIDMiddleware(opts.RequestIDGenerator))
		streamMiddlewares = append(streamMiddlewares, RequestIDStreamMiddleware(opts.RequestIDGenerator))
	}

	middlewares = append(middlewares, NormalizeRequestMiddleware(opts.DefaultModel))
	streamMiddlewares = append(streamMiddlewares, NormalizeRequestStreamMiddleware(opts.DefaultModel))

	if opts.Capabilities != nil && opts.Capabilities.Resolver != nil {
		capUnary, capStream := CapabilityNegotiationMiddleware(*opts.Capabilities)
		middlewares = append(middlewares, capUnary)
		streamMiddlewares = append(streamMiddlewares, capStream)
	}

	if len(opts.Observers) > 0 {
		observer := compositeLifecycleObserver{observers: opts.Observers}
		observerUnary, observerStream := LifecycleObserverMiddleware(observer)
		middlewares = append(middlewares, observerUnary)
		streamMiddlewares = append(streamMiddlewares, observerStream)
	}

	if opts.Timeout > 0 {
		middlewares = append(middlewares, TimeoutMiddleware(opts.Timeout))
		streamMiddlewares = append(streamMiddlewares, TimeoutStreamMiddleware(opts.Timeout))
	}

	if opts.Retry != nil && opts.Retry.MaxAttempts > 1 {
		retryUnary, retryStream := RetryMiddlewareWithConfig(*opts.Retry)
		middlewares = append(middlewares, retryUnary)
		streamMiddlewares = append(streamMiddlewares, retryStream)
	}

	if opts.RateLimit != nil {
		rateLimit, rateLimitStream, closer := RateLimitMiddlewares(opts.RateLimit.Limit, opts.RateLimit.Burst)
		middlewares = append(middlewares, rateLimit)
		streamMiddlewares = append(streamMiddlewares, rateLimitStream)
		closers = append(closers, closer)
	}

	if len(middlewares) == 0 && len(streamMiddlewares) == 0 {
		return engine
	}

	return ChainWithStreamAndClosers(engine, middlewares, streamMiddlewares, closers)
}

// NormalizeRequestMiddleware ensures diagnostics and downstream middleware see
// the effective request shape for unary operations.
func NormalizeRequestMiddleware(defaultModel string) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			return next(ctx, normalizeRequestForOperation(req, defaultModel, false))
		}
	}
}

// NormalizeRequestStreamMiddleware ensures diagnostics and downstream middleware
// see the effective request shape for stream operations.
func NormalizeRequestStreamMiddleware(defaultModel string) StreamMiddleware {
	return func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			return next(ctx, normalizeRequestForOperation(req, defaultModel, true))
		}
	}
}

type middlewareEngine struct {
	inner             Engine
	middlewares       []Middleware
	streamMiddlewares []StreamMiddleware
	closers           []func()
	closeOnce         sync.Once
}

func (m *middlewareEngine) Generate(ctx context.Context, req *Request) (*Response, error) {
	handler := m.inner.Generate
	for i := len(m.middlewares) - 1; i >= 0; i-- {
		handler = m.middlewares[i](handler)
	}
	return handler(ctx, req)
}

func (m *middlewareEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	handler := m.inner.Stream
	for i := len(m.streamMiddlewares) - 1; i >= 0; i-- {
		handler = m.streamMiddlewares[i](handler)
	}
	return handler(ctx, req)
}

func (m *middlewareEngine) Close() {
	m.closeOnce.Do(func() {
		for _, closer := range m.closers {
			closer()
		}
	})
}

// RateLimitMiddleware 创建速率限制中间件（自动管理资源）
func RateLimitMiddleware(limit float64, burst int) (Middleware, func()) {
	limiter := newSimpleLimiter(limit, burst)
	runtime.SetFinalizer(limiter, func(l *simpleLimiter) { l.Close() })
	unary, _, closer := rateLimitMiddlewaresFromLimiter(limiter)
	return unary, closer
}

// RateLimitMiddlewares creates unary and stream middlewares sharing one limiter.
func RateLimitMiddlewares(limit float64, burst int) (Middleware, StreamMiddleware, func()) {
	limiter := newSimpleLimiter(limit, burst)
	runtime.SetFinalizer(limiter, func(l *simpleLimiter) { l.Close() })
	return rateLimitMiddlewaresFromLimiter(limiter)
}

func rateLimitMiddlewaresFromLimiter(limiter *simpleLimiter) (Middleware, StreamMiddleware, func()) {
	unary := func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			if err := limiter.Wait(ctx); err != nil {
				return nil, fmt.Errorf("rate limit exceeded: %w", err)
			}
			return next(ctx, req)
		}
	}
	stream := func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			if err := limiter.Wait(ctx); err != nil {
				return nil, fmt.Errorf("rate limit exceeded: %w", err)
			}
			return next(ctx, req)
		}
	}
	closer := func() { limiter.Close() }
	return unary, stream, closer
}

// RateLimitStreamMiddleware 创建流式请求的速率限制中间件（自动管理资源）
func RateLimitStreamMiddleware(limit float64, burst int) (StreamMiddleware, func()) {
	limiter := newSimpleLimiter(limit, burst)
	runtime.SetFinalizer(limiter, func(l *simpleLimiter) { l.Close() })
	_, stream, closer := rateLimitMiddlewaresFromLimiter(limiter)
	return stream, closer
}

// simpleLimiter 简单令牌桶限流器
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

func (l *simpleLimiter) Close() {
	l.once.Do(func() {
		close(l.stop)
	})
}

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

// LoggingMiddleware 创建日志中间件
func LoggingMiddleware(logger Logger) Middleware {
	unary, _ := lifecycleObserverMiddlewareWithOwner(NewLogObserver(logger), telemetryOwnerLogging)
	return unary
}

// LoggingStreamMiddleware 创建流式日志中间件
func LoggingStreamMiddleware(logger Logger) StreamMiddleware {
	_, stream := lifecycleObserverMiddlewareWithOwner(NewLogObserver(logger), telemetryOwnerLogging)
	return stream
}

// TimeoutMiddleware 创建超时控制中间件
func TimeoutMiddleware(timeout time.Duration) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			ctx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()

			resp, err := next(ctx, req)
			if err != nil {
				if ctx.Err() == context.DeadlineExceeded {
					return nil, NewLLMError(ErrCodeTimeout, "request timeout", ctx.Err())
				}
				return nil, err
			}
			return resp, nil
		}
	}
}

// TimeoutStreamMiddleware 创建流式请求的超时控制中间件
func TimeoutStreamMiddleware(timeout time.Duration) StreamMiddleware {
	return func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			ctx, cancel := context.WithTimeout(ctx, timeout)

			iter, err := next(ctx, req)
			if err != nil {
				cancel()
				if ctx.Err() == context.DeadlineExceeded {
					return nil, NewLLMError(ErrCodeTimeout, "stream request timeout", ctx.Err())
				}
				return nil, err
			}

			return &timeoutStreamIterator{inner: iter, cancel: cancel, ctx: ctx}, nil
		}
	}
}

type timeoutStreamIterator struct {
	inner      StreamIterator
	cancel     context.CancelFunc
	ctx        context.Context
	ctxStopped bool
	closeOnce  sync.Once
	closeErr   error
}

func (t *timeoutStreamIterator) Next() bool {
	if t.ctxStopped {
		return false
	}

	type nextResult struct {
		ok bool
	}
	resultCh := make(chan nextResult, 1)
	go func() {
		resultCh <- nextResult{ok: t.inner.Next()}
	}()

	select {
	case result := <-resultCh:
		if !result.ok && t.ctx.Err() == context.DeadlineExceeded {
			t.ctxStopped = true
			_ = t.Close()
		}
		if !result.ok {
			_ = t.Close()
		}
		return result.ok
	case <-t.ctx.Done():
		t.ctxStopped = true
		interruptStreamIterator(t.inner, NewLLMError(ErrCodeTimeout, "stream read timeout", t.ctx.Err()))
		_ = t.Close()
		return false
	}
}
func (t *timeoutStreamIterator) Chunk() []byte       { return t.inner.Chunk() }
func (t *timeoutStreamIterator) Text() string        { return t.inner.Text() }
func (t *timeoutStreamIterator) FullText() string    { return t.inner.FullText() }
func (t *timeoutStreamIterator) Usage() *Usage       { return t.inner.Usage() }
func (t *timeoutStreamIterator) Response() *Response { return t.inner.Response() }
func (t *timeoutStreamIterator) Err() error {
	err := t.inner.Err()
	if t.ctxStopped {
		if err != nil {
			return NewLLMError(ErrCodeTimeout, "stream read timeout", err)
		}
		return NewLLMError(ErrCodeTimeout, "stream read timeout", nil)
	}
	if err != nil && t.ctx.Err() == context.DeadlineExceeded {
		return NewLLMError(ErrCodeTimeout, "stream read timeout", err)
	}
	return err
}
func (t *timeoutStreamIterator) Close() error {
	t.closeOnce.Do(func() {
		t.closeErr = t.inner.Close()
		t.cancel()
	})
	return t.closeErr
}
func (t *timeoutStreamIterator) Interrupt(err error) {
	t.ctxStopped = true
	interruptStreamIterator(t.inner, err)
	_ = t.Close()
}

// RequestIDMiddleware 创建请求ID追踪中间件
func RequestIDMiddleware(generator func() string) Middleware {
	if generator == nil {
		generator = func() string { return generateRequestID() }
	}
	return func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			reqID := GetRequestID(ctx)
			if reqID == "" {
				reqID = generator()
				ctx = context.WithValue(ctx, ctxKeyRequestID{}, reqID)
			}

			cloned := cloneRequestForMetadata(req)
			if cloned.Meta == nil {
				cloned.Meta = make(map[string]any)
			}
			cloned.Meta["request_id"] = reqID

			resp, err := next(ctx, cloned)
			if err != nil {
				return nil, err
			}

			if resp != nil {
				if resp.Meta == nil {
					resp.Meta = make(map[string]any)
				}
				resp.Meta["request_id"] = reqID
			}

			return resp, err
		}
	}
}

type ctxKeyRequestID struct{}

type telemetryOwner string

const (
	telemetryOwnerLogging  telemetryOwner = "logging"
	telemetryOwnerObserver telemetryOwner = "observer"
)

type ctxKeyTelemetryOwner struct{}

func claimTelemetryOwnership(ctx context.Context, owner telemetryOwner) (context.Context, bool) {
	if existing, ok := ctx.Value(ctxKeyTelemetryOwner{}).(telemetryOwner); ok {
		return ctx, existing == owner
	}
	return context.WithValue(ctx, ctxKeyTelemetryOwner{}, owner), true
}

// RequestIDStreamMiddleware 创建流式请求的请求ID追踪中间件
func RequestIDStreamMiddleware(generator func() string) StreamMiddleware {
	if generator == nil {
		generator = func() string { return generateRequestID() }
	}
	return func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			reqID := GetRequestID(ctx)
			if reqID == "" {
				reqID = generator()
				ctx = context.WithValue(ctx, ctxKeyRequestID{}, reqID)
			}

			cloned := cloneRequestForMetadata(req)
			if cloned.Meta == nil {
				cloned.Meta = make(map[string]any)
			}
			cloned.Meta["request_id"] = reqID

			iter, err := next(ctx, cloned)
			if err != nil {
				return nil, err
			}

			return &requestIDStreamIterator{inner: iter, requestID: reqID}, nil
		}
	}
}

type requestIDStreamIterator struct {
	inner     StreamIterator
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
func (r *requestIDStreamIterator) Chunk() []byte    { return r.inner.Chunk() }
func (r *requestIDStreamIterator) Text() string     { return r.inner.Text() }
func (r *requestIDStreamIterator) FullText() string { return r.inner.FullText() }
func (r *requestIDStreamIterator) Err() error       { return r.inner.Err() }
func (r *requestIDStreamIterator) Usage() *Usage    { return r.inner.Usage() }
func (r *requestIDStreamIterator) Close() error     { return r.closeOnce.Do(r.inner.Close) }
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

// GetRequestID 从上下文获取请求ID
func GetRequestID(ctx context.Context) string {
	if id, ok := ctx.Value(ctxKeyRequestID{}).(string); ok {
		return id
	}
	return ""
}

// WithRequestID stores a request id in context so logs, observers, and
// middleware can correlate work even when requests originate outside the
// standard slm middleware chain.
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

func normalizeRequestForOperation(req *Request, defaultModel string, stream bool) *Request {
	if req == nil {
		return nil
	}

	model := strings.TrimSpace(req.Model)
	resolvedModel := model
	if resolvedModel == "" {
		resolvedModel = strings.TrimSpace(defaultModel)
	}

	if req.Stream == stream && model == resolvedModel {
		return req
	}

	clone := cloneRequestShallow(req)
	clone.Stream = stream
	if model == "" {
		clone.Model = resolvedModel
	}
	return clone
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
