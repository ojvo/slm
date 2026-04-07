package slm

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

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

type middlewareEngine struct {
	inner             Engine
	middlewares       []Middleware
	streamMiddlewares []StreamMiddleware
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

// RateLimitMiddleware 创建速率限制中间件（自动管理资源）
func RateLimitMiddleware(limit float64, burst int) (Middleware, func()) {
	limiter := newSimpleLimiter(limit, burst)
	mw := func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			if err := limiter.Wait(ctx); err != nil {
				return nil, fmt.Errorf("rate limit exceeded: %w", err)
			}
			return next(ctx, req)
		}
	}
	closer := func() { limiter.Close() }
	return mw, closer
}

// RateLimitStreamMiddleware 创建流式请求的速率限制中间件（自动管理资源）
func RateLimitStreamMiddleware(limit float64, burst int) (StreamMiddleware, func()) {
	limiter := newSimpleLimiter(limit, burst)
	mw := func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			if err := limiter.Wait(ctx); err != nil {
				return nil, fmt.Errorf("rate limit exceeded: %w", err)
			}
			return next(ctx, req)
		}
	}
	closer := func() { limiter.Close() }
	return mw, closer
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
	select {
	case <-l.stop:
		return
	default:
		close(l.stop)
	}
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
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// LoggingMiddleware 创建日志中间件
func LoggingMiddleware(logger Logger) Middleware {
	if logger == nil {
		logger = &NopLogger{}
	}
	return func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			logger.Debug("LLM request start", "model", req.Model)
			start := time.Now()

			resp, err := next(ctx, req)

			duration := time.Since(start)
			if err != nil {
				logger.Error("LLM request failed", "error", err, "duration", duration)
			} else {
				logger.Debug("LLM request completed",
					"duration", duration,
					"tokens", resp.Usage.TotalTokens,
					"finish_reason", resp.FinishReason,
				)
			}

			return resp, err
		}
	}
}

// TimeoutMiddleware 创建超时控制中间件
func TimeoutMiddleware(timeout time.Duration) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			ctx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()

			respCh := make(chan *Response, 1)
			errCh := make(chan error, 1)

			go func() {
				resp, err := next(ctx, req)
				if err != nil {
					errCh <- err
					return
				}
				respCh <- resp
			}()

			select {
			case resp := <-respCh:
				return resp, nil
			case err := <-errCh:
				return nil, err
			case <-ctx.Done():
				return nil, NewLLMError(ErrCodeTimeout, "request timeout", ctx.Err())
			}
		}
	}
}

// RequestIDMiddleware 创建请求ID追踪中间件
func RequestIDMiddleware(generator func() string) Middleware {
	if generator == nil {
		generator = func() string { return generateRequestID() }
	}
	return func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			reqID := generator()

			ctx = context.WithValue(ctx, ctxKeyRequestID{}, reqID)
			if req.Meta == nil {
				req.Meta = make(map[string]any)
			}
			req.Meta["request_id"] = reqID

			resp, err := next(ctx, req)
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

// GetRequestID 从上下文获取请求ID
func GetRequestID(ctx context.Context) string {
	if id, ok := ctx.Value(ctxKeyRequestID{}).(string); ok {
		return id
	}
	return ""
}

func generateRequestID() string {
	return fmt.Sprintf("req_%d_%d", time.Now().UnixNano(), requestCounter.Add(1))
}

var requestCounter atomic.Int64
