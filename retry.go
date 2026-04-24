package slm

import (
	"context"
	"fmt"
	"math/rand"
	"time"
)

// RetryConfig 重试配置。
//
// 对流式请求，重试只发生在首个有效 chunk 产出之前。
// 一旦已经向调用方返回过任何 chunk，后续断流将直接透传错误，
// 避免在缺少协议级去重能力时发生重复输出或状态错乱。
type RetryConfig struct {
	MaxAttempts int
	Backoff     func(attempt int) time.Duration
	IsRetryable func(error) bool
	WrapError   func(msg string, cause error) error
}

// DefaultRetryConfig 返回默认重试配置
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts: 3,
		Backoff:     ExponentialBackoff,
		IsRetryable: IsRetryableError,
	}
}

// ExponentialBackoff 指数退避策略
func ExponentialBackoff(attempt int) time.Duration {
	if attempt >= 30 {
		return maxBackoff
	}
	d := time.Duration(1<<uint(attempt)) * 100 * time.Millisecond
	if d > maxBackoff {
		return maxBackoff
	}
	return d
}

// ExponentialBackoffWithJitter 指数退避 + 随机抖动策略
// 防止多客户端同时重试导致的惊群效应
func ExponentialBackoffWithJitter(attempt int) time.Duration {
	base := ExponentialBackoff(attempt)
	jitter := time.Duration(rand.Int63n(int64(base) / 2))
	result := base + jitter
	if result > maxBackoff {
		return maxBackoff
	}
	return result
}

const maxBackoff = 30 * time.Second

// RetryMiddlewareWithConfig 使用自定义配置的重试中间件
func RetryMiddlewareWithConfig(cfg RetryConfig) (Middleware, StreamMiddleware) {
	if cfg.MaxAttempts < 1 {
		cfg.MaxAttempts = 1
	}
	if cfg.Backoff == nil {
		cfg.Backoff = ExponentialBackoff
	}
	if cfg.IsRetryable == nil {
		cfg.IsRetryable = IsRetryableError
	}

	unary := func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			return retryUnary(ctx, req, next, cfg)
		}
	}

	stream := func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			return retryStream(ctx, req, next, cfg)
		}
	}

	return unary, stream
}

// retryUnary 统一的重试逻辑（Unary）
func retryUnary(ctx context.Context, req *Request, next Handler, cfg RetryConfig) (*Response, error) {
	var lastErr error
	for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
		if err := checkContext(ctx, cfg.WrapError); err != nil {
			return nil, err
		}

		retryReq := cloneRequest(req)
		resp, err := next(ctx, retryReq)
		if err == nil {
			return resp, nil
		}

		lastErr = err
		if shouldStopRetry(attempt, cfg.MaxAttempts, err, cfg.IsRetryable) {
			break
		}

		if err := waitRetry(ctx, attempt, cfg); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

// retryStream 统一的重试逻辑（Stream）
func retryStream(ctx context.Context, req *Request, next StreamHandler, cfg RetryConfig) (StreamIterator, error) {
	var lastErr error
	for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
		if err := checkContext(ctx, cfg.WrapError); err != nil {
			return nil, err
		}

		retryReq := cloneRequest(req)
		iter, err := next(ctx, retryReq)
		if err == nil {
			return &retryStreamIterator{
				ctx:     ctx,
				req:     req,
				next:    next,
				cfg:     cfg,
				attempt: attempt,
				inner:   iter,
			}, nil
		}

		lastErr = err
		if shouldStopRetry(attempt, cfg.MaxAttempts, err, cfg.IsRetryable) {
			break
		}

		if err := waitRetry(ctx, attempt, cfg); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

type retryStreamIterator struct {
	ctx                 context.Context
	req                 *Request
	next                StreamHandler
	cfg                 RetryConfig
	attempt             int
	inner               StreamIterator
	firstChunkDelivered bool
	closeOnce           syncOnceErr
}

func (r *retryStreamIterator) Next() bool {
	if r.firstChunkDelivered {
		ok := r.inner.Next()
		if !ok {
			_ = r.Close()
		}
		return ok
	}
	return r.nextUntilFirstChunk()
}

func (r *retryStreamIterator) nextUntilFirstChunk() bool {
	for {
		if r.inner.Next() {
			r.firstChunkDelivered = true
			return true
		}

		err := r.inner.Err()
		if err == nil {
			_ = r.Close()
			return false
		}

		if shouldStopRetry(r.attempt, r.cfg.MaxAttempts, err, r.cfg.IsRetryable) {
			return false
		}

		r.inner.Close()

		iter, attempt, err := r.openNextStreamAttempt()
		if err != nil {
			r.inner = &errorIterator{err: err}
			return false
		}
		r.inner = iter
		r.attempt = attempt
	}
}

func (r *retryStreamIterator) openNextStreamAttempt() (StreamIterator, int, error) {
	var lastErr error
	for attempt := r.attempt + 1; attempt <= r.cfg.MaxAttempts; attempt++ {
		if err := checkContext(r.ctx, r.cfg.WrapError); err != nil {
			return nil, 0, err
		}

		if err := waitRetry(r.ctx, attempt, r.cfg); err != nil {
			return nil, 0, err
		}

		retryReq := cloneRequest(r.req)
		iter, err := r.next(r.ctx, retryReq)
		if err != nil {
			lastErr = err
			if shouldStopRetry(attempt, r.cfg.MaxAttempts, err, r.cfg.IsRetryable) {
				break
			}
			continue
		}
		return iter, attempt, nil
	}

	if lastErr != nil {
		return nil, 0, fmt.Errorf("max retries exceeded before first stream chunk: %w", lastErr)
	}
	return nil, 0, nil
}

func (r *retryStreamIterator) Chunk() []byte       { return r.inner.Chunk() }
func (r *retryStreamIterator) Text() string        { return r.inner.Text() }
func (r *retryStreamIterator) FullText() string    { return r.inner.FullText() }
func (r *retryStreamIterator) Err() error          { return r.inner.Err() }
func (r *retryStreamIterator) Usage() *Usage       { return r.inner.Usage() }
func (r *retryStreamIterator) Response() *Response { return r.inner.Response() }
func (r *retryStreamIterator) Close() error        { return r.closeOnce.Do(r.inner.Close) }
func (r *retryStreamIterator) Interrupt(err error) { interruptStreamIterator(r.inner, err) }

type errorIterator struct {
	err error
}

func (e *errorIterator) Next() bool          { return false }
func (e *errorIterator) Chunk() []byte       { return nil }
func (e *errorIterator) Text() string        { return "" }
func (e *errorIterator) FullText() string    { return "" }
func (e *errorIterator) Err() error          { return e.err }
func (e *errorIterator) Usage() *Usage       { return nil }
func (e *errorIterator) Response() *Response { return nil }
func (e *errorIterator) Close() error        { return nil }

// checkContext 检查上下文是否已取消
func checkContext(ctx context.Context, wrapError func(msg string, cause error) error) error {
	if ctx.Err() != nil {
		if wrapError != nil {
			return wrapError("context cancelled", ctx.Err())
		}
		return ctx.Err()
	}
	return nil
}

// shouldStopRetry 判断是否应该停止重试
func shouldStopRetry(attempt, maxAttempts int, err error, isRetryable func(error) bool) bool {
	if attempt >= maxAttempts {
		return true
	}
	if isRetryable == nil || !isRetryable(err) {
		return true
	}
	return false
}

// waitRetry 等待重试间隔
func waitRetry(ctx context.Context, attempt int, cfg RetryConfig) error {
	delay := cfg.Backoff(attempt)
	select {
	case <-ctx.Done():
		if cfg.WrapError != nil {
			return cfg.WrapError("context cancelled", ctx.Err())
		}
		return ctx.Err()
	case <-time.After(delay):
		return nil
	}
}
