package slm

import (
	"context"
	"fmt"
	"time"
)

// RetryConfig 重试配置
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
	return time.Duration(1<<uint(attempt)) * 100 * time.Millisecond
}

// RetryMiddlewareWithConfig 使用自定义配置的重试中间件
func RetryMiddlewareWithConfig(cfg RetryConfig) (Middleware, StreamMiddleware) {
	if cfg.MaxAttempts <= 1 {
		cfg.MaxAttempts = 3
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
			return iter, nil
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
