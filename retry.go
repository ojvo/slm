package slm

import (
	"context"
	"fmt"
	"math/rand"
	"time"
)

type RetryConfig struct {
	MaxAttempts int
	Backoff     func(attempt int) time.Duration
	IsRetryable func(error) bool
	WrapError   func(msg string, cause error) error
}

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts: 3,
		Backoff:     ExponentialBackoff,
		IsRetryable: IsRetryableError,
	}
}

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

func RetryMiddlewareWithConfig(cfg RetryConfig) (Middleware, StreamMiddleware) {
	return retryMiddlewareWithConfig[*Request, *Response, StreamIterator](cfg,
		func(req *Request) *Request { return cloneRequestForMetadata(req) },
	)
}

func ResponseRetryMiddlewareWithConfig(cfg RetryConfig) (ResponseMiddleware, ResponseStreamMiddleware) {
	return retryMiddlewareWithConfig[*ResponseRequest, *ResponseObject, ResponseStream](cfg,
		func(req *ResponseRequest) *ResponseRequest { return cloneResponseRequest(req) },
	)
}

func retryMiddlewareWithConfig[Req any, Resp any, StreamResp any](
	cfg RetryConfig,
	cloneReq func(Req) Req,
) (PipelineMiddleware[Req, Resp], PipelineStreamMiddleware[Req, StreamResp]) {
	if cfg.MaxAttempts < 1 {
		cfg.MaxAttempts = 1
	}
	if cfg.Backoff == nil {
		cfg.Backoff = ExponentialBackoff
	}
	if cfg.IsRetryable == nil {
		cfg.IsRetryable = IsRetryableError
	}

	unary := func(next PipelineHandler[Req, Resp]) PipelineHandler[Req, Resp] {
		return func(ctx context.Context, req Req) (Resp, error) {
			var zero Resp
			var lastErr error
			for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
				if err := checkContext(ctx, cfg.WrapError); err != nil {
					return zero, err
				}
				retryReq := req
				if attempt > 1 {
					retryReq = cloneReq(req)
				}
				resp, err := next(ctx, retryReq)
				if err == nil {
					return resp, nil
				}
				lastErr = err
				if shouldStopRetry(attempt, cfg.MaxAttempts, err, cfg.IsRetryable) {
					break
				}
				if err := waitRetry(ctx, attempt, cfg); err != nil {
					return zero, err
				}
			}
			return zero, fmt.Errorf("max retries exceeded: %w", lastErr)
		}
	}

	stream := func(next PipelineStreamHandler[Req, StreamResp]) PipelineStreamHandler[Req, StreamResp] {
		return func(ctx context.Context, req Req) (StreamResp, error) {
			var zero StreamResp
			var lastErr error
			for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
				if err := checkContext(ctx, cfg.WrapError); err != nil {
					return zero, err
				}
				retryReq := req
				if attempt > 1 {
					retryReq = cloneReq(req)
				}
				result, err := next(ctx, retryReq)
				if err == nil {
					w := newRetryStreamWrapper(ctx, req, next, cfg, cloneReq, result, attempt)
					if wrapped, ok := any(w).(StreamResp); ok {
						return wrapped, nil
					}
					return result, nil
				}
				lastErr = err
				if shouldStopRetry(attempt, cfg.MaxAttempts, err, cfg.IsRetryable) {
					break
				}
				if err := waitRetry(ctx, attempt, cfg); err != nil {
					return zero, err
				}
			}
			return zero, fmt.Errorf("max retries exceeded: %w", lastErr)
		}
	}

	return unary, stream
}

type retryStreamWrapper[Req any, StreamResp any] struct {
	inner               StreamResp
	ctx                 context.Context
	req                 Req
	next                PipelineStreamHandler[Req, StreamResp]
	cfg                 RetryConfig
	cloneReq            func(Req) Req
	attempt             int
	firstChunkDelivered bool
	closeOnce           syncOnceErr
}

func newRetryStreamWrapper[Req any, StreamResp any](
	ctx context.Context,
	req Req,
	next PipelineStreamHandler[Req, StreamResp],
	cfg RetryConfig,
	cloneReq func(Req) Req,
	inner StreamResp,
	attempt int,
) *retryStreamWrapper[Req, StreamResp] {
	return &retryStreamWrapper[Req, StreamResp]{
		inner:    inner,
		ctx:      ctx,
		req:      req,
		next:     next,
		cfg:      cfg,
		cloneReq: cloneReq,
		attempt:  attempt,
	}
}

func (r *retryStreamWrapper[Req, StreamResp]) Next() bool {
	stream := asBaseStream(r.inner)
	if r.firstChunkDelivered {
		ok := stream.Next()
		if !ok {
			_ = r.Close()
		}
		return ok
	}
	return r.nextUntilFirstChunk()
}

func (r *retryStreamWrapper[Req, StreamResp]) nextUntilFirstChunk() bool {
	stream := asBaseStream(r.inner)
	for {
		if stream.Next() {
			r.firstChunkDelivered = true
			return true
		}
		err := stream.Err()
		if err == nil {
			_ = r.Close()
			return false
		}
		if shouldStopRetry(r.attempt, r.cfg.MaxAttempts, err, r.cfg.IsRetryable) {
			return false
		}
		stream.Close()
		iter, attempt, err := r.openNextAttempt()
		if err != nil {
			r.inner = any(&errorBaseStream{err: err}).(StreamResp)
			return false
		}
		r.inner = iter
		r.attempt = attempt
	}
}

func (r *retryStreamWrapper[Req, StreamResp]) openNextAttempt() (StreamResp, int, error) {
	var zero StreamResp
	var lastErr error
	for attempt := r.attempt + 1; attempt <= r.cfg.MaxAttempts; attempt++ {
		if err := checkContext(r.ctx, r.cfg.WrapError); err != nil {
			return zero, 0, err
		}
		if err := waitRetry(r.ctx, attempt, r.cfg); err != nil {
			return zero, 0, err
		}
		retryReq := r.cloneReq(r.req)
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
		return zero, 0, fmt.Errorf("max retries exceeded before first stream chunk: %w", lastErr)
	}
	return zero, 0, nil
}

func (r *retryStreamWrapper[Req, StreamResp]) Err() error { return asBaseStream(r.inner).Err() }
func (r *retryStreamWrapper[Req, StreamResp]) Close() error {
	return r.closeOnce.Do(func() error { return asBaseStream(r.inner).Close() })
}

func (r *retryStreamWrapper[Req, StreamResp]) Chunk() []byte {
	if si, ok := any(r.inner).(StreamIterator); ok {
		return si.Chunk()
	}
	return nil
}
func (r *retryStreamWrapper[Req, StreamResp]) Text() string {
	if si, ok := any(r.inner).(StreamIterator); ok {
		return si.Text()
	}
	return ""
}
func (r *retryStreamWrapper[Req, StreamResp]) FullText() string {
	if si, ok := any(r.inner).(StreamIterator); ok {
		return si.FullText()
	}
	return ""
}
func (r *retryStreamWrapper[Req, StreamResp]) Usage() *Usage {
	if si, ok := any(r.inner).(StreamIterator); ok {
		return si.Usage()
	}
	return nil
}
func (r *retryStreamWrapper[Req, StreamResp]) Response() *Response {
	if si, ok := any(r.inner).(StreamIterator); ok {
		return si.Response()
	}
	return nil
}
func (r *retryStreamWrapper[Req, StreamResp]) Current() ResponseEvent {
	if rs, ok := any(r.inner).(ResponseStream); ok {
		return rs.Current()
	}
	return ResponseEvent{}
}
func (r *retryStreamWrapper[Req, StreamResp]) Interrupt(err error) {
	if si, ok := any(r.inner).(InterruptibleStreamIterator); ok {
		si.Interrupt(err)
	}
}

type errorBaseStream struct{ err error }

func (e *errorBaseStream) Next() bool             { return false }
func (e *errorBaseStream) Err() error             { return e.err }
func (e *errorBaseStream) Close() error           { return nil }
func (e *errorBaseStream) Chunk() []byte          { return nil }
func (e *errorBaseStream) Text() string           { return "" }
func (e *errorBaseStream) FullText() string       { return "" }
func (e *errorBaseStream) Usage() *Usage          { return nil }
func (e *errorBaseStream) Response() *Response    { return nil }
func (e *errorBaseStream) Current() ResponseEvent { return ResponseEvent{} }

func asBaseStream(v any) BaseStream {
	if bs, ok := v.(BaseStream); ok {
		return bs
	}
	return nil
}

func checkContext(ctx context.Context, wrapError func(msg string, cause error) error) error {
	if ctx.Err() != nil {
		if wrapError != nil {
			return wrapError("context cancelled", ctx.Err())
		}
		return ctx.Err()
	}
	return nil
}

func shouldStopRetry(attempt, maxAttempts int, err error, isRetryable func(error) bool) bool {
	if attempt >= maxAttempts {
		return true
	}
	if isRetryable == nil || !isRetryable(err) {
		return true
	}
	return false
}

func waitRetry(ctx context.Context, attempt int, cfg RetryConfig) error {
	delay := cfg.Backoff(attempt)
	if delay <= 0 {
		if ctx.Err() != nil {
			if cfg.WrapError != nil {
				return cfg.WrapError("context cancelled", ctx.Err())
			}
			return ctx.Err()
		}
		return nil
	}
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
