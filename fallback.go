package slm

import (
	"context"
	"errors"
	"sync"
)

// IsOverloadError reports whether err indicates a provider-side overload
// (HTTP 429 Too Many Requests / 503 Overloaded). These errors signal that
// the primary model is temporarily unavailable and a fallback model should
// be used until the provider recovers.
//
// This is a narrower predicate than IsRetryableError: overload errors are
// retryable, but they also warrant model fallback because retrying the same
// model may keep hitting the same rate limit. Non-overload retryable errors
// (network timeouts, connection resets) do not trigger fallback — those are
// handled by RetryMiddleware.
func IsOverloadError(err error) bool {
	if err == nil {
		return false
	}
	var llmErr *LLMError
	if errors.As(err, &llmErr) {
		return llmErr.Code == ErrCodeOverloaded || llmErr.Code == ErrCodeRateLimit
	}
	return false
}

// FallbackConfig configures overload-based model fallback behavior.
// When the primary model returns consecutive overload errors, the
// middleware swaps to FallbackModel. After any successful call, it
// recovers to the primary model.
//
// This is the reactive counterpart to cofo's RoutingMiddleware:
//
//   - RoutingMiddleware: proactive model selection based on signals (flash→pro)
//   - FallbackMiddleware: reactive model swap based on overload errors (pro→fallback)
//
// The two are orthogonal and can be composed in the same middleware chain.
// FallbackMiddleware should be outermost so that routing selects the primary
// model first, and fallback overrides to the fallback model only when the
// primary is overloaded.
type FallbackConfig struct {
	// FallbackModel is the model name to use when the primary model is
	// overloaded. Required — if empty, the middleware is a no-op.
	FallbackModel string

	// OverloadThreshold is the number of consecutive overload errors
	// from the primary model before switching to FallbackModel.
	// Default 3. A value of 1 means "switch on first overload error".
	OverloadThreshold int

	// IsOverload determines whether an error counts as an overload signal.
	// Default: IsOverloadError.
	// Callers can override to add provider-specific overload detection.
	IsOverload func(error) bool
}

// fallbackState tracks consecutive overload failures and the current
// fallback status. It is guarded by a mutex because the middleware may
// be invoked concurrently (e.g., multiple sub-agents sharing an Engine).
type fallbackState struct {
	mu            sync.Mutex
	failures      int  // consecutive overload failures on primary model
	usingFallback bool // currently routing to fallback model
}

// maybeSwap overrides req.Model to the fallback model if currently in
// fallback mode. Returns the original request if not swapping.
//
// Uses cloneRequestShallow (value copy of the Request struct) because we
// only modify the Model field — slice/map fields share backing arrays but
// are not modified by this middleware, matching the pattern used by
// normalizeRequestForOperation.
func (s *fallbackState) maybeSwap(req *Request, fallbackModel string) *Request {
	s.mu.Lock()
	using := s.usingFallback
	s.mu.Unlock()
	if !using || fallbackModel == "" {
		return req
	}
	clone := cloneRequestShallow(req)
	clone.Model = fallbackModel
	return clone
}

// recordOutcome updates the fallback state based on the call result.
//
//   - Success                          → failures = 0; usingFallback = false (recover)
//   - Overload error while on primary  → failures++; if >= threshold → usingFallback = true
//   - Overload error while on fallback → no state change (already on fallback)
//   - Non-overload error               → no state change (handled by retry/reconnect)
func (s *fallbackState) recordOutcome(err error, isOverload func(error) bool, threshold int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err == nil {
		s.failures = 0
		s.usingFallback = false
		return
	}
	if !isOverload(err) {
		return
	}
	if s.usingFallback {
		return
	}
	s.failures++
	if s.failures >= threshold {
		s.usingFallback = true
	}
}

// FallbackMiddlewareWithConfig creates a pair of middlewares (unary + stream)
// that swap to a fallback model after consecutive overload errors and recover
// to the primary model after any success.
//
// Usage:
//
//	cfg := slm.FallbackConfig{
//	    FallbackModel:     "gpt-4o-mini",
//	    OverloadThreshold: 3,
//	}
//	unaryMW, streamMW := slm.FallbackMiddlewareWithConfig(cfg)
//	engine = slm.ChainWithStreamAndClosers(engine, []slm.Middleware{unaryMW}, []slm.StreamMiddleware{streamMW}, nil)
//
// Composition order (outer → inner): Fallback → Routing → Retry → Timeout → core.
//
// Stream behavior: outcome is recorded at stream-open time only. If the
// stream opens successfully but later fails mid-stream with an overload
// error, the failure is not recorded (mid-stream failures are usually
// network issues, not overload; and wrapping the iterator to detect
// completion would complicate the stream semantics). Callers needing
// mid-stream overload detection can compose with cofo's ReconnectStreamIterator.
func FallbackMiddlewareWithConfig(cfg FallbackConfig) (Middleware, StreamMiddleware) {
	if cfg.OverloadThreshold <= 0 {
		cfg.OverloadThreshold = 3
	}
	if cfg.IsOverload == nil {
		cfg.IsOverload = IsOverloadError
	}

	// No fallback model → no-op.
	if cfg.FallbackModel == "" {
		noopUnary := func(next Handler) Handler { return next }
		noopStream := func(next StreamHandler) StreamHandler { return next }
		return noopUnary, noopStream
	}

	st := &fallbackState{}
	fallbackModel := cfg.FallbackModel
	threshold := cfg.OverloadThreshold
	isOverload := cfg.IsOverload

	unary := func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			req = st.maybeSwap(req, fallbackModel)
			resp, err := next(ctx, req)
			st.recordOutcome(err, isOverload, threshold)
			return resp, err
		}
	}

	stream := func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			req = st.maybeSwap(req, fallbackModel)
			iter, err := next(ctx, req)
			st.recordOutcome(err, isOverload, threshold)
			return iter, err
		}
	}

	return unary, stream
}
