package slm

import (
	"context"
	"errors"
	"sync"
	"testing"
)

// fallbackEngine is a controllable engine for testing FallbackMiddleware.
// It records the model of each request it receives and returns the
// configured response/error (unary) or iterator/error (stream).
type fallbackEngine struct {
	mu       sync.Mutex
	models   []string // models seen by Generate/Stream, in order
	resp     *Response
	err      error
	streamIt StreamIterator
	streamErr error
}

func (e *fallbackEngine) Generate(ctx context.Context, req *Request) (*Response, error) {
	e.mu.Lock()
	e.models = append(e.models, req.Model)
	r, err := e.resp, e.err
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}
	if r != nil {
		return r, nil
	}
	return &Response{Content: "ok"}, nil
}

func (e *fallbackEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	e.mu.Lock()
	e.models = append(e.models, req.Model)
	it, err := e.streamIt, e.streamErr
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}
	if it != nil {
		return it, nil
	}
	return &stubIter{text: "ok"}, nil
}

func (e *fallbackEngine) Capabilities() *ProtocolCapabilities {
	return &ProtocolCapabilities{Description: "fallback test engine"}
}

func (e *fallbackEngine) lastModel() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.models) == 0 {
		return ""
	}
	return e.models[len(e.models)-1]
}

func (e *fallbackEngine) modelCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.models)
}

// TestIsOverloadError verifies that IsOverloadError correctly identifies
// overload errors (ErrCodeOverloaded, ErrCodeRateLimit) and rejects
// non-overload errors and non-LLMError errors.
func TestIsOverloadError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"overloaded", NewLLMError(ErrCodeOverloaded, "overloaded", nil), true},
		{"rate_limit", NewLLMError(ErrCodeRateLimit, "rate limit", nil), true},
		{"timeout", NewLLMError(ErrCodeTimeout, "timeout", nil), false},
		{"network", NewLLMError(ErrCodeNetwork, "network", nil), false},
		{"server", NewLLMError(ErrCodeServer, "server", nil), false},
		{"auth", NewLLMError(ErrCodeAuth, "auth", nil), false},
		{"internal", NewLLMError(ErrCodeInternal, "internal", nil), false},
		{"plain_error", errors.New("plain error"), false},
		{"wrapped_overloaded", &wrappedErr{inner: NewLLMError(ErrCodeOverloaded, "wrapped", nil)}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsOverloadError(tt.err)
			if got != tt.want {
				t.Errorf("IsOverloadError() = %v, want %v", got, tt.want)
			}
		})
	}
}

type wrappedErr struct{ inner error }

func (w *wrappedErr) Error() string { return w.inner.Error() }
func (w *wrappedErr) Unwrap() error { return w.inner }

// TestFallback_NoopWhenNoFallbackModel verifies that an empty FallbackModel
// makes the middleware a pass-through that does not swap models.
func TestFallback_NoopWhenNoFallbackModel(t *testing.T) {
	engine := &fallbackEngine{err: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	unary, stream := FallbackMiddlewareWithConfig(FallbackConfig{FallbackModel: ""})

	chainUnary := unary(engine.Generate)
	chainStream := stream(engine.Stream)

	// Five overload errors should not trigger any swap because FallbackModel is empty.
	for i := 0; i < 5; i++ {
		_, _ = chainUnary(context.Background(), &Request{Model: "primary"})
	}
	if got := engine.lastModel(); got != "primary" {
		t.Errorf("after 5 overload errors with empty FallbackModel, model = %q, want %q", got, "primary")
	}

	_, _ = chainStream(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "primary" {
		t.Errorf("stream: after overload with empty FallbackModel, model = %q, want %q", got, "primary")
	}
}

// TestFallback_NoSwapBeforeThreshold verifies that the model is not swapped
// until the overload threshold is reached.
func TestFallback_NoSwapBeforeThreshold(t *testing.T) {
	engine := &fallbackEngine{err: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	unary, _ := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel:     "fallback",
		OverloadThreshold: 3,
	})

	chain := unary(engine.Generate)

	// Two overload errors — below threshold, should still use primary.
	for i := 0; i < 2; i++ {
		_, _ = chain(context.Background(), &Request{Model: "primary"})
	}
	if got := engine.lastModel(); got != "primary" {
		t.Errorf("after 2 overload errors (threshold=3), model = %q, want %q", got, "primary")
	}
}

// TestFallback_SwapsAfterThreshold verifies that the model is swapped to
// the fallback model after the overload threshold is reached.
func TestFallback_SwapsAfterThreshold(t *testing.T) {
	engine := &fallbackEngine{err: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	unary, _ := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel:     "fallback",
		OverloadThreshold: 3,
	})

	chain := unary(engine.Generate)

	// Three overload errors — reaches threshold, next call should use fallback.
	for i := 0; i < 3; i++ {
		_, _ = chain(context.Background(), &Request{Model: "primary"})
	}
	// The 3rd error triggers usingFallback=true, but the 3rd call itself
	// still used primary (state is checked at START of call). The 4th call
	// should use fallback.
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "fallback" {
		t.Errorf("after 3 overload errors (threshold=3), 4th call model = %q, want %q", got, "fallback")
	}
}

// TestFallback_RecoverOnSuccess verifies that a successful call resets
// the fallback state back to primary.
func TestFallback_RecoverOnSuccess(t *testing.T) {
	// Phase 1: trigger fallback with threshold=1 and overload error.
	engine := &fallbackEngine{err: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	unary, _ := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel:     "fallback",
		OverloadThreshold: 1,
	})

	chain := unary(engine.Generate)

	// One overload error — threshold=1, next call should use fallback.
	_, _ = chain(context.Background(), &Request{Model: "primary"})

	// Phase 2: switch to success, call should use fallback (state already switched).
	engine.err = nil
	engine.resp = &Response{Content: "ok"}
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "fallback" {
		t.Errorf("after overload+success, call should still use fallback (state was switched before success), got %q", got)
	}

	// Phase 3: success was recorded, next call should recover to primary.
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "primary" {
		t.Errorf("after success, model should recover to primary, got %q", got)
	}
}

// TestFallback_NonOverloadErrorNoSwitch verifies that non-overload errors
// (e.g., network errors) do not trigger model swap.
func TestFallback_NonOverloadErrorNoSwitch(t *testing.T) {
	engine := &fallbackEngine{err: NewLLMError(ErrCodeNetwork, "network error", nil)}
	unary, _ := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel:     "fallback",
		OverloadThreshold: 1,
	})

	chain := unary(engine.Generate)

	// Multiple non-overload errors should not trigger swap.
	for i := 0; i < 5; i++ {
		_, _ = chain(context.Background(), &Request{Model: "primary"})
	}
	if got := engine.lastModel(); got != "primary" {
		t.Errorf("after 5 non-overload errors, model = %q, want %q", got, "primary")
	}
}

// TestFallback_CustomIsOverload verifies that a custom IsOverload predicate
// is used instead of the default.
func TestFallback_CustomIsOverload(t *testing.T) {
	engine := &fallbackEngine{err: NewLLMError(ErrCodeServer, "server error", nil)}
	customCalled := false
	unary, _ := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel:     "fallback",
		OverloadThreshold: 1,
		IsOverload: func(err error) bool {
			customCalled = true
			var llmErr *LLMError
			if errors.As(err, &llmErr) {
				return llmErr.Code == ErrCodeServer
			}
			return false
		},
	})

	chain := unary(engine.Generate)

	// ErrCodeServer is not an overload by default, but custom predicate treats it as one.
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if !customCalled {
		t.Fatal("custom IsOverload was not called")
	}

	// Now state should be switched (threshold=1), next call uses fallback.
	engine.err = nil
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "fallback" {
		t.Errorf("custom IsOverload: after server error, model = %q, want %q", got, "fallback")
	}
}

// TestFallback_DefaultsApplied verifies that default threshold (3) and
// default IsOverload (IsOverloadError) are applied when zero-valued.
func TestFallback_DefaultsApplied(t *testing.T) {
	engine := &fallbackEngine{err: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	unary, _ := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel: "fallback",
		// OverloadThreshold and IsOverload left as zero values
	})

	chain := unary(engine.Generate)

	// Default threshold is 3 — two errors should not trigger swap.
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "primary" {
		t.Errorf("default threshold=3: after 2 errors, model = %q, want %q", got, "primary")
	}

	// Third error reaches threshold, fourth call uses fallback.
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	engine.err = nil
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "fallback" {
		t.Errorf("default threshold=3: after 3 errors, 4th call model = %q, want %q", got, "fallback")
	}
}

// TestFallback_StreamSwapsAfterThreshold verifies that the stream middleware
// also swaps the model after the threshold is reached.
func TestFallback_StreamSwapsAfterThreshold(t *testing.T) {
	engine := &fallbackEngine{streamErr: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	_, stream := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel:     "fallback",
		OverloadThreshold: 1,
	})

	chain := stream(engine.Stream)

	// One stream-open overload error — threshold=1, next call should use fallback.
	_, _ = chain(context.Background(), &Request{Model: "primary"})

	// Now state is switched, next stream call should use fallback.
	engine.streamErr = nil
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "fallback" {
		t.Errorf("stream: after overload, model = %q, want %q", got, "fallback")
	}
}

// TestFallback_StreamNoopWhenNoFallbackModel verifies that the stream
// middleware is a no-op when FallbackModel is empty.
func TestFallback_StreamNoopWhenNoFallbackModel(t *testing.T) {
	engine := &fallbackEngine{streamErr: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	_, stream := FallbackMiddlewareWithConfig(FallbackConfig{FallbackModel: ""})

	chain := stream(engine.Stream)

	for i := 0; i < 5; i++ {
		_, _ = chain(context.Background(), &Request{Model: "primary"})
	}
	if got := engine.lastModel(); got != "primary" {
		t.Errorf("stream: after 5 overload errors with empty FallbackModel, model = %q, want %q", got, "primary")
	}
}

// TestFallback_ConcurrentSafe verifies that the middleware does not race
// when called concurrently. This is a best-effort race detector test.
func TestFallback_ConcurrentSafe(t *testing.T) {
	engine := &fallbackEngine{err: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	unary, _ := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel:     "fallback",
		OverloadThreshold: 2,
	})

	chain := unary(engine.Generate)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = chain(context.Background(), &Request{Model: "primary"})
		}()
	}
	wg.Wait()

	// After 10 concurrent overload errors, state should be usingFallback=true.
	// A subsequent call should use fallback.
	engine.err = nil
	_, _ = chain(context.Background(), &Request{Model: "primary"})
	if got := engine.lastModel(); got != "fallback" {
		t.Errorf("concurrent: after 10 overload errors, model = %q, want %q", got, "fallback")
	}
}

// TestFallback_OriginalRequestNotMutated verifies that maybeSwap does not
// modify the original request passed by the caller.
func TestFallback_OriginalRequestNotMutated(t *testing.T) {
	engine := &fallbackEngine{err: NewLLMError(ErrCodeOverloaded, "overloaded", nil)}
	unary, _ := FallbackMiddlewareWithConfig(FallbackConfig{
		FallbackModel:     "fallback",
		OverloadThreshold: 1,
	})

	chain := unary(engine.Generate)

	origReq := &Request{Model: "primary"}
	_, _ = chain(context.Background(), origReq)
	if origReq.Model != "primary" {
		t.Errorf("original request model was mutated: got %q, want %q", origReq.Model, "primary")
	}

	// After threshold reached, next call should swap but original should be unchanged.
	engine.err = nil
	_, _ = chain(context.Background(), origReq)
	if origReq.Model != "primary" {
		t.Errorf("original request model was mutated after swap: got %q, want %q", origReq.Model, "primary")
	}
}
