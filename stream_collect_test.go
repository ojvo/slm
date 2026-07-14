package slm

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// chunkIter is a controllable StreamIterator that yields a list of text
// chunks one by one. It optionally returns a terminal error after the
// last chunk.
type chunkIter struct {
	chunks []string
	idx    int
	err    error // returned by Err() after chunks are exhausted
	closed bool
}

func (i *chunkIter) Next() bool {
	if i.idx >= len(i.chunks) {
		return false
	}
	i.idx++
	return true
}
func (i *chunkIter) Chunk() []byte {
	if i.idx == 0 {
		return nil
	}
	return []byte(i.chunks[i.idx-1])
}
func (i *chunkIter) Text() string {
	if i.idx == 0 {
		return ""
	}
	return i.chunks[i.idx-1]
}
func (i *chunkIter) FullText() string {
	var sb strings.Builder
	for _, c := range i.chunks {
		sb.WriteString(c)
	}
	return sb.String()
}
func (i *chunkIter) Err() error { return i.err }
func (i *chunkIter) Close() error {
	i.closed = true
	return nil
}
func (i *chunkIter) Usage() *Usage       { return nil }
func (i *chunkIter) Response() *Response { return nil }

// streamCollectEngine is a controllable Engine for testing StreamAndCollect.
type streamCollectEngine struct {
	iter    StreamIterator
	iterErr error // returned by Stream (open error)
}

func (e *streamCollectEngine) Generate(context.Context, *Request) (*Response, error) {
	return &Response{Content: "ok"}, nil
}
func (e *streamCollectEngine) Stream(context.Context, *Request) (StreamIterator, error) {
	if e.iterErr != nil {
		return nil, e.iterErr
	}
	return e.iter, nil
}
func (e *streamCollectEngine) Capabilities() *ProtocolCapabilities {
	return &ProtocolCapabilities{Description: "stream collect test engine"}
}

// TestStreamAndCollect_BasicCollection verifies that StreamAndCollect
// accumulates all chunks into the returned string.
func TestStreamAndCollect_BasicCollection(t *testing.T) {
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: []string{"Hello", ", ", "world", "!"}},
	}

	got, err := StreamAndCollect(context.Background(), engine, &Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "Hello, world!"; got != want {
		t.Errorf("collected = %q, want %q", got, want)
	}
}

// TestStreamAndCollect_CallbackInvoked verifies that the callback is
// called for each chunk with the correct text.
func TestStreamAndCollect_CallbackInvoked(t *testing.T) {
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: []string{"a", "b", "c"}},
	}

	var seen []string
	ctx := WithStreamCallback(context.Background(), func(chunk string) error {
		seen = append(seen, chunk)
		return nil
	})

	_, err := StreamAndCollect(ctx, engine, &Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(seen) != 3 {
		t.Fatalf("callback called %d times, want 3", len(seen))
	}
	if want := []string{"a", "b", "c"}; !sliceEqual(seen, want) {
		t.Errorf("callback saw %v, want %v", seen, want)
	}
}

// TestStreamAndCollect_CallbackErrorStopsIteration verifies that a
// callback error stops iteration immediately and returns that error.
func TestStreamAndCollect_CallbackErrorStopsIteration(t *testing.T) {
	iter := &chunkIter{chunks: []string{"a", "b", "c"}}
	engine := &streamCollectEngine{iter: iter}

	cbErr := errors.New("callback stop")
	ctx := WithStreamCallback(context.Background(), func(chunk string) error {
		if chunk == "b" {
			return cbErr
		}
		return nil
	})

	_, err := StreamAndCollect(ctx, engine, &Request{})
	if !errors.Is(err, cbErr) {
		t.Errorf("error = %v, want %v", err, cbErr)
	}
	// "c" should not have been delivered to the callback.
	if iter.idx != 2 {
		t.Errorf("iteration continued after callback error: idx = %d, want 2", iter.idx)
	}
}

// TestStreamAndCollect_MaxSizeExceeded verifies that exceeding
// maxStreamContentSize returns an error.
func TestStreamAndCollect_MaxSizeExceeded(t *testing.T) {
	// Single chunk just over the limit.
	oversized := strings.Repeat("x", maxStreamContentSize+1)
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: []string{oversized}},
	}

	_, err := StreamAndCollect(context.Background(), engine, &Request{})
	if err == nil {
		t.Fatal("expected max size error, got nil")
	}
	if !strings.Contains(err.Error(), "max size") {
		t.Errorf("error = %v, want it to contain 'max size'", err)
	}
}

// TestStreamAndCollect_MaxSizeBoundary verifies that a stream exactly at
// the limit does NOT trigger the error (the check is strictly greater-than).
func TestStreamAndCollect_MaxSizeBoundary(t *testing.T) {
	atLimit := strings.Repeat("x", maxStreamContentSize)
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: []string{atLimit}},
	}

	got, err := StreamAndCollect(context.Background(), engine, &Request{})
	if err != nil {
		t.Fatalf("unexpected error at boundary: %v", err)
	}
	if len(got) != maxStreamContentSize {
		t.Errorf("collected length = %d, want %d", len(got), maxStreamContentSize)
	}
}

// TestStreamAndCollect_StreamError verifies that iter.Err() is propagated.
func TestStreamAndCollect_StreamError(t *testing.T) {
	streamErr := NewLLMError(ErrCodeServer, "stream failed", nil)
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: []string{"a"}, err: streamErr},
	}

	_, err := StreamAndCollect(context.Background(), engine, &Request{})
	if !errors.Is(err, streamErr) {
		t.Errorf("error = %v, want %v", err, streamErr)
	}
}

// TestStreamAndCollect_NilCallbackNoop verifies that StreamAndCollect
// works normally when no callback is injected.
func TestStreamAndCollect_NilCallbackNoop(t *testing.T) {
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: []string{"hello", "world"}},
	}

	// No WithStreamCallback — plain ctx.
	got, err := StreamAndCollect(context.Background(), engine, &Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "helloworld"; got != want {
		t.Errorf("collected = %q, want %q", got, want)
	}
}

// TestStreamAndCollect_RequestNotMutated verifies that the input request
// is not mutated (Stream field should remain false).
func TestStreamAndCollect_RequestNotMutated(t *testing.T) {
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: []string{"ok"}},
	}

	req := &Request{Model: "test-model", Stream: false}
	_, _ = StreamAndCollect(context.Background(), engine, req)
	if req.Stream {
		t.Errorf("input request Stream was mutated to true")
	}
	if req.Model != "test-model" {
		t.Errorf("input request Model was mutated: got %q", req.Model)
	}
}

// TestStreamAndCollect_NilRequestHandled verifies that a nil request
// does not panic (cloneRequest returns nil, StreamAndCollect creates &Request{}).
func TestStreamAndCollect_NilRequestHandled(t *testing.T) {
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: []string{"ok"}},
	}

	got, err := StreamAndCollect(context.Background(), engine, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "ok" {
		t.Errorf("collected = %q, want %q", got, "ok")
	}
}

// TestStreamAndCollect_StreamOpenError verifies that an error from
// engine.Stream (open failure) is propagated without calling the iterator.
func TestStreamAndCollect_StreamOpenError(t *testing.T) {
	openErr := NewLLMError(ErrCodeNetwork, "connection refused", nil)
	engine := &streamCollectEngine{iterErr: openErr}

	_, err := StreamAndCollect(context.Background(), engine, &Request{})
	if !errors.Is(err, openErr) {
		t.Errorf("error = %v, want %v", err, openErr)
	}
}

// TestStreamAndCollect_EmptyStream verifies that an empty stream (no chunks)
// returns an empty string without error.
func TestStreamAndCollect_EmptyStream(t *testing.T) {
	engine := &streamCollectEngine{
		iter: &chunkIter{chunks: nil},
	}

	got, err := StreamAndCollect(context.Background(), engine, &Request{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("collected = %q, want empty", got)
	}
}

// TestStreamAndCollect_IteratorClosed verifies that the iterator's Close
// method is called after StreamAndCollect completes.
func TestStreamAndCollect_IteratorClosed(t *testing.T) {
	iter := &chunkIter{chunks: []string{"ok"}}
	engine := &streamCollectEngine{iter: iter}

	_, _ = StreamAndCollect(context.Background(), engine, &Request{})
	if !iter.closed {
		t.Error("iterator was not closed")
	}
}

// TestStreamAndCollect_IteratorClosedOnError verifies that Close is called
// even when an error occurs.
func TestStreamAndCollect_IteratorClosedOnError(t *testing.T) {
	iter := &chunkIter{
		chunks: []string{"a"},
		err:    NewLLMError(ErrCodeServer, "fail", nil),
	}
	engine := &streamCollectEngine{iter: iter}

	_, _ = StreamAndCollect(context.Background(), engine, &Request{})
	if !iter.closed {
		t.Error("iterator was not closed on error")
	}
}

// TestWithStreamCallback_NilCallbackReturnsOriginalCtx verifies that
// passing a nil callback returns the original context unchanged.
func TestWithStreamCallback_NilCallbackReturnsOriginalCtx(t *testing.T) {
	ctx := context.Background()
	result := WithStreamCallback(ctx, nil)
	if result != ctx {
		t.Error("nil callback should return original context")
	}
}

// TestStreamCallbackFromCtx_NoCallbackReturnsNil verifies that extracting
// from a context without a callback returns nil.
func TestStreamCallbackFromCtx_NoCallbackReturnsNil(t *testing.T) {
	cb := StreamCallbackFromCtx(context.Background())
	if cb != nil {
		t.Error("expected nil callback from plain context")
	}
}

// TestStreamCallbackFromCtx_RoundTrip verifies that a callback injected
// via WithStreamCallback can be extracted via StreamCallbackFromCtx.
func TestStreamCallbackFromCtx_RoundTrip(t *testing.T) {
	original := StreamChunkCallback(func(s string) error { return nil })
	ctx := WithStreamCallback(context.Background(), original)
	extracted := StreamCallbackFromCtx(ctx)
	if extracted == nil {
		t.Fatal("expected non-nil callback after injection")
	}
	// Verify it's callable.
	if err := extracted("test"); err != nil {
		t.Errorf("extracted callback returned error: %v", err)
	}
}

func sliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
