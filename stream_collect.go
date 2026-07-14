package slm

import (
	"context"
	"fmt"
	"strings"
)

// maxStreamContentSize bounds the total accumulated text from a single
// StreamAndCollect call to avoid unbounded memory growth from a runaway
// stream. 10 MB matches the colm/slm default.
const maxStreamContentSize = 10 * 1024 * 1024

// StreamChunkCallback is invoked for each chunk received during StreamAndCollect.
// The chunk argument is the text content of the current chunk (iter.Text()).
// Returning a non-nil error immediately stops iteration and returns that
// error from StreamAndCollect (already-accumulated content is discarded).
type StreamChunkCallback func(chunk string) error

type streamCallbackKey struct{}

// WithStreamCallback injects cb into ctx so that a subsequent StreamAndCollect
// call (or any stream consumer that checks for the callback) will invoke it
// per chunk. If cb is nil, ctx is returned unchanged.
//
// This allows middleware or callers to observe stream progress without
// wrapping the StreamIterator, matching the context-injection pattern used
// by RetryMiddleware (ctxKeyRequestID) and timeout middleware.
func WithStreamCallback(ctx context.Context, cb StreamChunkCallback) context.Context {
	if cb == nil {
		return ctx
	}
	return context.WithValue(ctx, streamCallbackKey{}, cb)
}

// StreamCallbackFromCtx extracts a previously-injected StreamChunkCallback
// from ctx. Returns nil if no callback was injected.
func StreamCallbackFromCtx(ctx context.Context) StreamChunkCallback {
	v := ctx.Value(streamCallbackKey{})
	if v == nil {
		return nil
	}
	cb, _ := v.(StreamChunkCallback)
	return cb
}

// StreamAndCollect calls engine.Stream, iterates all chunks, and for each
// chunk:
//  1. Appends iter.Text() to an internal strings.Builder
//  2. Invokes the callback injected via WithStreamCallback (if any)
//
// It returns the accumulated full text.
//
// Behavior:
//   - The input request is NOT mutated. A deep clone is made and Stream is
//     set to true, matching the pattern used by StreamCall[T]. This follows
//     the "explicit over implicit" principle — callers don't need to remember
//     to set req.Stream = true before calling.
//   - If the accumulated text exceeds maxStreamContentSize (10 MB), iteration
//     stops and an error is returned.
//   - If iter.Err() is non-nil after iteration, that error is returned.
//   - If the callback returns a non-nil error, iteration stops immediately
//     and the callback's error is returned (accumulated content is discarded).
//
// StreamAndCollect vs StreamCall[T]:
//   - StreamCall[T] is for JSON-structured output (auto JSON extraction +
//     generic unmarshalling + reasoning/toolcall/usage tracking)
//   - StreamAndCollect is for plain-text output (no JSON extraction,
//     supports per-chunk callback, 10 MB size guard)
func StreamAndCollect(ctx context.Context, engine Engine, req *Request) (string, error) {
	clone := cloneRequest(req)
	if clone == nil {
		clone = &Request{}
	}
	clone.Stream = true

	iter, err := engine.Stream(ctx, clone)
	if err != nil {
		return "", err
	}
	defer iter.Close()

	cb := StreamCallbackFromCtx(ctx)
	var sb strings.Builder
	for iter.Next() {
		text := iter.Text()
		sb.WriteString(text)
		if sb.Len() > maxStreamContentSize {
			return "", fmt.Errorf("stream response exceeds max size (%d bytes)", maxStreamContentSize)
		}
		if cb != nil {
			if cbErr := cb(text); cbErr != nil {
				return "", cbErr
			}
		}
	}

	if err := iter.Err(); err != nil {
		return "", err
	}
	return sb.String(), nil
}
