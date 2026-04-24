package slm

import (
	"errors"
	"io"
	"sync"
)

const maxDataBufferSize = 10 * 1024 * 1024 // 10MB

// StreamParser is a function that parses a single SSE data line into a Response
type StreamParser func(event string, data []byte) (*Response, bool, error)

// SSEReader implements StreamIterator for SSE streams
type SSEReader struct {
	framer    *sseFrameReader
	parser    StreamParser
	current   *Response
	rawLine   []byte
	closer    io.Closer
	closeOnce sync.Once
	usage     *Usage
	err       error
	done      bool
}

// NewSSEReader creates a new SSEReader
func NewSSEReader(r io.ReadCloser, parser StreamParser) *SSEReader {
	return &SSEReader{
		framer: newSSEFrameReader(r),
		parser: parser,
		closer: r,
	}
}

// Next advances the iterator to the next response chunk.
func (r *SSEReader) Next() bool {
	if r.err != nil {
		return false
	}
	if r.done {
		return false
	}

	for {
		frame, ok, err := r.framer.Next()
		if err != nil {
			if err == io.EOF {
				r.done = true
				r.Close()
				return false
			}
			r.err = WrapOperationalError("stream read error", err)
			return false
		}
		if !ok {
			r.done = true
			r.Close()
			return false
		}
		if frame.Done {
			r.done = true
			r.Close()
			return false
		}
		yield, stop := r.dispatch(frame)
		if stop {
			return yield
		}
		if yield {
			return true
		}
	}
}

// Chunk returns the current raw line.
func (r *SSEReader) Chunk() []byte { return r.rawLine }

// Text returns the current content as string.
func (r *SSEReader) Text() string {
	if r.current != nil {
		return r.current.Content
	}
	return ""
}

// FullText returns the current content including reasoning.
func (r *SSEReader) FullText() string {
	if r.current != nil {
		if r.current.ReasoningContent != "" {
			return r.current.ReasoningContent + "\n" + r.current.Content
		}
		return r.current.Content
	}
	return ""
}

// Err returns any encountered error.
func (r *SSEReader) Err() error { return r.err }

// Close closes the reader and releases resources.
func (r *SSEReader) Close() error {
	var err error
	r.closeOnce.Do(func() {
		if r.closer != nil {
			err = r.closer.Close()
		}
	})
	return err
}

// Interrupt attempts to stop the stream promptly.
func (r *SSEReader) Interrupt(error) {
	_ = r.Close()
}

// Usage returns token usage after stream ends.
func (r *SSEReader) Usage() *Usage { return r.usage }

// Response returns the current parsed response chunk.
func (r *SSEReader) Response() *Response { return r.current }

func (r *SSEReader) dispatch(frame sseFrame) (bool, bool) {
	resp, done, err := r.parser(frame.Event, frame.Data)
	if err != nil {
		var llmErr *LLMError
		if errors.As(err, &llmErr) {
			r.err = err
		} else {
			r.err = NewLLMError(ErrCodeParse, "parse stream event", err)
		}
		return false, true
	}

	r.current = resp
	r.rawLine = append(r.rawLine[:0], frame.Data...)

	if resp != nil && hasUsage(resp.Usage) {
		r.usage = &resp.Usage
	}

	if resp != nil && resp.Content == "" && resp.ReasoningContent == "" && len(resp.ToolCalls) == 0 && resp.FinishReason == "" {
		return false, done
	}

	return true, done
}

func hasUsage(usage Usage) bool {
	return usage.PromptTokens > 0 || usage.CompletionTokens > 0 || usage.TotalTokens > 0
}
