package slm

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"sync"
)

const maxDataBufferSize = 10 * 1024 * 1024 // 10MB

// StreamParser is a function that parses a single SSE data line into a Response
type StreamParser func(event string, data []byte) (*Response, bool, error)

// SSEReader implements StreamIterator for SSE streams
type SSEReader struct {
	reader       *bufio.Reader
	parser       StreamParser
	current      *Response
	rawLine      []byte
	closer       io.Closer
	closeOnce    sync.Once
	usage        *Usage
	err          error
	currentEvent string
	dataBuffer   bytes.Buffer
	done         bool
}

// NewSSEReader creates a new SSEReader
func NewSSEReader(r io.ReadCloser, parser StreamParser) *SSEReader {
	return &SSEReader{
		reader: bufio.NewReader(r),
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
		line, err := r.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if len(line) > 0 {
					line = bytes.TrimSuffix(line, []byte("\n"))
					line = bytes.TrimSuffix(line, []byte("\r"))
					if r.processLine(line) {
						break
					}
				}
				if r.err != nil {
					return false
				}
				if r.dataBuffer.Len() > 0 {
					yield, _ := r.dispatch()
					if yield {
						return true
					}
				}
				r.done = true
				r.Close()
				return false
			}
			r.err = err
			return false
		}

		line = bytes.TrimSuffix(line, []byte("\n"))
		line = bytes.TrimSuffix(line, []byte("\r"))

		if len(line) == 0 {
			if r.dataBuffer.Len() > 0 {
				yield, stop := r.dispatch()
				if stop {
					return yield
				}
				if yield {
					return true
				}
				continue
			}
			r.currentEvent = ""
			continue
		}

		if r.processLine(line) {
			break
		}
		if r.err != nil {
			return false
		}
	}

	if r.dataBuffer.Len() > 0 {
		yield, _ := r.dispatch()
		if yield {
			return true
		}
	}
	r.done = true
	r.Close()
	return false
}

func (r *SSEReader) processLine(line []byte) bool {
	if bytes.HasPrefix(line, []byte("event:")) {
		r.currentEvent = string(bytes.TrimSpace(line[6:]))
		return false
	}

	if bytes.HasPrefix(line, []byte("data:")) {
		data := bytes.TrimSpace(line[5:])
		if string(data) == "[DONE]" {
			return true
		}
		if r.dataBuffer.Len()+len(data) > maxDataBufferSize {
			r.err = fmt.Errorf("SSE data buffer exceeded %d bytes", maxDataBufferSize)
			return false
		}
		if r.dataBuffer.Len() > 0 {
			r.dataBuffer.WriteByte('\n')
		}
		r.dataBuffer.Write(data)
		return false
	}

	return false
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

func (r *SSEReader) dispatch() (bool, bool) {
	data := r.dataBuffer.Bytes()
	r.dataBuffer.Reset()

	resp, done, err := r.parser(r.currentEvent, data)
	if err != nil {
		r.err = err
		return false, true
	}

	r.current = resp
	r.rawLine = make([]byte, len(data))
	copy(r.rawLine, data)

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
