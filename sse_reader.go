package slm

import (
	"bufio"
	"bytes"
	"io"
)

// StreamParser is a function that parses a single SSE data line into a Response
type StreamParser func(event string, data []byte) (*Response, bool, error)

// SSEReader implements StreamIterator for SSE streams
type SSEReader struct {
	reader       *bufio.Reader
	parser       StreamParser
	current      *Response
	rawLine      []byte
	closer       io.Closer
	usage        *Usage
	err          error
	currentEvent string
	dataBuffer   bytes.Buffer
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

	for {
		line, err := r.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if r.dataBuffer.Len() > 0 {
					yield, stop := r.dispatch()
					if stop {
						return false
					}
					return yield
				}
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
					return false
				}
				if yield {
					return true
				}
				continue
			}
			r.currentEvent = ""
			continue
		}

		if bytes.HasPrefix(line, []byte("event:")) {
			r.currentEvent = string(bytes.TrimSpace(line[6:]))
			continue
		}

		if bytes.HasPrefix(line, []byte("data:")) {
			data := bytes.TrimSpace(line[5:])
			if string(data) == "[DONE]" {
				if r.dataBuffer.Len() > 0 {
					yield, _ := r.dispatch()
					return yield
				}
				r.Close()
				return false
			}
			r.dataBuffer.Write(data)
			continue
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
	if r.closer != nil {
		return r.closer.Close()
	}
	return nil
}

// Usage returns token usage after stream ends.
func (r *SSEReader) Usage() *Usage { return r.usage }

func (r *SSEReader) dispatch() (bool, bool) {
	data := r.dataBuffer.Bytes()
	r.dataBuffer.Reset()

	resp, done, err := r.parser(r.currentEvent, data)
	if err != nil {
		r.err = err
		return false, true
	}

	r.current = resp
	r.rawLine = data

	if done && resp != nil && resp.Usage.PromptTokens > 0 {
		r.usage = &resp.Usage
	}

	return true, done
}
