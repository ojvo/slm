package slm

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"sync"
)

// sse frame

const maxDataBufferSize = 10 * 1024 * 1024

type sseFrame struct {
	Event string
	Data  []byte
	Done  bool
}

type sseFrameReader struct {
	reader       *bufio.Reader
	currentEvent string
	dataBuffer   bytes.Buffer
}

func newSSEFrameReader(r io.Reader) *sseFrameReader {
	return &sseFrameReader{reader: bufio.NewReader(r)}
}

func (r *sseFrameReader) Next() (sseFrame, bool, error) {
	for {
		line, err := r.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if len(line) > 0 {
					line = bytes.TrimSuffix(line, []byte("\n"))
					line = bytes.TrimSuffix(line, []byte("\r"))
					done, err := r.processLine(line)
					if err != nil {
						return sseFrame{}, false, err
					}
					if done {
						return sseFrame{Done: true}, true, nil
					}
				}
				if r.dataBuffer.Len() > 0 {
					return r.dispatch(), true, nil
				}
				return sseFrame{}, false, io.EOF
			}
			return sseFrame{}, false, err
		}

		line = bytes.TrimSuffix(line, []byte("\n"))
		line = bytes.TrimSuffix(line, []byte("\r"))

		if len(line) == 0 {
			if r.dataBuffer.Len() == 0 {
				r.currentEvent = ""
				continue
			}
			return r.dispatch(), true, nil
		}

		done, err := r.processLine(line)
		if err != nil {
			return sseFrame{}, false, err
		}
		if done {
			return sseFrame{Done: true}, true, nil
		}
	}
}

func (r *sseFrameReader) processLine(line []byte) (bool, error) {
	if bytes.HasPrefix(line, []byte(":")) {
		return false, nil
	}
	if bytes.HasPrefix(line, []byte("event:")) {
		r.currentEvent = string(bytes.TrimSpace(line[6:]))
		return false, nil
	}
	if bytes.HasPrefix(line, []byte("data:")) {
		data := bytes.TrimSpace(line[5:])
		if bytes.Equal(data, []byte("[DONE]")) {
			return true, nil
		}
		if r.dataBuffer.Len()+len(data) > maxDataBufferSize {
			return false, fmt.Errorf("SSE data buffer exceeded %d bytes", maxDataBufferSize)
		}
		if r.dataBuffer.Len() > 0 {
			r.dataBuffer.WriteByte('\n')
		}
		r.dataBuffer.Write(data)
	}
	return false, nil
}

type sseFrameResult struct {
	Frame sseFrame
	Done  bool
	Err   error
}

func consumeSSEFrame(framer *sseFrameReader) sseFrameResult {
	frame, ok, err := framer.Next()
	if err != nil {
		if err == io.EOF {
			return sseFrameResult{Done: true}
		}
		return sseFrameResult{Err: err}
	}
	if !ok {
		return sseFrameResult{Done: true}
	}
	if frame.Done {
		return sseFrameResult{Done: true}
	}
	return sseFrameResult{Frame: frame}
}

func (r *sseFrameReader) dispatch() sseFrame {
	frame := sseFrame{Event: r.currentEvent, Data: append([]byte(nil), r.dataBuffer.Bytes()...)}
	r.dataBuffer.Reset()
	r.currentEvent = ""
	return frame
}

// sse reader

type sseDispatchResult struct {
	Yield bool
	Stop  bool
	Err   error
}

type sseIteratorCore struct {
	framer    *sseFrameReader
	err       error
	done      bool
	wrapError func(error) error
	onDone    func()
}

func (c *sseIteratorCore) Next(dispatch func(sseFrame) sseDispatchResult) bool {
	if c.done || c.err != nil {
		return false
	}
	for {
		result := consumeSSEFrame(c.framer)
		if result.Done {
			c.done = true
			if c.onDone != nil {
				c.onDone()
			}
			return false
		}
		if result.Err != nil {
			if c.wrapError != nil {
				c.err = c.wrapError(result.Err)
			} else {
				c.err = result.Err
			}
			return false
		}
		dr := dispatch(result.Frame)
		if dr.Err != nil {
			c.err = dr.Err
			return false
		}
		if dr.Stop {
			return dr.Yield
		}
		if dr.Yield {
			return true
		}
	}
}

func (c *sseIteratorCore) Err() error { return c.err }

type StreamParser func(event string, data []byte) (*Response, bool, error)

type SSEReader struct {
	core      sseIteratorCore
	parser    StreamParser
	current   *Response
	rawLine   []byte
	closer    io.Closer
	closeOnce sync.Once
	usage     *Usage
}

func NewSSEReader(r io.ReadCloser, parser StreamParser) *SSEReader {
	reader := &SSEReader{
		parser: parser,
		closer: r,
	}
	reader.core = sseIteratorCore{
		framer: newSSEFrameReader(r),
		wrapError: func(err error) error {
			return WrapOperationalError("stream read error", err)
		},
		onDone: func() { reader.Close() },
	}
	return reader
}

func (r *SSEReader) Next() bool {
	return r.core.Next(r.dispatch)
}

func (r *SSEReader) Chunk() []byte { return r.rawLine }

func (r *SSEReader) Text() string {
	if r.current != nil {
		return r.current.Content
	}
	return ""
}

func (r *SSEReader) FullText() string {
	if r.current != nil {
		if r.current.ReasoningContent != "" {
			return r.current.ReasoningContent + "\n" + r.current.Content
		}
		return r.current.Content
	}
	return ""
}

func (r *SSEReader) Err() error { return r.core.Err() }

func (r *SSEReader) Close() error {
	var err error
	r.closeOnce.Do(func() {
		if r.closer != nil {
			err = r.closer.Close()
		}
	})
	return err
}

func (r *SSEReader) Interrupt(error) {
	_ = r.Close()
}

func (r *SSEReader) Usage() *Usage { return r.usage }

func (r *SSEReader) Response() *Response { return r.current }

func (r *SSEReader) dispatch(frame sseFrame) sseDispatchResult {
	resp, done, err := r.parser(frame.Event, frame.Data)
	if err != nil {
		var llmErr *LLMError
		if errors.As(err, &llmErr) {
			return sseDispatchResult{Err: err}
		}
		return sseDispatchResult{Err: NewLLMError(ErrCodeParse, "parse stream event", err)}
	}

	r.current = resp
	r.rawLine = append(r.rawLine[:0], frame.Data...)

	if resp != nil && hasUsage(resp.Usage) {
		r.usage = &resp.Usage
	}

	if resp != nil && resp.Content == "" && resp.ReasoningContent == "" && len(resp.ToolCalls) == 0 && resp.FinishReason == "" {
		return sseDispatchResult{Stop: done}
	}

	return sseDispatchResult{Yield: true, Stop: done}
}

func hasUsage(usage Usage) bool {
	return usage.PromptTokens > 0 || usage.CompletionTokens > 0 || usage.TotalTokens > 0
}
