package slm

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
)

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

func (r *sseFrameReader) dispatch() sseFrame {
	frame := sseFrame{Event: r.currentEvent, Data: append([]byte(nil), r.dataBuffer.Bytes()...)}
	r.dataBuffer.Reset()
	r.currentEvent = ""
	return frame
}
