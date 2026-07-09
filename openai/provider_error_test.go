package openai

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"ojv/slm"
)

func TestBuildProviderErrorEnvelope_Auth(t *testing.T) {
	err := slm.NewLLMError(slm.ErrCodeAuth, "bad key", nil)
	status, typ, code, payload := BuildProviderErrorEnvelope(err)
	if status != 401 || typ != "authentication_error" || code != slm.ErrCodeAuth.String() {
		t.Fatalf("unexpected mapping status=%d type=%q code=%q", status, typ, code)
	}
	if !strings.Contains(payload.Error.Message, "bad key") {
		t.Fatalf("unexpected message: %#v", payload)
	}
}

func TestBuildProviderErrorEnvelope_InvalidRequest(t *testing.T) {
	err := slm.NewLLMError(slm.ErrCodeUnsupportedCapability, "bad request", nil)
	status, typ, _, _ := BuildProviderErrorEnvelope(err)
	if status != 400 || typ != "invalid_request_error" {
		t.Fatalf("unexpected mapping status=%d type=%q", status, typ)
	}
}

func TestBuildProviderErrorEnvelope_RateLimit(t *testing.T) {
	err := slm.NewLLMError(slm.ErrCodeRateLimit, "too many", nil)
	status, typ, _, _ := BuildProviderErrorEnvelope(err)
	if status != 429 || typ != "rate_limit_error" {
		t.Fatalf("unexpected mapping status=%d type=%q", status, typ)
	}
}

func TestBuildProviderErrorEnvelope_NonLLM(t *testing.T) {
	err := errors.New("boom")
	status, typ, code, payload := BuildProviderErrorEnvelope(err)
	if status != 500 || typ != "server_error" || code != "" {
		t.Fatalf("unexpected mapping status=%d type=%q code=%q", status, typ, code)
	}
	if payload.Error.Message != "boom" {
		t.Fatalf("unexpected payload: %#v", payload)
	}
}

func TestWriteSSEData_UsesFrameTerminator(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteSSEData(&buf, map[string]any{"object": "chat.completion.chunk"}); err != nil {
		t.Fatalf("WriteSSEData() error = %v", err)
	}
	out := buf.String()
	if strings.Contains(out, `\\n`) {
		t.Fatalf("expected real newline delimiters, got %q", out)
	}
	if !strings.HasSuffix(out, "\n\n") {
		t.Fatalf("expected SSE frame terminator, got %q", out)
	}
}
