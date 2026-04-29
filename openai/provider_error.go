package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"ojv/slm"
)

// APIError is an OpenAI-compatible error object.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// APIErrorEnvelope is the standard OpenAI error envelope.
type APIErrorEnvelope struct {
	Error APIError `json:"error"`
}

// BuildProviderErrorEnvelope maps slm errors to OpenAI-compatible HTTP status and body.
func BuildProviderErrorEnvelope(err error) (int, string, string, APIErrorEnvelope) {
	status := http.StatusInternalServerError
	errorType := "server_error"
	code := ""
	message := ""
	if err != nil {
		message = err.Error()
	}

	var llmErr *slm.LLMError
	if errors.As(err, &llmErr) {
		code = llmErr.Code.String()
		switch llmErr.Code {
		case slm.ErrCodeAuth:
			status = http.StatusUnauthorized
			errorType = "authentication_error"
		case slm.ErrCodeInvalidModel, slm.ErrCodeInvalidConfig, slm.ErrCodeUnsupportedCapability, slm.ErrCodeContentFilter, slm.ErrCodeContextTooLong, slm.ErrCodeParse:
			status = http.StatusBadRequest
			errorType = "invalid_request_error"
		case slm.ErrCodeRateLimit:
			status = http.StatusTooManyRequests
			errorType = "rate_limit_error"
		case slm.ErrCodeTimeout:
			status = http.StatusGatewayTimeout
		case slm.ErrCodeOverloaded:
			status = http.StatusServiceUnavailable
		case slm.ErrCodeNetwork, slm.ErrCodeServer:
			status = http.StatusBadGateway
		case slm.ErrCodeCancelled:
			status = http.StatusRequestTimeout
		}
	}
	return status, errorType, code, APIErrorEnvelope{Error: APIError{Message: message, Type: errorType, Code: code}}
}

// WriteSSEData writes one SSE data frame with a JSON payload.
func WriteSSEData(w io.Writer, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}
