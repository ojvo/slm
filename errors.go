package slm

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
)

// ErrorCode represents error codes for LLM operations.
type ErrorCode int

const (
	ErrCodeRateLimit             ErrorCode = 1001
	ErrCodeTimeout               ErrorCode = 1002
	ErrCodeOverloaded            ErrorCode = 1003
	ErrCodeNetwork               ErrorCode = 1004
	ErrCodeServer                ErrorCode = 1005
	ErrCodeAuth                  ErrorCode = 2001
	ErrCodeInvalidModel          ErrorCode = 2002
	ErrCodeInvalidConfig         ErrorCode = 2003
	ErrCodeUnsupportedCapability ErrorCode = 2004
	ErrCodeContentFilter         ErrorCode = 3001
	ErrCodeContextTooLong        ErrorCode = 3002
	ErrCodeParse                 ErrorCode = 3003
	ErrCodeCancelled             ErrorCode = 4001
	ErrCodeInternal              ErrorCode = 4002
)

func (c ErrorCode) String() string {
	names := map[ErrorCode]string{
		ErrCodeRateLimit:             "RateLimit",
		ErrCodeTimeout:               "Timeout",
		ErrCodeOverloaded:            "Overloaded",
		ErrCodeNetwork:               "Network",
		ErrCodeServer:                "Server",
		ErrCodeAuth:                  "Auth",
		ErrCodeInvalidModel:          "InvalidModel",
		ErrCodeInvalidConfig:         "InvalidConfig",
		ErrCodeUnsupportedCapability: "UnsupportedCapability",
		ErrCodeContentFilter:         "ContentFilter",
		ErrCodeContextTooLong:        "ContextTooLong",
		ErrCodeParse:                 "Parse",
		ErrCodeCancelled:             "Cancelled",
		ErrCodeInternal:              "Internal",
	}
	if name, ok := names[c]; ok {
		return name
	}
	return fmt.Sprintf("Unknown(%d)", c)
}

// IsRetryable returns true if the error code indicates a retryable error.
func (c ErrorCode) IsRetryable() bool {
	return c >= 1000 && c < 2000
}

// LLMError represents an error from LLM operations.
type LLMError struct {
	Code    ErrorCode
	Message string
	Cause   error
}

func (e *LLMError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%d] %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("[%d] %s", e.Code, e.Message)
}

func (e *LLMError) Unwrap() error {
	return e.Cause
}

func (e *LLMError) Is(target error) bool {
	t, ok := target.(*LLMError)
	if !ok {
		return false
	}
	return e.Code == t.Code
}

// NewLLMError creates a new LLMError.
func NewLLMError(code ErrorCode, msg string, cause error) *LLMError {
	return &LLMError{Code: code, Message: msg, Cause: cause}
}

// IsRetryableError checks if an error is retryable.
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}
	var llmErr *LLMError
	if errors.As(err, &llmErr) {
		return llmErr.Code.IsRetryable()
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	return false
}

// WrapOperationalError normalizes transport and stream I/O failures into
// structured LLMError values when they match known timeout, cancellation, or
// network patterns.
func WrapOperationalError(message string, err error) error {
	if err == nil {
		return nil
	}
	var llmErr *LLMError
	if errors.As(err, &llmErr) {
		return err
	}
	return NewLLMError(classifyOperationalErrorCode(err), message, err)
}

func classifyOperationalErrorCode(err error) ErrorCode {
	if err == nil {
		return ErrCodeInternal
	}
	if errors.Is(err, context.Canceled) {
		return ErrCodeCancelled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return ErrCodeTimeout
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return ErrCodeTimeout
		}
		return ErrCodeNetwork
	}
	lower := strings.ToLower(err.Error())
	switch {
	case strings.Contains(lower, "client.timeout"):
		return ErrCodeTimeout
	case strings.Contains(lower, "context deadline exceeded"):
		return ErrCodeTimeout
	case strings.Contains(lower, "context canceled"), strings.Contains(lower, "context cancelled"):
		return ErrCodeCancelled
	default:
		return ErrCodeInternal
	}
}

func timeoutSource(err error) string {
	if err == nil {
		return ""
	}
	lower := strings.ToLower(err.Error())
	if strings.Contains(lower, "client.timeout") {
		return "http_client"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "context_deadline"
	}
	if errors.Is(err, context.Canceled) {
		return "context_cancelled"
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return "network_timeout"
	}
	return ""
}
