package slm

import (
	"log/slog"
	"time"
)

// Logger 是 LLM 包的日志接口
type Logger interface {
	Info(msg string, args ...any)
	Debug(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
}

// DefaultLogger 使用 slog 的默认实现
type DefaultLogger struct {
	*slog.Logger
}

// NewDefaultLogger 创建默认日志器
func NewDefaultLogger(logger *slog.Logger) Logger {
	return &DefaultLogger{Logger: logger}
}

func (l *DefaultLogger) Info(msg string, args ...any)  { l.Logger.Info(msg, args...) }
func (l *DefaultLogger) Debug(msg string, args ...any) { l.Logger.Debug(msg, args...) }
func (l *DefaultLogger) Warn(msg string, args ...any)  { l.Logger.Warn(msg, args...) }
func (l *DefaultLogger) Error(msg string, args ...any) { l.Logger.Error(msg, args...) }

// NopLogger 空日志器，用于禁用日志
type NopLogger struct{}

func (NopLogger) Info(string, ...any)  {}
func (NopLogger) Debug(string, ...any) {}
func (NopLogger) Warn(string, ...any)  {}
func (NopLogger) Error(string, ...any) {}

func resolvedLogger(logger Logger) Logger {
	if logger == nil {
		return NopLogger{}
	}
	return logger
}

func logRequestStart(logger Logger, msg, requestID string, requestFields []any) {
	logger = resolvedLogger(logger)
	args := append([]any{"request_id", requestID}, requestFields...)
	logger.Debug(msg, args...)
}

func logRequestFailure(logger Logger, msg string, duration time.Duration, requestID string, requestFields []any, err error) {
	logger = resolvedLogger(logger)
	args := append([]any{"duration", duration, "request_id", requestID}, requestFields...)
	args = append(args, ErrorDiagnosticFields(err)...)
	logger.Error(msg, args...)
}

func logRequestCompleted(logger Logger, msg string, duration time.Duration, requestID string, extra ...any) {
	logger = resolvedLogger(logger)
	args := []any{"duration", duration, "request_id", requestID}
	args = append(args, extra...)
	logger.Debug(msg, args...)
}
