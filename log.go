package slm

import (
	"log/slog"
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
