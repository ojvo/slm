package slm

import "time"

// Config 统一的 LLM 配置结构
type Config struct {
	Provider           ProviderConfig
	Transport          Transport
	Retry              RetryConfig
	Capabilities       *CapabilityNegotiationOptions
	Observers          []LifecycleObserver
	Timeout            time.Duration
	RequestIDEnabled   bool
	RequestIDGenerator func() string
	RateLimit          *RateLimitConfig
}

// DefaultConfig 返回默认配置
func DefaultConfig() Config {
	return Config{
		Provider: ProviderConfig{},
		Retry:    DefaultRetryConfig(),
	}
}

// WithProvider 设置 Provider 配置
func (c Config) WithProvider(cfg ProviderConfig) Config {
	c.Provider = cfg
	return c
}

// WithTransport 设置自定义 Transport
func (c Config) WithTransport(transport Transport) Config {
	c.Transport = transport
	return c
}

// WithRetry 设置重试配置
func (c Config) WithRetry(cfg RetryConfig) Config {
	c.Retry = cfg
	return c
}

// WithCapabilityNegotiation enables explicit provider/model capability validation.
func (c Config) WithCapabilityNegotiation(opts CapabilityNegotiationOptions) Config {
	c.Capabilities = &opts
	return c
}

// WithObserver appends a lifecycle observer for metrics, tracing, or audit.
func (c Config) WithObserver(observer LifecycleObserver) Config {
	if observer != nil {
		c.Observers = append(c.Observers, observer)
	}
	return c
}

// WithTimeout sets a total timeout around a request or stream session.
func (c Config) WithTimeout(timeout time.Duration) Config {
	c.Timeout = timeout
	return c
}

// WithRequestID enables request id propagation. Nil uses the default generator.
func (c Config) WithRequestID(generator func() string) Config {
	c.RequestIDEnabled = true
	c.RequestIDGenerator = generator
	return c
}

// WithRateLimit enables a shared rate limiter for unary and streaming calls.
func (c Config) WithRateLimit(limit float64, burst int) Config {
	c.RateLimit = &RateLimitConfig{Limit: limit, Burst: burst}
	return c
}

// BuildEngineWithTransport 根据配置和 Transport 构建 Engine。
// 如果配置了 Transport，使用 NewOpenAIWithTransport 创建引擎；
// 否则使用 ProviderConfig 的 Endpoint/APIKey 创建 HTTP 传输。
func (c Config) BuildEngineWithTransport() (Engine, error) {
	var engine Engine
	if c.Transport != nil {
		if c.Provider.DefaultModel == "" {
			return nil, NewLLMError(ErrCodeInvalidConfig, "DefaultModel is required when using custom Transport", nil)
		}
		engine = NewOpenAIWithTransport(c.Transport, c.Provider.DefaultModel)
	} else {
		if c.Provider.Endpoint == "" {
			return nil, NewLLMError(ErrCodeInvalidConfig, "Endpoint is required when Transport is not set", nil)
		}
		if c.Provider.DefaultModel == "" {
			return nil, NewLLMError(ErrCodeInvalidConfig, "DefaultModel is required", nil)
		}
		engine = NewOpenAIProtocol(c.Provider.Endpoint, c.Provider.APIKey, c.Provider.DefaultModel)
	}

	return c.applyMiddleware(engine), nil
}

func (c Config) BuildResponsesEngine() (*OpenAIResponsesEngine, error) {
	var engine *OpenAIResponsesEngine
	if c.Transport != nil {
		if c.Provider.DefaultModel == "" {
			return nil, NewLLMError(ErrCodeInvalidConfig, "DefaultModel is required when using custom Transport", nil)
		}
		engine = NewOpenAIResponsesWithTransport(c.Transport, c.Provider.DefaultModel)
	} else {
		if c.Provider.Endpoint == "" {
			return nil, NewLLMError(ErrCodeInvalidConfig, "Endpoint is required when Transport is not set", nil)
		}
		if c.Provider.DefaultModel == "" {
			return nil, NewLLMError(ErrCodeInvalidConfig, "DefaultModel is required", nil)
		}
		engine = NewOpenAIResponsesProtocol(c.Provider.Endpoint, c.Provider.APIKey, c.Provider.DefaultModel)
	}

	return c.applyResponsesMiddleware(engine), nil
}

// applyMiddleware 统一应用标准中间件
func (c Config) applyMiddleware(engine Engine) Engine {
	retry := c.Retry
	return ApplyStandardMiddleware(engine, StandardMiddlewareOptions{
		DefaultModel:       c.Provider.DefaultModel,
		Capabilities:       c.resolvedCapabilities(),
		Observers:          c.Observers,
		EnableRequestID:    c.RequestIDEnabled,
		RequestIDGenerator: c.RequestIDGenerator,
		CrossCutting: CrossCuttingMiddlewareOptions{
			Retry:     &retry,
			Timeout:   c.Timeout,
			RateLimit: c.RateLimit,
		},
	})
}

func (c Config) resolvedCapabilities() *CapabilityNegotiationOptions {
	if c.Capabilities == nil {
		return nil
	}
	clone := *c.Capabilities
	if clone.DefaultModel == "" {
		clone.DefaultModel = c.Provider.DefaultModel
	}
	return &clone
}

func (c Config) applyResponsesMiddleware(engine *OpenAIResponsesEngine) *OpenAIResponsesEngine {
	retry := c.Retry
	return ApplyStandardResponseMiddleware(engine, StandardMiddlewareOptions{
		DefaultModel:       c.Provider.DefaultModel,
		Capabilities:       c.resolvedCapabilities(),
		Observers:          c.Observers,
		EnableRequestID:    c.RequestIDEnabled,
		RequestIDGenerator: c.RequestIDGenerator,
		CrossCutting: CrossCuttingMiddlewareOptions{
			Retry:     &retry,
			Timeout:   c.Timeout,
			RateLimit: c.RateLimit,
		},
	})
}

type ProviderConfig struct {
	Endpoint     string
	DefaultModel string
	APIKey       string
}
