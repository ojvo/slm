package slm

// Config 统一的 LLM 配置结构
type Config struct {
	Provider ProviderConfig
	Retry    RetryConfig
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

// WithRetry 设置重试配置
func (c Config) WithRetry(cfg RetryConfig) Config {
	c.Retry = cfg
	return c
}

// BuildEngine 根据配置构建 Engine
func (c Config) BuildEngine(createEngine func(ProviderConfig) (Engine, error)) (Engine, error) {
	engine, err := createEngine(c.Provider)
	if err != nil {
		return nil, err
	}

	if c.Retry.MaxAttempts > 1 {
		retryUnary, retryStream := RetryMiddlewareWithConfig(c.Retry)
		return ChainWithStream(engine,
			[]Middleware{retryUnary},
			[]StreamMiddleware{retryStream},
		), nil
	}

	return engine, nil
}

// ProviderConfig 提供商配置
type ProviderConfig struct {
	Endpoint     string
	DefaultModel string
	APIKey       string
}
