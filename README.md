# SLM - Simple LLM Client Library

一个简洁、类型安全的 Go 语言大语言模型客户端库，提供统一的 API 抽象层，支持 OpenAI 协议兼容的服务。

## 特性

- **统一抽象** - 单一 `Engine` 接口支持多种 LLM 提供商
- **协议-传输分离** - `Transport` 接口解耦通信层，同一协议引擎可搭配不同传输方式
- **类型安全** - 泛型调用 `Call[T]` 自动解析结构化输出
- **流式响应** - 完整的 SSE 流式迭代器支持
- **中间件系统** - 内置重试、限流、日志、超时、请求追踪
- **工具调用** - 完整的 Function Calling 支持
- **多模态** - 支持文本和图片混合输入
- **推理模型** - 支持 o1/o3 等推理模型的思维链提取
- **零依赖** - 仅使用 Go 标准库

## 安装

```bash
go get ojv/sto/slm
```

## 快速开始

### 基础对话

```go
package main

import (
    "context"
    "fmt"
    "ojv/sto/slm"
)

func main() {
    engine := slm.NewOpenAIProtocol(
        "https://api.openai.com/v1",
        "your-api-key",
        "gpt-4o-mini",
    )

    resp, err := engine.Generate(context.Background(), &slm.Request{
        Messages: []slm.Message{
            slm.NewTextMessage(slm.RoleUser, "Hello, world!"),
        },
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Content)
}
```

### 流式输出

```go
iter, err := engine.Stream(ctx, &slm.Request{
    Messages: []slm.Message{
        slm.NewTextMessage(slm.RoleUser, "Tell me a story"),
    },
})
if err != nil {
    panic(err)
}
defer iter.Close()

for iter.Next() {
    fmt.Print(iter.Text())
}
```

### 类型安全调用

```go
type Sentiment struct {
    Text      string   `json:"text"`
    Sentiment string   `json:"sentiment"`
    Score     float64  `json:"score"`
    Keywords  []string `json:"keywords"`
}

result, err := slm.Call[Sentiment](ctx, engine, &slm.Request{
    Messages: []slm.Message{
        slm.NewTextMessage(slm.RoleUser, `Analyze: "This product is amazing!"`),
    },
})
// result 是 *Sentiment 类型，可直接使用
fmt.Printf("Sentiment: %s (%.1f)\n", result.Sentiment, result.Score)
```

## Transport 架构

SLM 采用**协议-传输分离**设计：`OpenAIEngine` 只负责 OpenAI 协议的编解码，HTTP 通信和认证由 `Transport` 接口实现。这使得同一个协议引擎可以搭配不同的传输方式。

```
┌─────────────────────────────────┐
│  slm 包 (零外部依赖)            │
│                                 │
│  Transport 接口                 │
│  ├─ HTTPTransport (标准HTTP)    │
│  OpenAIEngine (只管协议编解码)   │
│  Config.WithTransport()         │
└──────────┬──────────────────────┘
           │ 依赖倒置：slm 不知道 copilot 的存在
           ▼
┌─────────────────────────────────┐
│  copilot 包 (依赖 slm)          │
│                                 │
│  NewTransport(*Client)          │
│  └─ 返回 slm.Transport 接口     │
│     自动注入 OAuth + Intent     │
└─────────────────────────────────┘
```

### Transport 接口

```go
type Transport interface {
    Do(ctx context.Context, method, path string, headers map[string]string, body any) (*http.Response, error)
}
```

### 四种使用方式

#### 方式一：HTTP 直连（向后兼容）

最简单的用法，一行代码创建引擎，内部自动创建 `HTTPTransport`。

```go
engine := slm.NewOpenAIProtocol(
    "https://api.openai.com/v1",
    "your-api-key",
    "gpt-4o-mini",
)
```

#### 方式二：自定义 HTTP Transport

需要自定义 HTTP 客户端（如代理、连接池、超时）时使用。

```go
client := &http.Client{
    Timeout: 60 * time.Second,
    Transport: &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
    },
}

transport := slm.NewHTTPTransportWithClient(client, "https://api.openai.com/v1", "your-api-key")
engine := slm.NewOpenAIWithTransport(transport, "gpt-4o-mini")
```

#### 方式三：Copilot 传输（依赖倒置）

通过 GitHub Copilot 的 OAuth 认证访问 API，无需手动管理 Bearer token。`copilot` 包提供 `slm.Transport` 适配器，`slm` 包完全不知道 `copilot` 的存在。

```go
import (
    "ojv/sto/copilot"
    "ojv/sto/slm"
)

client := copilot.NewClient()
if err := client.Auth(ctx); err != nil {
    panic(err)
}

transport := copilot.NewTransport(client)
engine := slm.NewOpenAIWithTransport(transport, "gpt-4o")
```

#### 方式四：Config 构建

通过配置构建器统一管理 Transport、Provider 和中间件。

```go
// HTTP 直连配置
cfg := slm.DefaultConfig().
    WithProvider(slm.ProviderConfig{
        Endpoint:     "https://api.openai.com/v1",
        APIKey:       "your-api-key",
        DefaultModel: "gpt-4o-mini",
    }).
    WithRetry(slm.RetryConfig{MaxAttempts: 3})
engine, _ := cfg.BuildEngineWithTransport()

// Copilot 传输配置
client := copilot.NewClient()
client.Auth(ctx)

cfg := slm.DefaultConfig().
    WithTransport(copilot.NewTransport(client)).
    WithProvider(slm.ProviderConfig{DefaultModel: "gpt-4o"})
engine, _ := cfg.BuildEngineWithTransport()
```

## 核心概念

### Engine 接口

```go
type Engine interface {
    Generate(ctx context.Context, req *Request) (*Response, error)
    Stream(ctx context.Context, req *Request) (StreamIterator, error)
}
```

流式迭代器默认只需要实现 `StreamIterator`。如果底层实现支持被中间件主动打断，还可以额外实现可选增强接口：

```go
type InterruptibleStreamIterator interface {
    StreamIterator
    Interrupt(error)
}
```

像超时中间件这类包装器会优先调用 `Interrupt(error)`，以便底层尽快打断阻塞中的 `Next()`。

### Request 结构

```go
type Request struct {
    Model            string        // 模型名称
    Messages         []Message     // 消息列表
    Temperature      float64       // 温度参数
    TopP             float64       // Top-p 采样
    MaxTokens        int           // 最大输出 token
    Stop             []string      // 停止词
    Stream           bool          // 流式模式
    JSONMode         bool          // JSON 输出模式
    Tools            []Tool        // 工具定义
    ExtraBody        map[string]any // 额外请求参数
}
```

### Message 类型

```go
// 文本消息
slm.NewTextMessage(slm.RoleUser, "Hello")

// 系统消息
slm.NewTextMessage(slm.RoleSystem, "You are a helpful assistant")

// 图片消息
slm.NewImageMessage("https://example.com/image.png")

// 多模态消息
msg := slm.Message{
    Role: slm.RoleUser,
    Content: []slm.ContentPart{
        slm.TextPart("What's in this image?"),
        slm.ImagePart{URL: "https://example.com/image.png"},
    },
}

// 工具响应消息
slm.NewToolMessage("call_id", `{"result": "success"}`)
```

## 中间件系统

### 重试中间件

```go
retryUnary, retryStream := slm.RetryMiddlewareWithConfig(slm.RetryConfig{
    MaxAttempts: 3,
    Backoff:     slm.ExponentialBackoff,
    IsRetryable: slm.IsRetryableError,
})

// 对流式请求，重试只发生在首个有效 chunk 返回之前。
// 一旦已经向调用方交付过任何 chunk，后续断流会直接返回错误，
// 不会自动续传，以避免重复输出或上下文错乱。
// 如需禁用重试，可显式设置 MaxAttempts: 1。
```

### 限流中间件

```go
rateLimit, closer := slm.RateLimitMiddleware(10, 5)  // 10 QPS, burst 5
defer closer()

// 如果希望 unary 和 stream 共用一个 limiter，使用共享版本
rateLimitUnary, rateLimitStream, closer := slm.RateLimitMiddlewares(10, 5)
defer closer()
```

### 日志中间件

```go
logger := slm.NewDefaultLogger(slog.New(slog.NewTextHandler(os.Stdout, nil)))
loggingMW := slm.LoggingMiddleware(logger)
```

### 超时中间件

```go
timeoutMW := slm.TimeoutMiddleware(30 * time.Second)
```

### 请求追踪中间件

```go
requestIDMW := slm.RequestIDMiddleware(nil)  // 使用默认 ID 生成器
```

### 组合中间件

```go
engine := slm.ChainWithStream(baseEngine,
    []slm.Middleware{rateLimit, retry, logging},
    []slm.StreamMiddleware{rateLimitStream, retryStream},
)
```

### 标准中间件链

如果没有非常特殊的排序需求，优先使用标准链构建器：

```go
engine := slm.ApplyStandardMiddleware(baseEngine, slm.StandardMiddlewareOptions{
    EnableRequestID: true,
    Capabilities: &slm.CapabilityNegotiationOptions{
        Resolver: slm.StaticCapabilityResolver{
            "gpt-4o-mini": {Supports: slm.CapabilitySet{JSONMode: true, ToolCalls: true, Vision: true, Reasoning: true}},
        },
        DefaultModel: "gpt-4o-mini",
        RequireKnown: true,
    },
    Logger:          logger,
    Timeout:         30 * time.Second,
    Retry:           &slm.RetryConfig{MaxAttempts: 3},
    RateLimit:       &slm.RateLimitConfig{Limit: 10, Burst: 5},
})
```

标准链固定顺序为：

`request_id -> capability negotiation -> lifecycle observers -> logging -> timeout -> retry -> rate limit -> engine`

### 生命周期观察器

如果要接入指标、追踪、审计，不建议再复制一层日志式中间件。优先使用统一生命周期观察器：

```go
type Observer struct{}

func (Observer) OnRequestStart(ctx context.Context, event slm.LifecycleEvent)   {}
func (Observer) OnRequestFinish(ctx context.Context, event slm.LifecycleEvent)  {}
func (Observer) OnStreamStart(ctx context.Context, event slm.LifecycleEvent)    {}
func (Observer) OnStreamConnected(ctx context.Context, event slm.LifecycleEvent) {}
func (Observer) OnStreamFinish(ctx context.Context, event slm.LifecycleEvent)   {}

engine := slm.ApplyStandardMiddleware(baseEngine, slm.StandardMiddlewareOptions{
    EnableRequestID: true,
    Observers:       []slm.LifecycleObserver{Observer{}},
})
```

`LifecycleEvent` 会统一提供 `request_id`、`model`、`request`、`response`、`duration`、`error` 和 `stream` 标记。

如果当前请求启用了能力协商，还可以通过 `slm.GetNegotiatedCapabilities(event.Context)` 读取本次请求的显式能力协商结果，包括：最终模型、请求能力、支持能力、`Known`，以及 `State` / `Stale` 这类解析元数据。当前内置状态里已经包含低基数的 `Source`（如 `static`、`catalog`）、可选的 `RefreshedAt` 快照时间，以及目录刷新失败但允许继续服务时的 `Stale` 标记。

如果需要官方 metrics / trace 适配器，而不是自己手写 observer，可以直接使用：

```go
meter := NewYourMeter()
tracer := NewYourTracer()

engine := slm.ApplyStandardMiddleware(baseEngine, slm.StandardMiddlewareOptions{
    Observers: []slm.LifecycleObserver{
        slm.NewMetricsObserver(meter, slm.MetricsObserverOptions{Namespace: "slm"}),
        slm.NewTraceObserver(tracer, slm.TraceObserverOptions{SpanPrefix: "slm"}),
    },
})
```

这两个适配器只依赖 `slm` 自己定义的最小 `Meter` / `Tracer` 接口，不直接耦合 OpenTelemetry、Prometheus 或其他具体后端。

当同时启用了 `CapabilityNegotiationMiddleware` 时，官方适配器会自动补充能力协商属性，包括 `capabilities.source`、`capabilities.known`、`capabilities.stale`，以及请求侧的 `cap.json_mode` / `cap.tool_calls` / `cap.vision` / `cap.reasoning`。像 `RefreshedAt` 这类高基数时间元数据会保留在 `NegotiatedCapabilities.State` 中，避免直接变成 metrics label。

## 显式能力协商

`slm` 现在支持将 `json_mode`、`tool_calls`、`vision`、`reasoning` 从“隐式约定”提升为显式模型能力：

```go
resolver := slm.StaticCapabilityResolver{
    "gpt-4o-mini": {
        Supports: slm.CapabilitySet{
            JSONMode:  true,
            ToolCalls: true,
            Vision:    true,
            Reasoning: true,
        },
    },
    "gpt-3.5": {
        Supports: slm.CapabilitySet{
            JSONMode: true,
        },
    },
}

cfg := slm.DefaultConfig().
    WithProvider(slm.ProviderConfig{DefaultModel: "gpt-4o-mini"}).
    WithCapabilityNegotiation(slm.CapabilityNegotiationOptions{
        Resolver:     resolver,
        RequireKnown: true,
    })
```

请求能力会自动从请求结构中识别：

- `Request.JSONMode` -> `json_mode`
- `Request.Tools` -> `tool_calls`
- `ImagePart` -> `vision`
- `Request.Reasoning` 或兼容的 `ExtraBody["reasoning"]` / `ExtraBody["reasoning_effort"]` -> `reasoning`

如果模型不支持所请求的能力，会在真正发起 provider 调用之前返回 `ErrCodeUnsupportedCapability`。

如果能力信息来自运行时模型目录，可以直接使用 `slm.NewCatalogCapabilityResolver(...)` 构建通用 catalog resolver。它内置缓存、TTL 刷新、并发合并和可选的 stale-on-error 策略；当开启 `AllowStaleOnError` 且目录刷新失败时，本次协商结果会通过 `NegotiatedCapabilities.Stale`、`NegotiatedCapabilities.State.RefreshedAt` 和 observer attributes 显式暴露“当前能力来自陈旧目录、且快照形成于何时”的状态，而不是要求上层主动轮询 resolver 内部错误状态。

如果你在 `slm` 之外实现自定义 resolver，又希望把 freshness、来源或其他解析元数据一起带入协商上下文，可以额外实现可选接口 `slm.CapabilityResolverWithState`。标准协商中间件会优先读取它，并把结果写入 `NegotiatedCapabilities.State`。内置 `StaticCapabilityResolver` 和 `CatalogCapabilityResolver` 已分别使用 `static` / `catalog` 作为默认 `Source`。

如果能力信息来自 provider 模型目录，而不是手写静态表，建议在 `slm` 外层做桥接。仓库中已经提供了 Copilot 目录桥接包：

```go
import (
    "ojv/sto/copilot"
    "ojv/sto/copilot/slmbridge"
    "ojv/sto/slm"
)

models := []copilot.Model{
    {
        ID: "gpt-4.1",
        Capabilities: copilot.ModelCapabilities{
            Supports: copilot.ModelSupports{
                StructuredOutputs: true,
                ToolCalls:         true,
                Vision:            true,
                AdaptiveThinking:  true,
            },
        },
    },
}

resolver := slm.ChainCapabilityResolvers(
    slmbridge.StaticResolver(models),
    slm.StaticCapabilityResolver{
        "*": {Supports: slm.CapabilitySet{JSONMode: true}},
    },
)
```

这样 provider 专属模型目录仍留在桥接包中，`slm` 核心只消费通用 `CapabilityResolver` 抽象。

如果需要自定义目录加载器而不引入 provider 桥接，可以直接使用：

```go
resolver := slm.NewCatalogCapabilityResolver(loadCatalog, slm.CapabilityCatalogResolverOptions{
    CacheTTL:          5 * time.Minute,
    AllowStaleOnError: true,
})

engine := slm.ApplyStandardMiddleware(baseEngine, slm.StandardMiddlewareOptions{
    Capabilities: &slm.CapabilityNegotiationOptions{
        Resolver:     resolver,
        DefaultModel: "gpt-4o-mini",
        RequireKnown: true,
    },
    Observers: []slm.LifecycleObserver{
        slm.NewMetricsObserver(meter, slm.MetricsObserverOptions{Namespace: "slm"}),
    },
})
```

上层如果需要区分“能力未知”和“能力已知但目录已 stale”，优先读取 `GetNegotiatedCapabilities(ctx)` 的 `Known` / `Stale`，而不是直接依赖具体 resolver 实现。

## 配置构建器

```go
cfg := slm.DefaultConfig().
    WithProvider(slm.ProviderConfig{
        Endpoint:     "https://api.openai.com/v1",
        DefaultModel: "gpt-4o-mini",
    }).
    WithCapabilityNegotiation(slm.CapabilityNegotiationOptions{
        Resolver: slm.StaticCapabilityResolver{
            "gpt-4o-mini": {Supports: slm.CapabilitySet{JSONMode: true, ToolCalls: true, Vision: true, Reasoning: true}},
        },
        RequireKnown: true,
    }).
    WithRetry(slm.RetryConfig{
        MaxAttempts: 3,
    }).
    WithTimeout(30 * time.Second).
    WithRequestID(nil).
    WithObserver(Observer{}).
    WithRateLimit(10, 5)

engine, err := cfg.BuildEngine(func(p slm.ProviderConfig) (slm.Engine, error) {
    return slm.NewOpenAIProtocol(p.Endpoint, apiKey, p.DefaultModel), nil
})
```

## 工具调用

```go
tools := []slm.Tool{
    {
        Name:        "get_weather",
        Description: "Get weather for a location",
        Parameters: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "location": map[string]any{"type": "string"},
            },
            "required": []string{"location"},
        },
    },
}

resp, _ := engine.Generate(ctx, &slm.Request{
    Messages: messages,
    Tools:    tools,
})

for _, tc := range resp.ToolCalls {
    fmt.Printf("Tool: %s(%s)\n", tc.Name, tc.Arguments)
    // 执行工具并返回结果
    messages = append(messages, slm.NewToolMessage(tc.ID, toolResult))
}
```

## 错误处理

```go
var llmErr *slm.LLMError
if errors.As(err, &llmErr) {
    switch llmErr.Code {
    case slm.ErrCodeRateLimit:
        // 速率限制
    case slm.ErrCodeAuth:
        // 认证失败
    case slm.ErrCodeContextTooLong:
        // 上下文过长
    }
    
    if llmErr.Code.IsRetryable() {
        // 可重试错误
    }
}
```

### 错误码

| 代码 | 常量 | 说明 | 可重试 |
|------|------|------|--------|
| 1001 | `ErrCodeRateLimit` | 速率限制 | ✓ |
| 1002 | `ErrCodeTimeout` | 请求超时 | ✓ |
| 1003 | `ErrCodeOverloaded` | 服务过载 | ✓ |
| 1004 | `ErrCodeNetwork` | 网络错误 | ✓ |
| 2001 | `ErrCodeAuth` | 认证失败 | ✗ |
| 2002 | `ErrCodeInvalidModel` | 无效模型 | ✗ |
| 2003 | `ErrCodeInvalidConfig` | 配置错误 | ✗ |
| 2004 | `ErrCodeUnsupportedCapability` | 模型能力不支持请求特性 | ✗ |
| 3001 | `ErrCodeContentFilter` | 内容过滤 | ✗ |
| 3002 | `ErrCodeContextTooLong` | 上下文过长 | ✗ |

## 自定义 HTTP 客户端

```go
client := &http.Client{
    Timeout: 60 * time.Second,
    Transport: &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
    },
}

transport := slm.NewHTTPTransportWithClient(client, baseURL, apiKey)
engine := slm.NewOpenAIWithTransport(transport, model)
```

## 示例程序

项目包含丰富的示例代码：

```bash
# Transport 架构示例
go run ./cmd/transport-demo

# 基础示例
go run ./cmd/slm -example basic
go run ./cmd/slm -example streaming
go run ./cmd/slm -example json_mode

# 高级示例
go run ./cmd/slm -example tool_calling
go run ./cmd/slm -example generic
go run ./cmd/slm -example reasoning

# 生产级示例
go run ./cmd/slm -example middleware
go run ./cmd/slm -example error_handling
go run ./cmd/slm -example config

# 运行所有示例
go run ./cmd/slm -example all
```

## API 兼容性

支持所有 OpenAI 协议兼容的服务：

- OpenAI API
- Azure OpenAI
- GitHub Models (通过 Copilot Transport)
- Ollama
- vLLM
- LM Studio
- 其他兼容服务

## 项目结构

```
slm/
├── call.go              # 泛型调用函数
├── config.go            # 配置构建器
├── driver_openai.go     # OpenAI 协议实现
├── engine.go            # 核心接口、Transport 接口定义
├── errors.go            # 错误码和错误处理
├── log.go               # 日志接口
├── middleware.go         # 中间件系统
├── retry.go             # 重试机制
├── sse_reader.go        # SSE 流式读取器
├── transport_http.go    # HTTPTransport 实现
├── transport_test.go    # Transport 测试
└── cmd/
    ├── slm/
    │   └── main.go      # 示例程序
    └── transport-demo/
        └── main.go      # Transport 架构示例
```

## 设计原则

1. **简洁优先** - 最小化 API 表面积，降低学习成本
2. **协议-传输分离** - `Transport` 接口解耦通信层，协议引擎可搭配不同传输方式
3. **依赖倒置** - `slm` 定义接口，`copilot` 提供实现，避免反向依赖
4. **类型安全** - 利用 Go 泛型提供编译时类型检查
5. **可组合** - 中间件系统支持灵活的功能组合
6. **可观测** - 内置日志、追踪支持
7. **生产就绪** - 内置重试、限流、超时控制

## 许可证

MIT License
