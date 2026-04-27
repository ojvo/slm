# SLM - Simple LLM Client Library

一个面向生产的、极简且类型安全的 Go LLM 客户端。

SLM 聚焦三件事：

- 协议编解码（OpenAI 兼容 Chat + Responses）
- 传输抽象（HTTP 或自定义 Transport）
- 中间件组合（能力协商、观察器、超时、重试、限流、请求 ID）

无第三方依赖，仅标准库。

## 安装

```bash
go get ojv/slm
```

## 快速开始

### 1) Chat: 基础调用

```go
engine := slm.NewOpenAIProtocol(
    "https://api.openai.com/v1",
    "your-api-key",
    "gpt-4o-mini",
)

resp, err := engine.Generate(ctx, &slm.Request{
    Messages: []slm.Message{
        slm.NewTextMessage(slm.RoleUser, "Hello"),
    },
})
if err != nil {
    panic(err)
}
fmt.Println(resp.Content)
```

### 2) Chat: 流式输出

```go
iter, err := engine.Stream(ctx, &slm.Request{
    Messages: []slm.Message{
        slm.NewTextMessage(slm.RoleUser, "Tell me a short story"),
    },
})
if err != nil {
    panic(err)
}
defer iter.Close()

for iter.Next() {
    fmt.Print(iter.Text())
}
if err := iter.Err(); err != nil {
    panic(err)
}
```

### 3) Responses: 一次性创建

```go
re := slm.NewOpenAIResponsesProtocol(
    "https://api.openai.com/v1",
    "your-api-key",
    "gpt-4o-mini",
)

resp, err := re.Create(ctx, &slm.ResponseRequest{
    Input: []slm.ResponseInputItem{
        slm.NewTextResponseInputItem("user", "Summarize Go interfaces in one sentence."),
    },
    Reasoning: &slm.ResponseReasoning{Effort: "medium", Summary: "auto"},
})
if err != nil {
    panic(err)
}

for _, item := range resp.Output {
    for _, part := range item.Content {
        fmt.Println(part.Text)
    }
}
```

### 4) 泛型结构化输出

```go
type Sentiment struct {
    Text      string   `json:"text"`
    Sentiment string   `json:"sentiment"`
    Score     float64  `json:"score"`
    Keywords  []string `json:"keywords"`
}

result, err := slm.Call[Sentiment](ctx, engine, &slm.Request{
    JSONMode: true,
    Messages: []slm.Message{
        slm.NewTextMessage(slm.RoleUser, `Analyze: "This product is amazing!"`),
    },
})
if err != nil {
    panic(err)
}
fmt.Printf("%s %.1f\n", result.Sentiment, result.Score)
```

## 架构概览

SLM 使用协议与传输分离：

- 协议层：`OpenAIEngine` / `OpenAIResponsesEngine` 只做请求体构建、响应解析、SSE 事件处理
- 传输层：`Transport` 只负责 HTTP 通信与认证
- 组合层：中间件统一处理横切关注点

```go
type Transport interface {
    Do(ctx context.Context, method, path string, headers map[string]string, body []byte) (*http.Response, error)
}
```

## Chat 与 Responses 对照

| 维度 | Chat | Responses |
|---|---|---|
| 引擎 | `Engine` | `OpenAIResponsesEngine` |
| Unary | `Generate(ctx, *Request)` | `Create(ctx, *ResponseRequest)` |
| Stream | `Stream(ctx, *Request)` | `Stream(ctx, *ResponseRequest)` |
| 工具类型 | `[]Tool` | `[]ResponseTool` |
| 推理类型 | `*ReasoningOptions` | `*ResponseReasoning`（别名） |

补充说明：

- `ResponseReasoning` 是 `ReasoningOptions` 的类型别名，字段保持一致（`Effort`/`Summary`）。
- `ResponseTool` 采用 `/responses` 的扁平函数字段格式。

## Responses 进阶

### 1) 结构化工具

```go
tool := slm.NewResponseFunctionTool(slm.Tool{
    Name:        "classify_topic",
    Description: "Classify text topic",
    Parameters: map[string]any{
        "type": "object",
        "properties": map[string]any{
            "text": map[string]any{"type": "string"},
        },
        "required": []string{"text"},
    },
})

req := &slm.ResponseRequest{
    Input: []slm.ResponseInputItem{slm.NewTextResponseInputItem("user", "Analyze this text")},
    Tools: []slm.ResponseTool{tool},
}
```

### 2) 多段输入（typed parts）

```go
req := &slm.ResponseRequest{
    Input: []slm.ResponseInputItem{
        slm.NewMultiPartResponseInputItem("user", []slm.ResponseInputContentPart{
            slm.ResponseInputTextPart{Type: "input_text", Text: "Summarize:"},
            slm.ResponseInputTextPart{Type: "input_text", Text: "Go interfaces decouple behavior."},
        }),
    },
}
```

### 3) 流式事件 helper

```go
stream, err := re.Stream(ctx, &slm.ResponseRequest{
    Input:  []slm.ResponseInputItem{slm.NewTextResponseInputItem("user", "List 3 tips")},
    Stream: true,
})
if err != nil {
    panic(err)
}
defer stream.Close()

for stream.Next() {
    ev := stream.Current()
    if ev.IsOutputTextDelta() {
        fmt.Print(ev.Delta)
    }
    if done := ev.CompletedResponse(); done != nil {
        fmt.Printf("\nstatus=%s outputs=%d\n", done.Status, len(done.Output))
    }
}
if err := stream.Err(); err != nil {
    panic(err)
}
```

`CompletedResponse()` 会返回规范化后的对象：若 `status == "completed"` 且 `output == nil`，将标准化为空切片，简化调用方分支判断。

## 中间件系统

推荐通过标准入口应用：

```go
wrapped := slm.ApplyStandardMiddleware(baseEngine, slm.StandardMiddlewareOptions{
    DefaultModel:       "gpt-4o-mini",
    EnableRequestID:    true,
    RequestIDGenerator: nil,
    Capabilities: &slm.CapabilityNegotiationOptions{
        Resolver:     resolver,
        DefaultModel: "gpt-4o-mini",
        RequireKnown: true,
    },
    Observers: []slm.LifecycleObserver{observer},
    CrossCutting: slm.CrossCuttingMiddlewareOptions{
        Timeout:   30 * time.Second,
        Retry:     &slm.RetryConfig{MaxAttempts: 3},
        RateLimit: &slm.RateLimitConfig{Limit: 10, Burst: 5},
    },
})
```

顺序固定为：

`request_id -> normalize -> capability negotiation -> observers -> timeout -> retry -> rate limit -> engine`

Responses 版本：

```go
responses := slm.ApplyStandardResponseMiddleware(baseResponsesEngine, slm.StandardMiddlewareOptions{
    DefaultModel: "gpt-4o-mini",
    CrossCutting: slm.CrossCuttingMiddlewareOptions{
        Timeout:   30 * time.Second,
        RateLimit: &slm.RateLimitConfig{Limit: 10, Burst: 5},
    },
})
defer responses.Close()
```

## 能力协商

将 `json_mode`、`tool_calls`、`vision`、`reasoning` 显式化并在请求前校验：

```go
resolver := slm.ChainCapabilityResolvers(
    slm.StaticCapabilityResolver{
        "gpt-4o-mini": {
            Supports: slm.CapabilitySet{
                JSONMode:  true,
                ToolCalls: true,
                Vision:    true,
                Reasoning: true,
            },
        },
    },
    slm.StaticCapabilityResolver{
        "*": {Supports: slm.CapabilitySet{JSONMode: true}},
    },
)
```

若能力不满足，会在真正调用上游前返回 `ErrCodeUnsupportedCapability`。

也支持目录驱动的解析器：

```go
resolver := slm.NewCatalogCapabilityResolver(loadCatalog, slm.CapabilityCatalogResolverOptions{
    CacheTTL:          5 * time.Minute,
    AllowStaleOnError: true,
})
```

## 可观测性与诊断

生命周期观察器统一覆盖 unary + stream：

```go
type LifecycleObserver interface {
    OnRequestStart(context.Context, LifecycleEvent)
    OnRequestFinish(context.Context, LifecycleEvent)
    OnStreamStart(context.Context, LifecycleEvent)
    OnStreamConnected(context.Context, LifecycleEvent)
    OnStreamFinish(context.Context, LifecycleEvent)
}
```

可直接使用：

- `NewLogObserver(logger)`
- `NewMetricsObserver(meter, opts)`
- `NewTraceObserver(tracer, opts)`

诊断辅助：

- `RequestDiagnosticFields(*Request)`
- `ResponseRequestDiagnosticFields(*ResponseRequest)`

后者用于 Responses 请求的结构化日志字段输出（input_items/tools/reasoning/max_output_tokens/store 等）。

## 配置构建器

```go
cfg := slm.DefaultConfig().
    WithProvider(slm.ProviderConfig{
        Endpoint:     "https://api.openai.com/v1",
        APIKey:       apiKey,
        DefaultModel: "gpt-4o-mini",
    }).
    WithCapabilityNegotiation(slm.CapabilityNegotiationOptions{
        Resolver:     resolver,
        RequireKnown: true,
    }).
    WithRetry(slm.RetryConfig{MaxAttempts: 3}).
    WithTimeout(30 * time.Second).
    WithRequestID(nil).
    WithRateLimit(10, 5)

engine, err := cfg.BuildEngineWithTransport()
if err != nil {
    panic(err)
}

responsesEngine, err := cfg.BuildResponsesEngine()
if err != nil {
    panic(err)
}
defer responsesEngine.Close()
```

## 错误处理

```go
var llmErr *slm.LLMError
if errors.As(err, &llmErr) {
    if llmErr.Code.IsRetryable() {
        // retryable: rate_limit/timeout/network/server/overloaded
    }
}
```

常见错误码：

- `ErrCodeRateLimit`
- `ErrCodeTimeout`
- `ErrCodeOverloaded`
- `ErrCodeNetwork`
- `ErrCodeServer`
- `ErrCodeAuth`
- `ErrCodeInvalidModel`
- `ErrCodeInvalidConfig`
- `ErrCodeUnsupportedCapability`
- `ErrCodeContentFilter`
- `ErrCodeContextTooLong`
- `ErrCodeParse`
- `ErrCodeCancelled`
- `ErrCodeInternal`

## 示例程序

运行：

```bash
go run ./cmd/slm -example basic
```

完整样例位于 `cmd/slm/main.go`，包含 7 大类、25 个场景（基础、进阶、生产、模式、专项、增强接口、基础设施），其中 Responses 示例已覆盖：

- unary create
- typed multipart input
- structured tool definition
- stream + `ResponseEvent` helpers

## 项目结构

```
slm/
├── adapter.go
├── call.go
├── capabilities.go
├── capability_catalog_resolver.go
├── codec.go
├── config.go
├── diagnostics.go
├── driver_openai.go
├── errors.go
├── helper.go
├── log.go
├── middleware.go
├── model.go
├── observability.go
├── observability_adapters.go
├── retry.go
├── sse_frame.go
├── sse_reader.go
├── transport_http.go
├── types.go
└── cmd/slm/main.go
```

## 设计原则

1. 极简架构：协议层只负责编解码，横切能力统一走中间件
2. 类型安全：泛型与显式类型减少运行时 JSON 出错面
3. 可组合：能力协商、观测、限流、重试、超时可按需叠加
4. 可观测：统一生命周期事件覆盖 unary/stream/chat/responses
5. 生产可用：内置错误分类、重试判断、超时与请求追踪
6. 零依赖：仅 Go 标准库

## 许可证

MIT License
