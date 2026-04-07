# SLM - Simple LLM Client Library

一个简洁、类型安全的 Go 语言大语言模型客户端库，提供统一的 API 抽象层，支持 OpenAI 协议兼容的服务。

## 特性

- **统一抽象** - 单一 `Engine` 接口支持多种 LLM 提供商
- **类型安全** - 泛型调用 `Call[T]` 自动解析结构化输出
- **流式响应** - 完整的 SSE 流式迭代器支持
- **中间件系统** - 内置重试、限流、日志、超时、请求追踪
- **工具调用** - 完整的 Function Calling 支持
- **多模态** - 支持文本和图片混合输入
- **推理模型** - 支持 o1/o3 等推理模型的思维链提取
- **零依赖** - 仅使用 Go 标准库

## 安装

```bash
go get slm
```

## 快速开始

### 基础对话

```go
package main

import (
    "context"
    "fmt"
    "slm"
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

## 核心概念

### Engine 接口

```go
type Engine interface {
    Generate(ctx context.Context, req *Request) (*Response, error)
    Stream(ctx context.Context, req *Request) (StreamIterator, error)
}
```

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
```

### 限流中间件

```go
rateLimit, closer := slm.RateLimitMiddleware(10, 5)  // 10 QPS, burst 5
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

## 配置构建器

```go
cfg := slm.DefaultConfig().
    WithProvider(slm.ProviderConfig{
        Endpoint:     "https://api.openai.com/v1",
        DefaultModel: "gpt-4o-mini",
    }).
    WithRetry(slm.RetryConfig{
        MaxAttempts: 3,
    })

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
| 3001 | `ErrCodeContentFilter` | 内容过滤 | ✗ |
| 3002 | `ErrCodeContextTooLong` | 上下文过长 | ✗ |

## 自定义 HTTP 客户端

```go
baseEngine := slm.NewOpenAIProtocol(baseURL, apiKey, model).(*slm.OpenAIEngine)
baseEngine.Client = &http.Client{
    Timeout: 60 * time.Second,
    Transport: &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
    },
}
```

## 示例程序

项目包含丰富的示例代码：

```bash
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
- GitHub Models
- Ollama
- vLLM
- LM Studio
- 其他兼容服务

## 项目结构

```
slm/
├── call.go            # 泛型调用函数
├── config.go          # 配置构建器
├── driver_openai.go   # OpenAI 协议实现
├── engine.go          # 核心接口和类型定义
├── errors.go          # 错误码和错误处理
├── log.go             # 日志接口
├── middleware.go      # 中间件系统
├── retry.go           # 重试机制
├── sse_reader.go      # SSE 流式读取器
├── go.mod
└── cmd/
    └── slm/
        └── main.go    # 示例程序
```

## 设计原则

1. **简洁优先** - 最小化 API 表面积，降低学习成本
2. **类型安全** - 利用 Go 泛型提供编译时类型检查
3. **可组合** - 中间件系统支持灵活的功能组合
4. **可观测** - 内置日志、追踪支持
5. **生产就绪** - 内置重试、限流、超时控制

## 许可证

MIT License
