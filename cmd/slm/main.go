package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"ojv/sto/copilot"
	"ojv/sto/copilot/slmbridge"
	"ojv/sto/slm"
)

var defaultAPIKey = ""

func getAPIKey() string {
	if key := os.Getenv("SLM_API_KEY"); key != "" {
		return key
	}
	return defaultAPIKey
}

func main() {
	example := flag.String("example", "basic", "Available examples:\n\n"+
		"  01-BASIC (入门)\n"+
		"    basic          Simple chat with token usage\n"+
		"    parameters     Temperature, max_tokens, top_p tuning\n"+
		"    json_mode      Force JSON output format\n\n"+
		"  02-ADVANCED (高级)\n"+
		"    streaming      Real-time stream output\n"+
		"    responses      OpenAI responses API\n"+
		"    tool_calling   Multi-tool definition and execution\n"+
		"    generic        Type-safe Call[T] generics\n"+
		"    reasoning      o1/o3 reasoning content extraction\n\n"+
		"  03-PRODUCTION (生产级)\n"+
		"    middleware     Retry + RateLimit chain\n"+
		"    error_handling Error codes and recovery\n"+
		"    config         Builder pattern configuration\n"+
		"    custom_http    Custom HTTP client settings\n"+
		"    capabilities   Explicit model capability negotiation\n"+
		"    observability  Official metrics/trace observer adapters\n\n"+
		"  04-PATTERNS (设计模式)\n"+
		"    conversation  Multi-turn context management\n"+
		"    batch          Concurrent requests with isolation\n"+
		"    cancel         Context timeout and cancellation\n"+
		"    accumulator    Stream accumulation with buffering\n\n"+
		"  05-SPECIALIZED (专项)\n"+
		"    multimodal     Vision: image + text\n"+
		"    few_shot       Few-shot learning templates\n\n"+
		"  06-ENHANCED (增强接口)\n"+
		"    simple_call    SimpleCall/Chat convenience wrappers\n"+
		"    full_text      FullText() for reasoning models\n\n"+
		"  07-INFRASTRUCTURE (基础设施)\n"+
		"    logging        Logger interface + LoggingMiddleware\n"+
		"    timeout        TimeoutMiddleware for request control\n"+
		"    request_id     RequestIDMiddleware for tracing\n\n"+
		"  all             Run all examples sequentially\n")

	flag.Parse()

	engine := slm.NewOpenAIProtocol(
		"https://models.inference.ai.azure.com",
		getAPIKey(),
		"gpt-4o-mini",
	)

	ctx := context.Background()

	switch *example {
	case "all":
		runAll(ctx, engine)
	case "basic":
		runBasic(ctx, engine)
	case "parameters":
		runParameters(ctx, engine)
	case "json_mode":
		runJSONMode(ctx, engine)

	case "streaming":
		runStreaming(ctx, engine)
	case "responses":
		runResponses(ctx)
	case "tool_calling":
		runToolCalling(ctx, engine)
	case "generic":
		runGeneric(ctx, engine)
	case "reasoning":
		runReasoning(ctx, engine)

	case "middleware":
		runMiddleware(engine)
	case "error_handling":
		runErrorHandling(ctx, engine)
	case "config":
		runConfig()
	case "custom_http":
		runCustomHTTP()
	case "capabilities":
		runCapabilities(ctx, engine)
	case "observability":
		runObservability(ctx, engine)

	case "conversation":
		runConversation(ctx, engine)
	case "batch":
		runBatch(ctx, engine)
	case "cancel":
		runCancel(ctx, engine)
	case "accumulator":
		runAccumulator(ctx, engine)

	case "multimodal":
		runMultimodal(ctx, engine)
	case "few_shot":
		runFewShot(ctx, engine)

	case "simple_call":
		runSimpleCall(ctx, engine)
	case "full_text":
		runFullText(ctx, engine)

	case "logging":
		runLogging(ctx, engine)
	case "timeout":
		runTimeout(ctx, engine)
	case "request_id":
		runRequestID(ctx, engine)

	default:
		fmt.Printf("Unknown example: %s\n", *example)
		os.Exit(1)
	}
}

func runAll(ctx context.Context, engine slm.Engine) {
	examples := []struct {
		name string
		fn   func()
	}{
		{"basic", func() { runBasic(ctx, engine) }},
		{"parameters", func() { runParameters(ctx, engine) }},
		{"json_mode", func() { runJSONMode(ctx, engine) }},
		{"streaming", func() { runStreaming(ctx, engine) }},
		{"responses", func() { runResponses(ctx) }},
		{"tool_calling", func() { runToolCalling(ctx, engine) }},
		{"generic", func() { runGeneric(ctx, engine) }},
		{"reasoning", func() { runReasoning(ctx, engine) }},
		{"middleware", func() { runMiddleware(engine) }},
		{"error_handling", func() { runErrorHandling(ctx, engine) }},
		{"config", func() { runConfig() }},
		{"custom_http", func() { runCustomHTTP() }},
		{"capabilities", func() { runCapabilities(ctx, engine) }},
		{"observability", func() { runObservability(ctx, engine) }},
		{"conversation", func() { runConversation(ctx, engine) }},
		{"batch", func() { runBatch(ctx, engine) }},
		{"cancel", func() { runCancel(ctx, engine) }},
		{"accumulator", func() { runAccumulator(ctx, engine) }},
		{"multimodal", func() { runMultimodal(ctx, engine) }},
		{"few_shot", func() { runFewShot(ctx, engine) }},
		{"simple_call", func() { runSimpleCall(ctx, engine) }},
		{"full_text", func() { runFullText(ctx, engine) }},
		{"logging", func() { runLogging(ctx, engine) }},
		{"timeout", func() { runTimeout(ctx, engine) }},
		{"request_id", func() { runRequestID(ctx, engine) }},
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Printf("  RUNNING ALL EXAMPLES (%d total)\n", len(examples))
	fmt.Println(strings.Repeat("=", 60))

	start := time.Now()
	for i, ex := range examples {
		fmt.Printf("\n[%d/%d] Running: %s\n", i+1, len(examples), ex.name)
		ex.fn()
		//time.Sleep(500 * time.Millisecond)
		time.Sleep(7 * time.Second)
	}

	fmt.Printf("\n" + strings.Repeat("=", 60))
	fmt.Printf("  ALL DONE in %v\n", time.Since(start).Round(time.Millisecond))
	fmt.Println(strings.Repeat("=", 60))
}

// ============================================================
// 01-BASIC: 基础入门示例
// ============================================================

func runBasic(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  BASIC: Simple Chat")
	fmt.Println("========================================")

	req := &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleSystem, "You are a helpful assistant. Reply concisely."),
			slm.NewTextMessage(slm.RoleUser, "What is 2+2?"),
		},
	}

	resp, err := engine.Generate(ctx, req)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Response:\n  %s\n\n", resp.Content)
	fmt.Printf("Metadata:\n")
	fmt.Printf("  FinishReason: %s\n", resp.FinishReason)
	fmt.Printf("  Tokens: prompt=%d completion=%d total=%d\n",
		resp.Usage.PromptTokens,
		resp.Usage.CompletionTokens,
		resp.Usage.TotalTokens,
	)
}

func runParameters(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  PARAMETERS: Temperature & MaxTokens")
	fmt.Println("========================================")

	prompt := "Describe the color blue in one word."

	tests := []struct {
		name     string
		req      *slm.Request
		expected string
	}{
		{
			name: "Low temperature (creative)",
			req: &slm.Request{
				Temperature: slm.Float64(0.9),
				MaxTokens:   20,
				Messages: []slm.Message{
					slm.NewTextMessage(slm.RoleUser, prompt),
				},
			},
			expected: "more varied output",
		},
		{
			name: "Zero temperature (deterministic)",
			req: &slm.Request{
				Temperature: slm.Float64(0.0),
				MaxTokens:   20,
				Messages: []slm.Message{
					slm.NewTextMessage(slm.RoleUser, prompt),
				},
			},
			expected: "consistent output",
		},
		{
			name: "Short response (max_tokens=5)",
			req: &slm.Request{
				MaxTokens: 5,
				Messages: []slm.Message{
					slm.NewTextMessage(slm.RoleUser, prompt),
				},
			},
			expected: "truncated at 5 tokens",
		},
	}

	for _, test := range tests {
		test.req.Model = "gpt-4o-mini"
		fmt.Printf("--- %s ---\n", test.name)
		start := time.Now()
		resp, err := engine.Generate(ctx, test.req)
		if err != nil {
			fmt.Printf("  Error: %v\n\n", err)
			continue
		}
		fmt.Printf("  Response: %q\n", resp.Content)
		fmt.Printf("  Tokens: %d | Time: %v | Expected: %s\n\n",
			resp.Usage.TotalTokens, time.Since(start), test.expected)
	}
}

func runJSONMode(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  JSON MODE: Structured Output")
	fmt.Println("========================================")

	type Sentiment struct {
		Text      string   `json:"text"`
		Sentiment string   `json:"sentiment"`
		Score     float64  `json:"score"`
		Keywords  []string `json:"keywords"`
	}

	result, err := slm.Call[Sentiment](ctx, engine, &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleSystem, `Return JSON only. No markdown or explanation.`),
			slm.NewTextMessage(slm.RoleUser, `Analyze sentiment of: "This product is amazing! Best purchase I've made this year."`),
		},
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	data, _ := json.MarshalIndent(result, "", "  ")
	fmt.Printf("Parsed Result:\n%s\n", string(data))
	fmt.Printf("\nType-safe access:\n")
	fmt.Printf("  Text: %s\n", result.Text)
	fmt.Printf("  Sentiment: %s (%.1f/10)\n", result.Sentiment, result.Score)
	fmt.Printf("  Keywords: %v\n", result.Keywords)
}

// ============================================================
// 02-ADVANCED: 高级功能示例
// ============================================================

func runStreaming(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  STREAMING: Real-time Output")
	fmt.Println("========================================")

	req := &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleUser, "Count from 1 to 10, each on new line."),
		},
	}

	iter, err := engine.Stream(ctx, req)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer iter.Close()

	fmt.Print("Stream: [")
	for iter.Next() {
		chunk := iter.Text()
		if chunk != "" {
			fmt.Print(chunk)
		}
	}
	fmt.Println("]")

	if err := iter.Err(); err != nil {
		fmt.Printf("Stream error: %v\n", err)
		return
	}

	if usage := iter.Usage(); usage != nil {
		fmt.Printf("Usage: prompt=%d completion=%d\n",
			usage.PromptTokens, usage.CompletionTokens)
	}
}

func runResponses(ctx context.Context) {
	fmt.Println("========================================")
	fmt.Println("  RESPONSES: OpenAI Responses API")
	fmt.Println("========================================")

	engine := slm.NewOpenAIResponsesProtocolWithOptions(
		"https://models.inference.ai.azure.com",
		getAPIKey(),
		slm.OpenAIResponsesOptions{
			DefaultModel: "gpt-4o-mini",
			Capabilities: &slm.CapabilityNegotiationOptions{
				Resolver: slm.StaticCapabilityResolver{
					"gpt-4o-mini": {
						Supports: slm.CapabilitySet{JSONMode: true, ToolCalls: true, Reasoning: true},
					},
				},
				DefaultModel: "gpt-4o-mini",
				RequireKnown: true,
			},
		},
	)

	resp, err := engine.Create(ctx, &slm.ResponseRequest{
		Input: []slm.ResponseInputItem{{
			Role:    "user",
			Content: "Return compact JSON with keys summary and use_case for Go interfaces.",
		}},
		Reasoning: &slm.ResponseReasoning{Effort: "medium"},
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Status: %s\n", resp.Status)
	fmt.Printf("Model: %s\n", resp.Model)
	for _, item := range resp.Output {
		for _, content := range item.Content {
			if content.Text != "" {
				fmt.Printf("Output: %s\n", content.Text)
			}
		}
	}
	if resp.Usage != nil {
		fmt.Printf("Usage: input=%d output=%d total=%d\n", resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens)
	}
	fmt.Println("✓ /responses now demonstrates the same explicit capability negotiation entry as chat")
}

func runToolCalling(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  TOOL CALLING: Multi-tool Execution")
	fmt.Println("========================================")

	tools := []slm.Tool{
		{
			Name:        "get_weather",
			Description: "Get current weather for a location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string"},
					"unit":     map[string]any{"type": "string", "enum": []string{"celsius", "fahrenheit"}},
				},
				"required": []string{"location"},
			},
		},
		{
			Name:        "calculate",
			Description: "Evaluate a math expression",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"expression": map[string]any{"type": "string"},
				},
				"required": []string{"expression"},
			},
		},
		{
			Name:        "search_codebase",
			Description: "Search for code patterns in a repository",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query":       map[string]any{"type": "string"},
					"file_type":   map[string]any{"type": "string"},
					"max_results": map[string]any{"type": "integer"},
				},
				"required": []string{"query"},
			},
		},
	}

	messages := []slm.Message{
		slm.NewTextMessage(slm.RoleSystem, `Use tools when needed. Return final answer after all tools complete.`),
		slm.NewTextMessage(slm.RoleUser, `What's the weather in Tokyo and calculate 256*1024? Also search for "error handling" patterns.`),
	}

	for round := 1; round <= 5; round++ {
		fmt.Printf("--- Round %d ---\n", round)

		resp, err := engine.Generate(ctx, &slm.Request{
			Model:    "gpt-4o-mini",
			Messages: messages,
			Tools:    tools,
		})
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}

		if len(resp.ToolCalls) == 0 {
			fmt.Printf("Final Answer: %s\n", resp.Content)
			break
		}

		messages = append(messages, slm.Message{
			Role:      slm.RoleAssistant,
			ToolCalls: resp.ToolCalls,
		})

		for _, tc := range resp.ToolCalls {
			fmt.Printf("  → Tool: %s(%s)\n", tc.Name, tc.Arguments)

			var result string
			switch tc.Name {
			case "get_weather":
				result = `{"location":"Tokyo","temperature":22,"unit":"celsius","condition":"cloudy"}`
			case "calculate":
				result = `{"expression":"256*1024","result":262144}`
			case "search_codebase":
				result = `{"results":[{"file":"errors.go","line":42,"snippet":"if err != nil { return err }"}],"total":3}`
			default:
				result = `{"error":"unknown"}`
			}

			messages = append(messages, slm.NewToolMessage(tc.ID, result))
			fmt.Printf("    Result: %s\n", result[:min(len(result), 60)])
		}
	}
}

func runGeneric(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  GENERIC: Type-safe Call[T]")
	fmt.Println("========================================")

	type Issue struct {
		File     string `json:"file"`
		Line     int    `json:"line"`
		Severity string `json:"severity"`
		Message  string `json:"message"`
		Fix      string `json:"fix"`
	}
	type CodeReview struct {
		Score       int      `json:"score"`
		Category    string   `json:"category"`
		Issues      []Issue  `json:"issues"`
		Suggestions []string `json:"suggestions"`
	}

	code := `func process(data []int) int {
    result := 0
    for i := 0; i < len(data); i++ {
        result += data[i]
    }
    return result
}`

	prompt := "Review this Go code and return JSON only (no markdown):\n\n```go\n" + code + "\n```\n\nReturn exactly this JSON format:\n{\"score\":7,\"category\":\"performance\",\"issues\":[],\"suggestions\":[\"suggestion1\",\"suggestion2\"]}\n\nRules:\n- score: integer 1-10\n- category: one of \"performance\", \"readability\", \"correctness\", \"security\"\n- issues: array of {file,line,severity,message,fix}\n- suggestions: array of strings only"

	result, err := slm.Call[CodeReview](ctx, engine, &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleSystem, "You are a Go code reviewer. Return valid JSON only."),
			slm.NewTextMessage(slm.RoleUser, prompt),
		},
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Score: %d/10 [%s]\n", result.Score, result.Category)
	fmt.Printf("\nIssues (%d):\n", len(result.Issues))
	for _, issue := range result.Issues {
		fmt.Printf("  [%s:%d] %s: %s\n  Fix: %s\n\n",
			issue.File, issue.Line, issue.Severity, issue.Message, issue.Fix)
	}
	fmt.Printf("Suggestions (%d):\n", len(result.Suggestions))
	for _, s := range result.Suggestions {
		fmt.Printf("  - %s\n", s)
	}
}

func runReasoning(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  REASONING: o1/o3 Chain-of-Thought")
	fmt.Println("========================================")

	req := &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleUser, `A train leaves Station A traveling east at 80 km/h.
Another train leaves Station B traveling west at 100 km/h.
The stations are 360 km apart.
When will they meet?
Show your reasoning step by step.`),
		},
	}

	resp, err := engine.Generate(ctx, req)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Answer:\n%s\n", resp.Content)

	if resp.ReasoningContent != "" {
		fmt.Printf("Reasoning (hidden thought):\n---\n%s\n---\n", resp.ReasoningContent)
	} else {
		fmt.Println("(No separate reasoning content - model may not support it)")
	}
}

// ============================================================
// 03-PRODUCTION: 生产级示例
// ============================================================

func runMiddleware(engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  MIDDLEWARE: Retry + RateLimit Chain")
	fmt.Println("========================================")

	retryUnary, retryStream := slm.RetryMiddlewareWithConfig(slm.RetryConfig{
		MaxAttempts: 3,
		Backoff:     func(attempt int) time.Duration { return time.Duration(1<<uint(attempt)) * 150 * time.Millisecond },
	})

	rateLimit, rateLimitCloser := slm.RateLimitMiddleware(5, 3)
	defer rateLimitCloser()

	rateLimitStream, rateLimitStreamCloser := slm.RateLimitStreamMiddleware(5, 3)
	defer rateLimitStreamCloser()

	wrapped := slm.ChainWithStream(engine,
		[]slm.Middleware{rateLimit, retryUnary},
		[]slm.StreamMiddleware{rateLimitStream, retryStream},
	)

	fmt.Println("Chain: Request → [RateLimit(5 QPS)] → [Retry(3x)] → Engine")

	ctx := context.Background()
	prompts := []string{
		"What is 1+1?",
		"Name a red fruit.",
		"Capital of France?",
		"Largest ocean?",
		"Speed of light?",
	}

	var totalLatency atomic.Int64
	successCount := atomic.Int32{}

	var wg sync.WaitGroup
	for i, p := range prompts {
		wg.Add(1)
		go func(idx int, prompt string) {
			defer wg.Done()
			start := time.Now()

			resp, err := wrapped.Generate(ctx, &slm.Request{
				Model:    "gpt-4o-mini",
				Messages: []slm.Message{slm.NewTextMessage(slm.RoleUser, prompt)},
			})

			latency := time.Since(start)
			totalLatency.Add(latency.Nanoseconds())

			if err != nil {
				fmt.Printf("[%d] FAIL %v | %v\n", idx, latency, err)
				return
			}
			successCount.Add(1)
			fmt.Printf("[%d] OK   %v | %s\n", idx, latency, resp.Content)
		}(i, p)
	}
	wg.Wait()

	fmt.Printf("\nSummary: %d/%d success | Total: %v | Avg: %v\n",
		successCount.Load(), len(prompts),
		time.Duration(totalLatency.Load()), time.Duration(totalLatency.Load())/time.Duration(len(prompts)))
}

func runErrorHandling(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  ERROR HANDLING: Code Classification")
	fmt.Println("========================================")

	scenarios := []struct {
		name string
		req  *slm.Request
	}{
		{
			name: "Invalid Model Name",
			req: &slm.Request{
				Model: "nonexistent-model-v99",
				Messages: []slm.Message{
					slm.NewTextMessage(slm.RoleUser, "test"),
				},
			},
		},
		{
			name: "Empty Messages Array",
			req: &slm.Request{
				Model:    "gpt-4o-mini",
				Messages: []slm.Message{},
			},
		},
		{
			name: "Valid Request (baseline)",
			req: &slm.Request{
				Model: "gpt-4o-mini",
				Messages: []slm.Message{
					slm.NewTextMessage(slm.RoleUser, "Say OK"),
				},
			},
		},
	}

	for _, s := range scenarios {
		fmt.Printf("--- %s ---\n", s.name)
		start := time.Now()

		resp, err := engine.Generate(ctx, s.req)
		latency := time.Since(start)

		if err == nil {
			fmt.Printf("  ✅ Success (%v)\n", latency)
			fmt.Printf("  Content: %s\n", resp.Content[:min(len(resp.Content), 50)])
			continue
		}

		fmt.Printf("  ❌ Error (%v): %v\n", latency, err)

		var llmErr *slm.LLMError
		if errors.As(err, &llmErr) {
			fmt.Printf("  Code: %d (%s)\n", llmErr.Code, codeName(llmErr.Code))
			fmt.Printf("  Retryable: %v\n", llmErr.Code.IsRetryable())
			if llmErr.Cause != nil {
				fmt.Printf("  Cause: %v\n", llmErr.Cause)
			}
		}

		fmt.Printf("  IsRetryableError(): %v\n", slm.IsRetryableError(err))
		fmt.Println()
	}
}

func codeName(c slm.ErrorCode) string {
	names := map[slm.ErrorCode]string{
		slm.ErrCodeRateLimit:             "Rate Limit",
		slm.ErrCodeTimeout:               "Timeout",
		slm.ErrCodeOverloaded:            "Overloaded",
		slm.ErrCodeNetwork:               "Network",
		slm.ErrCodeServer:                "Server Error",
		slm.ErrCodeAuth:                  "Auth Failed",
		slm.ErrCodeInvalidModel:          "Invalid Model",
		slm.ErrCodeInvalidConfig:         "Bad Config",
		slm.ErrCodeUnsupportedCapability: "Unsupported Capability",
		slm.ErrCodeContentFilter:         "Content Filtered",
		slm.ErrCodeContextTooLong:        "Context Too Long",
		slm.ErrCodeParse:                 "Parse Error",
		slm.ErrCodeInternal:              "Internal Error",
		slm.ErrCodeCancelled:             "Cancelled",
	}
	if n, ok := names[c]; ok {
		return n
	}
	return fmt.Sprintf("Unknown(%d)", c)
}

func runConfig() {
	fmt.Println("========================================")
	fmt.Println("  CONFIG: Builder Pattern")
	fmt.Println("========================================")

	cfg := slm.DefaultConfig().
		WithProvider(slm.ProviderConfig{
			Endpoint:     "https://models.inference.ai.azure.com",
			DefaultModel: "gpt-4o-mini",
		}).
		WithRetry(slm.RetryConfig{
			MaxAttempts: 3,
			Backoff:     slm.ExponentialBackoff,
		})

	data, _ := json.MarshalIndent(cfg, "", "  ")
	fmt.Printf("Config:\n%s\n", string(data))

	engine, err := cfg.BuildEngine(func(p slm.ProviderConfig) (slm.Engine, error) {
		return slm.NewOpenAIProtocol(p.Endpoint, getAPIKey(), p.DefaultModel), nil
	})
	if err != nil {
		fmt.Printf("Build failed: %v\n", err)
		return
	}
	_ = engine
	fmt.Println("✅ Engine built with middleware applied automatically")
}

func runCustomHTTP() {
	fmt.Println("========================================")
	fmt.Println("  CUSTOM HTTP: Transport Settings")
	fmt.Println("========================================")

	client := &http.Client{
		Timeout: 45 * time.Second,
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			MaxIdleConns:          20,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
			ForceAttemptHTTP2:     true,
		},
	}

	transport := slm.NewHTTPTransportWithClient(client,
		"https://models.inference.ai.azure.com",
		getAPIKey(),
	)
	engine := slm.NewOpenAIWithTransport(transport, "gpt-4o-mini")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	start := time.Now()
	resp, err := engine.Generate(ctx, &slm.Request{
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleUser, "Say 'Custom HTTP OK'"),
		},
	})
	latency := time.Since(start)

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("✅ Custom HTTP client working!\n")
	fmt.Printf("   Response: %s\n", resp.Content)
	fmt.Printf("   Latency: %v\n", latency)
	fmt.Printf("\nCustom settings applied:\n")
	fmt.Printf("   - Timeout: 45s (default: 60s)\n")
	fmt.Printf("   - MaxIdleConns: 20\n")
	fmt.Printf("   - HTTP/2: enabled\n")
	fmt.Printf("   - KeepAlive: 30s\n")
}

func runCapabilities(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  CAPABILITIES: Explicit Negotiation")
	fmt.Println("========================================")

	resolver := slm.ChainCapabilityResolvers(
		slmbridge.StaticResolver([]copilot.Model{
			{
				ID: "gpt-4o-mini",
				Capabilities: copilot.ModelCapabilities{
					Supports: copilot.ModelSupports{
						StructuredOutputs: true,
						ToolCalls:         true,
						Vision:            true,
						AdaptiveThinking:  true,
						ReasoningEffort:   []string{"low", "medium", "high"},
					},
				},
			},
		}),
		slm.StaticCapabilityResolver{
			"*": {Supports: slm.CapabilitySet{JSONMode: true}},
		},
	)

	fmt.Println("Resolver chain:")
	fmt.Println("  1. Copilot model catalog bridge -> slm.CapabilityResolver")
	fmt.Println("  2. Generic slm fallback for unknown models")

	capabilityEngine := slm.NewOpenAIProtocolWithOptions(
		"https://models.inference.ai.azure.com",
		getAPIKey(),
		slm.OpenAIOptions{
			DefaultModel: "gpt-4o-mini",
			Capabilities: &slm.CapabilityNegotiationOptions{
				Resolver:     resolver,
				DefaultModel: "gpt-4o-mini",
				RequireKnown: true,
			},
		},
	)
	wrapped := slm.ApplyStandardMiddleware(capabilityEngine, slm.StandardMiddlewareOptions{
		Observers: []slm.LifecycleObserver{capabilityPrinterObserver{}},
	})

	fmt.Println("--- Preflight reject before provider call ---")
	_, err := wrapped.Generate(ctx, &slm.Request{
		Model: "json-only-model",
		Tools: []slm.Tool{{
			Name:        "lookup",
			Description: "lookup a record",
			Parameters:  map[string]any{"type": "object"},
		}},
		Messages: []slm.Message{slm.NewTextMessage(slm.RoleUser, "use the tool")},
	})
	if err != nil {
		fmt.Printf("Rejected: %v\n", err)
	}

	fmt.Println("\n--- Allowed request with explicit reasoning + JSON mode ---")
	resp, err := wrapped.Generate(ctx, &slm.Request{
		JSONMode:  true,
		Reasoning: &slm.ReasoningOptions{Effort: "medium"},
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleSystem, "Return compact JSON only."),
			slm.NewTextMessage(slm.RoleUser, `Summarize Go interfaces in JSON with fields summary and use_case.`),
		},
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Println("✓ Request was preflight-validated against explicit model capabilities")
}

func runObservability(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  OBSERVABILITY: Metrics + Trace Adapters")
	fmt.Println("========================================")

	now := time.Date(2026, 4, 24, 12, 0, 0, 0, time.UTC)
	refreshes := 0
	resolver := slm.NewCatalogCapabilityResolver(func(context.Context) ([]slm.ModelCapabilities, error) {
		refreshes++
		if refreshes == 1 {
			return []slm.ModelCapabilities{{
				Model:    "gpt-4o-mini",
				Supports: slm.CapabilitySet{JSONMode: true, ToolCalls: true, Vision: true, Reasoning: true},
			}}, nil
		}
		return nil, fmt.Errorf("demo catalog refresh failed")
	}, slm.CapabilityCatalogResolverOptions{
		CacheTTL:          time.Minute,
		Now:               func() time.Time { return now },
		AllowStaleOnError: true,
	})
	if _, _, err := resolver.ResolveCapabilities(ctx, "gpt-4o-mini"); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	now = now.Add(2 * time.Minute)

	meter := &exampleMeter{}
	tracer := &exampleTracer{}
	wrapped := slm.ApplyStandardMiddleware(engine, slm.StandardMiddlewareOptions{
		EnableRequestID:    true,
		RequestIDGenerator: func() string { return "demo_observe_req" },
		Capabilities: &slm.CapabilityNegotiationOptions{
			Resolver:     resolver,
			DefaultModel: "gpt-4o-mini",
		},
		Observers: []slm.LifecycleObserver{
			slm.NewMetricsObserver(meter, slm.MetricsObserverOptions{Namespace: "demo"}),
			slm.NewTraceObserver(tracer, slm.TraceObserverOptions{SpanPrefix: "demo"}),
		},
	})

	resp, err := wrapped.Generate(ctx, &slm.Request{
		Model:    "gpt-4o-mini",
		JSONMode: true,
		Messages: []slm.Message{slm.NewTextMessage(slm.RoleUser, "Return valid JSON: {\"status\":\"ok\"}")},
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Println("Simulated capability catalog: refresh expired, reload failed, stale fallback remained available")
	fmt.Println("\nMetrics:")
	for _, line := range meter.lines {
		fmt.Printf("  %s\n", line)
	}
	fmt.Println("  note: capability negotiation attributes now show capabilities.source=catalog, capabilities.known=true, and capabilities.stale=true")
	fmt.Println("\nSpans:")
	for _, line := range tracer.lines {
		fmt.Printf("  %s\n", line)
	}
}

type capabilityPrinterObserver struct{}

func (capabilityPrinterObserver) OnRequestStart(_ context.Context, event slm.LifecycleEvent) {
	if negotiated, ok := slm.GetNegotiatedCapabilities(event.Context); ok {
		refreshedAt := ""
		if !negotiated.State.RefreshedAt.IsZero() {
			refreshedAt = negotiated.State.RefreshedAt.Format(time.RFC3339)
		}
		fmt.Printf("Negotiated model=%s requested=%+v supported=%+v known=%v source=%s refreshed_at=%s stale=%v\n", negotiated.Model, negotiated.Requested, negotiated.Supported, negotiated.Known, negotiated.State.Source, refreshedAt, negotiated.Stale)
	}
}

func (capabilityPrinterObserver) OnRequestFinish(context.Context, slm.LifecycleEvent)   {}
func (capabilityPrinterObserver) OnStreamStart(context.Context, slm.LifecycleEvent)     {}
func (capabilityPrinterObserver) OnStreamConnected(context.Context, slm.LifecycleEvent) {}
func (capabilityPrinterObserver) OnStreamFinish(context.Context, slm.LifecycleEvent)    {}

type exampleMeter struct {
	mu    sync.Mutex
	lines []string
}

type exampleCounter struct {
	meter *exampleMeter
	name  string
}

type exampleHistogram struct {
	meter *exampleMeter
	name  string
}

func (m *exampleMeter) Int64Counter(name, description, unit string) slm.Int64Counter {
	return &exampleCounter{meter: m, name: name}
}

func (m *exampleMeter) Float64Histogram(name, description, unit string) slm.Float64Histogram {
	return &exampleHistogram{meter: m, name: name}
}

func (c *exampleCounter) Add(_ context.Context, value int64, attrs ...slm.Attribute) {
	c.meter.record(fmt.Sprintf("counter %s += %d %s", c.name, value, formatAttributes(attrs)))
}

func (h *exampleHistogram) Record(_ context.Context, value float64, attrs ...slm.Attribute) {
	h.meter.record(fmt.Sprintf("histogram %s = %.3f %s", h.name, value, formatAttributes(attrs)))
}

func (m *exampleMeter) record(line string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lines = append(m.lines, line)
}

type exampleTracer struct {
	mu    sync.Mutex
	lines []string
}

type exampleSpan struct {
	tracer *exampleTracer
	name   string
	attrs  []slm.Attribute
	errors []error
}

func (t *exampleTracer) Start(ctx context.Context, name string, attrs ...slm.Attribute) (context.Context, slm.Span) {
	return ctx, &exampleSpan{tracer: t, name: name, attrs: append([]slm.Attribute(nil), attrs...)}
}

func (s *exampleSpan) SetAttributes(attrs ...slm.Attribute) {
	s.attrs = append(s.attrs, attrs...)
}

func (s *exampleSpan) RecordError(err error) {
	if err != nil {
		s.errors = append(s.errors, err)
	}
}

func (s *exampleSpan) End() {
	line := fmt.Sprintf("span %s attrs=%s", s.name, formatAttributes(s.attrs))
	if len(s.errors) > 0 {
		line += fmt.Sprintf(" errors=%d", len(s.errors))
	}
	s.tracer.mu.Lock()
	defer s.tracer.mu.Unlock()
	s.tracer.lines = append(s.tracer.lines, line)
}

func formatAttributes(attrs []slm.Attribute) string {
	if len(attrs) == 0 {
		return "{}"
	}
	parts := make([]string, 0, len(attrs))
	for _, attr := range attrs {
		parts = append(parts, fmt.Sprintf("%s=%v", attr.Key, attr.Value))
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// ============================================================
// 04-PATTERNS: 设计模式示例
// ============================================================

func runConversation(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  CONVERSATION: Multi-turn Context")
	fmt.Println("========================================")

	history := []slm.Message{
		slm.NewTextMessage(slm.RoleSystem, `You are a helpful coding tutor.
Explain concepts clearly with simple examples.
Keep responses under 100 words.`),
	}

	dialogue := []string{
		"What is a closure in Go?",
		"Can you show me a concrete example?",
		"How is that different from a regular function variable?",
		"When would I use closures vs methods?",
	}

	for turn, userMsg := range dialogue {
		fmt.Printf("\n── Turn %d ──\n", turn+1)
		history = append(history, slm.NewTextMessage(slm.RoleUser, userMsg))

		resp, err := engine.Generate(ctx, &slm.Request{
			Model:    "gpt-4o-mini",
			Messages: history,
		})
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}

		fmt.Printf("Assistant: %s\n\n", resp.Content)
		history = append(history, slm.Message{
			Role:    slm.RoleAssistant,
			Content: []slm.ContentPart{slm.TextPart(resp.Content)},
		})
	}

	fmt.Printf("Total messages in context: %d\n", len(history))
}

func runBatch(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  BATCH: Concurrent Requests")
	fmt.Println("========================================")

	tasks := []struct {
		ID     string
		Prompt string
	}{
		{"T1", "Translate 'hello' to French"},
		{"T2", "Translate 'goodbye' to Spanish"},
		{"T3", "Translate 'thank you' to German"},
		{"T4", "Translate 'please' to Italian"},
		{"T5", "Translate 'sorry' to Japanese"},
		{"T6", "Summarize 'concurrency' in 5 words"},
	}

	const maxConcurrent = 3
	sem := make(chan struct{}, maxConcurrent)
	var wg sync.WaitGroup
	results := make(chan BatchResult, len(tasks))

	start := time.Now()

	for _, task := range tasks {
		wg.Add(1)
		go func(id, prompt string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			t0 := time.Now()
			resp, err := engine.Generate(ctx, &slm.Request{
				Model:    "gpt-4o-mini",
				Messages: []slm.Message{slm.NewTextMessage(slm.RoleUser, prompt)},
			})

			results <- BatchResult{ID: id, Response: resp, Err: err, Latency: time.Since(t0)}
		}(task.ID, task.Prompt)
	}
	wg.Wait()
	close(results)

	total := time.Since(start)
	var success int
	fmt.Printf("%-6s %8v  %s %s\n", "ID", "Latency", "Result", "")
	fmt.Println(strings.Repeat("-", 55))

	for r := range results {
		status := "✅"
		content := ""
		if r.Err != nil {
			status = "❌"
			content = r.Err.Error()[:min(len(r.Err.Error()), 35)]
		} else {
			content = "(no content)"
			if r.Response != nil {
				content = strings.TrimSpace(r.Response.Content)[:min(len(strings.TrimSpace(r.Response.Content)), 35)]
			}
			success++
		}
		fmt.Printf("%-6s %8v  %s %s\n", r.ID, r.Latency, status, content)
	}

	fmt.Printf("\n%d/%d success | Total: %v | Concurrency: %d\n", success, len(tasks), total, maxConcurrent)
}

type BatchResult struct {
	ID       string
	Response *slm.Response
	Err      error
	Latency  time.Duration
}

func runCancel(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  CANCEL: Timeout & Cancellation")
	fmt.Println("========================================")

	longReq := &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleUser, "Write a 500-word story about a dragon."),
		},
	}

	fmt.Println("--- Test 1: Context timeout (2s) ---")
	ctx1, cancel1 := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel1()

	t1 := time.Now()
	_, err := engine.Generate(ctx1, longReq)
	fmt.Printf("After %v: %v\n\n", time.Since(t1), err)

	fmt.Println("--- Test 2: Manual cancel (500ms) ---")
	ctx2, cancel2 := context.WithCancel(context.Background())
	go func() { time.Sleep(500 * time.Millisecond); cancel2() }()

	t2 := time.Now()
	_, err = engine.Generate(ctx2, longReq)
	fmt.Printf("After %v: %v\n\n", time.Since(t2), err)

	fmt.Println("--- Test 3: Stream with early stop ---")
	ctx3, cancel3 := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel3()

	iter, err := engine.Stream(ctx3, &slm.Request{
		Model:    "gpt-4o-mini",
		Messages: []slm.Message{slm.NewTextMessage(slm.RoleUser, "List numbers 1-50")},
	})
	if err != nil {
		fmt.Printf("Stream error: %v\n", err)
		return
	}
	defer iter.Close()

	buf := strings.Builder{}
	count := 0
	for iter.Next() {
		buf.WriteString(iter.Text())
		count++
		if count >= 15 {
			cancel3()
			fmt.Printf("Cancelled after %d chunks\n", count)
			break
		}
	}
	fmt.Printf("Collected: %d bytes | Error: %v\n", buf.Len(), iter.Err())
}

func runAccumulator(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  ACCUMULATOR: Buffered Stream Collection")
	fmt.Println("========================================")

	iter, err := engine.Stream(ctx, &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleUser, "Write a haiku about programming."),
		},
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer iter.Close()

	type Accumulator struct {
		FullText   strings.Builder
		ChunkCount int
		TokenEst   int
		StartTime  time.Time
		EndTime    time.Time
	}

	acc := Accumulator{StartTime: time.Now()}
	const avgCharsPerToken = 4

	for iter.Next() {
		chunk := iter.Text()
		acc.FullText.WriteString(chunk)
		acc.ChunkCount++
	}
	acc.EndTime = time.Now()
	acc.TokenEst = acc.FullText.Len() / avgCharsPerToken

	if err := iter.Err(); err != nil {
		fmt.Printf("Stream error: %v\n", err)
		return
	}

	fmt.Printf("Chunks received: %d\n", acc.ChunkCount)
	fmt.Printf("Total characters: %d (~%d tokens)\n", acc.FullText.Len(), acc.TokenEst)
	fmt.Printf("Duration: %v\n", acc.EndTime.Sub(acc.StartTime))
	fmt.Printf("Throughput: %.0f chars/sec\n",
		float64(acc.FullText.Len())/acc.EndTime.Sub(acc.StartTime).Seconds())
	fmt.Printf("\nContent:\n%s\n", acc.FullText.String())

	if usage := iter.Usage(); usage != nil {
		fmt.Printf("Actual tokens: prompt=%d completion=%d\n",
			usage.PromptTokens, usage.CompletionTokens)
	}
}

// ============================================================
// 05-SPECIALIZED: 专项场景示例
// ============================================================

func runMultimodal(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  MULTIMODAL: Vision (Image + Text)")
	fmt.Println("========================================")

	imgURL := []string{
		"https://raw.githubusercontent.com/microsoft/vscode/main/resources/win32/code_70x70.png",
		"https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1280px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
	}

	req := &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			{
				Role: slm.RoleUser,
				Content: []slm.ContentPart{
					slm.TextPart("What do you see in this image? Describe it in 2 sentences."),
					slm.ImagePart{URL: imgURL[0]},
				},
			},
		},
	}

	resp, err := engine.Generate(ctx, req)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		var le *slm.LLMError
		if errors.As(err, &le) {
			fmt.Printf("Note: Vision requires vision-capable model. Code=%d\n", le.Code)
		}
		return
	}

	fmt.Printf("Description:\n%s\n", resp.Content)
}

func runFewShot(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  FEW-SHOT: Learning from Examples")
	fmt.Println("========================================")

	type Translation struct {
		Input  string `json:"input"`
		Output string `json:"output"`
	}

	systemPrompt := `You are a translator. Follow the pattern exactly and respond in JSON format.
Input: hello
Output: bonjour

Input: goodbye  
Output: au revoir

Now translate:`
	userPrompt := "thank you very much"

	result, err := slm.Call[Translation](ctx, engine, &slm.Request{
		Model:       "gpt-4o-mini",
		Temperature: slm.Float64(0.0),
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleSystem, systemPrompt),
			slm.NewTextMessage(slm.RoleUser, userPrompt),
		},
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Input:  %s\n", result.Input)
	fmt.Printf("Output: %s\n", result.Output)
}

// ============================================================
// 06-ENHANCED: 增强接口示例
// ============================================================

func runSimpleCall(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  SIMPLE CALL: Convenience Wrappers")
	fmt.Println("========================================")

	fmt.Println("--- SimpleCall: One-liner text query ---")
	answer, err := slm.SimpleCall(ctx, engine, "What is the capital of Japan? Reply with just the city name.")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("Answer: %s\n\n", answer)

	fmt.Println("--- Chat: Multi-turn conversation ---")
	messages := []slm.Message{
		slm.NewTextMessage(slm.RoleSystem, "You are a math tutor. Be concise."),
		slm.NewTextMessage(slm.RoleUser, "What is 15 * 7?"),
	}

	resp1, err := slm.Chat(ctx, engine, messages)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("Q1: 15 * 7 = %s\n", resp1.Content)

	messages = append(messages,
		slm.Message{Role: slm.RoleAssistant, Content: []slm.ContentPart{slm.TextPart(resp1.Content)}},
		slm.NewTextMessage(slm.RoleUser, "Now what about 15 * 8?"),
	)

	resp2, err := slm.Chat(ctx, engine, messages)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("Q2: 15 * 8 = %s\n", resp2.Content)
}

func runFullText(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  FULL TEXT: Reasoning Content Extraction")
	fmt.Println("========================================")

	req := &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleUser, `Solve step by step:
If a store sells apples at $3 each and oranges at $2 each,
and I buy 5 apples and 8 oranges, how much do I spend?
Show your work.`),
		},
	}

	iter, err := engine.Stream(ctx, req)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer iter.Close()

	fmt.Println("--- Streaming Output ---")

	var fullContent strings.Builder
	var hasReasoning bool

	for iter.Next() {
		text := iter.Text()
		fullText := iter.FullText()

		if text != "" {
			fmt.Print(text)
		}

		if fullText != "" && text != fullText {
			hasReasoning = true
		}

		fullContent.WriteString(text)
	}

	fmt.Println()

	if err := iter.Err(); err != nil {
		fmt.Printf("Stream error: %v\n", err)
		return
	}

	if usage := iter.Usage(); usage != nil {
		fmt.Printf("\nUsage: prompt=%d completion=%d total=%d\n",
			usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
	}

	if hasReasoning {
		fmt.Println("\n✓ Model provided reasoning content (visible via FullText())")
	} else {
		fmt.Println("\nℹ This model doesn't expose separate reasoning content")
		fmt.Println("  (o1/o3 models would show reasoning via FullText())")
	}
}

// ============================================================
// 07-INFRASTRUCTURE: 基础设施示例（Logger/Timeout/RequestID）
// ============================================================

type MyLogger struct {
	prefix string
}

func (l *MyLogger) formatArgs(args ...any) string {
	if len(args) == 0 {
		return ""
	}
	var b strings.Builder
	for i := 0; i < len(args); i += 2 {
		if i+1 < len(args) {
			b.WriteString(fmt.Sprintf(" %v=%v", args[i], args[i+1]))
		}
	}
	return b.String()
}

func (l *MyLogger) Info(msg string, args ...any) {
	fmt.Printf("[INFO] %s %s%s\n", l.prefix, msg, l.formatArgs(args...))
}
func (l *MyLogger) Debug(msg string, args ...any) {
	fmt.Printf("[DEBUG] %s %s%s\n", l.prefix, msg, l.formatArgs(args...))
}
func (l *MyLogger) Warn(msg string, args ...any) {
	fmt.Printf("[WARN] %s %s%s\n", l.prefix, msg, l.formatArgs(args...))
}
func (l *MyLogger) Error(msg string, args ...any) {
	fmt.Printf("[ERROR] %s %s%s\n", l.prefix, msg, l.formatArgs(args...))
}

func runLogging(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  LOGGING: Logger Interface + Middleware")
	fmt.Println("========================================")

	logger := &MyLogger{prefix: "LLM"}

	loggingMW := slm.LoggingMiddleware(logger)

	wrapped := slm.Chain(engine, loggingMW)

	fmt.Println("--- Request with automatic logging ---")

	resp, err := wrapped.Generate(ctx, &slm.Request{
		Model: "gpt-4o-mini",
		Messages: []slm.Message{
			slm.NewTextMessage(slm.RoleUser, "Say 'Hello World'"),
		},
	})

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("\nResponse: %s\n", resp.Content)
	fmt.Printf("\n✅ All LLM requests are now automatically logged!\n")
	fmt.Println("   - Request start time and model")
	fmt.Println("   - Response duration and token usage")
	fmt.Println("   - Error details if any failure")
}

func runTimeout(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  TIMEOUT: Request Timeout Control")
	fmt.Println("========================================")

	timeoutMW := slm.TimeoutMiddleware(5 * time.Second)

	wrapped := slm.Chain(engine, timeoutMW)

	tests := []struct {
		name   string
		prompt string
	}{
		{
			name:   "Quick request (< 5s)",
			prompt: "Reply with just 'OK'",
		},
		{
			name:   "Simulated slow request",
			prompt: "Count from 1 to 1000000 very slowly",
		},
	}

	for _, test := range tests {
		fmt.Printf("--- %s ---\n", test.name)

		func() {
			reqCtx, cancel := context.WithTimeout(ctx, 6*time.Second)
			defer cancel()

			start := time.Now()
			resp, err := wrapped.Generate(reqCtx, &slm.Request{
				Model: "gpt-4o-mini",
				Messages: []slm.Message{
					slm.NewTextMessage(slm.RoleUser, test.prompt),
				},
			})
			latency := time.Since(start)

			if err != nil {
				var llmErr *slm.LLMError
				if errors.As(err, &llmErr) && llmErr.Code == slm.ErrCodeTimeout {
					fmt.Printf("  ⏰ Timed out after %v (as expected)\n", latency)
					fmt.Printf("     Error: %v\n\n", llmErr)
					return
				}
				fmt.Printf("  ❌ Error (%v): %v\n\n", latency, err)
				return
			}

			fmt.Printf("  ✅ Success in %v\n", latency)
			fmt.Printf("  Response: %s\n\n", resp.Content[:min(len(resp.Content), 50)])
		}()
	}

	fmt.Println("💡 Benefits of TimeoutMiddleware:")
	fmt.Println("   - Prevents indefinite hanging on slow responses")
	fmt.Println("   - Returns structured timeout error (ErrCodeTimeout)")
	fmt.Println("   - Automatic goroutine cleanup (no leaks)")
}

func runRequestID(ctx context.Context, engine slm.Engine) {
	fmt.Println("========================================")
	fmt.Println("  REQUEST ID: Distributed Tracing")
	fmt.Println("========================================")

	requestIDMW := slm.RequestIDMiddleware(nil)

	wrapped := slm.Chain(engine, requestIDMW)

	prompts := []string{
		"What is 2+2?",
		"Capital of Japan?",
		"Largest planet?",
	}

	for i, prompt := range prompts {
		fmt.Printf("--- Request #%d ---\n", i+1)

		resp, err := wrapped.Generate(ctx, &slm.Request{
			Model: "gpt-4o-mini",
			Messages: []slm.Message{
				slm.NewTextMessage(slm.RoleUser, prompt),
			},
		})

		if err != nil {
			fmt.Printf("  Error: %v\n\n", err)
			continue
		}

		if reqID, ok := resp.Meta["request_id"].(string); ok {
			fmt.Printf("  📋 Request ID: %s\n", reqID)
		}

		fmt.Printf("  Response: %s\n\n", resp.Content)
	}

	fmt.Println("💡 Benefits of RequestIDMiddleware:")
	fmt.Println("   - Automatic unique ID generation per request")
	fmt.Println("   - Stored in Meta for logging/monitoring")
	fmt.Println("   - Can retrieve via GetRequestID(ctx) in middleware chain")
	fmt.Println("   - Essential for distributed tracing and debugging")
}
