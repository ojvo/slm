package slm

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

// -----------------------------------------------------------------------------
// Content And Messages
// -----------------------------------------------------------------------------

// Role 定义消息角色
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentPart 定义多模态内容接口
type ContentPart interface {
	isContentPart()
}

// TextPart 纯文本内容
type TextPart string

func (t TextPart) isContentPart() {}

// ImagePart 图片内容
type ImagePart struct {
	URL    string `json:"url,omitempty"`
	Base64 string `json:"base64,omitempty"`
	Detail string `json:"detail,omitempty"`
	MIME   string `json:"-"` // MIME type, e.g. "image/png", "image/jpeg". Defaults to "image/png"
}

func (i ImagePart) isContentPart() {}

// APIToolCall represents a tool/function call emitted by the model.
type APIToolCall struct {
	Index     int    `json:"index,omitempty"`
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool defines a callable tool exposed to the model.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

// Message is the package-level message model for chat-oriented requests.
type Message struct {
	Role       Role
	Content    []ContentPart
	Name       string
	ToolCalls  []APIToolCall
	ToolCallID string
}

// -----------------------------------------------------------------------------
// Chat Request / Response Model
// -----------------------------------------------------------------------------

// Request is the package-level request model for the chat/completions path.
// It is also the primary request shape consumed by the Engine interface.
type Request struct {
	Model            string
	Messages         []Message
	Temperature      *float64
	TopP             *float64
	MaxTokens        int
	Stop             []string
	PresencePenalty  *float64
	FrequencyPenalty *float64
	Stream           bool
	JSONMode         bool
	Reasoning        *ReasoningOptions
	Tools            []Tool
	Meta             map[string]any
	ExtraBody        map[string]any
}

// Response is the normalized chat response returned by Engine.
type Response struct {
	Content          string
	ReasoningContent string
	Usage            Usage
	FinishReason     string
	ToolCalls        []APIToolCall
	Meta             map[string]any
}

// Usage reports token accounting for chat/completions requests.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// -----------------------------------------------------------------------------
// Responses API Model
// -----------------------------------------------------------------------------

// ResponseReasoning carries explicit reasoning controls for the /responses API.
type ResponseReasoning struct {
	Effort  string
	Summary string
}

// ResponseInputItem is one input item for the /responses API.
type ResponseInputItem struct {
	Role    string
	Content string
}

// ResponseRequest is the package-level request model for the OpenAI /responses API.
type ResponseRequest struct {
	Model           string
	Input           []ResponseInputItem
	Stream          bool
	Store           bool
	MaxOutputTokens int
	Reasoning       *ResponseReasoning
	Tools           []json.RawMessage
	ExtraBody       map[string]any
}

// ResponseOutputContent is one content block in a /responses output item.
type ResponseOutputContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// ResponseOutput is one output item returned by the /responses API.
type ResponseOutput struct {
	Type    string                  `json:"type"`
	ID      string                  `json:"id,omitempty"`
	Role    string                  `json:"role,omitempty"`
	Status  string                  `json:"status,omitempty"`
	Content []ResponseOutputContent `json:"content,omitempty"`
	Summary []ResponseOutputContent `json:"summary,omitempty"`
}

// ResponseUsage reports token accounting for the /responses API.
type ResponseUsage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
}

// ResponseObject is the package-level response model for the /responses API.
type ResponseObject struct {
	ID          string
	Object      string
	Status      string
	Model       string
	Output      []ResponseOutput
	Usage       *ResponseUsage
	CreatedAt   time.Time
	CompletedAt time.Time
}

// ResponseEvent is one streamed /responses event.
type ResponseEvent struct {
	Type     string
	Delta    string
	Response *ResponseObject
	Item     *ResponseOutput
	Err      error
}

// ResponseStream is the streaming interface for the /responses API.
type ResponseStream interface {
	Next() bool
	Current() ResponseEvent
	Err() error
	Close() error
}

// -----------------------------------------------------------------------------
// Runtime Interfaces
// -----------------------------------------------------------------------------

// Transport 抽象 LLM API 的 HTTP 通信层。
// OpenAIEngine 只负责 OpenAI 协议的编解码，实际的 HTTP 通信和认证
// 由 Transport 实现。这使得同一个协议引擎可以搭配不同的传输方式：
//   - HTTPTransport: 标准 Bearer token + HTTP 直连
//   - CopilotTransport: GitHub OAuth + token 自动刷新
type Transport interface {
	Do(ctx context.Context, method, path string, headers map[string]string, body []byte) (*http.Response, error)
}

// StreamIterator 流式迭代器接口
type StreamIterator interface {
	Next() bool
	Chunk() []byte
	Text() string
	FullText() string
	Err() error
	Close() error
	Usage() *Usage
	Response() *Response
}

// InterruptibleStreamIterator is an optional capability for stream iterators
// that can be actively interrupted by middleware such as timeout wrappers.
//
// Implementations should make best efforts to unblock any in-flight Next call
// promptly after Interrupt is invoked. The provided error is advisory and can be
// ignored if the iterator surfaces its own terminal error.
type InterruptibleStreamIterator interface {
	StreamIterator
	Interrupt(error)
}

// Handler 处理函数定义 (用于中间件)
type Handler func(ctx context.Context, req *Request) (*Response, error)

// Engine LLM 核心引擎接口
type Engine interface {
	Generate(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) (StreamIterator, error)
}
