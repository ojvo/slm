package slm

import (
	"context"
	"net/http"
	"strings"
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

type RequestIdentity struct {
	Model     string         `json:"model"`
	Meta      map[string]any `json:"meta,omitempty"`
	ExtraBody map[string]any `json:"extra_body,omitempty"`
}

func (r RequestIdentity) GetModel() string { return r.Model }

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

func (r *Request) GetModel() string { return r.Model }

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

// ResponseReasoning is an alias for ReasoningOptions used by the /responses API.
// Both chat and responses paths share the same Effort/Summary fields.
type ResponseReasoning = ReasoningOptions

// ResponseTool represents a tool passed to the /responses API.
// The fields map directly to the /responses wire format where function properties
// are promoted to the top level (unlike /chat/completions which nests them under "function").
// Use NewResponseFunctionTool to build one from an existing Tool definition.
type ResponseTool struct {
	Type        string `json:"type"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// NewResponseFunctionTool converts a Tool into a ResponseTool ready for the /responses API.
func NewResponseFunctionTool(t Tool) ResponseTool {
	return ResponseTool{
		Type:        "function",
		Name:        t.Name,
		Description: t.Description,
		Parameters:  t.Parameters,
	}
}

// ResponseInputItem is one input item for the /responses API.
type ResponseInputItem struct {
	Role    string
	Content any
}

func NewTextResponseInputItem(role, text string) ResponseInputItem {
	return ResponseInputItem{Role: role, Content: text}
}

func NewMultiPartResponseInputItem(role string, parts []ResponseInputContentPart) ResponseInputItem {
	return ResponseInputItem{Role: role, Content: parts}
}

type ResponseInputContentPart interface {
	isResponseInputContentPart()
}

type ResponseInputTextPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (ResponseInputTextPart) isResponseInputContentPart() {}

type ResponseInputImagePart struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url,omitempty"`
}

func (ResponseInputImagePart) isResponseInputContentPart() {}

// ResponseRequest is the package-level request model for the OpenAI /responses API.
type ResponseRequest struct {
	Model           string
	Input           []ResponseInputItem
	Stream          bool
	Store           bool
	MaxOutputTokens int
	Reasoning       *ResponseReasoning
	Tools           []ResponseTool
	ExtraBody       map[string]any
}

func (r *ResponseRequest) GetModel() string { return r.Model }

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

// IsOutputTextDelta reports whether this event carries output text delta chunks.
// Providers may emit slightly different prefixes, so suffix matching is used.
func (e ResponseEvent) IsOutputTextDelta() bool {
	t := strings.TrimSpace(strings.ToLower(e.Type))
	if t == "response.output_text.delta" || t == "output_text.delta" {
		return true
	}
	return strings.HasSuffix(t, ".output_text.delta")
}

// CompletedResponse returns a normalized completed response object when present.
func (e ResponseEvent) CompletedResponse() *ResponseObject {
	if e.Response == nil {
		return nil
	}
	return normalizeCompletedResponseObject(e.Response)
}

// BaseStream is the minimal streaming interface shared by both chat and
// responses streams.  Timeout middleware and other cross-cutting
// stream decorators operate on this interface to avoid duplicating
// goroutine+channel+select logic for each stream type.
type BaseStream interface {
	Next() bool
	Err() error
	Close() error
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

type streamIteratorWrapper struct {
	inner StreamIterator
}

func (w *streamIteratorWrapper) Chunk() []byte {
	if w.inner != nil {
		return w.inner.Chunk()
	}
	return nil
}
func (w *streamIteratorWrapper) Text() string {
	if w.inner != nil {
		return w.inner.Text()
	}
	return ""
}
func (w *streamIteratorWrapper) FullText() string {
	if w.inner != nil {
		return w.inner.FullText()
	}
	return ""
}
func (w *streamIteratorWrapper) Err() error {
	if w.inner != nil {
		return w.inner.Err()
	}
	return nil
}
func (w *streamIteratorWrapper) Usage() *Usage {
	if w.inner != nil {
		return w.inner.Usage()
	}
	return nil
}
func (w *streamIteratorWrapper) Response() *Response {
	if w.inner != nil {
		return w.inner.Response()
	}
	return nil
}
func (w *streamIteratorWrapper) Close() error {
	if w.inner != nil {
		return w.inner.Close()
	}
	return nil
}

type responseStreamWrapper struct {
	inner ResponseStream
}

func (w *responseStreamWrapper) Next() bool {
	if w.inner != nil {
		return w.inner.Next()
	}
	return false
}
func (w *responseStreamWrapper) Current() ResponseEvent {
	if w.inner != nil {
		return w.inner.Current()
	}
	return ResponseEvent{}
}
func (w *responseStreamWrapper) Err() error {
	if w.inner != nil {
		return w.inner.Err()
	}
	return nil
}
func (w *responseStreamWrapper) Close() error {
	if w.inner != nil {
		return w.inner.Close()
	}
	return nil
}

// Engine LLM 核心引擎接口
type Engine interface {
	Generate(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) (StreamIterator, error)
}

type protocolBase struct {
	transport    Transport
	defaultModel string
}

func (b *protocolBase) resolveModel(model string) (string, error) {
	m := resolveRequestedModel(model, b.defaultModel)
	if m == "" {
		return "", NewLLMError(ErrCodeInvalidModel, "model is required", nil)
	}
	return m, nil
}

func (b *protocolBase) doPost(ctx context.Context, path string, headers map[string]string, body []byte) (*http.Response, error) {
	return b.transport.Do(ctx, http.MethodPost, path, headers, body)
}
