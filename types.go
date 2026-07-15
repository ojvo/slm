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

// ThinkingPart represents extended thinking content from reasoning models
// (DeepSeek-R1 reasoning_content, Anthropic Claude extended thinking).
// It is a ContentPart so thinking can be replayed in tool loops: the Codec
// layer converts it to the appropriate wire format (thinking block for
// Claude, reasoning_content field for OpenAI/DeepSeek).
//
// Signature is required for Anthropic thinking block replay; other drivers
// ignore it. It is analogous to how ImagePart.Base64 is a wire-transport
// detail that some drivers use and others ignore.
type ThinkingPart struct {
	Content   string
	Signature string // Anthropic extended thinking signature (required for replay)
}

func (t ThinkingPart) isContentPart() {}

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

// ToolSpec is an alias for Tool for compatibility with consumers that
// use the ToolSpec name. Prefer Tool in new code.
type ToolSpec = Tool

// FunctionSpec is a generic interface for tool function specifications.
// Any type with Name, Description, and Parameters fields can satisfy this
// interface, enabling conversion from application-specific tool spec types
// to slm.ToolSpec without slm depending on those types.
type FunctionSpec interface {
	GetName() string
	GetDescription() string
	GetParameters() any
}

// ToolsFromSpecs converts a slice of FunctionSpec implementations to []ToolSpec.
// This enables consumers to convert their tool spec types without slm depending
// on application-specific types.
func ToolsFromSpecs[S FunctionSpec](specs []S) []ToolSpec {
	result := make([]ToolSpec, len(specs))
	for i, s := range specs {
		result[i] = ToolSpec{
			Name:        s.GetName(),
			Description: s.GetDescription(),
			Parameters:  s.GetParameters(),
		}
	}
	return result
}

// Message is the package-level message model for chat-oriented requests.
type Message struct {
	Role       Role
	Content    []ContentPart
	Name       string
	ToolCalls  []APIToolCall
	ToolCallID string
	Meta       map[string]any // application-layer extension metadata (not serialized to LLM requests)
}

// SetMeta sets a single extension metadata key-value pair.
func (m *Message) SetMeta(key string, val any) {
	if m.Meta == nil {
		m.Meta = make(map[string]any)
	}
	m.Meta[key] = val
}

// GetMeta returns the extension metadata value for the key, or nil if absent.
func (m *Message) GetMeta(key string) any {
	if m.Meta == nil {
		return nil
	}
	return m.Meta[key]
}

// GetMetaString returns the extension metadata value as a string, or "" if
// absent or not a string.
func (m *Message) GetMetaString(key string) string {
	v := m.GetMeta(key)
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// GetMetaStrings returns the extension metadata value as a []string, or nil
// if absent or not a []string.
func (m *Message) GetMetaStrings(key string) []string {
	v := m.GetMeta(key)
	if s, ok := v.([]string); ok {
		return s
	}
	return nil
}

// Text returns the concatenated text content of the message.
// For messages with multiple content parts, only TextPart values are concatenated.
func (m Message) Text() string {
	var b strings.Builder
	for _, p := range m.Content {
		if t, ok := p.(TextPart); ok {
			b.WriteString(string(t))
		}
	}
	return b.String()
}

// ReasoningContent returns the concatenated thinking content from ThinkingPart
// values in the message. This is the protocol-agnostic way to access extended
// thinking content that was previously stored as a wire-format field.
func (m Message) ReasoningContent() string {
	var b strings.Builder
	for _, p := range m.Content {
		if tp, ok := p.(ThinkingPart); ok {
			b.WriteString(tp.Content)
		}
	}
	return b.String()
}

// ThinkingSignature returns the signature from the first ThinkingPart in the
// message, if any. Anthropic extended thinking requires this signature for
// replay in tool loops.
func (m Message) ThinkingSignature() string {
	for _, p := range m.Content {
		if tp, ok := p.(ThinkingPart); ok {
			return tp.Signature
		}
	}
	return ""
}

// SetReasoningContent replaces any existing ThinkingPart values in the
// message content with a single ThinkingPart containing the given content
// and signature. If content is empty, all ThinkingPart values are removed.
func (m *Message) SetReasoningContent(content, signature string) {
	var rest []ContentPart
	for _, p := range m.Content {
		if _, ok := p.(ThinkingPart); !ok {
			rest = append(rest, p)
		}
	}
	if content == "" && signature == "" {
		m.Content = rest
		return
	}
	// ThinkingPart is placed at the start to match Anthropic's wire ordering
	// (thinking blocks precede other content).
	m.Content = append([]ContentPart{ThinkingPart{Content: content, Signature: signature}}, rest...)
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
	// Capabilities holds protocol-specific parameters not represented in standard fields.
	// Use this for advanced parameters like reasoning_effort, thinking_budget, etc.
	// Replaces legacy ExtraBody field while maintaining backward compatibility.
	Capabilities map[string]any
}

func (r *Request) GetModel() string { return r.Model }

func (r *Request) ValidateFor(engine Engine) error {
	if engine == nil {
		return NewLLMError(ErrCodeInvalidConfig, "engine is nil", nil)
	}
	return ValidateCapabilities(r.Capabilities, engine.Capabilities(), "this protocol")
}

// Response is the normalized chat response returned by Engine.
type Response struct {
	Content           string
	ReasoningContent  string
	ThinkingSignature string // Anthropic extended thinking signature (populated on non-streaming responses)
	Usage             Usage
	FinishReason      string
	ToolCalls         []APIToolCall
	Meta              map[string]any
}

// Usage reports token accounting for chat/completions requests.
type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	CacheHitTokens      int                  `json:"prompt_cache_hit_tokens,omitempty"` // DeepSeek top-level
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`   // OpenAI/DeepSeek nested
}

// PromptTokensDetails holds nested cache hit data (OpenAI format).
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// CacheMissTokens returns the number of prompt tokens that were NOT served
// from cache. It normalizes across DeepSeek (top-level CacheHitTokens) and
// OpenAI (nested PromptTokensDetails.CachedTokens) formats.
func (u Usage) CacheMissTokens() int {
	hit := u.CacheHitTokens
	if hit == 0 && u.PromptTokensDetails != nil {
		hit = u.PromptTokensDetails.CachedTokens
	}
	miss := u.PromptTokens - hit
	if miss < 0 {
		return 0
	}
	return miss
}

// TotalContextTokens returns the total context token occupancy including cache
// hits. Cached tokens still occupy the model's context window and must not be
// ignored — a fully-cached Claude session would look nearly empty if only
// PromptTokens were checked, causing auto-compact to never trigger.
//
// For Anthropic: PromptTokens excludes cache_read, so the result = input +
// output + cache_read (accurate).
// For OpenAI: PromptTokens may already include cached_tokens, so the result
// double-counts cache hits (overestimate). Overestimating is safe for compact
// triggers (early compact is better than missed compact).
func (u Usage) TotalContextTokens() int {
	return u.PromptTokens + u.CompletionTokens + u.CacheHitTokens
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
	// Capabilities holds protocol-specific parameters not represented in standard fields.
	Capabilities map[string]any
}

func (r *ResponseRequest) GetModel() string { return r.Model }

func (r *ResponseRequest) ValidateFor(engine ResponsesEngine) error {
	if engine == nil {
		return NewLLMError(ErrCodeInvalidConfig, "engine is nil", nil)
	}
	return ValidateCapabilities(r.Capabilities, engine.Capabilities(), "Responses API")
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
// 引擎只负责协议的编解码，实际的 HTTP 通信和认证
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

func (w *streamIteratorWrapper) Next() bool          { return w.inner.Next() }
func (w *streamIteratorWrapper) Chunk() []byte       { return w.inner.Chunk() }
func (w *streamIteratorWrapper) Text() string        { return w.inner.Text() }
func (w *streamIteratorWrapper) FullText() string    { return w.inner.FullText() }
func (w *streamIteratorWrapper) Err() error          { return w.inner.Err() }
func (w *streamIteratorWrapper) Usage() *Usage       { return w.inner.Usage() }
func (w *streamIteratorWrapper) Response() *Response { return w.inner.Response() }
func (w *streamIteratorWrapper) Close() error        { return w.inner.Close() }

type responseStreamWrapper struct {
	inner ResponseStream
}

func (w *responseStreamWrapper) Next() bool             { return w.inner.Next() }
func (w *responseStreamWrapper) Current() ResponseEvent { return w.inner.Current() }
func (w *responseStreamWrapper) Err() error             { return w.inner.Err() }
func (w *responseStreamWrapper) Close() error           { return w.inner.Close() }

// ParameterRange describes valid values or bounds for a protocol parameter.
type ParameterRange struct {
	Min    float64  // Minimum value (for numeric params)
	Max    float64  // Maximum value (for numeric params)
	Values []string // Allowed values (for enum params)
}

// ProtocolCapabilities describes what parameters a protocol/model supports.
// It enables transparent capability negotiation across different LLM providers.
type ProtocolCapabilities struct {
	// SupportedParameters lists protocol-specific parameters beyond standard Request fields.
	// Standard parameters (Temperature, TopP, MaxTokens, etc.) are always supported
	// if supported by the Request type. This maps additional parameters to their valid ranges.
	SupportedParameters map[string]ParameterRange
	// ParameterMapping maps generic parameter names to protocol-specific wire names.
	// E.g., {"reasoning_effort": "reasoning_effort"} for OpenAI o1.
	ParameterMapping map[string]string
	// ConflictingParameters lists groups of parameters that cannot be used together.
	// E.g., [["reasoning", "stream"]] means reasoning and stream cannot both be true.
	ConflictingParameters [][]string
	// Description provides human-readable info about the protocol capabilities.
	Description string
}

func cloneProtocolCapabilities(template ProtocolCapabilities) *ProtocolCapabilities {
	clone := ProtocolCapabilities{
		SupportedParameters:   make(map[string]ParameterRange, len(template.SupportedParameters)),
		ParameterMapping:      make(map[string]string, len(template.ParameterMapping)),
		ConflictingParameters: make([][]string, len(template.ConflictingParameters)),
		Description:           template.Description,
	}

	for key, value := range template.SupportedParameters {
		clonedRange := ParameterRange{Min: value.Min, Max: value.Max}
		if len(value.Values) > 0 {
			clonedRange.Values = append([]string(nil), value.Values...)
		}
		clone.SupportedParameters[key] = clonedRange
	}

	for key, value := range template.ParameterMapping {
		clone.ParameterMapping[key] = value
	}

	for i, conflictGroup := range template.ConflictingParameters {
		clone.ConflictingParameters[i] = append([]string(nil), conflictGroup...)
	}

	return &clone
}

// Engine LLM 核心引擎接口
type Engine interface {
	Generate(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) (StreamIterator, error)
	// Capabilities returns the protocol capabilities supported by this engine.
	// Returns nil if the engine does not declare capabilities (fallback mode).
	Capabilities() *ProtocolCapabilities
}

// ResponsesEngine is the interface for Responses API engines.
// It handles reasoning-focused completion requests separately from chat requests.
type ResponsesEngine interface {
	Create(ctx context.Context, req *ResponseRequest) (*ResponseObject, error)
	Stream(ctx context.Context, req *ResponseRequest) (ResponseStream, error)
	Close() error
	// Capabilities returns the protocol capabilities supported by this responses engine.
	Capabilities() *ProtocolCapabilities
}

type protocolBase struct {
	transport    Transport
	defaultModel string
}

func (b *protocolBase) resolveModel(model string) (string, error) {
	m := ResolveRequestedModel(model, b.defaultModel)
	if m == "" {
		return "", NewLLMError(ErrCodeInvalidModel, "model is required", nil)
	}
	return m, nil
}

func (b *protocolBase) doPost(ctx context.Context, path string, headers map[string]string, body []byte) (*http.Response, error) {
	return b.transport.Do(ctx, http.MethodPost, path, headers, body)
}
