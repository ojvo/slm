package slm

import (
	"context"
	"net/http"
)

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

// Message 统一消息结构
type Message struct {
	Role       Role
	Content    []ContentPart
	Name       string
	ToolCalls  []APIToolCall
	ToolCallID string
}

// NewTextMessage 创建文本消息
func NewTextMessage(role Role, text string) Message {
	return Message{
		Role:    role,
		Content: []ContentPart{TextPart(text)},
	}
}

// NewToolMessage 创建工具响应消息
func NewToolMessage(toolCallID, content string) Message {
	return Message{
		Role:       RoleTool,
		Content:    []ContentPart{TextPart(content)},
		ToolCallID: toolCallID,
	}
}

// NewAssistantMessage 创建助手消息（支持工具调用）
func NewAssistantMessage(content string, toolCalls ...APIToolCall) Message {
	msg := Message{
		Role:    RoleAssistant,
		Content: []ContentPart{TextPart(content)},
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	return msg
}

// NewImageMessage 创建图片消息
func NewImageMessage(imageURL string) Message {
	return Message{
		Role:    RoleUser,
		Content: []ContentPart{ImagePart{URL: imageURL}},
	}
}

// Request 统一请求结构
type Request struct {
	Model            string
	Messages         []Message
	Temperature      float64
	TopP             float64
	MaxTokens        int
	Stop             []string
	PresencePenalty  float64
	FrequencyPenalty float64
	Stream           bool
	JSONMode         bool
	Tools            []Tool
	Meta             map[string]any
	ExtraBody        map[string]any
}

// Response 统一响应结构
type Response struct {
	Content          string
	ReasoningContent string
	Usage            Usage
	FinishReason     string
	ToolCalls        []APIToolCall
	Meta             map[string]any
}

// Usage Token 使用统计
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// BaseEngine 包含所有 Driver 共有的字段
type BaseEngine struct {
	Client       *http.Client
	BaseURL      string
	APIKey       string
	AuthHeader   string
	AuthPrefix   string
	DefaultModel string
	ExtraHeader  map[string]string
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
}

// Handler 处理函数定义 (用于中间件)
type Handler func(ctx context.Context, req *Request) (*Response, error)

// Engine LLM 核心引擎接口
type Engine interface {
	Generate(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) (StreamIterator, error)
}
