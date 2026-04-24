package slm

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Call 泛型调用：直接请求并解析结果为结构体 T
func Call[T any](ctx context.Context, engine Engine, req *Request) (*T, error) {
	clone := cloneRequest(req)
	if clone == nil {
		clone = &Request{}
	}
	clone.JSONMode = true
	resp, err := engine.Generate(ctx, clone)
	if err != nil {
		return nil, err
	}

	var result T
	content := extractJSON(strings.TrimSpace(resp.Content))
	if content == "" {
		return nil, fmt.Errorf("empty response from LLM")
	}
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	return &result, nil
}

// CallWithPrompt 使用提示词模板的泛型调用
func CallWithPrompt[T any](ctx context.Context, engine Engine, prompt string) (*T, error) {
	return Call[T](ctx, engine, &Request{
		Messages: []Message{
			NewTextMessage(RoleUser, prompt),
		},
	})
}

// StreamCall 泛型流式调用，收集完整响应后解析
func StreamCall[T any](ctx context.Context, engine Engine, req *Request) (*T, error) {
	result, err := StreamCallFull[T](ctx, engine, req)
	if err != nil {
		return nil, err
	}
	return result.Result, nil
}

// StreamResult 流式调用的完整结果
type StreamResult[T any] struct {
	Result           *T
	ReasoningContent string
	ToolCalls        []APIToolCall
	Usage            *Usage
}

// StreamCallFull 泛型流式调用，收集完整响应后解析，返回包含元数据的完整结果
func StreamCallFull[T any](ctx context.Context, engine Engine, req *Request) (*StreamResult[T], error) {
	clone := cloneRequest(req)
	if clone == nil {
		clone = &Request{}
	}
	clone.JSONMode = true
	clone.Stream = true

	iter, err := engine.Stream(ctx, clone)
	if err != nil {
		return nil, err
	}
	defer iter.Close()

	var content strings.Builder
	var reasoning strings.Builder
	toolCallMap := make(map[int]*toolCallAccum)
	for iter.Next() {
		content.WriteString(iter.Text())
		if resp := iter.Response(); resp != nil {
			if resp.ReasoningContent != "" {
				reasoning.WriteString(resp.ReasoningContent)
			}
			for i, tc := range resp.ToolCalls {
				mergeToolCallDelta(toolCallMap, resolveToolCallIndex(i, tc), tc)
			}
		}
	}

	if err := iter.Err(); err != nil {
		return nil, err
	}

	var toolCalls []APIToolCall
	toolCallIndexes := sortedToolCallIndexes(toolCallMap)
	for _, index := range toolCallIndexes {
		acc := toolCallMap[index]
		toolCalls = append(toolCalls, APIToolCall{
			Index:     index,
			ID:        acc.id,
			Type:      acc.typ,
			Name:      acc.name,
			Arguments: acc.arguments.String(),
		})
	}

	var result T
	text := extractJSON(strings.TrimSpace(content.String()))
	if text == "" {
		return nil, fmt.Errorf("empty response from LLM")
	}
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	return &StreamResult[T]{
		Result:           &result,
		ReasoningContent: reasoning.String(),
		ToolCalls:        toolCalls,
		Usage:            iter.Usage(),
	}, nil
}

func cloneRequest(req *Request) *Request {
	if req == nil {
		return nil
	}

	msgs := make([]Message, len(req.Messages))
	for i, msg := range req.Messages {
		msgs[i] = Message{
			Role:       msg.Role,
			Content:    append([]ContentPart(nil), msg.Content...),
			Name:       msg.Name,
			ToolCalls:  append([]APIToolCall(nil), msg.ToolCalls...),
			ToolCallID: msg.ToolCallID,
		}
	}
	tools := make([]Tool, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = Tool{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  deepCopyAny(t.Parameters),
		}
	}

	var extraBody map[string]any
	if len(req.ExtraBody) > 0 {
		extraBody = make(map[string]any, len(req.ExtraBody))
		for k, v := range req.ExtraBody {
			extraBody[k] = deepCopyAny(v)
		}
	}

	var meta map[string]any
	if len(req.Meta) > 0 {
		meta = make(map[string]any, len(req.Meta))
		for k, v := range req.Meta {
			meta[k] = deepCopyAny(v)
		}
	}

	return &Request{
		Model:            req.Model,
		Messages:         msgs,
		Temperature:      cloneFloat64Ptr(req.Temperature),
		TopP:             cloneFloat64Ptr(req.TopP),
		MaxTokens:        req.MaxTokens,
		Stop:             append([]string(nil), req.Stop...),
		PresencePenalty:  cloneFloat64Ptr(req.PresencePenalty),
		FrequencyPenalty: cloneFloat64Ptr(req.FrequencyPenalty),
		Stream:           req.Stream,
		JSONMode:         req.JSONMode,
		Reasoning:        cloneReasoningOptions(req.Reasoning),
		Tools:            tools,
		Meta:             meta,
		ExtraBody:        extraBody,
	}
}

func cloneReasoningOptions(reasoning *ReasoningOptions) *ReasoningOptions {
	if reasoning == nil {
		return nil
	}
	clone := *reasoning
	return &clone
}

// SimpleCall 简单文本调用
func SimpleCall(ctx context.Context, engine Engine, prompt string) (string, error) {
	resp, err := engine.Generate(ctx, &Request{
		Messages: []Message{
			NewTextMessage(RoleUser, prompt),
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// Chat 简单对话（保留上下文）
func Chat(ctx context.Context, engine Engine, messages []Message) (*Response, error) {
	return engine.Generate(ctx, &Request{
		Messages: messages,
	})
}

func extractJSON(s string) string {
	if s == "" {
		return ""
	}
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if s[0] == '{' || s[0] == '[' {
		return s
	}

	if strings.HasPrefix(s, "```") {
		firstNewline := strings.Index(s, "\n")
		if firstNewline >= 0 {
			s = s[firstNewline+1:]
		}
		if idx := strings.LastIndex(s, "```"); idx >= 0 {
			s = s[:idx]
		}
		s = strings.TrimSpace(s)
		if s != "" && (s[0] == '{' || s[0] == '[') {
			return s
		}
	}

	if idx := strings.Index(s, "{"); idx >= 0 {
		return extractBalanced(s[idx:], '{', '}')
	}
	if idx := strings.Index(s, "["); idx >= 0 {
		return extractBalanced(s[idx:], '[', ']')
	}
	return s
}

func extractBalanced(s string, open, close byte) string {
	depth := 0
	inStr := false
	escape := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escape {
			escape = false
			continue
		}
		if c == '\\' && inStr {
			escape = true
			continue
		}
		if c == '"' {
			inStr = !inStr
			continue
		}
		if inStr {
			continue
		}
		if c == open {
			depth++
		} else if c == close {
			depth--
			if depth == 0 {
				return s[:i+1]
			}
		}
	}
	return s
}

func deepCopyAny(v any) any {
	if v == nil {
		return nil
	}
	data, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var result any
	if err := json.Unmarshal(data, &result); err != nil {
		return v
	}
	return result
}

func cloneFloat64Ptr(p *float64) *float64 {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

type toolCallAccum struct {
	id        string
	typ       string
	name      string
	arguments strings.Builder
}

func mergeToolCallDelta(m map[int]*toolCallAccum, idx int, tc APIToolCall) {
	acc, ok := m[idx]
	if !ok {
		acc = &toolCallAccum{}
		m[idx] = acc
	}
	if tc.ID != "" {
		acc.id = tc.ID
	}
	if tc.Type != "" {
		acc.typ = tc.Type
	}
	if tc.Name != "" {
		acc.name = tc.Name
	}
	acc.arguments.WriteString(tc.Arguments)
}

func resolveToolCallIndex(ordinal int, tc APIToolCall) int {
	if tc.Index > 0 {
		return tc.Index
	}
	return ordinal
}

func sortedToolCallIndexes(m map[int]*toolCallAccum) []int {
	indexes := make([]int, 0, len(m))
	for index := range m {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	return indexes
}
