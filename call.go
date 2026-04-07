package slm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// Call 泛型调用：直接请求并解析结果为结构体 T
func Call[T any](ctx context.Context, engine Engine, req *Request) (*T, error) {
	clone := cloneRequest(req)
	clone.JSONMode = true
	resp, err := engine.Generate(ctx, clone)
	if err != nil {
		return nil, err
	}

	var result T
	content := strings.TrimSpace(resp.Content)
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
	clone := cloneRequest(req)
	clone.JSONMode = true
	clone.Stream = true

	iter, err := engine.Stream(ctx, clone)
	if err != nil {
		return nil, err
	}
	defer iter.Close()

	var content strings.Builder
	for iter.Next() {
		content.WriteString(iter.Text())
	}

	if err := iter.Err(); err != nil {
		return nil, err
	}

	var result T
	text := strings.TrimSpace(content.String())
	if text == "" {
		return nil, fmt.Errorf("empty response from LLM")
	}
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	return &result, nil
}

func cloneRequest(req *Request) *Request {
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
	copy(tools, req.Tools)

	var extraBody map[string]any
	if len(req.ExtraBody) > 0 {
		extraBody = make(map[string]any, len(req.ExtraBody))
		for k, v := range req.ExtraBody {
			extraBody[k] = v
		}
	}

	var meta map[string]any
	if len(req.Meta) > 0 {
		meta = make(map[string]any, len(req.Meta))
		for k, v := range req.Meta {
			meta[k] = v
		}
	}

	return &Request{
		Model:            req.Model,
		Messages:         msgs,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		MaxTokens:        req.MaxTokens,
		Stop:             append([]string(nil), req.Stop...),
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
		Stream:           req.Stream,
		JSONMode:         req.JSONMode,
		Tools:            tools,
		Meta:             meta,
		ExtraBody:        extraBody,
	}
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
