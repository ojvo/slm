package slm

import (
	"encoding/json"
)

// jsonContentPart is the JSON representation of a ContentPart.
type jsonContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *jsonImageURL `json:"image_url,omitempty"`
}

type jsonImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// jsonMessage is the JSON intermediate representation of Message.
// Content can be string, []jsonContentPart, or nil.
// reasoning_content and thinking_signature are serialized at the top level
// for compatibility with the colm format and DeepSeek API expectations.
type jsonMessage struct {
	Role              string        `json:"role"`
	Content           any           `json:"content"`
	Name              string        `json:"name,omitempty"`
	ToolCalls         []APIToolCall `json:"tool_calls,omitempty"`
	ToolCallID        string        `json:"tool_call_id,omitempty"`
	ReasoningContent  any           `json:"reasoning_content,omitempty"`
	ThinkingSignature string        `json:"thinking_signature,omitempty"`
}

// MarshalJSON implements json.Marshaler, serializing ContentPart interface
// into OpenAI-compatible format. Pure-text messages serialize content as a
// string; messages with images serialize as an array. ThinkingPart content
// is serialized via the top-level reasoning_content/thinking_signature fields.
func (m Message) MarshalJSON() ([]byte, error) {
	var content any
	hasImage := false
	for _, part := range m.Content {
		if _, ok := part.(ImagePart); ok {
			hasImage = true
			break
		}
	}

	switch {
	case len(m.Content) == 0:
		if m.Role == RoleAssistant {
			content = ""
		} else {
			content = nil
		}
	case hasImage:
		parts := make([]jsonContentPart, 0, len(m.Content))
		for _, part := range m.Content {
			switch p := part.(type) {
			case TextPart:
				parts = append(parts, jsonContentPart{Type: "text", Text: string(p)})
			case ImagePart:
				url := p.URL
				if url == "" {
					url = p.Base64
				}
				parts = append(parts, jsonContentPart{Type: "image_url", ImageURL: &jsonImageURL{URL: url, Detail: p.Detail}})
			}
		}
		content = parts
	default:
		var text string
		for _, part := range m.Content {
			if t, ok := part.(TextPart); ok {
				text += string(t)
			}
		}
		content = text
	}

	// Extract thinking content for top-level fields.
	var reasoningContent any
	thinkingText := m.ReasoningContent()
	thinkingSig := m.ThinkingSignature()
	switch {
	case thinkingText != "":
		reasoningContent = thinkingText
	case m.Role == RoleAssistant:
		reasoningContent = ""
	default:
		reasoningContent = nil
	}

	return json.Marshal(jsonMessage{
		Role:              string(m.Role),
		Content:           content,
		Name:              m.Name,
		ToolCalls:         m.ToolCalls,
		ToolCallID:        m.ToolCallID,
		ReasoningContent:  reasoningContent,
		ThinkingSignature: thinkingSig,
	})
}

// UnmarshalJSON implements json.Unmarshaler, restoring Message from JSON.
// Supports content as string, []jsonContentPart, or nil.
func (m *Message) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role              string          `json:"role"`
		Content           json.RawMessage `json:"content"`
		Name              string          `json:"name"`
		ToolCalls         []APIToolCall   `json:"tool_calls"`
		ToolCallID        string          `json:"tool_call_id"`
		ReasoningContent  string          `json:"reasoning_content"`
		ThinkingSignature string          `json:"thinking_signature"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	m.Role = Role(raw.Role)
	m.Name = raw.Name
	m.ToolCalls = raw.ToolCalls
	m.ToolCallID = raw.ToolCallID

	if len(raw.Content) == 0 || string(raw.Content) == "null" {
		if raw.ReasoningContent != "" || raw.ThinkingSignature != "" {
			m.SetReasoningContent(raw.ReasoningContent, raw.ThinkingSignature)
		}
		return nil
	}

	// Try to parse content as string.
	if raw.Content[0] == '"' {
		var text string
		if err := json.Unmarshal(raw.Content, &text); err != nil {
			return err
		}
		m.Content = []ContentPart{TextPart(text)}
		if raw.ReasoningContent != "" || raw.ThinkingSignature != "" {
			m.SetReasoningContent(raw.ReasoningContent, raw.ThinkingSignature)
		}
		return nil
	}

	// Parse as []jsonContentPart.
	var parts []jsonContentPart
	if err := json.Unmarshal(raw.Content, &parts); err != nil {
		return err
	}
	m.Content = make([]ContentPart, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case "text":
			m.Content = append(m.Content, TextPart(p.Text))
		case "image_url":
			if p.ImageURL != nil {
				img := ImagePart{Detail: p.ImageURL.Detail}
				if len(p.ImageURL.URL) > 5 && p.ImageURL.URL[:5] == "data:" {
					img.Base64 = p.ImageURL.URL
				} else {
					img.URL = p.ImageURL.URL
				}
				m.Content = append(m.Content, img)
			}
		}
	}
	if raw.ReasoningContent != "" || raw.ThinkingSignature != "" {
		m.SetReasoningContent(raw.ReasoningContent, raw.ThinkingSignature)
	}
	return nil
}
