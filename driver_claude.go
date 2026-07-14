package slm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
)

var claudeChatCapabilitiesTemplate = ProtocolCapabilities{
	SupportedParameters: map[string]ParameterRange{
		"thinking_budget": {
			Min: 1024,
			Max: 10000,
		},
	},
	ParameterMapping: map[string]string{
		"thinking_budget": "budget_tokens",
	},
	ConflictingParameters: [][]string{},
	Description:           "Anthropic Claude Messages API (supports claude-3-5, claude-3, etc.)",
}

type claudeEngine struct {
	adapter *genericAdapter[*Request, *Response, StreamIterator]
}

func newClaudeEngine(transport Transport, defaultModel string) Engine {
	base := protocolBase{transport: transport, defaultModel: defaultModel}
	codec := claudeCodecInst
	return &claudeEngine{adapter: newClaudeChatAdapter(base, codec)}
}

func (e *claudeEngine) Generate(ctx context.Context, req *Request) (*Response, error) {
	return e.adapter.generate(ctx, req)
}

func (e *claudeEngine) Stream(ctx context.Context, req *Request) (StreamIterator, error) {
	return e.adapter.stream(ctx, req)
}

func (e *claudeEngine) Capabilities() *ProtocolCapabilities {
	return cloneProtocolCapabilities(claudeChatCapabilitiesTemplate)
}

func newClaudeChatAdapter(base protocolBase, codec *claudeCodec) *genericAdapter[*Request, *Response, StreamIterator] {
	return &genericAdapter[*Request, *Response, StreamIterator]{
		base:     base,
		path:     "/messages",
		validate: validateChatRequest,
		buildBody: func(req *Request, model string, stream bool) ([]byte, error) {
			return codec.BuildChatRequestBody(req, model, stream)
		},
		decodeResp: func(resp *http.Response) (*Response, error) {
			var claudeResp claudeResponse
			if err := decodeJSONResponse(resp, &claudeResp); err != nil {
				return nil, err
			}
			return codec.ConvertChatResponse(&claudeResp), nil
		},
		createStream: func(resp *http.Response) (StreamIterator, error) {
			return newClaudeStreamIterator(resp, codec), nil
		},
	}
}

// -------------------------------------------------------------------------
// Claude Chat Wire Types
// -------------------------------------------------------------------------

type claudeTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

type claudeImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type claudeContent struct {
	Type      string             `json:"type"`
	Text      string             `json:"text,omitempty"`
	Thinking  string             `json:"thinking,omitempty"`
	Signature string             `json:"signature,omitempty"` // Anthropic extended thinking signature (required for replay)
	ID        string             `json:"id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Input     any                `json:"input,omitempty"`
	Source    *claudeImageSource `json:"source,omitempty"`
	ToolUseID string             `json:"tool_use_id,omitempty"`
	Content   string             `json:"content,omitempty"`
}

type claudeMessage struct {
	Role    string          `json:"role"`
	Content []claudeContent `json:"content"`
}

type claudeChoice struct {
	Content      []claudeContent `json:"content"`
	StopReason   string          `json:"stop_reason"`
	StopSequence *string         `json:"stop_sequence,omitempty"`
}

type claudeResponse struct {
	ID         string          `json:"id"`
	Type       string          `json:"type"`
	Role       string          `json:"role"`
	Content    []claudeContent `json:"content"`
	Model      string          `json:"model"`
	Usage      *Usage          `json:"usage,omitempty"`
	StopReason string          `json:"stop_reason"`
}

type claudeStreamEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index,omitempty"`
	Message      *claudeResponse `json:"message,omitempty"`
	ContentBlock *claudeContent  `json:"content_block,omitempty"`
	Delta        *claudeDelta    `json:"delta,omitempty"`
}

type claudeDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
}

// -------------------------------------------------------------------------
// Claude Stream Handler
// -------------------------------------------------------------------------

type claudeStreamIterator struct {
	resp              *http.Response
	framer            *sseFrameReader
	codec             *claudeCodec
	chunk             []byte
	fullText          string
	fullReasoning     string
	thinkingSignature string
	usage             *Usage
	current           *Response
	err               error
	done              bool
}

func newClaudeStreamIterator(resp *http.Response, codec *claudeCodec) *claudeStreamIterator {
	return &claudeStreamIterator{
		resp:   resp,
		framer: newSSEFrameReader(resp.Body),
		codec:  codec,
	}
}

func (s *claudeStreamIterator) Next() bool {
	if s.done || s.err != nil {
		return false
	}

	for {
		result := consumeSSEFrame(s.framer)
		if result.Done {
			s.done = true
			return false
		}
		if result.Err != nil {
			wrapped := WrapOperationalError("stream read error", result.Err)
			var llmErr *LLMError
			if errors.As(wrapped, &llmErr) && (llmErr.Code == ErrCodeTimeout || llmErr.Code == ErrCodeCancelled || llmErr.Code == ErrCodeNetwork) {
				s.err = wrapped
			} else {
				s.err = NewLLMError(ErrCodeParse, result.Err.Error(), nil)
			}
			return false
		}
		if s.dispatch(result.Frame) {
			return true
		}
		if s.done || s.err != nil {
			return false
		}
	}
}

func (s *claudeStreamIterator) dispatch(frame sseFrame) bool {
	var event claudeStreamEvent
	if err := json.Unmarshal(frame.Data, &event); err != nil {
		s.err = NewLLMError(ErrCodeParse, "parse stream event", err)
		return false
	}

	switch event.Type {
	case "message_start":
		if event.Message != nil && event.Message.Usage != nil {
			s.usage = event.Message.Usage
		}
		return false

	case "content_block_delta":
		if event.Delta == nil {
			return false
		}
		s.chunk = nil
		s.current = &Response{}

		switch event.Delta.Type {
		case "text_delta":
			s.chunk = []byte(event.Delta.Text)
			s.fullText += event.Delta.Text
			s.current.Content = event.Delta.Text
		case "input_json_delta":
			s.chunk = []byte(event.Delta.PartialJSON)
			s.current.Content = event.Delta.PartialJSON
		case "thinking_delta":
			s.chunk = []byte(event.Delta.Thinking)
			s.fullReasoning += event.Delta.Thinking
			s.current.ReasoningContent = event.Delta.Thinking
		case "signature_delta":
			s.thinkingSignature += event.Delta.Signature
			s.current.ThinkingSignature = event.Delta.Signature
		}
		return len(s.chunk) > 0

	case "message_delta":
		if event.Message != nil && event.Message.Usage != nil {
			s.usage = event.Message.Usage
		}
		return false

	case "message_stop":
		s.done = true
		return false

	default:
		return false
	}
}

func (s *claudeStreamIterator) Chunk() []byte      { return s.chunk }
func (s *claudeStreamIterator) Text() string       { return s.current.Content }
func (s *claudeStreamIterator) FullText() string   { return s.fullText }
func (s *claudeStreamIterator) Current() *Response { return s.current }
func (s *claudeStreamIterator) Err() error         { return s.err }
func (s *claudeStreamIterator) Usage() *Usage      { return s.usage }

func (s *claudeStreamIterator) Response() *Response {
	if s.current == nil {
		if s.usage != nil {
			return &Response{Usage: *s.usage}
		}
		return &Response{}
	}
	if s.usage != nil && s.current.Usage == (Usage{}) {
		s.current.Usage = *s.usage
	}
	// Populate accumulated reasoning content and signature from the full stream
	if s.fullReasoning != "" && s.current.ReasoningContent == "" {
		s.current.ReasoningContent = s.fullReasoning
	}
	if s.thinkingSignature != "" && s.current.ThinkingSignature == "" {
		s.current.ThinkingSignature = s.thinkingSignature
	}
	return s.current
}

func (s *claudeStreamIterator) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}

// -------------------------------------------------------------------------
// Claude Codec
// -------------------------------------------------------------------------

var claudeCodecInst = &claudeCodec{}

type claudeCodec struct{}

func convertMessagesToClaude(messages []Message) []claudeMessage {
	if len(messages) == 0 {
		return nil
	}

	var result []claudeMessage
	for _, msg := range messages {
		cm := claudeMessage{Role: string(msg.Role)}

		for _, part := range msg.Content {
			switch p := part.(type) {
			case ThinkingPart:
				// Emit thinking block only when signature is present —
				// Anthropic requires the signature for replay; sending
				// a thinking block without it causes an API error.
				if p.Signature != "" {
					cm.Content = append(cm.Content, claudeContent{
						Type:      "thinking",
						Thinking:  p.Content,
						Signature: p.Signature,
					})
				}
			case TextPart:
				cm.Content = append(cm.Content, claudeContent{
					Type: "text",
					Text: string(p),
				})
			case ImagePart:
				if p.Base64 != "" {
					cm.Content = append(cm.Content, claudeContent{
						Type: "image",
						Source: &claudeImageSource{
							Type:      "base64",
							MediaType: DefaultString(p.MIME, "image/png"),
							Data:      p.Base64,
						},
					})
				}
			}
		}

		for _, tc := range msg.ToolCalls {
			var input any
			if err := json.Unmarshal([]byte(tc.Arguments), &input); err != nil {
				input = tc.Arguments
			}
			cm.Content = append(cm.Content, claudeContent{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Name,
				Input: input,
			})
		}

		if len(cm.Content) > 0 {
			result = append(result, cm)
		}
	}

	return result
}

func convertToolsToClaudeSchema(tools []Tool) []claudeTool {
	if len(tools) == 0 {
		return nil
	}

	var result []claudeTool
	for _, t := range tools {
		schema := t.Parameters
		if schema == nil {
			schema = map[string]any{}
		}
		result = append(result, claudeTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: toMap(schema),
		})
	}
	return result
}

func (c *claudeCodec) BuildChatRequestBody(req *Request, model string, stream bool) ([]byte, error) {
	messages := convertMessagesToClaude(req.Messages)

	reqMap := map[string]any{
		"model":    model,
		"messages": messages,
	}

	if stream {
		reqMap["stream"] = true
	}

	putFloat64PtrField(reqMap, "temperature", req.Temperature)
	putFloat64PtrField(reqMap, "top_p", req.TopP)
	putPositiveIntField(reqMap, "max_tokens", req.MaxTokens)
	putSliceField(reqMap, "stop_sequences", req.Stop)

	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		thinking := map[string]any{"type": "enabled"}
		if val, ok := req.Capabilities["thinking_budget"]; ok {
			thinking["budget_tokens"] = val
		}
		putAnyField(reqMap, "thinking", thinking)
	}

	putAnyField(reqMap, "tools", convertToolsToClaudeSchema(req.Tools))

	for k, v := range req.Capabilities {
		if k != "thinking_budget" {
			reqMap[k] = v
		}
	}

	return json.Marshal(reqMap)
}

func (c *claudeCodec) ConvertChatResponse(resp any) *Response {
	claudeResp, ok := resp.(*claudeResponse)
	if !ok {
		return &Response{}
	}

	content := ""
	reasoningContent := ""
	thinkingSignature := ""
	var toolCalls []APIToolCall

	for _, block := range claudeResp.Content {
		switch block.Type {
		case "text":
			content += block.Text
		case "thinking":
			reasoningContent += block.Thinking
			if block.Signature != "" {
				thinkingSignature = block.Signature
			}
		case "tool_use":
			args, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, APIToolCall{
				ID:        block.ID,
				Type:      "function",
				Name:      block.Name,
				Arguments: string(args),
			})
		}
	}

	usage := Usage{}
	if claudeResp.Usage != nil {
		usage = *claudeResp.Usage
	}

	finishReason := claudeResp.StopReason
	if claudeResp.StopReason == "stop_sequence" {
		finishReason = "stop_sequence"
	}

	return &Response{
		Content:           content,
		ReasoningContent:  reasoningContent,
		ThinkingSignature: thinkingSignature,
		FinishReason:      finishReason,
		ToolCalls:         toolCalls,
		Usage:             usage,
	}
}
