package slm

type ProtocolType string

const (
	ProtocolOpenAI ProtocolType = "openai"
	ProtocolClaude ProtocolType = "claude"
)

func NewEngine(protocol ProtocolType, transport Transport, defaultModel string) Engine {
	switch protocol {
	case ProtocolClaude:
		return newClaudeEngine(transport, defaultModel)
	default:
		return newOpenAIEngine(transport, defaultModel)
	}
}

func NewEngineWithEndpoint(protocol ProtocolType, baseURL, apiKey, defaultModel string) Engine {
	return NewEngine(protocol, NewHTTPTransport(baseURL, apiKey), defaultModel)
}

func NewResponsesEngine(protocol ProtocolType, transport Transport, defaultModel string) ResponsesEngine {
	switch protocol {
	case ProtocolOpenAI:
		return newOpenAIResponsesEngine(transport, defaultModel)
	default:
		return nil
	}
}

func NewResponsesEngineWithEndpoint(protocol ProtocolType, baseURL, apiKey, defaultModel string) ResponsesEngine {
	return NewResponsesEngine(protocol, NewHTTPTransport(baseURL, apiKey), defaultModel)
}
