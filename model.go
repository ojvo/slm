package slm

import (
	"context"
	"encoding/json"
	"io"
	"time"
)

type ModelLimits struct {
	MaxContextWindowTokens      int `json:"max_context_window_tokens,omitempty"`
	MaxOutputTokens             int `json:"max_output_tokens,omitempty"`
	MaxNonStreamingOutputTokens int `json:"max_non_streaming_output_tokens,omitempty"`
	MaxPromptTokens             int `json:"max_prompt_tokens,omitempty"`
}

func (l ModelLimits) Any() bool {
	return l.MaxContextWindowTokens > 0 || l.MaxOutputTokens > 0 || l.MaxNonStreamingOutputTokens > 0 || l.MaxPromptTokens > 0
}

func (l ModelLimits) EffectiveMaxOutputTokens(streaming bool) int {
	if streaming && l.MaxNonStreamingOutputTokens > 0 && l.MaxOutputTokens == 0 {
		return l.MaxNonStreamingOutputTokens
	}
	return l.MaxOutputTokens
}

type Model struct {
	ID                 string         `json:"id"`
	Object             string         `json:"object"`
	OwnedBy            string         `json:"owned_by"`
	SupportedEndpoints []string       `json:"supported_endpoints,omitempty"`
	Capabilities       CapabilitySet  `json:"capabilities,omitempty"`
	Limits             ModelLimits    `json:"limits,omitempty"`
	Meta               map[string]any `json:"metadata,omitempty"`
	Created            int64          `json:"created,omitempty"`
}

func (m Model) ToCapabilities() ModelCapabilities {
	return ModelCapabilities{
		Model:    m.ID,
		Supports: m.Capabilities,
		Limits:   m.Limits,
		Meta:     cloneMap(m.Meta),
	}
}

type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

func ParseModelsResponse(data []byte) (ModelsResponse, error) {
	var resp ModelsResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return ModelsResponse{}, err
	}
	return resp, nil
}

func (r ModelsResponse) ToCapabilities() []ModelCapabilities {
	if len(r.Data) == 0 {
		return nil
	}
	caps := make([]ModelCapabilities, len(r.Data))
	for i, m := range r.Data {
		caps[i] = m.ToCapabilities()
	}
	return caps
}

type ModelInfo struct {
	ID           string
	OwnedBy      string
	Capabilities CapabilitySet
	Limits       ModelLimits
	Endpoints    []string
	CreatedAt    time.Time
}

func (m Model) Info() ModelInfo {
	var createdAt time.Time
	if m.Created > 0 {
		createdAt = time.Unix(m.Created, 0)
	}
	return ModelInfo{
		ID:           m.ID,
		OwnedBy:      m.OwnedBy,
		Capabilities: m.Capabilities,
		Limits:       m.Limits,
		Endpoints:    cloneStringSlice(m.SupportedEndpoints),
		CreatedAt:    createdAt,
	}
}

func ModelsResponseToCatalogLoader(resp ModelsResponse) CapabilityCatalogLoader {
	caps := resp.ToCapabilities()
	return func(ctx context.Context) ([]ModelCapabilities, error) {
		return caps, nil
	}
}

func FetchModelsCatalog(ctx context.Context, transport Transport, baseURL string) (ModelsResponse, error) {
	resp, err := transport.Do(ctx, "GET", baseURL+"/models", nil, nil)
	if err != nil {
		return ModelsResponse{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return ModelsResponse{}, NewLLMError(ErrCodeServer, "failed to fetch models catalog", nil)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return ModelsResponse{}, WrapOperationalError("read models response", err)
	}
	return ParseModelsResponse(data)
}
