package openai

import "ojv/slm"

// Model is an OpenAI-compatible model listing item used at API boundaries.
// Pointer fields keep JSON output compact by omitting empty capabilities/limits.
type Model struct {
	ID                 string             `json:"id"`
	Object             string             `json:"object"`
	OwnedBy            string             `json:"owned_by"`
	SupportedEndpoints []string           `json:"supported_endpoints,omitempty"`
	Capabilities       *slm.CapabilitySet `json:"capabilities,omitempty"`
	Limits             *slm.ModelLimits   `json:"limits,omitempty"`
	Metadata           map[string]any     `json:"metadata,omitempty"`
}

type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}
