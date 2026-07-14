package slm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"

	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestNegotiateRequestCapabilities_Idempotent(t *testing.T) {
	var calls atomic.Int32
	resolver := CapabilityResolverFunc(func(ctx context.Context, model string) (ModelCapabilities, bool, error) {
		_ = ctx
		calls.Add(1)
		return ModelCapabilities{
			Model: model,
			Supports: CapabilitySet{
				JSONMode:  true,
				ToolCalls: true,
				Vision:    true,
				Reasoning: true,
			},
		}, true, nil
	})

	opts := CapabilityNegotiationOptions{
		Resolver:     resolver,
		DefaultModel: "gpt-4o-mini",
		RequireKnown: true,
	}

	req := &Request{
		Messages: []Message{NewTextMessage(RoleUser, "hello")},
		JSONMode: true,
	}

	ctx, req, err := NegotiateRequestCapabilities(context.Background(), req, opts)
	if err != nil {
		t.Fatalf("first negotiation failed: %v", err)
	}
	if req.Model == "" {
		t.Fatalf("first negotiation should inject model")
	}

	_, req, err = NegotiateRequestCapabilities(ctx, req, opts)
	if err != nil {
		t.Fatalf("second negotiation failed: %v", err)
	}
	if req.Model == "" {
		t.Fatalf("second negotiation should preserve model")
	}

	if got := calls.Load(); got != 1 {
		t.Fatalf("resolver called %d times, want 1", got)
	}
}

func TestNegotiateResponseCapabilities_Idempotent(t *testing.T) {
	var calls atomic.Int32
	resolver := CapabilityResolverFunc(func(ctx context.Context, model string) (ModelCapabilities, bool, error) {
		_ = ctx
		calls.Add(1)
		return ModelCapabilities{
			Model: model,
			Supports: CapabilitySet{
				ToolCalls: true,
				Reasoning: true,
			},
		}, true, nil
	})

	opts := CapabilityNegotiationOptions{
		Resolver:     resolver,
		DefaultModel: "gpt-4o-mini",
		RequireKnown: true,
	}

	req := &ResponseRequest{
		Input: []ResponseInputItem{{Role: "user", Content: "hello"}},
		Tools: []ResponseTool{{Type: "function"}},
	}

	ctx, req, err := NegotiateResponseCapabilities(context.Background(), req, opts)
	if err != nil {
		t.Fatalf("first negotiation failed: %v", err)
	}
	if req.Model == "" {
		t.Fatalf("first negotiation should inject model")
	}

	_, req, err = NegotiateResponseCapabilities(ctx, req, opts)
	if err != nil {
		t.Fatalf("second negotiation failed: %v", err)
	}
	if req.Model == "" {
		t.Fatalf("second negotiation should preserve model")
	}

	if got := calls.Load(); got != 1 {
		t.Fatalf("resolver called %d times, want 1", got)
	}
}

type negotiationVisibilityObserver struct {
	startVisible  bool
	finishVisible bool
}

func (o *negotiationVisibilityObserver) OnRequestStart(_ context.Context, event LifecycleEvent) {
	_, o.startVisible = GetNegotiatedCapabilities(event.Context)
}

func (o *negotiationVisibilityObserver) OnRequestFinish(_ context.Context, event LifecycleEvent) {
	_, o.finishVisible = GetNegotiatedCapabilities(event.Context)
}

func (o *negotiationVisibilityObserver) OnStreamStart(context.Context, LifecycleEvent)     {}
func (o *negotiationVisibilityObserver) OnStreamConnected(context.Context, LifecycleEvent) {}
func (o *negotiationVisibilityObserver) OnStreamFinish(context.Context, LifecycleEvent)    {}

func supportedJSONResolver() CapabilityResolver {
	return CapabilityResolverFunc(func(_ context.Context, model string) (ModelCapabilities, bool, error) {
		return ModelCapabilities{
			Model: model,
			Supports: CapabilitySet{
				JSONMode: true,
			},
		}, true, nil
	})
}

func TestCapabilityNegotiationVisibility_RecommendedOrder(t *testing.T) {
	observer := &negotiationVisibilityObserver{}
	opts := CapabilityNegotiationOptions{
		Resolver:     supportedJSONResolver(),
		DefaultModel: "gpt-4o-mini",
		RequireKnown: true,
	}

	engine := ApplyStandardMiddleware(stubEngine{}, StandardMiddlewareOptions{
		DefaultModel: "gpt-4o-mini",
		Capabilities: &opts,
		Observers:    []LifecycleObserver{observer},
	})

	_, err := engine.Generate(context.Background(), &Request{
		Model:    "gpt-4o-mini",
		JSONMode: true,
		Messages: []Message{NewTextMessage(RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}

	if !observer.startVisible {
		t.Fatalf("expected negotiated capabilities visible at request start in recommended order")
	}
	if !observer.finishVisible {
		t.Fatalf("expected negotiated capabilities visible at request finish in recommended order")
	}
}

func TestCapabilityNegotiationVisibility_ObserverBeforeCapability(t *testing.T) {
	observer := &negotiationVisibilityObserver{}
	opts := CapabilityNegotiationOptions{
		Resolver:     supportedJSONResolver(),
		DefaultModel: "gpt-4o-mini",
		RequireKnown: true,
	}
	capUnary, _ := CapabilityNegotiationMiddleware(opts)
	obsUnary, _ := LifecycleObserverMiddleware(observer)

	engine := ChainWithStreamAndClosers(stubEngine{}, []Middleware{obsUnary, capUnary}, nil, nil)

	_, err := engine.Generate(context.Background(), &Request{
		Model:    "gpt-4o-mini",
		JSONMode: true,
		Messages: []Message{NewTextMessage(RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}

	if observer.startVisible {
		t.Fatalf("expected negotiated capabilities hidden at request start when observer runs before capability middleware")
	}
	if observer.finishVisible {
		t.Fatalf("expected negotiated capabilities hidden at request finish when observer runs before capability middleware")
	}
}

type noopTransport struct{}

func (noopTransport) Do(context.Context, string, string, map[string]string, []byte) (*http.Response, error) {
	return nil, nil
}

func TestBuildEngineWithTransport_CapabilityNegotiationUsesMiddlewarePath(t *testing.T) {
	resolver := CapabilityResolverFunc(func(context.Context, string) (ModelCapabilities, bool, error) {
		return ModelCapabilities{Supports: CapabilitySet{JSONMode: true}}, true, nil
	})

	cfg := DefaultConfig().
		WithTransport(noopTransport{}).
		WithProvider(ProviderConfig{DefaultModel: "gpt-4o-mini"}).
		WithCapabilityNegotiation(CapabilityNegotiationOptions{Resolver: resolver, RequireKnown: true})

	engine, err := cfg.BuildEngineWithTransport()
	if err != nil {
		t.Fatalf("build engine with transport: %v", err)
	}

	chain, ok := engine.(*enginePipelineBridge)
	if !ok {
		t.Fatalf("engine type = %T, want *enginePipelineBridge", engine)
	}

	pipeline, ok := chain.pipeline.(*genericPipelineEngine[*Request, *Response, StreamIterator])
	if !ok {
		t.Fatalf("pipeline type = %T, want *genericPipelineEngine", chain.pipeline)
	}

	adapter, ok := pipeline.inner.(enginePipelineAdapter)
	if !ok {
		t.Fatalf("inner type = %T, want enginePipelineAdapter", pipeline.inner)
	}

	_, ok = adapter.Engine.(*openAIEngine)
	if !ok {
		t.Fatalf("inner engine type = %T, want *openAIEngine", adapter.Engine)
	}

	if len(pipeline.middlewares) == 0 {
		t.Fatalf("expected middleware chain to contain capability negotiation middleware")
	}
}
func TestModelLimits_Any(t *testing.T) {
	tests := []struct {
		name     string
		limits   ModelLimits
		expected bool
	}{
		{"empty", ModelLimits{}, false},
		{"max_context", ModelLimits{MaxContextWindowTokens: 1000}, true},
		{"max_output", ModelLimits{MaxOutputTokens: 100}, true},
		{"all", ModelLimits{MaxContextWindowTokens: 1000, MaxOutputTokens: 100, MaxNonStreamingOutputTokens: 50, MaxPromptTokens: 500}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.limits.Any(); got != tt.expected {
				t.Errorf("Any() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestModelLimits_EffectiveMaxOutputTokens(t *testing.T) {
	tests := []struct {
		name      string
		limits    ModelLimits
		streaming bool
		expected  int
	}{
		{"empty streaming", ModelLimits{}, true, 0},
		{"empty non-streaming", ModelLimits{}, false, 0},
		{"max_output only streaming", ModelLimits{MaxOutputTokens: 100}, true, 100},
		{"max_output only non-streaming", ModelLimits{MaxOutputTokens: 100}, false, 100},
		{"non_streaming_only streaming", ModelLimits{MaxNonStreamingOutputTokens: 50}, true, 50},
		{"non_streaming_only non-streaming", ModelLimits{MaxNonStreamingOutputTokens: 50}, false, 0},
		{"both streaming", ModelLimits{MaxOutputTokens: 100, MaxNonStreamingOutputTokens: 50}, true, 100},
		{"both non-streaming", ModelLimits{MaxOutputTokens: 100, MaxNonStreamingOutputTokens: 50}, false, 100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.limits.EffectiveMaxOutputTokens(tt.streaming); got != tt.expected {
				t.Errorf("EffectiveMaxOutputTokens(%v) = %v, want %v", tt.streaming, got, tt.expected)
			}
		})
	}
}

func TestModel_ToCapabilities(t *testing.T) {
	model := Model{
		ID: "gpt-4o",
		Capabilities: CapabilitySet{
			JSONMode:  true,
			ToolCalls: true,
			Vision:    true,
			Reasoning: true,
		},
		Limits: ModelLimits{
			MaxContextWindowTokens: 128000,
			MaxOutputTokens:        16384,
		},
		Meta: map[string]any{"owner": "openai"},
	}

	caps := model.ToCapabilities()

	if caps.Model != model.ID {
		t.Errorf("Model = %v, want %v", caps.Model, model.ID)
	}
	if caps.Supports != model.Capabilities {
		t.Errorf("Supports = %v, want %v", caps.Supports, model.Capabilities)
	}
	if caps.Limits != model.Limits {
		t.Errorf("Limits = %v, want %v", caps.Limits, model.Limits)
	}
	if caps.Meta["owner"] != "openai" {
		t.Errorf("Meta = %v, want openai", caps.Meta)
	}
}

func TestModelsResponse_ToCapabilities(t *testing.T) {
	resp := ModelsResponse{
		Object: "list",
		Data: []Model{
			{ID: "gpt-4o", Capabilities: CapabilitySet{JSONMode: true}},
			{ID: "gpt-4o-mini", Capabilities: CapabilitySet{JSONMode: true, ToolCalls: true}},
		},
	}

	caps := resp.ToCapabilities()

	if len(caps) != 2 {
		t.Fatalf("len = %v, want 2", len(caps))
	}
	if caps[0].Model != "gpt-4o" {
		t.Errorf("caps[0].Model = %v, want gpt-4o", caps[0].Model)
	}
	if caps[1].Model != "gpt-4o-mini" {
		t.Errorf("caps[1].Model = %v, want gpt-4o-mini", caps[1].Model)
	}
}

func TestParseModelsResponse(t *testing.T) {
	data := `{"object":"list","data":[{"id":"gpt-4o","object":"model","owned_by":"openai","capabilities":{"json_mode":true,"tool_calls":true},"limits":{"max_context_window_tokens":128000}}]}`

	resp, err := ParseModelsResponse([]byte(data))
	if err != nil {
		t.Fatalf("ParseModelsResponse() error = %v", err)
	}

	if resp.Object != "list" {
		t.Errorf("Object = %v, want list", resp.Object)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("len(Data) = %v, want 1", len(resp.Data))
	}
	if resp.Data[0].ID != "gpt-4o" {
		t.Errorf("ID = %v, want gpt-4o", resp.Data[0].ID)
	}
	if !resp.Data[0].Capabilities.JSONMode {
		t.Error("Capabilities.JSONMode = false, want true")
	}
	if resp.Data[0].Limits.MaxContextWindowTokens != 128000 {
		t.Errorf("Limits.MaxContextWindowTokens = %v, want 128000", resp.Data[0].Limits.MaxContextWindowTokens)
	}
}

func TestModelsResponseToCatalogLoader(t *testing.T) {
	resp := ModelsResponse{
		Data: []Model{
			{ID: "gpt-4o", Capabilities: CapabilitySet{JSONMode: true}},
		},
	}

	loader := ModelsResponseToCatalogLoader(resp)
	caps, err := loader(context.Background())
	if err != nil {
		t.Fatalf("loader() error = %v", err)
	}

	if len(caps) != 1 {
		t.Fatalf("len(caps) = %v, want 1", len(caps))
	}
	if caps[0].Model != "gpt-4o" {
		t.Errorf("Model = %v, want gpt-4o", caps[0].Model)
	}
}

func TestCatalogCapabilityResolver_ListModelCapabilities(t *testing.T) {
	resolver := NewCatalogCapabilityResolver(func(ctx context.Context) ([]ModelCapabilities, error) {
		return []ModelCapabilities{
			{Model: "gpt-4o", Supports: CapabilitySet{JSONMode: true}, Limits: ModelLimits{MaxOutputTokens: 16384}},
			{Model: "gpt-4o-mini", Supports: CapabilitySet{JSONMode: true, ToolCalls: true}},
		}, nil
	}, CapabilityCatalogResolverOptions{})

	caps, state, err := resolver.ListModelCapabilities(context.Background())
	if err != nil {
		t.Fatalf("ListModelCapabilities() error = %v", err)
	}

	if len(caps) != 2 {
		t.Fatalf("len(caps) = %v, want 2", len(caps))
	}
	if caps[0].Model != "gpt-4o" {
		t.Errorf("caps[0].Model = %v, want gpt-4o", caps[0].Model)
	}
	if caps[0].Limits.MaxOutputTokens != 16384 {
		t.Errorf("caps[0].Limits.MaxOutputTokens = %v, want 16384", caps[0].Limits.MaxOutputTokens)
	}
	if state.Source != CapabilitySourceCatalog {
		t.Errorf("state.Source = %v, want %v", state.Source, CapabilitySourceCatalog)
	}
}

func TestCatalogCapabilityResolver_ListModelCapabilitiesReturnsDefensiveCopy(t *testing.T) {
	resolver := NewCatalogCapabilityResolver(func(ctx context.Context) ([]ModelCapabilities, error) {
		return []ModelCapabilities{
			{Model: "gpt-4o", Supports: CapabilitySet{JSONMode: true}},
		}, nil
	}, CapabilityCatalogResolverOptions{})

	caps1, _, _ := resolver.ListModelCapabilities(context.Background())
	caps1[0].Model = "modified"

	caps2, _, _ := resolver.ListModelCapabilities(context.Background())

	if caps2[0].Model != "gpt-4o" {
		t.Errorf("catalog was mutated: caps2[0].Model = %v, want gpt-4o", caps2[0].Model)
	}
}

func TestCatalogCapabilityResolver_ResolveCapabilitiesWithLimits(t *testing.T) {
	resolver := NewCatalogCapabilityResolver(func(ctx context.Context) ([]ModelCapabilities, error) {
		return []ModelCapabilities{
			{
				Model:    "gpt-4o",
				Supports: CapabilitySet{JSONMode: true, Vision: true},
				Limits:   ModelLimits{MaxContextWindowTokens: 128000, MaxOutputTokens: 16384},
			},
		}, nil
	}, CapabilityCatalogResolverOptions{})

	caps, known, err := resolver.ResolveCapabilities(context.Background(), "gpt-4o")
	if err != nil {
		t.Fatalf("ResolveCapabilities() error = %v", err)
	}
	if !known {
		t.Fatal("known = false, want true")
	}
	if caps.Limits.MaxContextWindowTokens != 128000 {
		t.Errorf("Limits.MaxContextWindowTokens = %v, want 128000", caps.Limits.MaxContextWindowTokens)
	}
	if caps.Limits.MaxOutputTokens != 16384 {
		t.Errorf("Limits.MaxOutputTokens = %v, want 16384", caps.Limits.MaxOutputTokens)
	}
}

func TestCatalogCapabilityResolver_ImplementsCapabilityCatalog(t *testing.T) {
	var _ CapabilityCatalog = (*CatalogCapabilityResolver)(nil)
}

func TestModelCreatedAt(t *testing.T) {
	model := Model{
		ID:           "gpt-4o",
		OwnedBy:      "openai",
		Capabilities: CapabilitySet{JSONMode: true},
		Limits:       ModelLimits{MaxOutputTokens: 16384},
		Created:      1700000000,
	}

	if model.ID != "gpt-4o" {
		t.Errorf("ID = %v, want gpt-4o", model.ID)
	}
	if model.OwnedBy != "openai" {
		t.Errorf("OwnedBy = %v, want openai", model.OwnedBy)
	}
	if model.CreatedAt().Year() != 2023 {
		t.Errorf("CreatedAt().Year() = %v, want 2023", model.CreatedAt().Year())
	}
}

func TestCatalogCapabilityResolver_CacheTTL(t *testing.T) {
	var callCount int
	now := time.Now()
	resolver := NewCatalogCapabilityResolver(func(ctx context.Context) ([]ModelCapabilities, error) {
		callCount++
		return []ModelCapabilities{{Model: "gpt-4o"}}, nil
	}, CapabilityCatalogResolverOptions{
		CacheTTL: time.Hour,
		Now:      func() time.Time { return now },
	})

	resolver.ListModelCapabilities(context.Background())
	resolver.ListModelCapabilities(context.Background())

	if callCount != 1 {
		t.Errorf("callCount = %v, want 1 (should cache)", callCount)
	}

	now = now.Add(2 * time.Hour)
	resolver.ListModelCapabilities(context.Background())

	if callCount != 2 {
		t.Errorf("callCount = %v, want 2 (should refresh after TTL)", callCount)
	}
}
func TestResponseEvent_IsOutputTextDelta(t *testing.T) {
	tests := []struct {
		name  string
		type_ string
		want  bool
	}{
		{name: "canonical", type_: "response.output_text.delta", want: true},
		{name: "short", type_: "output_text.delta", want: true},
		{name: "prefixed", type_: "vendor.response.output_text.delta", want: true},
		{name: "trim + uppercase", type_: "  RESPONSE.OUTPUT_TEXT.DELTA  ", want: true},
		{name: "different event", type_: "response.completed", want: false},
		{name: "empty", type_: "", want: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := ResponseEvent{Type: tc.type_}
			if got := e.IsOutputTextDelta(); got != tc.want {
				t.Fatalf("IsOutputTextDelta() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNewResponseFunctionTool(t *testing.T) {
	base := Tool{
		Name:        "lookup",
		Description: "lookup by id",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id": map[string]any{"type": "string"},
			},
		},
	}

	got := NewResponseFunctionTool(base)
	if got.Type != "function" {
		t.Fatalf("Type = %q, want function", got.Type)
	}
	if got.Name != base.Name {
		t.Fatalf("Name = %q, want %q", got.Name, base.Name)
	}
	if got.Description != base.Description {
		t.Fatalf("Description = %q, want %q", got.Description, base.Description)
	}
	if got.Parameters == nil {
		t.Fatal("Parameters = nil, want copied value")
	}
}

func diagnosticFieldsToMap(t *testing.T, fields []any) map[string]any {
	t.Helper()
	if len(fields)%2 != 0 {
		t.Fatalf("diagnostic fields length = %d, want even", len(fields))
	}
	out := make(map[string]any, len(fields)/2)
	for i := 0; i < len(fields); i += 2 {
		key, ok := fields[i].(string)
		if !ok {
			t.Fatalf("fields[%d] key type = %T, want string", i, fields[i])
		}
		out[key] = fields[i+1]
	}
	return out
}

func TestResponseRequestDiagnosticFields(t *testing.T) {
	req := &ResponseRequest{
		Model: "gpt-4o-mini",
		Input: []ResponseInputItem{{
			Role: "user",
			Content: []ResponseInputContentPart{
				ResponseInputTextPart{Type: "input_text", Text: "hello"},
				ResponseInputImagePart{Type: "input_image", ImageURL: "https://example.com/a.png"},
			},
		}},
		Stream:          true,
		Store:           true,
		MaxOutputTokens: 256,
		Reasoning:       &ResponseReasoning{Effort: "medium", Summary: "auto"},
		Tools:           []ResponseTool{{Type: "function", Name: "lookup"}},
		Capabilities:    map[string]any{"trace": true},
	}

	fields := diagnosticFieldsToMap(t, ResponseRequestDiagnosticFields(req))
	if got := fields["model"]; got != "gpt-4o-mini" {
		t.Fatalf("model = %v, want gpt-4o-mini", got)
	}
	if got := fields["stream"]; got != true {
		t.Fatalf("stream = %v, want true", got)
	}
	if got := fields["input_items"]; got != 1 {
		t.Fatalf("input_items = %v, want 1", got)
	}
	if got := fields["vision_parts"]; got != 1 {
		t.Fatalf("vision_parts = %v, want 1", got)
	}
	if got := fields["tools"]; got != 1 {
		t.Fatalf("tools = %v, want 1", got)
	}
	if got := fields["reasoning"]; got != "medium" {
		t.Fatalf("reasoning = %v, want medium", got)
	}
	if got := fields["reasoning_summary"]; got != "auto" {
		t.Fatalf("reasoning_summary = %v, want auto", got)
	}
	if got := fields["max_output_tokens"]; got != 256 {
		t.Fatalf("max_output_tokens = %v, want 256", got)
	}
	if got := fields["store"]; got != true {
		t.Fatalf("store = %v, want true", got)
	}
	if got := fields["capabilities_keys"]; got != 1 {
		t.Fatalf("capabilities_keys = %v, want 1", got)
	}
}

func TestEventDiagnosticFields_PrefersResponseRequest(t *testing.T) {
	event := LifecycleEvent{
		Request: &Request{
			Model:    "chat-model",
			Messages: []Message{NewTextMessage(RoleUser, "chat")},
		},
		ResponseRequest: &ResponseRequest{
			Model: "resp-model",
			Input: []ResponseInputItem{{Role: "user", Content: "resp"}},
		},
	}

	fields := diagnosticFieldsToMap(t, eventDiagnosticFields(event))
	if got := fields["model"]; got != "resp-model" {
		t.Fatalf("model = %v, want resp-model", got)
	}
	if _, ok := fields["input_items"]; !ok {
		t.Fatal("missing input_items from response diagnostics")
	}
	if _, ok := fields["messages"]; ok {
		t.Fatal("unexpected chat diagnostic field messages when response request is present")
	}
}

func TestResponseEvent_CompletedResponse(t *testing.T) {
	t.Run("nil response", func(t *testing.T) {
		e := ResponseEvent{}
		if got := e.CompletedResponse(); got != nil {
			t.Fatalf("CompletedResponse() = %v, want nil", got)
		}
	})

	t.Run("completed normalizes nil output", func(t *testing.T) {
		resp := &ResponseObject{Status: "completed"}
		e := ResponseEvent{Response: resp}
		got := e.CompletedResponse()
		if got == nil {
			t.Fatalf("CompletedResponse() = nil, want object")
		}
		if got.Output == nil {
			t.Fatalf("CompletedResponse().Output = nil, want empty slice")
		}
		if len(got.Output) != 0 {
			t.Fatalf("CompletedResponse().Output len = %d, want 0", len(got.Output))
		}
	})

	t.Run("non-completed keeps nil output", func(t *testing.T) {
		resp := &ResponseObject{Status: "in_progress"}
		e := ResponseEvent{Response: resp}
		got := e.CompletedResponse()
		if got == nil {
			t.Fatalf("CompletedResponse() = nil, want object")
		}
		if got.Output != nil {
			t.Fatalf("CompletedResponse().Output = %v, want nil", got.Output)
		}
	})
}

func TestNormalizeCompletedResponseObject(t *testing.T) {
	t.Run("nil input", func(t *testing.T) {
		if got := normalizeCompletedResponseObject(nil); got != nil {
			t.Fatalf("normalizeCompletedResponseObject(nil) = %v, want nil", got)
		}
	})

	t.Run("completed with nil output", func(t *testing.T) {
		resp := &ResponseObject{Status: "completed"}
		got := normalizeCompletedResponseObject(resp)
		if got == nil {
			t.Fatalf("normalizeCompletedResponseObject() = nil, want object")
		}
		if got.Output == nil {
			t.Fatalf("Output = nil, want empty slice")
		}
	})

	t.Run("completed with output preserved", func(t *testing.T) {
		resp := &ResponseObject{Status: "completed", Output: []ResponseOutput{{Type: "message"}}}
		got := normalizeCompletedResponseObject(resp)
		if len(got.Output) != 1 {
			t.Fatalf("Output len = %d, want 1", len(got.Output))
		}
	})
}

type stubResponseTransport struct{}

func (stubResponseTransport) Do(_ context.Context, _ string, _ string, _ map[string]string, _ []byte) (*http.Response, error) {
	body := `{"id":"resp_123","object":"response","status":"completed","model":"gpt-4o-mini","output":[]}`
	return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body))}, nil
}

func newStubRespEngine() ResponsesEngine {
	base := protocolBase{
		transport:    stubResponseTransport{},
		defaultModel: "gpt-4o-mini",
	}
	codec := openaiCodec
	return &openAIResponsesEngine{adapter: newResponsesAdapter(base, codec)}
}

func TestApplyStandardResponseMiddleware(t *testing.T) {
	tests := []struct {
		name   string
		opts   StandardMiddlewareOptions
		assert func(t *testing.T, resp *ResponseObject)
	}{
		{
			name: "normalize",
			opts: StandardMiddlewareOptions{DefaultModel: "gpt-4o-mini"},
			assert: func(t *testing.T, resp *ResponseObject) {
				t.Helper()
				if resp.Model != "gpt-4o-mini" {
					t.Fatalf("model = %s, want gpt-4o-mini", resp.Model)
				}
			},
		},
		{
			name: "timeout",
			opts: StandardMiddlewareOptions{
				DefaultModel: "gpt-4o-mini",
				CrossCutting: CrossCuttingMiddlewareOptions{Timeout: 5 * time.Second},
			},
			assert: func(t *testing.T, resp *ResponseObject) {
				t.Helper()
				if resp.Status != "completed" {
					t.Fatalf("status = %s, want completed", resp.Status)
				}
			},
		},
		{
			name: "rate_limit",
			opts: StandardMiddlewareOptions{
				DefaultModel: "gpt-4o-mini",
				CrossCutting: CrossCuttingMiddlewareOptions{RateLimit: &RateLimitConfig{Limit: 10, Burst: 5}},
			},
			assert: func(t *testing.T, resp *ResponseObject) {
				t.Helper()
				if resp.Status != "completed" {
					t.Fatalf("status = %s, want completed", resp.Status)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := newStubRespEngine()
			wrapped := ApplyStandardResponseMiddleware(engine, tt.opts)

			resp, err := wrapped.Create(context.Background(), &ResponseRequest{
				Input: []ResponseInputItem{{Role: "user", Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("create failed: %v", err)
			}
			tt.assert(t, resp)
		})
	}
}
func benchmarkRequestFixture() *Request {
	return &Request{
		Model:       "gpt-4o-mini",
		Temperature: Float64(0.7),
		TopP:        Float64(0.9),
		MaxTokens:   512,
		Stop:        []string{"END"},
		Messages: []Message{
			NewTextMessage(RoleSystem, "You are a benchmarking assistant."),
			{
				Role: RoleUser,
				Content: []ContentPart{
					TextPart("Summarize this payload."),
					ImagePart{URL: "https://example.com/image.png", Detail: "low"},
				},
			},
		},
		Tools: []Tool{
			{
				Name:        "search",
				Description: "search in corpus",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{"type": "string"},
						"limit": map[string]any{"type": "integer"},
					},
					"required": []any{"query"},
				},
			},
		},
		Meta: map[string]any{
			"trace": map[string]any{"span_id": "abc", "attempt": 1},
		},
		Capabilities: map[string]any{
			"response_format": map[string]any{"type": "json_object"},
			"reasoning":       map[string]any{"effort": "medium"},
		},
	}
}

func BenchmarkCloneRequest(b *testing.B) {
	req := benchmarkRequestFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cloneRequest(req)
	}
}

func BenchmarkNormalizeRequest_NoMutation(b *testing.B) {
	req := benchmarkRequestFixture()
	req.Stream = false
	req.Model = "gpt-4o-mini"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeRequestForOperation(req, "gpt-4o-mini", false)
	}
}

func BenchmarkNormalizeRequest_WithMutation(b *testing.B) {
	req := benchmarkRequestFixture()
	req.Stream = false
	req.Model = ""
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeRequestForOperation(req, "gpt-4o-mini", true)
	}
}

func BenchmarkRequestIDMiddleware_Generate(b *testing.B) {
	req := benchmarkRequestFixture()
	next := func(ctx context.Context, req *Request) (*Response, error) {
		_ = ctx
		_ = req
		return &Response{}, nil
	}
	h := RequestIDMiddleware(nil)(next)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = h(context.Background(), req)
	}
}

func BenchmarkRetryUnary_FirstAttemptSuccess(b *testing.B) {
	req := benchmarkRequestFixture()
	next := func(ctx context.Context, req *Request) (*Response, error) {
		_ = ctx
		_ = req
		return &Response{}, nil
	}

	unary, _ := RetryMiddlewareWithConfig(RetryConfig{
		MaxAttempts: 3,
		Backoff:     func(int) time.Duration { return 0 },
		IsRetryable: func(error) bool { return true },
	})
	h := unary(next)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = h(context.Background(), req)
	}
}

func BenchmarkRetryUnary_OneRetryThenSuccess(b *testing.B) {
	req := benchmarkRequestFixture()
	unary, _ := RetryMiddlewareWithConfig(RetryConfig{
		MaxAttempts: 3,
		Backoff:     func(int) time.Duration { return 0 },
		IsRetryable: func(error) bool { return true },
	})
	attempt := 0
	next := func(ctx context.Context, req *Request) (*Response, error) {
		_ = ctx
		_ = req
		attempt++
		if attempt == 1 {
			return nil, errors.New("transient")
		}
		return &Response{}, nil
	}
	h := unary(next)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		attempt = 0
		_, _ = h(context.Background(), req)
	}
}

type benchmarkStreamIterator struct {
	chunks []string
	idx    int
	err    error
}

func (i *benchmarkStreamIterator) Next() bool {
	if i.idx < len(i.chunks) {
		i.idx++
		return true
	}
	return false
}

func (i *benchmarkStreamIterator) Chunk() []byte {
	if i.idx == 0 || i.idx > len(i.chunks) {
		return nil
	}
	return []byte(i.chunks[i.idx-1])
}

func (i *benchmarkStreamIterator) Text() string {
	if i.idx == 0 || i.idx > len(i.chunks) {
		return ""
	}
	return i.chunks[i.idx-1]
}

func (i *benchmarkStreamIterator) FullText() string { return i.Text() }
func (i *benchmarkStreamIterator) Err() error       { return i.err }
func (i *benchmarkStreamIterator) Close() error     { return nil }
func (i *benchmarkStreamIterator) Usage() *Usage    { return nil }
func (i *benchmarkStreamIterator) Response() *Response {
	if i.idx == 0 || i.idx > len(i.chunks) {
		return nil
	}
	return &Response{Content: i.chunks[i.idx-1]}
}

func BenchmarkRetryStream_FirstAttemptSuccess(b *testing.B) {
	req := benchmarkRequestFixture()
	_, stream := RetryMiddlewareWithConfig(RetryConfig{
		MaxAttempts: 3,
		Backoff:     func(int) time.Duration { return 0 },
		IsRetryable: func(error) bool { return true },
	})

	next := func(ctx context.Context, req *Request) (StreamIterator, error) {
		_ = ctx
		_ = req
		return &benchmarkStreamIterator{chunks: []string{"ok"}}, nil
	}
	h := stream(next)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		iter, err := h(context.Background(), req)
		if err != nil {
			b.Fatal(err)
		}
		for iter.Next() {
			_ = iter.Text()
		}
		_ = iter.Err()
		_ = iter.Close()
	}
}

func BenchmarkRetryStream_OneRetryBeforeFirstChunk(b *testing.B) {
	req := benchmarkRequestFixture()
	_, stream := RetryMiddlewareWithConfig(RetryConfig{
		MaxAttempts: 3,
		Backoff:     func(int) time.Duration { return 0 },
		IsRetryable: func(error) bool { return true },
	})

	attempt := 0
	next := func(ctx context.Context, req *Request) (StreamIterator, error) {
		_ = ctx
		_ = req
		attempt++
		if attempt == 1 {
			return &benchmarkStreamIterator{err: errors.New("transient")}, nil
		}
		return &benchmarkStreamIterator{chunks: []string{"ok"}}, nil
	}
	h := stream(next)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		attempt = 0
		iter, err := h(context.Background(), req)
		if err != nil {
			b.Fatal(err)
		}
		for iter.Next() {
			_ = iter.Text()
		}
		_ = iter.Err()
		_ = iter.Close()
	}
}
func TestMultipleObservers_BothFire(t *testing.T) {
	tests := []struct {
		name   string
		run    func(t *testing.T, engine Engine)
		assert func(t *testing.T, observer *countingObserver)
	}{
		{
			name: "unary",
			run: func(t *testing.T, engine Engine) {
				t.Helper()
				_, err := engine.Generate(context.Background(), &Request{Messages: []Message{NewTextMessage(RoleUser, "hi")}})
				if err != nil {
					t.Fatalf("generate failed: %v", err)
				}
			},
			assert: func(t *testing.T, observer *countingObserver) {
				t.Helper()
				if got := observer.requestStart.Load(); got != 1 {
					t.Fatalf("observer request start = %d, want 1", got)
				}
				if got := observer.requestFinish.Load(); got != 1 {
					t.Fatalf("observer request finish = %d, want 1", got)
				}
			},
		},
		{
			name: "stream",
			run: func(t *testing.T, engine Engine) {
				t.Helper()
				iter, err := engine.Stream(context.Background(), &Request{Messages: []Message{NewTextMessage(RoleUser, "hi")}})
				if err != nil {
					t.Fatalf("stream failed: %v", err)
				}
				for iter.Next() {
					_ = iter.Text()
				}
				if err := iter.Close(); err != nil {
					t.Fatalf("stream close failed: %v", err)
				}
			},
			assert: func(t *testing.T, observer *countingObserver) {
				t.Helper()
				if got := observer.streamStart.Load(); got != 1 {
					t.Fatalf("observer stream start = %d, want 1", got)
				}
				if got := observer.streamConnected.Load(); got != 1 {
					t.Fatalf("observer stream connected = %d, want 1", got)
				}
				if got := observer.streamFinish.Load(); got != 1 {
					t.Fatalf("observer stream finish = %d, want 1", got)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			observer := &countingObserver{}

			engine := ApplyStandardMiddleware(stubEngine{}, StandardMiddlewareOptions{
				Observers: []LifecycleObserver{observer},
			})

			tt.run(t, engine)
			tt.assert(t, observer)
		})
	}
}

type countingLogger struct {
	debug atomic.Int32
	error atomic.Int32
}

func (l *countingLogger) Info(string, ...any)  {}
func (l *countingLogger) Warn(string, ...any)  {}
func (l *countingLogger) Debug(string, ...any) { l.debug.Add(1) }
func (l *countingLogger) Error(string, ...any) { l.error.Add(1) }

type countingObserver struct {
	requestStart    atomic.Int32
	requestFinish   atomic.Int32
	streamStart     atomic.Int32
	streamConnected atomic.Int32
	streamFinish    atomic.Int32
}

func (o *countingObserver) OnRequestStart(context.Context, LifecycleEvent)  { o.requestStart.Add(1) }
func (o *countingObserver) OnRequestFinish(context.Context, LifecycleEvent) { o.requestFinish.Add(1) }
func (o *countingObserver) OnStreamStart(context.Context, LifecycleEvent)   { o.streamStart.Add(1) }
func (o *countingObserver) OnStreamConnected(context.Context, LifecycleEvent) {
	o.streamConnected.Add(1)
}
func (o *countingObserver) OnStreamFinish(context.Context, LifecycleEvent) { o.streamFinish.Add(1) }

type stubEngine struct{}

func (stubEngine) Generate(context.Context, *Request) (*Response, error) {
	return &Response{Content: "ok"}, nil
}

func (stubEngine) Stream(context.Context, *Request) (StreamIterator, error) {
	return &stubIter{text: "ok"}, nil
}

func (stubEngine) Capabilities() *ProtocolCapabilities {
	return &ProtocolCapabilities{
		SupportedParameters: map[string]ParameterRange{},
		Description:         "Stub engine for testing",
	}
}

type stubIter struct {
	text string
	done bool
}

func (i *stubIter) Next() bool {
	if i.done {
		return false
	}
	i.done = true
	return true
}
func (i *stubIter) Chunk() []byte       { return []byte(i.text) }
func (i *stubIter) Text() string        { return i.text }
func (i *stubIter) FullText() string    { return i.text }
func (i *stubIter) Err() error          { return nil }
func (i *stubIter) Close() error        { return nil }
func (i *stubIter) Usage() *Usage       { return nil }
func (i *stubIter) Response() *Response { return &Response{Content: i.text} }

// ============================================================================
// Protocol Capabilities Tests
// ============================================================================

func TestRequest_ValidateFor(t *testing.T) {
	tests := []struct {
		name      string
		req       *Request
		engine    Engine
		wantError bool
	}{
		{
			name:      "nil_engine",
			req:       &Request{},
			engine:    nil,
			wantError: true,
		},
		{
			name:      "engine_with_nil_capabilities",
			req:       &Request{},
			engine:    &stubEngine{},
			wantError: false, // Falls back to no validation
		},
		{
			name: "supported_parameter",
			req: &Request{
				Capabilities: map[string]any{
					"reasoning_effort": "high",
				},
			},
			engine:    NewEngineWithEndpoint(ProtocolOpenAI, "https://api.openai.com/v1", "key", "gpt-4"),
			wantError: false,
		},
		{
			name: "unsupported_parameter",
			req: &Request{
				Capabilities: map[string]any{
					"unsupported_param": "value",
				},
			},
			engine:    NewEngineWithEndpoint(ProtocolOpenAI, "https://api.openai.com/v1", "key", "gpt-4"),
			wantError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.req.ValidateFor(tc.engine)
			if tc.wantError && err == nil {
				t.Errorf("ValidateFor() = nil, want error")
			}
			if !tc.wantError && err != nil {
				t.Errorf("ValidateFor() = %v, want nil", err)
			}
		})
	}
}

func TestEngine_Capabilities(t *testing.T) {
	engine := NewEngineWithEndpoint(ProtocolOpenAI, "https://api.openai.com/v1", "key", "gpt-4")
	caps := engine.Capabilities()

	if caps == nil {
		t.Fatalf("Capabilities() = nil, want *ProtocolCapabilities")
	}

	if caps.Description == "" {
		t.Errorf("Description is empty")
	}

	// Verify reasoning_effort is supported
	if _, ok := caps.SupportedParameters["reasoning_effort"]; !ok {
		t.Errorf("reasoning_effort not in SupportedParameters")
	}
}

func TestEngine_Capabilities_DefensiveCopy(t *testing.T) {
	engine := NewEngineWithEndpoint(ProtocolOpenAI, "https://api.openai.com/v1", "key", "gpt-4")
	first := engine.Capabilities()
	second := engine.Capabilities()

	first.Description = "mutated"
	delete(first.SupportedParameters, "reasoning_effort")
	first.ParameterMapping["reasoning_effort"] = "changed"

	if second.Description == "mutated" {
		t.Fatalf("Capabilities description leaked across calls")
	}
	if _, ok := second.SupportedParameters["reasoning_effort"]; !ok {
		t.Fatalf("SupportedParameters leaked mutation across calls")
	}
	if got := second.ParameterMapping["reasoning_effort"]; got != "reasoning_effort" {
		t.Fatalf("ParameterMapping leaked mutation across calls: %v", got)
	}
}

func TestResponseRequest_ValidateFor(t *testing.T) {
	tests := []struct {
		name      string
		req       *ResponseRequest
		engine    ResponsesEngine
		wantError bool
	}{
		{
			name:      "nil_engine",
			req:       &ResponseRequest{},
			engine:    nil,
			wantError: true,
		},
		{
			name: "valid_request",
			req: &ResponseRequest{
				Model: "gpt-4o",
				Input: []ResponseInputItem{
					NewTextResponseInputItem("user", "test"),
				},
				Capabilities: map[string]any{},
			},
			engine:    NewResponsesEngineWithEndpoint(ProtocolOpenAI, "https://api.openai.com/v1", "key", "gpt-4o"),
			wantError: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.req.ValidateFor(tc.engine)
			if tc.wantError && err == nil {
				t.Errorf("ValidateFor() = nil, want error")
			}
			if !tc.wantError && err != nil {
				t.Errorf("ValidateFor() = %v, want nil", err)
			}
		})
	}
}

func TestResponsesEngine_Capabilities(t *testing.T) {
	engine := NewResponsesEngineWithEndpoint(ProtocolOpenAI, "https://api.openai.com/v1", "key", "gpt-4o")
	caps := engine.Capabilities()

	if caps == nil {
		t.Fatalf("Capabilities() = nil, want *ProtocolCapabilities")
	}

	if caps.Description == "" {
		t.Errorf("Description is empty")
	}

	// Verify it's different from Chat API description
	chatEngine := NewEngineWithEndpoint(ProtocolOpenAI, "https://api.openai.com/v1", "key", "gpt-4")
	chatCaps := chatEngine.Capabilities()

	if caps.Description == chatCaps.Description {
		t.Errorf("ResponsesEngine and Engine have same description, should be different")
	}
}

// ============================================================================
// Claude Driver Tests
// ============================================================================

// TestClaudeEngine_Capabilities verifies Claude engine declares capabilities correctly.
func TestClaudeEngine_Capabilities(t *testing.T) {
	engine := NewEngine(ProtocolClaude, nil, "claude-3-5-sonnet")
	caps := engine.Capabilities()

	if caps == nil {
		t.Fatal("expected non-nil capabilities")
	}

	// Verify Claude-specific parameter
	if _, ok := caps.SupportedParameters["thinking_budget"]; !ok {
		t.Error("expected thinking_budget parameter")
	}

	// Verify the mapping exists
	if _, ok := caps.ParameterMapping["thinking_budget"]; !ok {
		t.Error("expected thinking_budget mapping")
	}

	if caps.Description == "" {
		t.Error("expected non-empty description")
	}

	t.Logf("Claude capabilities: %s", caps.Description)
}

// TestClaude_MessageConversion verifies message conversion to Claude format.
func TestClaude_MessageConversion(t *testing.T) {
	// Test basic text message
	messages := []Message{
		{
			Role:    RoleUser,
			Content: []ContentPart{TextPart("Hello Claude")},
		},
	}

	claudeMessages := convertMessagesToClaude(messages)
	if len(claudeMessages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(claudeMessages))
	}

	if claudeMessages[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", claudeMessages[0].Role)
	}

	if len(claudeMessages[0].Content) != 1 {
		t.Errorf("expected 1 content block, got %d", len(claudeMessages[0].Content))
	}

	if claudeMessages[0].Content[0].Type != "text" {
		t.Errorf("expected type 'text', got %q", claudeMessages[0].Content[0].Type)
	}

	if claudeMessages[0].Content[0].Text != "Hello Claude" {
		t.Errorf("expected text 'Hello Claude', got %q", claudeMessages[0].Content[0].Text)
	}

	t.Log("Claude message conversion verified")
}

// TestClaude_ToolConversion verifies tool schema conversion.
func TestClaude_ToolConversion(t *testing.T) {
	tools := []Tool{
		{
			Name:        "search",
			Description: "Search the web",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type": "string",
					},
				},
			},
		},
	}

	claudeTools := convertToolsToClaudeSchema(tools)
	if len(claudeTools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(claudeTools))
	}

	if claudeTools[0].Name != "search" {
		t.Errorf("expected tool name 'search', got %q", claudeTools[0].Name)
	}

	if claudeTools[0].Description != "Search the web" {
		t.Errorf("expected description 'Search the web', got %q", claudeTools[0].Description)
	}

	if claudeTools[0].InputSchema == nil {
		t.Error("expected non-nil InputSchema")
	}

	t.Log("Claude tool schema conversion verified")
}

// TestOpenAI_And_Claude_UniversalTemplate demonstrates both engines use the same
// architecture despite different request/response formats.
func TestOpenAI_And_Claude_UniversalTemplate(t *testing.T) {
	// Both use the same architecture
	openaiEngine := NewEngine(ProtocolOpenAI, nil, "gpt-4o")
	claudeEngine := NewEngine(ProtocolClaude, nil, "claude-3-5-sonnet")

	// Both implement the Engine interface
	var _ Engine = openaiEngine
	var _ Engine = claudeEngine

	// Both declare capabilities using the same ProtocolCapabilities system
	openaiCaps := openaiEngine.Capabilities()
	claudeCaps := claudeEngine.Capabilities()

	if openaiCaps == nil || claudeCaps == nil {
		t.Fatal("both engines must declare capabilities")
	}

	// Both have different parameter sets reflecting their protocol
	_, openaiHasReasoning := openaiCaps.SupportedParameters["reasoning_effort"]
	_, claudeHasThinking := claudeCaps.SupportedParameters["thinking_budget"]

	if !openaiHasReasoning {
		t.Error("OpenAI should support reasoning_effort")
	}
	if !claudeHasThinking {
		t.Error("Claude should support thinking_budget")
	}

	// The same Request model works with both via ValidateFor
	req := &Request{
		Model:    "gpt-4o",
		Messages: []Message{{Role: RoleUser, Content: []ContentPart{TextPart("test")}}},
		Capabilities: map[string]any{
			"reasoning_effort": "high",
		},
	}

	// This would validate correctly for OpenAI
	if err := req.ValidateFor(openaiEngine); err != nil {
		t.Errorf("OpenAI should accept reasoning_effort: %v", err)
	}

	// But would fail for Claude (which doesn't support it)
	if err := req.ValidateFor(claudeEngine); err == nil {
		t.Error("Claude should reject reasoning_effort")
	}

	t.Log("✓ Universal template verified: OpenAI and Claude share the same Engine interface and ProtocolCapabilities system")
}

// catalog

func TestCloneMetadata(t *testing.T) {
	cloned := CloneMetadata(map[string]any{"a": 1})
	if cloned["a"] != 1 {
		t.Fatalf("unexpected clone: %#v", cloned)
	}
	cloned["a"] = 2
	if CloneMetadata(nil) != nil {
		t.Fatal("expected nil clone for nil input")
	}
}

func TestAttachCatalogStateMetadata(t *testing.T) {
	meta := map[string]any{"vendor": "openai"}
	out := AttachCatalogStateMetadata(meta, CapabilityResolverState{Source: CapabilitySourceCatalog, Stale: true}, "capabilities.chat_catalog")
	if out["vendor"] != "openai" {
		t.Fatalf("expected preserved metadata, got %#v", out)
	}
	if out["capabilities.chat_catalog_source"] != CapabilitySourceCatalog {
		t.Fatalf("missing source key, got %#v", out)
	}
	if out["capabilities.chat_catalog_stale"] != true {
		t.Fatalf("missing stale key, got %#v", out)
	}
}

func TestAttachCatalogStateMetadata_DefaultPrefixAndNoSource(t *testing.T) {
	out := AttachCatalogStateMetadata(nil, CapabilityResolverState{Source: CapabilitySourceCatalog}, "")
	if out["capabilities.catalog_source"] != CapabilitySourceCatalog {
		t.Fatalf("expected default prefix source, got %#v", out)
	}
	if out["capabilities.catalog_stale"] != false {
		t.Fatalf("expected default prefix stale=false, got %#v", out)
	}

	base := map[string]any{"k": "v"}
	unchanged := AttachCatalogStateMetadata(base, CapabilityResolverState{}, "capabilities.chat_catalog")
	if unchanged["k"] != "v" || len(unchanged) != 1 {
		t.Fatalf("expected unchanged metadata when source missing, got %#v", unchanged)
	}
}

func TestIndexModelCapabilities(t *testing.T) {
	index := IndexModelCapabilities([]ModelCapabilities{
		{Model: " gpt-4.1 "},
		{Model: ""},
		{Model: "gpt-4.1", Supports: CapabilitySet{JSONMode: true}},
	})
	if len(index) != 1 {
		t.Fatalf("len(index) = %d, want 1", len(index))
	}
	if _, ok := index["gpt-4.1"]; !ok {
		t.Fatalf("expected trimmed key gpt-4.1, got %#v", index)
	}
	if !index["gpt-4.1"].Supports.JSONMode {
		t.Fatalf("expected last duplicate to win, got %#v", index["gpt-4.1"])
	}
}

func TestMergeModelCapabilities(t *testing.T) {
	merged, ok := MergeModelCapabilities(
		"gpt-4.1",
		map[string]ModelCapabilities{
			"gpt-4.1": {
				Model:    "gpt-4.1",
				Supports: CapabilitySet{JSONMode: true, ToolCalls: true},
				Limits:   ModelLimits{MaxContextWindowTokens: 200000, MaxOutputTokens: 16000},
				Meta:     map[string]any{"vendor": "openai", "priority": 1},
			},
		},
		map[string]ModelCapabilities{
			"gpt-4.1": {
				Model:    "gpt-4.1",
				Supports: CapabilitySet{Reasoning: true},
				Limits:   ModelLimits{MaxOutputTokens: 32000, MaxPromptTokens: 120000},
				Meta:     map[string]any{"priority": 2, "region": "us"},
			},
		},
	)
	if !ok {
		t.Fatal("expected merged model")
	}
	if !merged.Supports.JSONMode || !merged.Supports.ToolCalls || !merged.Supports.Reasoning {
		t.Fatalf("unexpected supports: %#v", merged.Supports)
	}
	if merged.Limits.MaxContextWindowTokens != 200000 || merged.Limits.MaxOutputTokens != 32000 || merged.Limits.MaxPromptTokens != 120000 {
		t.Fatalf("unexpected limits: %#v", merged.Limits)
	}
	if merged.Meta["priority"] != 1 {
		t.Fatalf("expected first metadata value to win, got %#v", merged.Meta)
	}
	if merged.Meta["region"] != "us" || merged.Meta["vendor"] != "openai" {
		t.Fatalf("expected metadata merge, got %#v", merged.Meta)
	}
}

func TestMergeModelCapabilities_MissingModel(t *testing.T) {
	_, ok := MergeModelCapabilities("missing", map[string]ModelCapabilities{"gpt-4.1": {Model: "gpt-4.1"}})
	if ok {
		t.Fatal("expected missing model merge to return false")
	}
}

func TestMergeModelCapabilities_TrimmedModelID(t *testing.T) {
	merged, ok := MergeModelCapabilities(
		"  gpt-4.1  ",
		map[string]ModelCapabilities{"gpt-4.1": {Model: "gpt-4.1", Supports: CapabilitySet{JSONMode: true}}},
	)
	if !ok {
		t.Fatal("expected merge to resolve trimmed model id")
	}
	if !merged.Supports.JSONMode {
		t.Fatalf("expected merged supports from trimmed id, got %#v", merged.Supports)
	}
}

func TestMergeModelCapabilities_EmptyModelID(t *testing.T) {
	_, ok := MergeModelCapabilities("   ", map[string]ModelCapabilities{"gpt-4.1": {Model: "gpt-4.1"}})
	if ok {
		t.Fatal("expected empty model id to return false")
	}
}

// request

func TestRequestIDFromHTTP_UsesHeaderFirst(t *testing.T) {
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("X-Request-Id", "req_from_header")
	id := RequestIDFromHTTP(req, func() string { return "generated" })
	if id != "req_from_header" {
		t.Fatalf("expected header id, got %q", id)
	}
}

func TestRequestIDFromHTTP_UsesGeneratorFallback(t *testing.T) {
	req := httptest.NewRequest("GET", "/", nil)
	id := RequestIDFromHTTP(req, func() string { return "req_generated" })
	if id != "req_generated" {
		t.Fatalf("expected generated id, got %q", id)
	}
}

func TestRequestIDFromHTTP_NilGeneratorUsesDefault(t *testing.T) {
	id := RequestIDFromHTTP(nil, nil)
	if strings.TrimSpace(id) == "" {
		t.Fatal("expected non-empty request id")
	}
	if !strings.HasPrefix(id, "req_") {
		t.Fatalf("expected default req_ prefix, got %q", id)
	}
}

func TestGenerateRequestIDWithPrefix(t *testing.T) {
	id := GenerateRequestIDWithPrefix("req_proxy")
	if !strings.HasPrefix(id, "req_proxy_") {
		t.Fatalf("expected custom prefix, got %q", id)
	}
	fallback := GenerateRequestIDWithPrefix("   ")
	if !strings.HasPrefix(fallback, "req_") {
		t.Fatalf("expected fallback req_ prefix, got %q", fallback)
	}
}

// ============================================================================
// Regression Tests (R1-R5)
// ============================================================================

// R1: go vet lock-copy — timeout stream wrapper uses pointer to avoid copying sync.Once
func TestTimeoutStreamWrapper_PointerBase(t *testing.T) {
	iter := &stubIter{text: "ok"}
	result := wrapTimeoutStream(iter, func() {}, context.Background(), "read timeout", nil)
	ts, ok := result.(*timeoutStreamIterator)
	if !ok {
		t.Fatalf("expected *timeoutStreamIterator, got %T", result)
	}
	if ts.base == nil {
		t.Fatal("expected non-nil *timeoutBaseStream")
	}
	ts.Next()
	ts.Close()
}

// R1: go vet lock-copy — observed stream wrapper uses pointer to avoid copying sync.Once
func TestObservedStreamWrapper_PointerCore(t *testing.T) {
	observer := &countingObserver{}
	var iter StreamIterator = &stubIter{text: "ok"}
	wrapped := wrapObservedStream[*Request, *Response, StreamIterator](observer, context.Background(), time.Now(), LifecycleEvent{}, iter, chatEventBuilder)
	os, ok := wrapped.(*observedChatStream)
	if !ok {
		t.Fatalf("expected *observedChatStream, got %T", wrapped)
	}
	if os.core == nil {
		t.Fatal("expected non-nil *observedStreamCore")
	}
	os.Next()
	os.Close()
	if got := observer.streamFinish.Load(); got != 1 {
		t.Fatalf("observer stream finish = %d, want 1", got)
	}
}

// R2: FetchModelsCatalog passes relative "/models" path to transport (not full URL)
func TestFetchModelsCatalog_RelativePath(t *testing.T) {
	var capturedPath string
	transport := &pathRecordingTransport{
		pathFn: func(p string) { capturedPath = p },
	}
	_, err := FetchModelsCatalog(context.Background(), transport)
	if err != nil {
		t.Fatalf("FetchModelsCatalog() error = %v", err)
	}
	if capturedPath != "/models" {
		t.Errorf("path = %q, want %q", capturedPath, "/models")
	}
}

type pathRecordingTransport struct {
	pathFn   func(string)
	response *http.Response
}

func (t *pathRecordingTransport) Do(_ context.Context, _, path string, _ map[string]string, _ []byte) (*http.Response, error) {
	if t.pathFn != nil {
		t.pathFn(path)
	}
	if t.response != nil {
		return t.response, nil
	}
	return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"object":"list","data":[]}`))}, nil
}

// R3: resolveToolCallIndex uses explicit index 0 instead of falling back to ordinal
func TestResolveToolCallIndex_ZeroIndex(t *testing.T) {
	tests := []struct {
		name     string
		ordinal  int
		tc       APIToolCall
		expected int
	}{
		{"explicit_0", 5, APIToolCall{Index: 0}, 0},
		{"explicit_1", 0, APIToolCall{Index: 1}, 1},
		{"explicit_3", 0, APIToolCall{Index: 3}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveToolCallIndex(tt.ordinal, tt.tc); got != tt.expected {
				t.Errorf("resolveToolCallIndex(%d, Index=%d) = %d, want %d", tt.ordinal, tt.tc.Index, got, tt.expected)
			}
		})
	}
}

// R4: rate limiter middleware returns LLMError (not bare fmt.Errorf) for classification
func TestRateLimiter_ErrorIsLLMError(t *testing.T) {
	unary, _, closer := RateLimitMiddlewares(1, 1)
	defer closer()

	next := func(context.Context, *Request) (*Response, error) { return &Response{}, nil }
	h := unary(next)
	req := &Request{Messages: []Message{NewTextMessage(RoleUser, "hi")}}

	// First call consumes the only token
	if _, err := h(context.Background(), req); err != nil {
		t.Fatalf("first call failed: %v", err)
	}

	// Second call with cancelled context should return LLMError(ErrCodeCancelled)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := h(ctx, req)
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	var llmErr *LLMError
	if !errors.As(err, &llmErr) {
		t.Fatalf("expected *LLMError, got %T: %v", err, err)
	}
	if llmErr.Code != ErrCodeCancelled {
		t.Errorf("code = %v, want %v", llmErr.Code, ErrCodeCancelled)
	}
}

// R5: ExponentialBackoffWithJitter must not panic for any attempt value
func TestExponentialBackoffWithJitter_NoPanic(t *testing.T) {
	for attempt := 0; attempt < 35; attempt++ {
		d := ExponentialBackoffWithJitter(attempt)
		if d < 0 {
			t.Errorf("attempt %d: negative duration %v", attempt, d)
		}
		if d > maxBackoff {
			t.Errorf("attempt %d: duration %v exceeds max %v", attempt, d, maxBackoff)
		}
	}
}


func TestResponseInputDisplayText_String(t *testing.T) {
	item := ResponseInputItem{Role: "user", Content: "hello"}
	if got := ResponseInputDisplayText(item); got != "hello" {
		t.Fatalf("ResponseInputDisplayText() = %q, want hello", got)
	}
}

func TestResponseInputDisplayText_Parts(t *testing.T) {
	item := ResponseInputItem{
		Role: "user",
		Content: []ResponseInputContentPart{
			ResponseInputTextPart{Type: "input_text", Text: "hello "},
			ResponseInputTextPart{Type: "input_text", Text: "world"},
		},
	}
	if got := ResponseInputDisplayText(item); got != "hello world" {
		t.Fatalf("ResponseInputDisplayText(parts) = %q, want hello world", got)
	}
}

func TestResponseWireTools_MapsToAnySlice(t *testing.T) {
	if got := ResponseWireTools(nil); got != nil {
		t.Fatalf("ResponseWireTools(nil) = %#v, want nil", got)
	}
	tools := []ResponseTool{{Type: "function", Name: "lookup"}}
	mapped := ResponseWireTools(tools)
	if len(mapped) != 1 {
		t.Fatalf("ResponseWireTools() len = %d, want 1", len(mapped))
	}
	tool, ok := mapped[0].(ResponseTool)
	if !ok || tool.Name != "lookup" {
		t.Fatalf("ResponseWireTools()[0] = %#v, want ResponseTool{Name:lookup}", mapped[0])
	}
}

func TestSummarizeNormalizedChatRequest(t *testing.T) {
	summary := SummarizeNormalizedChatRequest(&Request{
		Model: "gpt-4.1",
		Messages: []Message{{
			Role:      RoleUser,
			Content:   []ContentPart{TextPart("look"), ImagePart{URL: "https://example.com/image.png", Detail: "high"}},
			ToolCalls: []APIToolCall{{ID: "call_1", Type: "function", Name: "lookup", Arguments: `{}`}},
		}},
		Tools:    []Tool{{Name: "lookup"}},
		JSONMode: true,
	})
	if !strings.Contains(summary, `vision_parts=1`) || !strings.Contains(summary, `tool_calls=1`) || !strings.Contains(summary, `json_mode=true`) {
		t.Fatalf("expected summary to include normalized fields, got %s", summary)
	}
}

func TestSummarizeNormalizedResponsesRequest(t *testing.T) {
	summary := SummarizeNormalizedResponsesRequest(&ResponseRequest{
		Model:           "gpt-5-mini",
		Input:           []ResponseInputItem{{Role: "user", Content: "hi"}},
		Tools:           []ResponseTool{{Type: "function", Name: "lookup"}},
		Reasoning:       &ResponseReasoning{Effort: "medium"},
		MaxOutputTokens: 256,
		Store:           true,
	})
	if !strings.Contains(summary, `input_items=1`) || !strings.Contains(summary, `tools=1`) || !strings.Contains(summary, `max_output_tokens=256`) {
		t.Fatalf("expected summary to include response fields, got %s", summary)
	}
}


func TestDetectContextOverflow_OpenAIStyle(t *testing.T) {
	body := `{"error":{"code":"context_length_exceeded","message":"This model's maximum context length is 128000 tokens."}}`
	if !DetectContextOverflow(body) {
		t.Error("expected DetectContextOverflow to return true for OpenAI context_length_exceeded")
	}
}

func TestDetectContextOverflow_AnthropicStyle(t *testing.T) {
	body := `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 137500 tokens > 200000 maximum"}}`
	if !DetectContextOverflow(body) {
		t.Error("expected DetectContextOverflow to return true for Anthropic 'prompt is too long'")
	}
}

func TestDetectContextOverflow_EmptyBody(t *testing.T) {
	if DetectContextOverflow("") {
		t.Error("expected DetectContextOverflow to return false for empty body")
	}
}

func TestDetectContextOverflow_UnrelatedError(t *testing.T) {
	body := `{"error":{"message":"rate limit exceeded"}}`
	if DetectContextOverflow(body) {
		t.Error("expected DetectContextOverflow to return false for unrelated error")
	}
}

func TestDetectContextOverflow_VariousPhrasings(t *testing.T) {
	cases := []string{
		"input is too long",
		"request too large",
		"too many tokens",
		"maximum context length exceeded",
		"exceeds the context window",
		"model_context_window_exceeded",
	}
	for _, c := range cases {
		if !DetectContextOverflow(c) {
			t.Errorf("expected DetectContextOverflow true for %q", c)
		}
	}
}

func TestParseOverflowTokens_AnthropicStyle(t *testing.T) {
	body := `{"error":{"message":"prompt is too long: 137500 tokens > 135000 maximum"}}`
	gap := ParseOverflowTokens(body)
	if gap.Actual != 137500 {
		t.Errorf("expected Actual 137500, got %d", gap.Actual)
	}
	if gap.Limit != 135000 {
		t.Errorf("expected Limit 135000, got %d", gap.Limit)
	}
}

func TestParseOverflowTokens_OpenAIStyle(t *testing.T) {
	body := `This model's maximum context length is 128000 tokens. However, your messages resulted in 145230 tokens.`
	gap := ParseOverflowTokens(body)
	if gap.Actual != 145230 {
		t.Errorf("expected Actual 145230, got %d", gap.Actual)
	}
	if gap.Limit != 128000 {
		t.Errorf("expected Limit 128000, got %d", gap.Limit)
	}
}

func TestParseOverflowTokens_NoNumbers(t *testing.T) {
	body := `context length exceeded`
	gap := ParseOverflowTokens(body)
	if gap.Actual != 0 || gap.Limit != 0 {
		t.Errorf("expected zero gap, got Actual=%d Limit=%d", gap.Actual, gap.Limit)
	}
}

func TestParseOverflowTokens_WithCommas(t *testing.T) {
	body := `prompt is too long: 1,375,000 tokens > 1,350,000 maximum`
	gap := ParseOverflowTokens(body)
	if gap.Actual != 1375000 {
		t.Errorf("expected Actual 1375000, got %d", gap.Actual)
	}
	if gap.Limit != 1350000 {
		t.Errorf("expected Limit 1350000, got %d", gap.Limit)
	}
}


// --- ThinkingPart content part ---

func TestThinkingPartIsContentPart(t *testing.T) {
	var p ContentPart = ThinkingPart{Content: "reasoning", Signature: "sig"}
	if p == nil {
		t.Fatal("ThinkingPart should satisfy ContentPart interface")
	}
}

// --- OpenAI convertMessages: ThinkingPart -> reasoning_content ---

func TestConvertMessages_ThinkingPartSetsReasoningContent(t *testing.T) {
	messages := []Message{
		{
			Role: RoleAssistant,
			Content: []ContentPart{
				ThinkingPart{Content: "Let me think about this", Signature: "sig123"},
				TextPart("The answer is 42"),
			},
		},
	}

	result := convertMessages(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	if result[0].ReasoningContent != "Let me think about this" {
		t.Errorf("expected ReasoningContent 'Let me think about this', got %q", result[0].ReasoningContent)
	}

	// Text content should still be present
	if s, ok := result[0].Content.(string); !ok || s != "The answer is 42" {
		t.Errorf("expected Content to be 'The answer is 42', got %v", result[0].Content)
	}
}

func TestConvertMessages_ThinkingPartOnlyNoText(t *testing.T) {
	messages := []Message{
		{
			Role: RoleAssistant,
			Content: []ContentPart{
				ThinkingPart{Content: "just thinking", Signature: "sig"},
			},
		},
	}

	result := convertMessages(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	if result[0].ReasoningContent != "just thinking" {
		t.Errorf("expected ReasoningContent 'just thinking', got %q", result[0].ReasoningContent)
	}

	// Content should be nil since ThinkingPart is extracted and no text remains
	if result[0].Content != nil {
		t.Errorf("expected nil Content, got %v", result[0].Content)
	}
}

func TestConvertMessages_NoThinkingPart(t *testing.T) {
	messages := []Message{
		{
			Role:    RoleUser,
			Content: []ContentPart{TextPart("hello")},
		},
	}

	result := convertMessages(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	if result[0].ReasoningContent != "" {
		t.Errorf("expected empty ReasoningContent, got %q", result[0].ReasoningContent)
	}
}

// --- Claude convertMessagesToClaude: ThinkingPart -> thinking block ---

func TestConvertMessagesToClaude_ThinkingPartEmitsThinkingBlock(t *testing.T) {
	messages := []Message{
		{
			Role: RoleAssistant,
			Content: []ContentPart{
				ThinkingPart{Content: "I should consider all options", Signature: "signature_abc"},
				TextPart("Here is my answer"),
			},
		},
	}

	result := convertMessagesToClaude(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	if len(result[0].Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(result[0].Content))
	}

	// Thinking block should be first
	thinking := result[0].Content[0]
	if thinking.Type != "thinking" {
		t.Errorf("expected first block type 'thinking', got %q", thinking.Type)
	}
	if thinking.Thinking != "I should consider all options" {
		t.Errorf("expected thinking content, got %q", thinking.Thinking)
	}
	if thinking.Signature != "signature_abc" {
		t.Errorf("expected signature 'signature_abc', got %q", thinking.Signature)
	}

	// Text block should be second
	text := result[0].Content[1]
	if text.Type != "text" {
		t.Errorf("expected second block type 'text', got %q", text.Type)
	}
	if text.Text != "Here is my answer" {
		t.Errorf("expected text 'Here is my answer', got %q", text.Text)
	}
}

func TestConvertMessagesToClaude_ThinkingPartWithoutSignatureSkipped(t *testing.T) {
	messages := []Message{
		{
			Role: RoleAssistant,
			Content: []ContentPart{
				ThinkingPart{Content: "thinking without sig", Signature: ""},
				TextPart("answer"),
			},
		},
	}

	result := convertMessagesToClaude(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	// Only the text block should be present — thinking block without signature
	// would cause an Anthropic API error
	if len(result[0].Content) != 1 {
		t.Fatalf("expected 1 content block (thinking skipped), got %d", len(result[0].Content))
	}
	if result[0].Content[0].Type != "text" {
		t.Errorf("expected type 'text', got %q", result[0].Content[0].Type)
	}
}

func TestConvertMessagesToClaude_ThinkingPartBeforeToolCalls(t *testing.T) {
	messages := []Message{
		{
			Role: RoleAssistant,
			Content: []ContentPart{
				ThinkingPart{Content: "I need to use a tool", Signature: "sig"},
				TextPart("Let me search for that"),
			},
			ToolCalls: []APIToolCall{
				{ID: "call_1", Type: "function", Name: "search", Arguments: `{"q":"test"}`},
			},
		},
	}

	result := convertMessagesToClaude(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	// Expected order: thinking, text, tool_use
	if len(result[0].Content) != 3 {
		t.Fatalf("expected 3 content blocks, got %d", len(result[0].Content))
	}

	if result[0].Content[0].Type != "thinking" {
		t.Errorf("expected first block 'thinking', got %q", result[0].Content[0].Type)
	}
	if result[0].Content[1].Type != "text" {
		t.Errorf("expected second block 'text', got %q", result[0].Content[1].Type)
	}
	if result[0].Content[2].Type != "tool_use" {
		t.Errorf("expected third block 'tool_use', got %q", result[0].Content[2].Type)
	}
}

// --- Claude ConvertChatResponse: thinking block -> Response ---

func TestClaudeConvertChatResponse_CapturesThinkingAndSignature(t *testing.T) {
	codec := claudeCodecInst
	claudeResp := &claudeResponse{
		Content: []claudeContent{
			{Type: "thinking", Thinking: "Deep reasoning here", Signature: "sig_xyz"},
			{Type: "text", Text: "The result is 42"},
		},
		StopReason: "end_turn",
		Usage:      &Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
	}

	resp := codec.ConvertChatResponse(claudeResp)

	if resp.ReasoningContent != "Deep reasoning here" {
		t.Errorf("expected ReasoningContent 'Deep reasoning here', got %q", resp.ReasoningContent)
	}
	if resp.ThinkingSignature != "sig_xyz" {
		t.Errorf("expected ThinkingSignature 'sig_xyz', got %q", resp.ThinkingSignature)
	}
	if resp.Content != "The result is 42" {
		t.Errorf("expected Content 'The result is 42', got %q", resp.Content)
	}
}

func TestClaudeConvertChatResponse_NoThinkingBlock(t *testing.T) {
	codec := claudeCodecInst
	claudeResp := &claudeResponse{
		Content: []claudeContent{
			{Type: "text", Text: "No thinking here"},
		},
		StopReason: "end_turn",
	}

	resp := codec.ConvertChatResponse(claudeResp)

	if resp.ReasoningContent != "" {
		t.Errorf("expected empty ReasoningContent, got %q", resp.ReasoningContent)
	}
	if resp.ThinkingSignature != "" {
		t.Errorf("expected empty ThinkingSignature, got %q", resp.ThinkingSignature)
	}
	if resp.Content != "No thinking here" {
		t.Errorf("expected Content 'No thinking here', got %q", resp.Content)
	}
}

func TestClaudeConvertChatResponse_MultipleThinkingBlocks(t *testing.T) {
	codec := claudeCodecInst
	claudeResp := &claudeResponse{
		Content: []claudeContent{
			{Type: "thinking", Thinking: "First thought", Signature: "sig1"},
			{Type: "thinking", Thinking: "Second thought", Signature: "sig2"},
			{Type: "text", Text: "Final answer"},
		},
		StopReason: "end_turn",
	}

	resp := codec.ConvertChatResponse(claudeResp)

	// Multiple thinking blocks should concatenate reasoning content
	expected := "First thoughtSecond thought"
	if resp.ReasoningContent != expected {
		t.Errorf("expected ReasoningContent %q, got %q", expected, resp.ReasoningContent)
	}
	// Last signature wins
	if resp.ThinkingSignature != "sig2" {
		t.Errorf("expected ThinkingSignature 'sig2', got %q", resp.ThinkingSignature)
	}
}

// --- Usage fields ---

func TestUsageCacheMissTokens_DeepSeekFormat(t *testing.T) {
	u := Usage{
		PromptTokens:     1000,
		CacheHitTokens:   800,
		CompletionTokens: 200,
	}
	if miss := u.CacheMissTokens(); miss != 200 {
		t.Errorf("expected CacheMissTokens 200, got %d", miss)
	}
}

func TestUsageCacheMissTokens_OpenAIFormat(t *testing.T) {
	u := Usage{
		PromptTokens:        1000,
		PromptTokensDetails: &PromptTokensDetails{CachedTokens: 500},
		CompletionTokens:    200,
	}
	if miss := u.CacheMissTokens(); miss != 500 {
		t.Errorf("expected CacheMissTokens 500, got %d", miss)
	}
}

func TestUsageCacheMissTokens_NoCache(t *testing.T) {
	u := Usage{
		PromptTokens:     1000,
		CompletionTokens: 200,
	}
	if miss := u.CacheMissTokens(); miss != 1000 {
		t.Errorf("expected CacheMissTokens 1000, got %d", miss)
	}
}

func TestUsageTotalContextTokens(t *testing.T) {
	u := Usage{
		PromptTokens:     1000,
		CompletionTokens: 500,
		CacheHitTokens:   300,
	}
	if total := u.TotalContextTokens(); total != 1800 {
		t.Errorf("expected TotalContextTokens 1800, got %d", total)
	}
}

func TestUsageJSONRoundTrip(t *testing.T) {
	u := Usage{
		PromptTokens:        1000,
		CompletionTokens:    500,
		TotalTokens:         1500,
		CacheHitTokens:      300,
		PromptTokensDetails: &PromptTokensDetails{CachedTokens: 300},
	}

	data, err := json.Marshal(u)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded Usage
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded.CacheHitTokens != 300 {
		t.Errorf("expected CacheHitTokens 300, got %d", decoded.CacheHitTokens)
	}
	if decoded.PromptTokensDetails == nil || decoded.PromptTokensDetails.CachedTokens != 300 {
		t.Errorf("expected PromptTokensDetails.CachedTokens 300, got %v", decoded.PromptTokensDetails)
	}
}

