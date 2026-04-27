package slm

import (
	"context"
	"errors"
	"io"
	"net/http"

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

	_, ok = adapter.Engine.(*OpenAIEngine)
	if !ok {
		t.Fatalf("inner engine type = %T, want *OpenAIEngine", adapter.Engine)
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

func TestModelInfo(t *testing.T) {
	model := Model{
		ID:           "gpt-4o",
		OwnedBy:      "openai",
		Capabilities: CapabilitySet{JSONMode: true},
		Limits:       ModelLimits{MaxOutputTokens: 16384},
		Created:      1700000000,
	}

	info := model.Info()

	if info.ID != "gpt-4o" {
		t.Errorf("ID = %v, want gpt-4o", info.ID)
	}
	if info.OwnedBy != "openai" {
		t.Errorf("OwnedBy = %v, want openai", info.OwnedBy)
	}
	if info.CreatedAt.Year() != 2023 {
		t.Errorf("CreatedAt.Year() = %v, want 2023", info.CreatedAt.Year())
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
		ExtraBody:       map[string]any{"trace": true},
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
	if got := fields["extra_body_keys"]; got != 1 {
		t.Fatalf("extra_body_keys = %v, want 1", got)
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

func newStubRespEngine() *OpenAIResponsesEngine {
	base := openAIBase{
		transport:    stubResponseTransport{},
		defaultModel: "gpt-4o-mini",
	}
	codec := NewOpenAICodec()
	return &OpenAIResponsesEngine{adapter: newResponsesAdapter(base, codec)}
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
		ExtraBody: map[string]any{
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
			logger := &countingLogger{}
			logObserver := NewLogObserver(logger)

			engine := ApplyStandardMiddleware(stubEngine{}, StandardMiddlewareOptions{
				Observers: []LifecycleObserver{logObserver, observer},
			})

			tt.run(t, engine)
			tt.assert(t, observer)
			if got := logger.debug.Load(); got == 0 {
				t.Fatalf("logger debug count = %d, want > 0", got)
			}
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
