package slm

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Attribute is a backend-agnostic observability attribute.
type Attribute struct {
	Key   string
	Value any
}

// ReasoningOptions defines explicit reasoning request settings.
//
// The zero value requests reasoning capability without provider-specific tuning.
// Effort is forwarded to OpenAI-compatible providers via "reasoning_effort".
type ReasoningOptions struct {
	Effort  string
	Summary string
}

// CapabilitySet represents the feature surface of a model or request.
type CapabilitySet struct {
	JSONMode  bool `json:"json_mode,omitempty"`
	ToolCalls bool `json:"tool_calls,omitempty"`
	Vision    bool `json:"vision,omitempty"`
	Reasoning bool `json:"reasoning,omitempty"`
}

// Any reports whether any capability bit is enabled.
func (c CapabilitySet) Any() bool {
	return c.JSONMode || c.ToolCalls || c.Vision || c.Reasoning
}

func (c CapabilitySet) Missing(requested CapabilitySet) []string {
	var missing []string
	if requested.JSONMode && !c.JSONMode {
		missing = append(missing, "json_mode")
	}
	if requested.ToolCalls && !c.ToolCalls {
		missing = append(missing, "tool_calls")
	}
	if requested.Vision && !c.Vision {
		missing = append(missing, "vision")
	}
	if requested.Reasoning && !c.Reasoning {
		missing = append(missing, "reasoning")
	}
	return missing
}

func (c CapabilitySet) attributes() []Attribute {
	return []Attribute{
		{Key: "cap.json_mode", Value: c.JSONMode},
		{Key: "cap.tool_calls", Value: c.ToolCalls},
		{Key: "cap.vision", Value: c.Vision},
		{Key: "cap.reasoning", Value: c.Reasoning},
	}
}

// ModelCapabilities declares the explicit feature surface of a concrete model.
type ModelCapabilities struct {
	Model    string
	Provider string
	Supports CapabilitySet
	Limits   ModelLimits
	Meta     map[string]any
}

func cloneModelCapabilities(caps ModelCapabilities) ModelCapabilities {
	return ModelCapabilities{
		Model:    caps.Model,
		Provider: caps.Provider,
		Supports: caps.Supports,
		Limits:   caps.Limits,
		Meta:     cloneMap(caps.Meta),
	}
}

// CapabilityCatalog exposes the current model capability catalog snapshot.
type CapabilityCatalog interface {
	ListModelCapabilities(ctx context.Context) ([]ModelCapabilities, CapabilityResolverState, error)
}

// CapabilityResolver resolves explicit capabilities for a concrete model name.
// The bool result reports whether the model was known to the resolver.
type CapabilityResolver interface {
	ResolveCapabilities(context.Context, string) (ModelCapabilities, bool, error)
}

// CapabilityResolverFunc adapts a function to CapabilityResolver.
type CapabilityResolverFunc func(context.Context, string) (ModelCapabilities, bool, error)

func (f CapabilityResolverFunc) ResolveCapabilities(ctx context.Context, model string) (ModelCapabilities, bool, error) {
	return f(ctx, model)
}

// CapabilityResolverState carries optional metadata about how capabilities were resolved.
type CapabilityResolverState struct {
	Source      string
	RefreshedAt time.Time
	Stale       bool
}

const (
	CapabilitySourceStatic  = "static"
	CapabilitySourceCatalog = "catalog"
)

// CapabilityResolverWithState is an optional extension for resolvers that can
// report metadata about the resolution result, such as whether the answer came
// from a stale catalog snapshot.
type CapabilityResolverWithState interface {
	ResolveCapabilitiesWithState(context.Context, string) (ModelCapabilities, bool, CapabilityResolverState, error)
}

// StaticCapabilityResolver resolves capabilities from an in-memory map.
// The "*" entry, when present, acts as a fallback for unknown models.
type StaticCapabilityResolver map[string]ModelCapabilities

func (r StaticCapabilityResolver) ResolveCapabilities(_ context.Context, model string) (ModelCapabilities, bool, error) {
	if caps, ok := r[model]; ok {
		if caps.Model == "" {
			caps.Model = model
		}
		return caps, true, nil
	}
	if caps, ok := r["*"]; ok {
		if caps.Model == "" {
			caps.Model = model
		}
		return caps, true, nil
	}
	return ModelCapabilities{}, false, nil
}

func (r StaticCapabilityResolver) ResolveCapabilitiesWithState(ctx context.Context, model string) (ModelCapabilities, bool, CapabilityResolverState, error) {
	caps, known, err := r.ResolveCapabilities(ctx, model)
	if !known || err != nil {
		return caps, known, CapabilityResolverState{}, err
	}
	return caps, true, CapabilityResolverState{Source: CapabilitySourceStatic}, nil
}

// ChainCapabilityResolvers tries resolvers in order and returns the first known model.
//
// This keeps slm independent of provider-specific discovery packages while still
// allowing callers to compose static defaults, cached catalogs, and live provider
// lookups into one capability negotiation source.
func ChainCapabilityResolvers(resolvers ...CapabilityResolver) CapabilityResolver {
	filtered := make([]CapabilityResolver, 0, len(resolvers))
	for _, resolver := range resolvers {
		if resolver != nil {
			filtered = append(filtered, resolver)
		}
	}
	if len(filtered) == 0 {
		return nil
	}
	return chainedCapabilityResolver{resolvers: filtered}
}

type chainedCapabilityResolver struct {
	resolvers []CapabilityResolver
}

func (r chainedCapabilityResolver) ResolveCapabilities(ctx context.Context, model string) (ModelCapabilities, bool, error) {
	caps, known, _, err := r.ResolveCapabilitiesWithState(ctx, model)
	return caps, known, err
}

func (r chainedCapabilityResolver) ResolveCapabilitiesWithState(ctx context.Context, model string) (ModelCapabilities, bool, CapabilityResolverState, error) {
	for _, resolver := range r.resolvers {
		if withState, ok := resolver.(CapabilityResolverWithState); ok {
			caps, known, state, err := withState.ResolveCapabilitiesWithState(ctx, model)
			if err != nil {
				return ModelCapabilities{}, false, CapabilityResolverState{}, err
			}
			if known {
				if caps.Model == "" {
					caps.Model = model
				}
				return caps, true, state, nil
			}
			continue
		}

		caps, known, err := resolver.ResolveCapabilities(ctx, model)
		if err != nil {
			return ModelCapabilities{}, false, CapabilityResolverState{}, err
		}
		if known {
			if caps.Model == "" {
				caps.Model = model
			}
			return caps, true, CapabilityResolverState{}, nil
		}
	}
	return ModelCapabilities{}, false, CapabilityResolverState{}, nil
}

// NegotiatedCapabilities captures the explicit request/model capability contract.
type NegotiatedCapabilities struct {
	Model     string
	Requested CapabilitySet
	Supported CapabilitySet
	Known     bool
	State     CapabilityResolverState
	Stale     bool
}

// CapabilityNegotiationOptions configures explicit model capability validation.
type CapabilityNegotiationOptions struct {
	Resolver          CapabilityResolver
	DefaultModel      string
	RequireKnown      bool
	RequireKnownModel bool
}

// DetectRequestedCapabilities extracts the capability requirements implied by a request.
func DetectRequestedCapabilities(req *Request) CapabilitySet {
	if req == nil {
		return CapabilitySet{}
	}

	requested := CapabilitySet{
		JSONMode:  req.JSONMode,
		ToolCalls: len(req.Tools) > 0,
		Vision:    ScanMessages(req.Messages).VisionParts > 0,
		Reasoning: req.Reasoning != nil || hasLegacyReasoningRequest(req.Capabilities),
	}

	return requested
}

// DetectRequestedResponseCapabilities extracts the capability requirements implied by a responses request.
func DetectRequestedResponseCapabilities(req *ResponseRequest) CapabilitySet {
	if req == nil {
		return CapabilitySet{}
	}
	return CapabilitySet{
		ToolCalls: len(req.Tools) > 0,
		Vision:    ScanResponseInput(req.Input).VisionParts > 0,
		Reasoning: req.Reasoning != nil || hasLegacyReasoningRequest(req.Capabilities),
	}
}

// CapabilityNegotiationMiddleware validates explicit request requirements against model capabilities.
func CapabilityNegotiationMiddleware(opts CapabilityNegotiationOptions) (Middleware, StreamMiddleware) {
	return capabilityNegotiationMiddleware[*Request, *Response, StreamIterator](opts, NegotiateRequestCapabilities)
}

// NegotiateRequestCapabilities validates and injects chat request capability negotiation state.
//
// This helper exposes the same capability negotiation logic used by middleware so callers can
// apply explicit model validation closer to a transport or protocol implementation when needed.
func NegotiateRequestCapabilities(ctx context.Context, req *Request, opts CapabilityNegotiationOptions) (context.Context, *Request, error) {
	return negotiateCapabilitiesForRequest(ctx, req, opts,
		func(r *Request) string { return r.Model },
		DetectRequestedCapabilities,
		func(r *Request, model string) *Request {
			clone := cloneRequestShallow(r)
			clone.Model = model
			return clone
		},
	)
}

func NegotiateResponseCapabilities(ctx context.Context, req *ResponseRequest, opts CapabilityNegotiationOptions) (context.Context, *ResponseRequest, error) {
	return negotiateCapabilitiesForRequest(ctx, req, opts,
		func(r *ResponseRequest) string { return r.Model },
		DetectRequestedResponseCapabilities,
		func(r *ResponseRequest, model string) *ResponseRequest {
			clone := cloneResponseRequest(r)
			clone.Model = model
			return clone
		},
	)
}

func negotiateCapabilitiesForRequest[T any](ctx context.Context, req T, opts CapabilityNegotiationOptions,
	getModel func(T) string,
	detectCapabilities func(T) CapabilitySet,
	assignModel func(T, string) T,
) (context.Context, T, error) {
	if any(req) == nil || opts.Resolver == nil {
		return ctx, req, nil
	}
	requested := detectCapabilities(req)
	if negotiated, ok := reusableNegotiatedCapabilities(ctx, getModel(req), requested, opts.DefaultModel); ok {
		if getModel(req) == "" && negotiated.Model != "" {
			req = assignModel(req, negotiated.Model)
		}
		return ctx, req, nil
	}

	ctx, model, err := negotiateCapabilities(ctx, getModel(req), requested, opts)
	if err != nil {
		return ctx, req, err
	}
	if getModel(req) == "" && model != "" {
		req = assignModel(req, model)
	}
	return ctx, req, nil
}

func negotiateCapabilities(ctx context.Context, requestedModel string, requested CapabilitySet, opts CapabilityNegotiationOptions) (context.Context, string, error) {
	model := requestedModel
	if model == "" {
		model = opts.DefaultModel
	}

	if model == "" {
		if requested.Any() || opts.RequireKnownModel {
			return ctx, model, NewLLMError(ErrCodeUnsupportedCapability, "cannot negotiate capabilities without a model", nil)
		}
		return withNegotiatedCapabilities(ctx, NegotiatedCapabilities{Requested: requested}), model, nil
	}

	var (
		caps  ModelCapabilities
		known bool
		state CapabilityResolverState
		err   error
	)
	if withState, ok := opts.Resolver.(CapabilityResolverWithState); ok {
		caps, known, state, err = withState.ResolveCapabilitiesWithState(ctx, model)
	} else {
		caps, known, err = opts.Resolver.ResolveCapabilities(ctx, model)
	}
	if err != nil {
		return ctx, model, err
	}

	negotiated := NegotiatedCapabilities{
		Model:     model,
		Requested: requested,
		Known:     known,
		State:     state,
		Stale:     state.Stale,
	}
	if known {
		if caps.Model == "" {
			caps.Model = model
		}
		model = caps.Model
		negotiated.Model = caps.Model
		negotiated.Supported = caps.Supports
		if missing := caps.Supports.Missing(requested); len(missing) > 0 {
			return ctx, model, NewLLMError(ErrCodeUnsupportedCapability, fmt.Sprintf("model %q does not support %s", caps.Model, strings.Join(missing, ", ")), nil)
		}
	} else if opts.RequireKnownModel || (opts.RequireKnown && requested.Any()) {
		return ctx, model, NewLLMError(ErrCodeUnsupportedCapability, fmt.Sprintf("capabilities for model %q are unknown", model), nil)
	}

	return withNegotiatedCapabilities(ctx, negotiated), model, nil
}

func ValidateCapabilities(requested map[string]any, caps *ProtocolCapabilities, protocolName string) error {
	if caps == nil {
		return nil
	}
	for param := range requested {
		if _, supported := caps.SupportedParameters[param]; !supported {
			return NewLLMError(ErrCodeInvalidConfig,
				fmt.Sprintf("parameter %q not supported by %s", param, protocolName), nil)
		}
	}
	for _, conflict := range caps.ConflictingParameters {
		count := 0
		var present []string
		for _, param := range conflict {
			if _, exists := requested[param]; exists {
				count++
				present = append(present, param)
			}
		}
		if count > 1 {
			return NewLLMError(ErrCodeInvalidConfig,
				fmt.Sprintf("conflicting parameters cannot be used together: %v", present), nil)
		}
	}
	return nil
}

func hasLegacyReasoningRequest(extraBody map[string]any) bool {
	if len(extraBody) == 0 {
		return false
	}
	_, hasReasoning := extraBody["reasoning"]
	_, hasReasoningEffort := extraBody["reasoning_effort"]
	return hasReasoning || hasReasoningEffort
}

type ctxKeyNegotiatedCapabilities struct{}

// GetNegotiatedCapabilities returns the explicit capability contract for the current call.
func GetNegotiatedCapabilities(ctx context.Context) (NegotiatedCapabilities, bool) {
	if ctx == nil {
		return NegotiatedCapabilities{}, false
	}
	negotiated, ok := ctx.Value(ctxKeyNegotiatedCapabilities{}).(NegotiatedCapabilities)
	return negotiated, ok
}

func withNegotiatedCapabilities(ctx context.Context, negotiated NegotiatedCapabilities) context.Context {
	return context.WithValue(ctx, ctxKeyNegotiatedCapabilities{}, negotiated)
}

func reusableNegotiatedCapabilities(ctx context.Context, requestedModel string, requested CapabilitySet, defaultModel string) (NegotiatedCapabilities, bool) {
	negotiated, ok := GetNegotiatedCapabilities(ctx)
	if !ok {
		return NegotiatedCapabilities{}, false
	}
	if negotiated.Requested != requested {
		return NegotiatedCapabilities{}, false
	}

	resolvedModel := strings.TrimSpace(requestedModel)
	if resolvedModel == "" {
		resolvedModel = strings.TrimSpace(defaultModel)
	}

	if resolvedModel != "" && negotiated.Model != "" && resolvedModel != negotiated.Model {
		return NegotiatedCapabilities{}, false
	}

	return negotiated, true
}

// catalog

// CapabilityCatalogLoader loads a provider-agnostic model capability catalog.
type CapabilityCatalogLoader func(context.Context) ([]ModelCapabilities, error)

// CapabilityCatalogMatcher resolves one model request against a loaded capability catalog.
type CapabilityCatalogMatcher func(model string, catalog []ModelCapabilities) (ModelCapabilities, bool)

// CapabilityCatalogResolverOptions configures catalog-backed capability resolution.
type CapabilityCatalogResolverOptions struct {
	CacheTTL          time.Duration
	Now               func() time.Time
	Match             CapabilityCatalogMatcher
	AllowStaleOnError bool
}

// CatalogCapabilityResolver resolves model capabilities from a cached catalog snapshot.
type CatalogCapabilityResolver struct {
	load              CapabilityCatalogLoader
	match             CapabilityCatalogMatcher
	now               func() time.Time
	allowStaleOnError bool

	mu             sync.RWMutex
	catalog        []ModelCapabilities
	lastRefresh    time.Time
	lastRefreshErr error
	refreshing     bool
	refreshWaiters []chan error
	cacheTTL       time.Duration
}

// NewCatalogCapabilityResolver creates a catalog-backed capability resolver with optional caching.
func NewCatalogCapabilityResolver(load CapabilityCatalogLoader, opts CapabilityCatalogResolverOptions) *CatalogCapabilityResolver {
	resolver := &CatalogCapabilityResolver{
		load:              load,
		match:             opts.Match,
		now:               opts.Now,
		cacheTTL:          opts.CacheTTL,
		allowStaleOnError: opts.AllowStaleOnError,
	}
	if resolver.match == nil {
		resolver.match = DefaultCapabilityCatalogMatch
	}
	if resolver.now == nil {
		resolver.now = time.Now
	}
	return resolver
}

// ResolveCapabilities returns explicit capabilities for a model from the cached or freshly loaded catalog.
func (r *CatalogCapabilityResolver) ResolveCapabilities(ctx context.Context, model string) (ModelCapabilities, bool, error) {
	caps, known, _, err := r.ResolveCapabilitiesWithState(ctx, model)
	return caps, known, err
}

// ResolveCapabilitiesWithState returns explicit capabilities together with resolver state.
func (r *CatalogCapabilityResolver) ResolveCapabilitiesWithState(ctx context.Context, model string) (ModelCapabilities, bool, CapabilityResolverState, error) {
	if r == nil || r.load == nil {
		return ModelCapabilities{}, false, CapabilityResolverState{}, nil
	}
	if r.shouldRefresh() {
		if err := r.refreshOnce(ctx); err != nil {
			return ModelCapabilities{}, false, CapabilityResolverState{}, err
		}
	}
	if caps, ok := r.lookup(model); ok {
		return normalizeCatalogMatch(model, caps), true, r.state(), nil
	}
	if err := r.refreshOnce(ctx); err != nil {
		return ModelCapabilities{}, false, CapabilityResolverState{}, err
	}
	if caps, ok := r.lookup(model); ok {
		return normalizeCatalogMatch(model, caps), true, r.state(), nil
	}
	return ModelCapabilities{}, false, r.state(), nil
}

// Refresh reloads the capability catalog.
func (r *CatalogCapabilityResolver) Refresh(ctx context.Context) error {
	if r == nil || r.load == nil {
		return nil
	}
	catalog, err := r.load(ctx)
	if err != nil {
		return err
	}
	copyCatalog := append([]ModelCapabilities(nil), catalog...)
	r.mu.Lock()
	r.catalog = copyCatalog
	r.lastRefresh = r.now()
	r.lastRefreshErr = nil
	r.mu.Unlock()
	return nil
}

// DefaultCapabilityCatalogMatch resolves exact IDs first, then a unique prefix match.
func DefaultCapabilityCatalogMatch(model string, catalog []ModelCapabilities) (ModelCapabilities, bool) {
	for _, caps := range catalog {
		if caps.Model == model {
			return caps, true
		}
	}
	var match ModelCapabilities
	matchCount := 0
	for _, caps := range catalog {
		if strings.HasPrefix(caps.Model, model) {
			match = caps
			matchCount++
		}
	}
	if matchCount == 1 {
		return match, true
	}
	return ModelCapabilities{}, false
}

func normalizeCatalogMatch(requested string, caps ModelCapabilities) ModelCapabilities {
	if caps.Model == "" {
		caps.Model = requested
	}
	return caps
}

func (r *CatalogCapabilityResolver) lookup(model string) (ModelCapabilities, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if len(r.catalog) == 0 {
		return ModelCapabilities{}, false
	}
	return r.match(model, r.catalog)
}

func (r *CatalogCapabilityResolver) shouldRefresh() bool {
	if r == nil {
		return false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if len(r.catalog) == 0 {
		return true
	}
	if r.cacheTTL <= 0 {
		return false
	}
	return r.now().Sub(r.lastRefresh) >= r.cacheTTL
}

// LastRefreshError returns the most recent catalog refresh failure, if any.
// It is only retained when AllowStaleOnError is enabled and a stale snapshot was kept in service.
func (r *CatalogCapabilityResolver) LastRefreshError() error {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.lastRefreshErr
}

// ListModelCapabilities returns the current catalog snapshot used for capability resolution.
//
// The resolver refreshes the cache when needed (same policy as ResolveCapabilities) and returns
// a defensive copy so callers can inspect model metadata without mutating resolver state.
func (r *CatalogCapabilityResolver) ListModelCapabilities(ctx context.Context) ([]ModelCapabilities, CapabilityResolverState, error) {
	if r == nil || r.load == nil {
		return nil, CapabilityResolverState{}, nil
	}
	if r.shouldRefresh() {
		if err := r.refreshOnce(ctx); err != nil {
			return nil, r.state(), err
		}
	}

	r.mu.RLock()
	defer r.mu.RUnlock()
	state := CapabilityResolverState{Source: CapabilitySourceCatalog, RefreshedAt: r.lastRefresh, Stale: len(r.catalog) > 0 && r.lastRefreshErr != nil}
	if len(r.catalog) == 0 {
		return nil, state, nil
	}

	catalog := make([]ModelCapabilities, 0, len(r.catalog))
	for _, caps := range r.catalog {
		catalog = append(catalog, cloneModelCapabilities(caps))
	}
	return catalog, state, nil
}

func (r *CatalogCapabilityResolver) state() CapabilityResolverState {
	if r == nil {
		return CapabilityResolverState{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return CapabilityResolverState{Source: CapabilitySourceCatalog, RefreshedAt: r.lastRefresh, Stale: len(r.catalog) > 0 && r.lastRefreshErr != nil}
}

func (r *CatalogCapabilityResolver) refreshOnce(ctx context.Context) error {
	if r == nil {
		return nil
	}
	waiter, leader := r.beginRefresh()
	if !leader {
		select {
		case err := <-waiter:
			return err
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	err := r.Refresh(ctx)
	if err != nil && r.allowStaleOnError && r.hasCatalog() {
		r.mu.Lock()
		r.lastRefreshErr = err
		r.mu.Unlock()
		r.finishRefresh(nil)
		return nil
	}
	r.finishRefresh(err)
	return err
}

func (r *CatalogCapabilityResolver) beginRefresh() (chan error, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.refreshing {
		r.refreshing = true
		return nil, true
	}
	waiter := make(chan error, 1)
	r.refreshWaiters = append(r.refreshWaiters, waiter)
	return waiter, false
}

func (r *CatalogCapabilityResolver) finishRefresh(err error) {
	r.mu.Lock()
	waiters := r.refreshWaiters
	r.refreshWaiters = nil
	r.refreshing = false
	r.mu.Unlock()
	for _, waiter := range waiters {
		waiter <- err
		close(waiter)
	}
}

func (r *CatalogCapabilityResolver) hasCatalog() bool {
	if r == nil {
		return false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.catalog) > 0
}

// IndexModelCapabilities builds a model-id keyed capability map.
//
// Empty model IDs are ignored. IDs are trimmed before indexing.
// The last duplicate wins.
func IndexModelCapabilities(items []ModelCapabilities) map[string]ModelCapabilities {
	if len(items) == 0 {
		return nil
	}
	result := make(map[string]ModelCapabilities, len(items))
	for _, item := range items {
		id := strings.TrimSpace(item.Model)
		if id == "" {
			continue
		}
		result[id] = item
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// MergeModelCapabilities merges the same model across multiple capability catalogs.
//
// Merge rules:
//  1. Supports flags are OR-merged.
//  2. Limits fields take the max value.
//  3. Meta keys keep the first-seen value (later catalogs do not overwrite).
//  4. Provider keeps the first-seen value.
//
// modelID is trimmed before lookup.
func MergeModelCapabilities(modelID string, catalogs ...map[string]ModelCapabilities) (ModelCapabilities, bool) {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		return ModelCapabilities{}, false
	}
	var merged ModelCapabilities
	found := false
	for _, catalog := range catalogs {
		if len(catalog) == 0 {
			continue
		}
		caps, ok := catalog[modelID]
		if !ok {
			continue
		}
		if !found {
			merged = caps
			merged.Meta = cloneMap(caps.Meta)
			found = true
			continue
		}
		merged.Supports.JSONMode = merged.Supports.JSONMode || caps.Supports.JSONMode
		merged.Supports.ToolCalls = merged.Supports.ToolCalls || caps.Supports.ToolCalls
		merged.Supports.Vision = merged.Supports.Vision || caps.Supports.Vision
		merged.Supports.Reasoning = merged.Supports.Reasoning || caps.Supports.Reasoning
		merged.Limits.MaxContextWindowTokens = maxIntValue(merged.Limits.MaxContextWindowTokens, caps.Limits.MaxContextWindowTokens)
		merged.Limits.MaxOutputTokens = maxIntValue(merged.Limits.MaxOutputTokens, caps.Limits.MaxOutputTokens)
		merged.Limits.MaxNonStreamingOutputTokens = maxIntValue(merged.Limits.MaxNonStreamingOutputTokens, caps.Limits.MaxNonStreamingOutputTokens)
		merged.Limits.MaxPromptTokens = maxIntValue(merged.Limits.MaxPromptTokens, caps.Limits.MaxPromptTokens)
		if len(caps.Meta) > 0 {
			if merged.Meta == nil {
				merged.Meta = make(map[string]any, len(caps.Meta))
			}
			for key, value := range caps.Meta {
				if _, exists := merged.Meta[key]; !exists {
					merged.Meta[key] = value
				}
			}
		}
	}
	return merged, found
}

func maxIntValue(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func MetaString(metadata map[string]any, key string) string {
	if len(metadata) == 0 {
		return ""
	}
	value, _ := metadata[key].(string)
	return strings.TrimSpace(value)
}

func MetaBool(metadata map[string]any, key string) bool {
	if len(metadata) == 0 {
		return false
	}
	value, _ := metadata[key].(bool)
	return value
}

func MetaFloat64(metadata map[string]any, key string) float64 {
	if len(metadata) == 0 {
		return 0
	}
	switch value := metadata[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	default:
		return 0
	}
}

func MetaStringSlice(metadata map[string]any, key string) []string {
	if len(metadata) == 0 {
		return nil
	}
	raw, ok := metadata[key]
	if !ok {
		return nil
	}
	switch value := raw.(type) {
	case []string:
		return append([]string(nil), value...)
	case []any:
		result := make([]string, 0, len(value))
		for _, item := range value {
			text, _ := item.(string)
			text = strings.TrimSpace(text)
			if text != "" {
				result = append(result, text)
			}
		}
		return result
	default:
		return nil
	}
}

func CloneMetadata(metadata map[string]any) map[string]any {
	return cloneMap(metadata)
}

// AttachCatalogStateMetadata appends resolver state fields into metadata.
//
// When state.Source is empty, metadata is returned unchanged.
// keyPrefix defaults to "capabilities.catalog" when empty.
//
// Added keys:
//
//	<keyPrefix>_source
//	<keyPrefix>_stale
func AttachCatalogStateMetadata(metadata map[string]any, state CapabilityResolverState, keyPrefix string) map[string]any {
	if state.Source == "" {
		return metadata
	}
	keyPrefix = strings.TrimSpace(keyPrefix)
	if keyPrefix == "" {
		keyPrefix = "capabilities.catalog"
	}
	if metadata == nil {
		metadata = make(map[string]any, 2)
	}
	metadata[keyPrefix+"_source"] = state.Source
	metadata[keyPrefix+"_stale"] = state.Stale
	return metadata
}

func EndpointSupported(endpoint string, supportedEndpoints []string) bool {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return true
	}
	if len(supportedEndpoints) == 0 {
		if strings.EqualFold(endpoint, "/responses") || strings.EqualFold(endpoint, "ws:/responses") {
			return false
		}
		return true
	}
	for _, item := range supportedEndpoints {
		if strings.EqualFold(strings.TrimSpace(item), endpoint) {
			return true
		}
	}
	return false
}

func MetaSupportsEndpoint(metadata map[string]any, endpoint string) bool {
	return EndpointSupported(endpoint, MetaStringSlice(metadata, "copilot.supported_endpoints"))
}

func MetaMatchesRestrictedAccess(metadata map[string]any, accessTags []string) bool {
	restrictedTo := MetaStringSlice(metadata, "copilot.billing_restricted_to")
	if len(restrictedTo) == 0 {
		return true
	}
	if len(accessTags) == 0 {
		return false
	}
	allowed := make(map[string]struct{}, len(accessTags))
	for _, tag := range accessTags {
		tag = strings.ToLower(strings.TrimSpace(tag))
		if tag != "" {
			allowed[tag] = struct{}{}
		}
	}
	for _, item := range restrictedTo {
		if _, ok := allowed[strings.ToLower(strings.TrimSpace(item))]; ok {
			return true
		}
	}
	return false
}
