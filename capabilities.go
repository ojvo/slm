package slm

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// ReasoningOptions defines explicit reasoning request settings.
//
// The zero value requests reasoning capability without provider-specific tuning.
// Effort is forwarded to OpenAI-compatible providers via "reasoning_effort".
type ReasoningOptions struct {
	Effort string
}

// CapabilitySet represents the feature surface of a model or request.
type CapabilitySet struct {
	JSONMode  bool
	ToolCalls bool
	Vision    bool
	Reasoning bool
}

// Any reports whether any capability bit is enabled.
func (c CapabilitySet) Any() bool {
	return c.JSONMode || c.ToolCalls || c.Vision || c.Reasoning
}

func (c CapabilitySet) missing(requested CapabilitySet) []string {
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
	Supports CapabilitySet
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
		Reasoning: req.Reasoning != nil || hasLegacyReasoningRequest(req.ExtraBody),
	}

	for _, msg := range req.Messages {
		for _, part := range msg.Content {
			if _, ok := part.(ImagePart); ok {
				requested.Vision = true
				return requested
			}
		}
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
		Reasoning: req.Reasoning != nil || hasLegacyReasoningRequest(req.ExtraBody),
	}
}

// CapabilityNegotiationMiddleware validates explicit request requirements against model capabilities.
func CapabilityNegotiationMiddleware(opts CapabilityNegotiationOptions) (Middleware, StreamMiddleware) {
	passUnary := func(next Handler) Handler { return next }
	passStream := func(next StreamHandler) StreamHandler { return next }
	if opts.Resolver == nil {
		return passUnary, passStream
	}

	negotiate := func(ctx context.Context, req *Request) (context.Context, *Request, error) {
		return NegotiateRequestCapabilities(ctx, req, opts)
	}

	unary := func(next Handler) Handler {
		return func(ctx context.Context, req *Request) (*Response, error) {
			ctx, req, err := negotiate(ctx, req)
			if err != nil {
				return nil, err
			}
			return next(ctx, req)
		}
	}

	stream := func(next StreamHandler) StreamHandler {
		return func(ctx context.Context, req *Request) (StreamIterator, error) {
			ctx, req, err := negotiate(ctx, req)
			if err != nil {
				return nil, err
			}
			return next(ctx, req)
		}
	}

	return unary, stream
}

// NegotiateRequestCapabilities validates and injects chat request capability negotiation state.
//
// This helper exposes the same capability negotiation logic used by middleware so callers can
// apply explicit model validation closer to a transport or protocol implementation when needed.
func NegotiateRequestCapabilities(ctx context.Context, req *Request, opts CapabilityNegotiationOptions) (context.Context, *Request, error) {
	if req == nil || opts.Resolver == nil {
		return ctx, req, nil
	}
	requested := DetectRequestedCapabilities(req)
	ctx, model, err := negotiateCapabilities(ctx, req.Model, requested, opts)
	if err != nil {
		return ctx, req, err
	}
	if req.Model == "" && model != "" {
		clone := cloneRequest(req)
		clone.Model = model
		req = clone
	}
	return ctx, req, nil
}

// NegotiateResponseCapabilities validates and injects responses request capability negotiation state.
func NegotiateResponseCapabilities(ctx context.Context, req *ResponseRequest, opts CapabilityNegotiationOptions) (context.Context, *ResponseRequest, error) {
	if req == nil || opts.Resolver == nil {
		return ctx, req, nil
	}
	requested := DetectRequestedResponseCapabilities(req)
	ctx, model, err := negotiateCapabilities(ctx, req.Model, requested, opts)
	if err != nil {
		return ctx, req, err
	}
	if req.Model == "" && model != "" {
		clone := cloneResponseRequest(req)
		clone.Model = model
		req = clone
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
		if missing := caps.Supports.missing(requested); len(missing) > 0 {
			return ctx, model, NewLLMError(ErrCodeUnsupportedCapability, fmt.Sprintf("model %q does not support %s", caps.Model, strings.Join(missing, ", ")), nil)
		}
	} else if opts.RequireKnownModel || (opts.RequireKnown && requested.Any()) {
		return ctx, model, NewLLMError(ErrCodeUnsupportedCapability, fmt.Sprintf("capabilities for model %q are unknown", model), nil)
	}

	return withNegotiatedCapabilities(ctx, negotiated), model, nil
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
