package slm

import (
	"context"
	"strings"
	"sync"
	"time"
)

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
