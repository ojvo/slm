package slm

import "strings"

func NormalizeProvider(provider, fallback string) string {
	provider = strings.ToLower(strings.TrimSpace(provider))
	if provider == "" {
		provider = strings.ToLower(strings.TrimSpace(fallback))
	}
	return provider
}

func ProviderKey(provider string) string {
	return NormalizeProvider(provider, "")
}

func ProviderLabel(provider, suffix string) string {
	key := ProviderKey(provider)
	if key == "" {
		return ""
	}
	return key + "-" + suffix
}
