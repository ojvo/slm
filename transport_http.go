package slm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// HTTPTransport 通过标准 HTTP + Bearer token 通信的 Transport 实现。
// 这是从引擎提取的传输层，与协议层解耦。
type HTTPTransport struct {
	Client      *http.Client
	BaseURL     string
	APIKey      string
	AuthHeader  string
	AuthPrefix  string
	ExtraHeader map[string]string
	mu          sync.RWMutex
}

// NewHTTPTransport 创建标准 HTTP 传输层。
func NewHTTPTransport(baseURL, apiKey string) *HTTPTransport {
	baseURL = strings.TrimRight(baseURL, "/")
	return &HTTPTransport{
		Client:      newNoProxyHTTPClient(120 * time.Second),
		BaseURL:     baseURL,
		APIKey:      apiKey,
		AuthHeader:  "Authorization",
		AuthPrefix:  "Bearer ",
		ExtraHeader: make(map[string]string),
	}
}

func newNoProxyHTTPClient(timeout time.Duration) *http.Client {
	transport := &http.Transport{
		Proxy: func(*http.Request) (*url.URL, error) { return nil, nil },
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          10,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
	client := &http.Client{Transport: transport}
	if timeout > 0 {
		client.Timeout = timeout
	}
	return client
}

// NewHTTPTransportWithClient 创建使用自定义 HTTP 客户端的传输层。
func NewHTTPTransportWithClient(client *http.Client, baseURL, apiKey string) *HTTPTransport {
	baseURL = strings.TrimRight(baseURL, "/")
	return &HTTPTransport{
		Client:      client,
		BaseURL:     baseURL,
		APIKey:      apiKey,
		AuthHeader:  "Authorization",
		AuthPrefix:  "Bearer ",
		ExtraHeader: make(map[string]string),
	}
}

// Do 发送 HTTP 请求并返回原始响应。
func (t *HTTPTransport) Do(ctx context.Context, method, path string, headers map[string]string, body []byte) (*http.Response, error) {
	var bodyReader io.Reader
	if len(body) > 0 {
		bodyReader = bytes.NewReader(body)
	}

	t.mu.RLock()
	baseURL := t.BaseURL
	apiKey := t.APIKey
	authHeader := t.AuthHeader
	authPrefix := t.AuthPrefix
	var extraHeader map[string]string
	if len(t.ExtraHeader) > 0 {
		extraHeader = make(map[string]string, len(t.ExtraHeader))
		for k, v := range t.ExtraHeader {
			extraHeader[k] = v
		}
	}
	t.mu.RUnlock()

	url := baseURL + path
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	if len(body) > 0 {
		req.Header.Set("Content-Type", "application/json")
	}
	if apiKey != "" {
		req.Header.Set(authHeader, authPrefix+apiKey)
	}

	for k, v := range extraHeader {
		req.Header.Set(k, v)
	}

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := t.Client.Do(req)
	if err != nil {
		return nil, NewLLMError(ErrCodeNetwork, fmt.Sprintf("request failed: %v", err), err)
	}
	return resp, nil
}

// SetExtraHeader 设置额外的请求头。
func (t *HTTPTransport) SetExtraHeader(key, value string) {
	t.mu.Lock()
	if t.ExtraHeader == nil {
		t.ExtraHeader = make(map[string]string)
	}
	t.ExtraHeader[key] = value
	t.mu.Unlock()
}
