package slm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type requestValidator[Req any] func(Req) error
type requestBodyBuilder[Req any] func(Req, string, bool) ([]byte, error)
type responseDecoder[Resp any] func(*http.Response) (Resp, error)
type streamFactory[StreamResp any] func(*http.Response) (StreamResp, error)
type postValidator[Req any] func(context.Context, *http.Response, Req, string, bool) (*http.Response, error)

type genericAdapter[Req any, Resp any, StreamResp any] struct {
	base         protocolBase
	path         string
	validate     requestValidator[Req]
	buildBody    requestBodyBuilder[Req]
	decodeResp   responseDecoder[Resp]
	createStream streamFactory[StreamResp]
	postValidate postValidator[Req]
}

func (a *genericAdapter[Req, Resp, StreamResp]) generate(ctx context.Context, req Req) (Resp, error) {
	var zero Resp
	resp, err := a.doRequest(ctx, req, false)
	if err != nil {
		return zero, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return zero, llmErrorFromResponse(resp)
	}

	return a.decodeResp(resp)
}

func (a *genericAdapter[Req, Resp, StreamResp]) stream(ctx context.Context, req Req) (StreamResp, error) {
	var zero StreamResp
	resp, err := a.doRequest(ctx, req, true)
	if err != nil {
		return zero, err
	}

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		return zero, llmErrorFromResponse(resp)
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(strings.ToLower(contentType), "text/event-stream") {
		defer resp.Body.Close()
		return zero, llmErrorFromResponse(resp)
	}

	return a.createStream(resp)
}

func (a *genericAdapter[Req, Resp, StreamResp]) doRequest(ctx context.Context, req Req, stream bool) (*http.Response, error) {
	if err := a.validate(req); err != nil {
		return nil, err
	}

	model, err := a.base.resolveModel(a.extractModel(req))
	if err != nil {
		return nil, err
	}

	body, err := a.buildBody(req, model, stream)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	headers := streamRequestHeaders(stream)

	resp, err := a.base.doPost(ctx, a.path, headers, body)
	if err != nil {
		return nil, err
	}

	if a.postValidate != nil {
		return a.postValidate(ctx, resp, req, model, stream)
	}

	return resp, nil
}

func (a *genericAdapter[Req, Resp, StreamResp]) extractModel(req Req) string {
	type modelExtractor interface{ GetModel() string }
	if m, ok := any(req).(modelExtractor); ok {
		return m.GetModel()
	}
	return ""
}

func validateChatRequest(req *Request) error {
	if req == nil {
		return NewLLMError(ErrCodeInvalidConfig, "request is nil", nil)
	}
	if len(req.Messages) == 0 {
		return NewLLMError(ErrCodeInvalidConfig, "messages is required", nil)
	}
	return nil
}

func validateResponseRequest(req *ResponseRequest) error {
	if req == nil {
		return NewLLMError(ErrCodeInvalidConfig, "request is nil", nil)
	}
	if len(req.Input) == 0 {
		return NewLLMError(ErrCodeInvalidConfig, "input is required", nil)
	}
	return nil
}

func newChatAdapter(base protocolBase, codec *openAICodec) *genericAdapter[*Request, *Response, StreamIterator] {
	return &genericAdapter[*Request, *Response, StreamIterator]{
		base:     base,
		path:     "/chat/completions",
		validate: validateChatRequest,
		buildBody: func(req *Request, model string, stream bool) ([]byte, error) {
			return codec.BuildChatRequestBody(req, model, stream)
		},
		decodeResp: func(resp *http.Response) (*Response, error) {
			var oaiResp oaiResponse
			if err := decodeJSONResponse(resp, &oaiResp); err != nil {
				return nil, err
			}
			return codec.ConvertChatResponse(&oaiResp), nil
		},
		createStream: func(resp *http.Response) (StreamIterator, error) {
			return NewSSEReader(resp.Body, codec.ParseChatSSEChunk), nil
		},
		postValidate: func(ctx context.Context, resp *http.Response, req *Request, model string, stream bool) (*http.Response, error) {
			return chatPostValidate(ctx, resp, req, model, stream, codec, &base)
		},
	}
}

func newResponsesAdapter(base protocolBase, codec *openAICodec) *genericAdapter[*ResponseRequest, *ResponseObject, ResponseStream] {
	return &genericAdapter[*ResponseRequest, *ResponseObject, ResponseStream]{
		base:     base,
		path:     "/responses",
		validate: validateResponseRequest,
		buildBody: func(req *ResponseRequest, model string, stream bool) ([]byte, error) {
			return codec.BuildResponsesRequestBody(req, model, stream)
		},
		decodeResp: func(resp *http.Response) (*ResponseObject, error) {
			var decoded oaiResponseObject
			if err := decodeJSONResponse(resp, &decoded); err != nil {
				return nil, err
			}
			return codec.ConvertResponseObject(&decoded), nil
		},
		createStream: func(resp *http.Response) (ResponseStream, error) {
			return newOpenAIResponseStream(resp, codec), nil
		},
	}
}

func chatPostValidate(ctx context.Context, resp *http.Response, req *Request, model string, stream bool, codec *openAICodec, base *protocolBase) (*http.Response, error) {
	if resp.StatusCode != http.StatusBadRequest || req.Reasoning == nil {
		return resp, nil
	}

	bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	_ = resp.Body.Close()

	if !isReasoningEffortUnsupported(bodyBytes) {
		resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		resp.ContentLength = int64(len(bodyBytes))
		return resp, nil
	}

	fallbackReq := cloneRequestShallow(req)
	fallbackReq.Reasoning = nil

	fallbackBody, err := codec.BuildChatRequestBody(fallbackReq, model, stream)
	if err != nil {
		resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		resp.ContentLength = int64(len(bodyBytes))
		return resp, nil
	}

	headers := streamRequestHeaders(stream)
	fallbackResp, err := base.doPost(ctx, "/chat/completions", headers, fallbackBody)
	if err != nil {
		resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		resp.ContentLength = int64(len(bodyBytes))
		return resp, nil
	}
	return fallbackResp, nil
}
