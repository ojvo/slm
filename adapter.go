package slm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
)

type requestValidator[Req any] func(Req) error
type requestBodyBuilder[Req any] func(Req, string, bool) ([]byte, error)
type responseDecoder[Resp any] func(*http.Response) (Resp, error)
type streamFactory[StreamResp any] func(*http.Response, *OpenAICodec) (StreamResp, error)
type postValidator[Req any] func(*http.Response, Req, string, bool, *OpenAICodec, *protocolBase) (*http.Response, error)

type genericAdapter[Req any, Resp any, StreamResp any] struct {
	base         protocolBase
	codec        *OpenAICodec
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

	return a.createStream(resp, a.codec)
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
		return a.postValidate(resp, req, model, stream, a.codec, &a.base)
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

func chatPostValidate(resp *http.Response, req *Request, model string, stream bool, codec *OpenAICodec, base *protocolBase) (*http.Response, error) {
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
		return nil, fmt.Errorf("marshal fallback request: %w", err)
	}

	headers := streamRequestHeaders(stream)
	return base.doPost(context.Background(), "/chat/completions", headers, fallbackBody)
}

func newChatAdapter(base protocolBase, codec *OpenAICodec) *genericAdapter[*Request, *Response, StreamIterator] {
	return &genericAdapter[*Request, *Response, StreamIterator]{
		base:      base,
		codec:     codec,
		path:      "/chat/completions",
		validate:  validateChatRequest,
		buildBody: codec.BuildChatRequestBody,
		decodeResp: func(resp *http.Response) (*Response, error) {
			var oaiResp oaiResponse
			if err := decodeJSONResponse(resp, &oaiResp); err != nil {
				return nil, err
			}
			return codec.ConvertChatResponse(&oaiResp), nil
		},
		createStream: func(resp *http.Response, c *OpenAICodec) (StreamIterator, error) {
			return NewSSEReader(resp.Body, c.ParseChatSSEChunk), nil
		},
		postValidate: chatPostValidate,
	}
}

func newResponsesAdapter(base protocolBase, codec *OpenAICodec) *genericAdapter[*ResponseRequest, *ResponseObject, ResponseStream] {
	return &genericAdapter[*ResponseRequest, *ResponseObject, ResponseStream]{
		base:      base,
		codec:     codec,
		path:      "/responses",
		validate:  validateResponseRequest,
		buildBody: codec.BuildResponsesRequestBody,
		decodeResp: func(resp *http.Response) (*ResponseObject, error) {
			var decoded oaiResponseObject
			if err := decodeJSONResponse(resp, &decoded); err != nil {
				return nil, err
			}
			return codec.ConvertResponseObject(&decoded), nil
		},
		createStream: func(resp *http.Response, c *OpenAICodec) (ResponseStream, error) {
			return newOpenAIResponseStream(resp, c), nil
		},
	}
}
