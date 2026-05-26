import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib import error, request


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    usage: Dict[str, Any] | None = None


class ChatCompletionError(RuntimeError):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        body: str | None = None,
        usage: Dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.body = body
        self.usage = usage
        self.prompt_tokens = (usage or {}).get("prompt_tokens")
        self.completion_tokens = (usage or {}).get("completion_tokens")
        self.total_tokens = (usage or {}).get("total_tokens")
        self.cost_usd = (usage or {}).get("cost")


def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout_seconds: int = 300,
    api_key: str | None = None,
    extra_body: Dict[str, Any] | None = None,
) -> str:
    return chat_completion_with_usage(
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        extra_body=extra_body,
    ).content


def _resolve_api_key(base_url: str, api_key: str | None = None) -> str:
    if api_key:
        return api_key
    if "openrouter.ai" in base_url:
        return os.environ.get("OPENROUTER_API_KEY", "")
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or "dummy"
    )


def _headers(base_url: str, api_key: str | None = None) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_resolve_api_key(base_url, api_key)}",
    }
    if "openrouter.ai" in base_url:
        site_url = os.environ.get("OPENROUTER_SITE_URL")
        app_name = os.environ.get("OPENROUTER_APP_NAME", "cadbench-freecad-agent")
        if site_url:
            headers["HTTP-Referer"] = site_url
        headers["X-Title"] = app_name
    return headers


def chat_completion_with_usage(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout_seconds: int = 300,
    api_key: str | None = None,
    extra_body: Dict[str, Any] | None = None,
) -> ChatCompletionResult:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers=_headers(base_url, api_key),
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body[:1000] if body else exc.reason
        raise ChatCompletionError(
            f"HTTP Error {exc.code}: {exc.reason}: {detail}",
            status=exc.code,
            body=body,
        ) from exc
    except error.URLError as exc:
        raise ChatCompletionError(
            f"URL error while calling chat completion API: {exc}",
        ) from exc
    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ChatCompletionError(
            f"Chat completion API returned invalid JSON: {exc}",
            body=response_body[:4000],
        ) from exc
    usage = result.get("usage") or {}
    message = result.get("choices", [{}])[0].get("message") or {}
    content = message.get("content")
    if content is None:
        compact = {
            "id": result.get("id"),
            "model": result.get("model"),
            "provider": result.get("provider"),
            "choices": [
                {
                    "finish_reason": choice.get("finish_reason"),
                    "native_finish_reason": choice.get("native_finish_reason"),
                    "message": {
                        "role": (choice.get("message") or {}).get("role"),
                        "content": (choice.get("message") or {}).get("content"),
                        "refusal": (choice.get("message") or {}).get("refusal"),
                    },
                }
                for choice in result.get("choices", [])[:2]
            ],
            "usage": usage,
        }
        raise ChatCompletionError(
            "Chat completion response did not include visible assistant content.",
            body=json.dumps(compact),
            usage=usage,
        )
    return ChatCompletionResult(
        content=content,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cost_usd=usage.get("cost"),
        usage=usage,
    )


def extract_python_code(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip() + "\n"
    open_fence = re.match(r"\s*```(?:python|py)?\s*", text, flags=re.IGNORECASE)
    if open_fence:
        return text[open_fence.end() :].strip() + "\n"
    return text.strip() + "\n"


def estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
    char_count = sum(len(message.get("content", "")) for message in messages)
    return max(1, char_count // 4) + 4 * len(messages) + 8


def cap_max_tokens_for_context(
    messages: List[Dict[str, str]],
    max_tokens: int,
    context_window_tokens: int | None,
    safety_margin: int = 256,
) -> tuple[int, int | None]:
    if not context_window_tokens:
        return max_tokens, None
    estimated_prompt_tokens = estimate_message_tokens(messages)
    available = context_window_tokens - estimated_prompt_tokens - safety_margin
    return max(1, min(max_tokens, available)), estimated_prompt_tokens


def is_context_window_error(exc: ChatCompletionError) -> bool:
    text = f"{exc} {exc.body or ''}".lower()
    return (
        exc.status in (400, 413)
        and (
            "maximum context length" in text
            or "context length" in text
            or "context window" in text
            or re.search(r"\d+\s+input\s*\+\s*\d+\s+output\s*=", text) is not None
        )
    )


def reduced_max_tokens_after_context_error(
    current_max_tokens: int,
    error_text: str,
    min_response_tokens: int = 128,
    safety_margin: int = 64,
) -> int:
    patterns = [
        r"(\d+)\s+input\s*\+\s*(\d+)\s+output\s*=\s*\d+\s*>\s*(\d+)",
        r"(\d+)\s+in the messages,\s*(\d+)\s+in the completion.*?maximum context length is\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            prompt_tokens = int(match.group(1))
            max_context = int(match.group(3))
            allowed = max_context - prompt_tokens - safety_margin
            return max(1, min(current_max_tokens - 1, allowed))
    return max(min_response_tokens, current_max_tokens // 2)
