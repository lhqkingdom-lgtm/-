"""Reusable, minimal OpenAI-compatible JSON client.

The client owns only transport concerns.  Callers remain responsible for
domain-schema validation and for projecting local identifiers before a remote
call.  Credentials are never included in reprs or exception messages.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .config import DEFAULT_CONFIG, load_api_key, load_llm_config


MAX_PROVIDER_RESPONSE_BYTES = 1_000_000


class JsonClientError(RuntimeError):
    """The structured model boundary could not return a usable JSON object."""


class JsonClientConfigurationError(JsonClientError):
    """The endpoint configuration is unsafe or incomplete."""


class JsonClientTransportError(JsonClientError):
    """The provider request or response envelope failed."""


Transport = Callable[[str, Mapping[str, str], bytes, float], bytes]


class _NoRedirect(urlrequest.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def urllib_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> bytes:
    request = urlrequest.Request(url, data=body, headers=dict(headers), method="POST")
    opener = urlrequest.build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
    if len(payload) > MAX_PROVIDER_RESPONSE_BYTES:
        raise JsonClientTransportError("provider response exceeded the local size limit")
    return payload


_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(
        r"(?i)(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|"
        r"password|passwd|secret|密钥|令牌|密码|口令|凭据|私钥|授权)"
        r"\s*(?::|=|：|\bis\b(?:\s+set\s+to)?|\bequals?\b|是|为)\s*"
        r"(?:[\"'][^\"']+[\"']|[^\s,;，。；]+)"
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _assert_no_secrets(value: Any) -> None:
    if any(pattern.search(text) for text in _iter_strings(value) for pattern in _SECRET_PATTERNS):
        raise JsonClientTransportError("structured model payload may contain a secret")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_TRAILING_JSON_COMMA = re.compile(r",\s*([}\]])")
_MISNESTED_CONTENT_EVIDENCE = re.compile(
    r'("content"\s*:\s*\{\s*"statement"\s*:\s*("(?:\\.|[^"\\])*"))\s*,\s*"evidence"\s*:'
)
_MISNESTED_CONTENT_ACTION = re.compile(
    r'("content"\s*:\s*\{\s*"statement"\s*:\s*("(?:\\.|[^"\\])*"))\s*,\s*"action_hint"\s*:'
)


def _loads_model_json(content: str) -> dict[str, Any]:
    """Parse provider JSON content with a tiny formatting-only repair pass.

    Some JSON-mode providers still occasionally return a fenced object, a
    short prose prefix, or trailing commas.  This function never invents or
    rewrites domain content; it only strips wrappers and removes commas that
    are illegal immediately before ``}`` or ``]``.
    """

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if starts:
        text = text[min(starts) :]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        repaired = _TRAILING_JSON_COMMA.sub(r"\1", text)
        repaired = _MISNESTED_CONTENT_EVIDENCE.sub(
            r'\1},"status":"active","evidence":', repaired
        )
        repaired = _MISNESTED_CONTENT_ACTION.sub(
            r'\1},"status":"active","action_hint":', repaired
        )
        parsed = json.loads(repaired)
    parsed = _normalize_model_json_shape(parsed)
    if not isinstance(parsed, Mapping):
        raise JsonClientTransportError("provider JSON result must be an object")
    return dict(parsed)


def _normalize_model_json_shape(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_model_json_shape(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    result = {key: _normalize_model_json_shape(item) for key, item in value.items()}
    content = result.get("content")
    if (
        isinstance(content, Mapping)
        and isinstance(content.get("statement"), str)
        and (
            "evidence" in content
            or "action_hint" in content
            or "status" in content
        )
    ):
        result["content"] = {"statement": content["statement"]}
        if "status" not in result:
            result["status"] = content.get("status") or "active"
        for misplaced in ("evidence", "action_hint"):
            if misplaced not in result and misplaced in content:
                result[misplaced] = content[misplaced]
    return result


@dataclass(frozen=True)
class OpenAICompatibleJsonClient:
    """Send one stateless chat-completions request and return a JSON object."""

    base_url: str
    model: str
    api_key: str = field(default="", repr=False, compare=False)
    timeout_seconds: float = 45.0
    max_attempts: int = 3
    allow_remote: bool = False
    transport: Transport = field(default=urllib_transport, repr=False, compare=False)
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)
    jitter: Callable[[], float] = field(default=random.random, repr=False, compare=False)

    @classmethod
    def from_config(cls, path: Path = DEFAULT_CONFIG) -> "OpenAICompatibleJsonClient":
        config = load_llm_config(path)
        return cls(
            base_url=config.base_url,
            model=config.model,
            api_key=load_api_key(config),
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
            allow_remote=config.allow_remote,
        )

    def endpoint(self) -> str:
        url = self.base_url.rstrip("/")
        endpoint = url if url.endswith("/chat/completions") else url + "/chat/completions"
        parsed = urlparse.urlsplit(endpoint)
        if parsed.username is not None or parsed.password is not None:
            raise JsonClientConfigurationError(
                "model URL must not contain embedded credentials"
            )
        if parsed.query or parsed.fragment:
            raise JsonClientConfigurationError(
                "model URL cannot contain a query string or fragment"
            )
        host = (parsed.hostname or "").lower()
        loopback = host in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme == "http" and not loopback:
            raise JsonClientConfigurationError(
                "plain HTTP is allowed only for a loopback model service"
            )
        if parsed.scheme not in {"http", "https"} or not host:
            raise JsonClientConfigurationError(
                "model URL must be an absolute HTTP(S) URL"
            )
        if not loopback and not self.allow_remote:
            raise JsonClientConfigurationError(
                "remote model calls require allow_remote=true"
            )
        return endpoint

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, Any] | str,
        max_tokens: int = 1600,
        forbid_secrets: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise JsonClientConfigurationError("system_prompt must not be empty")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 16000:
            raise JsonClientConfigurationError("max_tokens must be between 1 and 16000")
        content = user_payload if isinstance(user_payload, str) else _canonical_json(user_payload)
        if not content.strip():
            raise JsonClientConfigurationError("user_payload must not be empty")
        if forbid_secrets:
            _assert_no_secrets({"system": system_prompt, "user": content})

        endpoint = self.endpoint()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        configured_attempts = max(1, self.max_attempts)
        attempts_made = 0
        last_error: Exception | None = None

        def wait_before_retry(attempt: int, retry_after: str | None = None) -> None:
            delay: float | None = None
            if retry_after:
                try:
                    delay = max(0.0, float(retry_after))
                except ValueError:
                    delay = None
            if delay is None:
                delay = 0.25 * (2 ** (attempt - 1))
                delay += min(max(self.jitter(), 0.0), 1.0) * 0.1
            self.sleeper(min(delay, 5.0))

        for attempt in range(1, configured_attempts + 1):
            attempts_made = attempt
            try:
                payload: dict[str, Any] = {
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "response_format": {"type": "json_object"},
                    "messages": messages,
                }
                if (urlparse.urlsplit(endpoint).hostname or "").lower() == "api.deepseek.com":
                    payload["thinking"] = {"type": "disabled"}
                body = _canonical_json(payload).encode("utf-8")
                raw = self.transport(endpoint, headers, body, self.timeout_seconds)
                if len(raw) > MAX_PROVIDER_RESPONSE_BYTES:
                    raise JsonClientTransportError(
                        "provider response exceeded the local size limit"
                    )
                envelope = json.loads(raw.decode("utf-8"))
                if not isinstance(envelope, Mapping):
                    raise JsonClientTransportError(
                        "provider response envelope must be an object"
                    )
                choice = envelope["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason in {
                    "length",
                    "content_filter",
                    "insufficient_system_resource",
                }:
                    raise JsonClientTransportError(
                        f"provider did not complete JSON output: {finish_reason}"
                    )
                response_content = choice["message"]["content"]
                if isinstance(response_content, list):
                    response_content = "".join(
                        item.get("text", "")
                        for item in response_content
                        if isinstance(item, Mapping)
                    )
                if not isinstance(response_content, str) or not response_content.strip():
                    raise JsonClientTransportError(
                        "provider returned empty JSON content"
                    )
                try:
                    result = _loads_model_json(response_content)
                except json.JSONDecodeError as error:
                    last_error = error
                    if attempt < configured_attempts:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": response_content[:2000],
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was not strict valid JSON "
                                    "accepted by JSON.parse/json.loads. Return the "
                                    "complete JSON object again. Do not use trailing "
                                    "commas, comments, Markdown, or extra text."
                                ),
                            }
                        )
                        wait_before_retry(attempt)
                        continue
                    raise JsonClientTransportError(
                        "provider returned malformed JSON content"
                    ) from error
                return result
            except JsonClientConfigurationError:
                raise
            except JsonClientTransportError:
                # A complete but invalid envelope is deterministic; do not pay
                # for repeated calls that cannot repair the local contract.
                raise
            except (KeyError, IndexError, TypeError, ValueError, UnicodeError) as error:
                raise JsonClientTransportError(
                    "provider returned an invalid JSON envelope"
                ) from error
            except urlerror.HTTPError as error:
                last_error = error
                try:
                    body = error.read(1000).decode("utf-8", errors="replace").strip()
                except Exception:
                    body = ""
                if error.code not in {408, 409, 429, 500, 502, 503, 504}:
                    suffix = f": {body}" if body else ""
                    raise JsonClientTransportError(f"provider HTTP {error.code}{suffix}") from error
                if attempt < configured_attempts:
                    wait_before_retry(attempt, error.headers.get("Retry-After"))
            except (urlerror.URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt < configured_attempts:
                    wait_before_retry(attempt)
        raise JsonClientTransportError(
            f"model request failed after {attempts_made} attempt(s)"
        ) from last_error
