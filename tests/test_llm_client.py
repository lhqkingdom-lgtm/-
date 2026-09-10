from __future__ import annotations

import json
import unittest
from urllib import error as urlerror

from memory_lab.llm_client import (
    JsonClientConfigurationError,
    JsonClientTransportError,
    OpenAICompatibleJsonClient,
)


def envelope(value: dict) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(value)},
                }
            ]
        }
    ).encode("utf-8")


class JsonClientTests(unittest.TestCase):
    def test_returns_json_object_and_deepseek_disables_thinking(self) -> None:
        captured = {}

        def transport(url, headers, body, timeout):
            captured.update(
                url=url,
                headers=dict(headers),
                body=json.loads(body),
                timeout=timeout,
            )
            return envelope({"ok": True})

        client = OpenAICompatibleJsonClient(
            base_url="https://api.deepseek.com",
            model="deepseek-test",
            api_key="credential-value-without-provider-prefix",
            allow_remote=True,
            transport=transport,
        )
        self.assertEqual(
            client.complete_json(system_prompt="return json", user_payload={"x": 1}),
            {"ok": True},
        )
        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["body"]["thinking"], {"type": "disabled"})
        self.assertEqual(captured["body"]["response_format"], {"type": "json_object"})
        self.assertIn("Authorization", captured["headers"])
        self.assertEqual(
            captured["headers"]["Content-Type"], "application/json; charset=utf-8"
        )
        self.assertNotIn("credential-value", repr(client))

    def test_transient_http_error_retries_without_exposing_key(self) -> None:
        calls = []
        sleeps = []

        def transport(url, headers, body, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise urlerror.HTTPError(url, 429, "rate limited", {"Retry-After": "0"}, None)
            return envelope({"done": True})

        client = OpenAICompatibleJsonClient(
            base_url="http://127.0.0.1:8000",
            model="local",
            api_key="private-key",
            transport=transport,
            sleeper=sleeps.append,
            jitter=lambda: 0,
        )
        self.assertEqual(
            client.complete_json(system_prompt="json", user_payload="{}"),
            {"done": True},
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [0.0])

    def test_unsafe_endpoints_and_secret_payload_fail_before_transport(self) -> None:
        called = []
        client = OpenAICompatibleJsonClient(
            base_url="http://example.com",
            model="x",
            allow_remote=True,
            transport=lambda *args: called.append(args),
        )
        with self.assertRaises(JsonClientConfigurationError):
            client.complete_json(system_prompt="json", user_payload={"x": 1})
        self.assertEqual(called, [])

        local = OpenAICompatibleJsonClient(
            base_url="http://127.0.0.1:8000",
            model="x",
            transport=lambda *args: called.append(args),
        )
        with self.assertRaisesRegex(JsonClientTransportError, "secret"):
            local.complete_json(
                system_prompt="json", user_payload={"token": "sk-1234567890abcdef"}
            )
        for natural_secret in (
            "API key is natural-language-secret",
            "访问令牌为 chinese-language-secret",
            "密码是 another-language-secret",
        ):
            with self.subTest(natural_secret=natural_secret), self.assertRaisesRegex(
                JsonClientTransportError, "secret"
            ):
                local.complete_json(system_prompt="json", user_payload=natural_secret)
        self.assertEqual(called, [])

    def test_invalid_or_truncated_provider_output_is_rejected(self) -> None:
        invalid = OpenAICompatibleJsonClient(
            base_url="http://localhost:8000",
            model="x",
            transport=lambda *_args: b'{"choices":[]}',
        )
        with self.assertRaises(JsonClientTransportError):
            invalid.complete_json(system_prompt="json", user_payload="{}")

        truncated = OpenAICompatibleJsonClient(
            base_url="http://localhost:8000",
            model="x",
            transport=lambda *_args: json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "{}"},
                        }
                    ]
                }
            ).encode(),
        )
        with self.assertRaisesRegex(JsonClientTransportError, "did not complete"):
            truncated.complete_json(system_prompt="json", user_payload="{}")


if __name__ == "__main__":
    unittest.main()
