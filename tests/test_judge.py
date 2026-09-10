from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urlerror

from memory_lab.judge import (
    INPUT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    PROMPT_VERSION,
    ApiJudge,
    JudgeTransportError,
    JudgeValidationError,
    MockJudge,
    ReplayJudge,
    ReplayMissError,
    request_hash,
    validate_judge_request,
    validate_judge_result,
)


def make_request() -> dict:
    text = "以后在所有对话中，回复统一使用英文。"
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "request_id": "llmreq_001",
        "claim": {
            "candidate_key": "admcand_001",
            "event_id": "event_001",
            "part_id": "part_001",
            "source_text": text,
            "exact_quote": text,
            "start_offset": 0,
            "end_offset": len(text),
        },
        "context": {
            "session_id": "session_001",
            "project_key": "project_current",
            "directory": r"D:\project-current",
            "occurred_at": 1_788_900_000_000,
        },
        "hard_signals": {
            "long_term": ["以后", "所有对话"],
            "temporary": [],
            "correction": [],
            "instruction_coverage": {"status": "none", "scope": None},
        },
        "current_memories": [
            {
                "memory_id": "memory_language",
                "stable_key": "preference.response.language",
                "scope": {"level": "global"},
                "revision_id": "revision_language_1",
                "value": "zh",
                "statement": "默认使用中文回复。",
                "open_conflict_values": [],
            }
        ],
        "pending_proposals": [],
    }


def make_result(*, relation: str = "conflict") -> dict:
    request = make_request()
    claim = request["claim"]
    return {
        "should_store": True,
        "type": "response_preference",
        "key": "preference.response.language",
        "scope": {"level": "global"},
        "value": {"normalized": "en", "statement": "默认使用英文回复。"},
        "relation": {
            "kind": relation,
            "target_kind": "memory" if relation != "new" else None,
            "target_id": "memory_language" if relation != "new" else None,
        },
        "evidence": [
            {
                "event_id": claim["event_id"],
                "part_id": claim["part_id"],
                "quote": claim["exact_quote"],
                "start_offset": claim["start_offset"],
                "end_offset": claim["end_offset"],
            }
        ],
        "confidence": 0.97,
        "reason": "这是跨会话的长期回复语言要求。",
    }


class JudgeValidationTests(unittest.TestCase):
    def test_request_requires_exact_source_offsets(self) -> None:
        request = make_request()
        request["claim"]["start_offset"] = 1
        with self.assertRaisesRegex(JudgeValidationError, "quote/offsets"):
            validate_judge_request(request)

    def test_valid_result_preserves_exact_evidence(self) -> None:
        result = validate_judge_result(make_result(), make_request())
        self.assertTrue(result["should_store"])
        self.assertEqual(result["relation"]["kind"], "conflict")
        self.assertEqual(result["evidence"][0]["quote"], make_request()["claim"]["exact_quote"])

    def test_fabricated_evidence_is_rejected(self) -> None:
        result = make_result()
        result["evidence"][0]["quote"] = "模型虚构的原文"
        with self.assertRaisesRegex(JudgeValidationError, "exactly match"):
            validate_judge_result(result, make_request())

    def test_fabricated_target_is_rejected(self) -> None:
        result = make_result()
        result["relation"]["target_id"] = "memory_does_not_exist"
        with self.assertRaisesRegex(JudgeValidationError, "whitelist"):
            validate_judge_result(result, make_request())

    def test_project_scope_cannot_jump_to_another_project(self) -> None:
        result = make_result(relation="new")
        result["scope"] = {"level": "project", "project_key": "project_other"}
        with self.assertRaisesRegex(JudgeValidationError, "another project"):
            validate_judge_result(result, make_request())

    def test_correction_without_explicit_signal_is_downgraded_to_conflict(self) -> None:
        result = validate_judge_result(make_result(relation="correction"), make_request())
        self.assertEqual(result["relation"]["kind"], "conflict")

    def test_temporary_and_full_coverage_are_hard_vetoes(self) -> None:
        for field in ("temporary", "coverage"):
            with self.subTest(field=field):
                request = make_request()
                if field == "temporary":
                    request["hard_signals"]["temporary"] = ["这次"]
                else:
                    request["hard_signals"]["instruction_coverage"]["status"] = "full"
                with self.assertRaises(JudgeValidationError):
                    validate_judge_result(make_result(), request)

    def test_non_memory_has_null_semantics_and_unrelated_relation(self) -> None:
        request = make_request()
        claim = request["claim"]
        result = {
            "should_store": False,
            "type": None,
            "key": None,
            "scope": None,
            "value": None,
            "relation": {"kind": "unrelated", "target_kind": None, "target_id": None},
            "evidence": [
                {
                    "event_id": claim["event_id"],
                    "part_id": claim["part_id"],
                    "quote": claim["exact_quote"],
                    "start_offset": 0,
                    "end_offset": len(claim["exact_quote"]),
                }
            ],
            "confidence": 0.91,
            "reason": "只是当前问题，不影响未来对话。",
        }
        self.assertFalse(validate_judge_result(result, request)["should_store"])

    def test_mock_and_replay_are_bound_to_exact_input(self) -> None:
        request = make_request()
        expected = make_result()
        self.assertEqual(MockJudge({"llmreq_001": expected}).judge(request)["value"]["normalized"], "en")

        digest = request_hash(request)
        replay = ReplayJudge(
            {
                digest: {
                    "request_hash": digest,
                    "prompt_version": PROMPT_VERSION,
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "response": expected,
                }
            }
        )
        self.assertEqual(replay.judge(request)["key"], "preference.response.language")
        changed = make_request()
        changed["request_id"] = "llmreq_changed"
        with self.assertRaises(ReplayMissError):
            replay.judge(changed)

    def test_api_adapter_uses_json_only_contract(self) -> None:
        captured: dict = {}

        def transport(url: str, headers: dict, body: bytes, timeout: float) -> bytes:
            provider_result = make_result()
            provider_result["relation"]["target_id"] = "memory_0"
            provider_result["evidence"][0]["event_id"] = "event_0"
            provider_result["evidence"][0]["part_id"] = "part_0"
            captured.update(
                url=url,
                authorization=headers.get("Authorization"),
                payload=json.loads(body.decode("utf-8")),
                timeout=timeout,
            )
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(provider_result, ensure_ascii=False)}}]}
            ).encode("utf-8")

        judge = ApiJudge(
            base_url="https://api.deepseek.com",
            model="test-model",
            api_key="secret-test-key",
            max_attempts=1,
            allow_remote=True,
            transport=transport,
        )
        result = judge.judge(make_request())

        self.assertEqual(result["value"]["normalized"], "en")
        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer secret-test-key")
        self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertIn("output_json_schema", captured["payload"]["messages"][1]["content"])
        self.assertIn("positive_json_example", captured["payload"]["messages"][1]["content"])
        outbound = captured["payload"]["messages"][1]["content"]
        self.assertNotIn(r"D:\project-current", outbound)
        self.assertNotIn("event_001", outbound)
        self.assertNotIn("part_001", outbound)
        self.assertNotIn("session_001", outbound)

    def test_api_key_is_hidden_from_repr(self) -> None:
        judge = ApiJudge(
            base_url="http://127.0.0.1:1234/v1",
            model="local-test-model",
            api_key="TOP-SECRET-TEST-VALUE",
        )
        self.assertNotIn("TOP-SECRET-TEST-VALUE", repr(judge))

    def test_only_atomic_quote_not_full_source_text_is_sent(self) -> None:
        request = make_request()
        quote = request["claim"]["exact_quote"]
        source = "不要外发的前文。" + quote + "不要外发的后文。"
        request["claim"]["source_text"] = source
        request["claim"]["start_offset"] = source.index(quote)
        request["claim"]["end_offset"] = source.index(quote) + len(quote)
        result = make_result()
        result["evidence"][0]["start_offset"] = request["claim"]["start_offset"]
        result["evidence"][0]["end_offset"] = request["claim"]["end_offset"]

        def transport(_url: str, _headers: dict, body: bytes, _timeout: float) -> bytes:
            payload_text = body.decode("utf-8")
            self.assertIn(quote, payload_text)
            self.assertNotIn("不要外发的前文", payload_text)
            self.assertNotIn("不要外发的后文", payload_text)
            provider_result = json.loads(json.dumps(result, ensure_ascii=False))
            provider_result["relation"]["target_id"] = "memory_0"
            provider_result["evidence"][0]["event_id"] = "event_0"
            provider_result["evidence"][0]["part_id"] = "part_0"
            return json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(provider_result, ensure_ascii=False)}}]}
            ).encode("utf-8")

        judge = ApiJudge(
            base_url="http://127.0.0.1:1234/v1",
            model="local-test-model",
            max_attempts=1,
            transport=transport,
        )
        self.assertEqual(judge.judge(request)["evidence"][0]["quote"], quote)

    def test_project_only_claim_cannot_be_widened_to_global(self) -> None:
        request = make_request()
        text = "仅在当前项目中，以后统一使用英文回复。"
        request["claim"].update(source_text=text, exact_quote=text, start_offset=0, end_offset=len(text))
        result = make_result(relation="new")
        result["evidence"][0].update(quote=text, start_offset=0, end_offset=len(text))
        with self.assertRaisesRegex(JudgeValidationError, "cannot be widened"):
            validate_judge_result(result, request)

    def test_unresolved_scope_limit_blocks_storage(self) -> None:
        request = make_request()
        request["hard_signals"]["scope_limit"] = {
            "allowed_levels": [],
            "reason": "unresolved_scope_evidence",
        }
        with self.assertRaisesRegex(JudgeValidationError, "scope limit"):
            validate_judge_result(make_result(), request)

    def test_relation_target_must_use_same_stable_key(self) -> None:
        result = make_result()
        result["key"] = "workflow.answer.confirmation"
        with self.assertRaisesRegex(JudgeValidationError, "same stable key"):
            validate_judge_result(result, make_request())

    def test_known_stable_key_uses_canonical_value(self) -> None:
        result = make_result()
        result["value"]["normalized"] = "English"
        with self.assertRaisesRegex(JudgeValidationError, "canonical value"):
            validate_judge_result(result, make_request())

    def test_secret_or_path_in_quote_never_reaches_transport(self) -> None:
        for text in (
            "记住 api_key=example-secret-value",
            "令牌 github_pat_" + "A" * 24,
            "令牌 ghp_" + "B" * 24,
            "令牌 sk-" + "C" * 24,
            "认证 Bearer " + "D" * 24,
            r"以后使用 D:\private\config.json",
        ):
            with self.subTest(text=text):
                calls = 0

                def transport(*_: object) -> bytes:
                    nonlocal calls
                    calls += 1
                    return b"{}"

                request = make_request()
                request["claim"].update(source_text=text, exact_quote=text, start_offset=0, end_offset=len(text))
                judge = ApiJudge(
                    base_url="http://127.0.0.1:1234/v1",
                    model="local-test-model",
                    transport=transport,
                )
                with self.assertRaises(JudgeValidationError):
                    judge.judge(request)
                self.assertEqual(calls, 0)

    def test_retry_after_is_respected_for_429(self) -> None:
        calls = 0
        waits: list[float] = []

        def transport(_url: str, _headers: dict, _body: bytes, _timeout: float) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urlerror.HTTPError(
                    "https://api.deepseek.com/chat/completions",
                    429,
                    "rate limited",
                    {"Retry-After": "2"},
                    None,
                )
            provider_result = make_result()
            provider_result["relation"]["target_id"] = "memory_0"
            provider_result["evidence"][0]["event_id"] = "event_0"
            provider_result["evidence"][0]["part_id"] = "part_0"
            return json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(provider_result, ensure_ascii=False)}}]}
            ).encode("utf-8")

        judge = ApiJudge(
            base_url="https://api.deepseek.com",
            model="test-model",
            api_key="test-only-key",
            allow_remote=True,
            transport=transport,
            sleeper=waits.append,
            jitter=lambda: 0.0,
        )
        self.assertTrue(judge.judge(make_request())["should_store"])
        self.assertEqual(calls, 2)
        self.assertEqual(waits, [2.0])

    def test_authorization_is_not_forwarded_across_redirect(self) -> None:
        captured_authorization: list[str | None] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path == "/capture":
                    captured_authorization.append(self.headers.get("Authorization"))
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/capture")
                self.end_headers()

            def log_message(self, _format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            judge = ApiJudge(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="local-test-model",
                api_key="redirect-test-secret",
                max_attempts=1,
            )
            with self.assertRaises(JudgeTransportError):
                judge.judge(make_request())
            self.assertEqual(captured_authorization, [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_scalar_or_truncated_provider_output_is_rejected(self) -> None:
        envelopes = (
            {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]},
            {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        )
        for envelope in envelopes:
            with self.subTest(envelope=envelope):
                judge = ApiJudge(
                    base_url="http://127.0.0.1:1234/v1",
                    model="local-test-model",
                    max_attempts=1,
                    transport=lambda *_args, value=envelope: json.dumps(value).encode("utf-8"),
                )
                with self.assertRaises(JudgeTransportError):
                    judge.judge(make_request())

    def test_remote_endpoint_requires_explicit_opt_in(self) -> None:
        judge = ApiJudge(
            base_url="https://api.deepseek.com",
            model="test-model",
            api_key="unused-test-key",
            max_attempts=1,
            transport=lambda *_: b"{}",
        )
        with self.assertRaisesRegex(Exception, "ALLOW_REMOTE"):
            judge.judge(make_request())

    def test_sensitive_claim_never_reaches_transport(self) -> None:
        calls = 0

        def transport(*_: object) -> bytes:
            nonlocal calls
            calls += 1
            return b"{}"

        request = make_request()
        request["hard_signals"]["sensitive"] = True
        judge = ApiJudge(
            base_url="http://127.0.0.1:1234/v1",
            model="local-test-model",
            transport=transport,
        )
        with self.assertRaisesRegex(JudgeValidationError, "before model transport"):
            judge.judge(request)
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
