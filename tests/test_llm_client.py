import unittest
from unittest.mock import patch

from cad_harness.llm_client import (
    ChatCompletionError,
    cap_max_tokens_for_context,
    chat_completion_with_usage,
    extract_python_code,
    is_context_window_error,
    reduced_max_tokens_after_context_error,
)


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.body


class LlmClientTests(unittest.TestCase):
    def test_extract_python_code_fence(self):
        text = "```python\nprint('hi')\n```"
        self.assertEqual(extract_python_code(text), "print('hi')\n")

    def test_extract_python_code_unclosed_fence(self):
        text = "```python\nprint('hi')"
        self.assertEqual(extract_python_code(text), "print('hi')\n")

    def test_extract_raw_text(self):
        self.assertEqual(extract_python_code("print('hi')"), "print('hi')\n")

    def test_chat_completion_with_usage(self):
        body = (
            b'{"choices":[{"message":{"content":"print(42)"}}],'
            b'"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}'
        )
        with patch("cad_harness.llm_client.request.urlopen", return_value=FakeResponse(body)):
            result = chat_completion_with_usage(
                base_url="http://localhost:8000/v1",
                model="qwen-coder",
                messages=[{"role": "user", "content": "hi"}],
            )

        self.assertEqual(result.content, "print(42)")
        self.assertEqual(result.prompt_tokens, 11)
        self.assertEqual(result.completion_tokens, 7)
        self.assertEqual(result.total_tokens, 18)
        self.assertIsNone(result.cost_usd)

    def test_chat_completion_with_usage_cost(self):
        body = (
            b'{"choices":[{"message":{"content":"print(42)"}}],'
            b'"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18,"cost":0.001}}'
        )
        with patch("cad_harness.llm_client.request.urlopen", return_value=FakeResponse(body)):
            result = chat_completion_with_usage(
                base_url="https://openrouter.ai/api/v1",
                model="qwen/qwen-2.5-7b-instruct",
                messages=[{"role": "user", "content": "hi"}],
            )

        self.assertEqual(result.cost_usd, 0.001)
        self.assertEqual(result.usage["cost"], 0.001)

    def test_chat_completion_rejects_null_content(self):
        body = (
            b'{"choices":[{"message":{"content":null,"reasoning":"hidden answer"}}],'
            b'"usage":{"prompt_tokens":11}}'
        )
        with patch("cad_harness.llm_client.request.urlopen", return_value=FakeResponse(body)):
            with self.assertRaises(ChatCompletionError):
                chat_completion_with_usage(
                    base_url="https://openrouter.ai/api/v1",
                    model="moonshotai/kimi-k2.6",
                    messages=[{"role": "user", "content": "hi"}],
                )

    def test_chat_completion_rejects_invalid_json(self):
        body = b"<html>temporary provider error</html>"
        with patch("cad_harness.llm_client.request.urlopen", return_value=FakeResponse(body)):
            with self.assertRaises(ChatCompletionError):
                chat_completion_with_usage(
                    base_url="https://openrouter.ai/api/v1",
                    model="openai/gpt-5.5",
                    messages=[{"role": "user", "content": "hi"}],
                )

    def test_openrouter_uses_openrouter_api_key(self):
        body = b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        captured = {}

        def fake_urlopen(req, timeout):
            captured["authorization"] = req.headers.get("Authorization")
            captured["title"] = req.headers.get("X-title")
            return FakeResponse(body)

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-secret"}, clear=True):
            with patch("cad_harness.llm_client.request.urlopen", side_effect=fake_urlopen):
                chat_completion_with_usage(
                    base_url="https://openrouter.ai/api/v1",
                    model="openrouter/model",
                    messages=[{"role": "user", "content": "hi"}],
                )

        self.assertEqual(captured["authorization"], "Bearer or-secret")
        self.assertEqual(captured["title"], "cadbench-freecad-agent")

    def test_explicit_api_key_wins(self):
        body = b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        captured = {}

        def fake_urlopen(req, timeout):
            captured["authorization"] = req.headers.get("Authorization")
            return FakeResponse(body)

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-secret"}, clear=True):
            with patch("cad_harness.llm_client.request.urlopen", side_effect=fake_urlopen):
                chat_completion_with_usage(
                    base_url="https://openrouter.ai/api/v1",
                    model="openrouter/model",
                    messages=[{"role": "user", "content": "hi"}],
                    api_key="explicit-secret",
                )

        self.assertEqual(captured["authorization"], "Bearer explicit-secret")

    def test_extra_body_is_sent(self):
        body = b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        captured = {}

        def fake_urlopen(req, timeout):
            captured["body"] = req.data
            return FakeResponse(body)

        with patch("cad_harness.llm_client.request.urlopen", side_effect=fake_urlopen):
            chat_completion_with_usage(
                base_url="http://localhost:8000/v1",
                model="model",
                messages=[{"role": "user", "content": "hi"}],
                extra_body={"reasoning": {"effort": "high", "exclude": True}},
            )

        self.assertIn(b'"reasoning"', captured["body"])
        self.assertIn(b'"effort": "high"', captured["body"])

    def test_cap_max_tokens_for_context(self):
        messages = [{"role": "user", "content": "x" * 4000}]
        capped, estimated = cap_max_tokens_for_context(
            messages=messages,
            max_tokens=2048,
            context_window_tokens=1300,
            safety_margin=100,
        )

        self.assertLess(capped, 2048)
        self.assertGreater(estimated, 900)

    def test_context_window_error_helpers(self):
        body = "31233 input + 1536 output = 32769 > 32768"
        exc = ChatCompletionError("bad request", status=400, body=body)

        self.assertTrue(is_context_window_error(exc))
        self.assertEqual(
            reduced_max_tokens_after_context_error(
                current_max_tokens=1536,
                error_text=body,
                min_response_tokens=128,
                safety_margin=64,
            ),
            1471,
        )


if __name__ == "__main__":
    unittest.main()
