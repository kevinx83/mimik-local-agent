import json
import unittest
from unittest.mock import patch

from agent import Config, LocalAgent, LocalInferenceClient


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __iter__(self):
        return iter(self.body.splitlines(keepends=True))

    def read(self):
        return self.body

    def close(self):
        self.closed = True


class LocalInferenceClientTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(base_url="http://local.test/v1", api_key="test-key", model="test-model")
        self.client = LocalInferenceClient(self.config)

    @patch("agent.urlopen")
    def test_chat_sends_openai_compatible_request(self, urlopen):
        urlopen.return_value = FakeResponse(
            json.dumps({"choices": [{"message": {"content": "Hello from mimik"}}]}).encode()
        )

        answer = self.client.chat([{"role": "user", "content": "Hello"}])

        self.assertEqual(answer, "Hello from mimik")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://local.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(
            json.loads(request.data),
            {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )

    @patch("agent.urlopen")
    def test_streaming_response_yields_text_deltas(self, urlopen):
        urlopen.return_value = FakeResponse(
            b'data: {"choices":[]}\n\n'
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        self.assertEqual("".join(self.client.chat([], stream=True)), "Hello world")


class LocalAgentTests(unittest.TestCase):
    def test_agent_retains_conversation_context(self):
        class FakeClient:
            def chat(self, messages, stream):
                self.messages = list(messages)
                self.stream = stream
                return "answer"

        client = FakeClient()
        agent = LocalAgent(client)
        self.assertEqual(agent.ask("question", stream=False), "answer")
        self.assertEqual(
            client.messages,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a concise local AI assistant. Explain your reasoning briefly "
                        "and provide practical answers. If counting words would help, respond "
                        'with exactly ACTION: word_count("text to count") and no other text.'
                    ),
                },
                {"role": "user", "content": "question"},
            ],
        )
        self.assertFalse(client.stream)
        self.assertEqual(agent.messages[-1], {"role": "assistant", "content": "answer"})

    def test_agent_executes_word_count_action_and_uses_tool_result(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def chat(self, messages, stream):
                self.calls.append((list(messages), stream))
                if len(self.calls) == 1:
                    return 'ACTION: word_count("one two three")'
                return "That is 3 words."

        client = FakeClient()
        agent = LocalAgent(client)

        self.assertEqual(agent.ask("How many words?", stream=False), "That is 3 words.")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1][0][-1], {"role": "tool", "content": "word_count(3)"})
        self.assertFalse(client.calls[1][1])


if __name__ == "__main__":
    unittest.main()
