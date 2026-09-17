#!/usr/bin/env python3
"""Small OpenAI-compatible agent for a local mimik inference endpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Config:
    base_url: str = "http://192.168.1.196:8083/mimik-ai/openai/v1"
    api_key: str = "1234"
    model: str = "smollm-360m"
    timeout: float = 120.0

    @classmethod
    def from_environment(cls) -> "Config":
        return cls(
            base_url=os.getenv("MIMIK_BASE_URL", cls.base_url).rstrip("/"),
            api_key=os.getenv("MIMIK_API_KEY", cls.api_key),
            model=os.getenv("MIMIK_MODEL", cls.model),
            timeout=float(os.getenv("MIMIK_TIMEOUT", cls.timeout)),
        )


class LocalInferenceClient:
    """Minimal client for the mimik OpenAI-compatible chat endpoint."""

    def __init__(self, config: Config):
        self.config = config

    def chat(self, messages: list[dict[str, str]], stream: bool = False) -> str | Iterator[str]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
        }
        request = Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            response = urlopen(request, timeout=self.config.timeout)
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"mimik endpoint returned HTTP {error.code}: {details}") from error
        except URLError as error:
            raise RuntimeError(f"could not reach mimik endpoint: {error.reason}") from error

        if stream:
            return self._stream_response(response)

        with response:
            body = json.load(response)
        return self._message_content(body)

    def _stream_response(self, response: Any) -> Iterator[str]:
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        finally:
            response.close()

    @staticmethod
    def _message_content(body: dict[str, Any]) -> str:
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"unexpected chat completion response: {body}") from error


class LocalAgent:
    """Conversation agent with one local tool and a bounded action loop."""

    ACTION_PATTERN = re.compile(r'^ACTION:\s*word_count\((?P<quote>["\'])(?P<text>.*?)\1\)\s*$', re.DOTALL)

    def __init__(self, client: LocalInferenceClient):
        self.client = client
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a concise local AI assistant. Explain your reasoning briefly "
                    "and provide practical answers. If counting words would help, respond "
                    'with exactly ACTION: word_count("text to count") and no other text.'
                ),
            }
        ]

    def ask(self, prompt: str, stream: bool = True) -> str:
        return "".join(self.ask_stream(prompt, stream=stream))

    def ask_stream(self, prompt: str, stream: bool = True) -> Iterator[str]:
        self.messages.append({"role": "user", "content": prompt})
        if self._may_need_tool(prompt):
            decision = self.client.chat(self.messages, stream=False)
            action = self._parse_action(decision)
            if action:
                self.messages.append({"role": "assistant", "content": decision})
                self.messages.append({"role": "tool", "content": self._run_tool(action)})
                result = self.client.chat(self.messages, stream=stream)
            else:
                result = decision
        else:
            result = self.client.chat(self.messages, stream=stream)

        if stream and not isinstance(result, str):
            chunks = []
            for chunk in result:
                chunks.append(chunk)
                yield chunk
            answer = "".join(chunks)
        else:
            answer = result
            yield answer

        self.messages.append({"role": "assistant", "content": answer})

    @staticmethod
    def _may_need_tool(prompt: str) -> bool:
        prompt = prompt.lower()
        return "word count" in prompt or "count words" in prompt or "how many words" in prompt

    @classmethod
    def _parse_action(cls, response: str) -> str | None:
        match = cls.ACTION_PATTERN.match(response.strip())
        return match.group("text") if match else None

    @staticmethod
    def _run_tool(text: str) -> str:
        return f"word_count({len(text.split())})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a local mimik model.")
    parser.add_argument("prompt", nargs="*", help="one prompt; omit to start an interactive session")
    parser.add_argument("--no-stream", action="store_true", help="wait for the complete response")
    args = parser.parse_args()

    agent = LocalAgent(LocalInferenceClient(Config.from_environment()))
    if args.prompt:
        prompt = " ".join(args.prompt)
        if args.no_stream:
            print(agent.ask(prompt, stream=False))
        else:
            for chunk in agent.ask_stream(prompt, stream=True):
                print(chunk, end="", flush=True)
            print()
        return 0

    print("mimik local agent. Type /exit to quit, /reset to clear context.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt == "/exit":
            return 0
        if prompt == "/reset":
            agent = LocalAgent(LocalInferenceClient(Config.from_environment()))
            print("Context reset.")
            continue
        if not prompt:
            continue
        try:
            print("assistant> ", end="", flush=True)
            for chunk in agent.ask_stream(prompt):
                print(chunk, end="", flush=True)
            print(flush=True)
        except RuntimeError as error:
            print(f"\nerror: {error}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
