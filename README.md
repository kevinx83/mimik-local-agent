# mimik Local AI Agent

A small Python agent that connects to the local OpenAI-compatible inference API exposed by mimOE Studio. It uses raw HTTP calls through Python's standard library, which keeps the BYO Framework path transparent and avoids an unnecessary dependency for a single-agent exercise.

## Prerequisites

1. Install and open [mimOE Studio](https://developer.mimik.com/mimOE-studio-early-access-download-v2).
2. In Model View, load the bundled `smollm-360m` model.
3. Open its API panel and confirm the host and port shown there.
4. Use Python 3.10 or newer.

The supplied API values are the defaults in this project:

- Base URL: `http://192.168.1.196:8083/mimik-ai/openai/v1`
- Model: `smollm-360m`
- API key: `1234`

The host is local-network-specific. Override it when Studio displays a different address:

```bash
export MIMIK_BASE_URL="http://<studio-host>:8083/mimik-ai/openai/v1"
export MIMIK_API_KEY="1234"
export MIMIK_MODEL="smollm-360m"
```

## Run

One prompt:

```bash
python3 agent.py "Explain what an OpenAI-compatible API is"
```

Interactive conversation:

```bash
python3 agent.py
```

Use `/reset` to start a fresh conversation and `/exit` to quit. The interactive mode writes response chunks as they arrive. Add `--no-stream` to wait for the complete response.

## Tests

The tests use a mocked HTTP response, so they run without mimOE Studio or network access:

```bash
python3 -m unittest -v
```

## Approach

The data flow is deliberately small:

1. `LocalAgent` owns the system prompt and conversation history.
2. Each user prompt is appended to the history and sent to the local model.
3. The model can choose the explicit action format `ACTION: word_count("text")`.
4. The agent parses that action, runs its local `word_count` tool, and sends the tool result back to the model for a final answer. This is one bounded tool round, so the loop cannot run away.
5. For the final answer, `LocalInferenceClient` either reads normal JSON or parses Server-Sent Events and the CLI prints each yielded chunk immediately.
6. The assistant response is appended to history, giving the next request conversational context.

I chose raw API calls instead of LangChain or another orchestration framework because this assignment has one model, one endpoint, and one small tool loop. The smaller dependency surface makes the connection, action parsing, and tool result visible, easier to debug, and easier to run on a machine hosting a local model. A framework could be added later if the agent grows to include tools, retrieval, or multiple agents.

## Traceable endpoint

mimOE Studio also exposes a traceable base URL in its API panel. To use it, set `MIMIK_BASE_URL` to the displayed `mimik-airouter/openai/v1` URL. The client code does not otherwise change because both endpoints use the same OpenAI-compatible contract.

## Project files

- `agent.py`: client, conversation agent, streaming parser, and CLI.
- `test_agent.py`: request, streaming, and conversation tests.
