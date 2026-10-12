# cortex-engine

A LIDA-style cognitive agent ("Clara"). A cheap attention cycle runs constantly.
Claude is called only to perceive new input, to deliberate when no learned habit
fits, and to carry out actions. Memory is mem0 with local embeddings.

## How it works

Each cycle:

1. **Perceive**: new input becomes workspace items, judged against her values,
   goals and what's on her mind.
2. **Attend**: attention codelets (one per goal, plus her current intention,
   urgency and novelty) form coalitions; the strongest is broadcast.
3. **Remember**: the broadcast is written as an episode and cues related memories.
4. **Act**: a learned habit fires if one fits; otherwise she deliberates.
   Either way, the action is carried out with tools: web search/fetch, her own
   workspace folder, sandboxed Python, revising her goals, asking her person a
   question.
5. **Appraise and learn**: she judges whether it worked (Claude, or the Jev
   decision model). Deliberated actions that work become habits.

With nothing new coming in, her mind wanders through her intention and goals.
Episodes are periodically consolidated into facts.

## Setup

```sh
poetry install
cp .env.example .env    # add ANTHROPIC_API_KEY; everything else has defaults
poetry run python lida-llm.py
```

Type a line and press Enter to talk to her while she runs. Ctrl-C stops her.

Running Python she writes needs [bubblewrap](https://github.com/containers/bubblewrap)
(`bwrap`). Without it, code execution is turned off; there is no unsandboxed fallback.

## Files

| Path | What it is |
|---|---|
| `lida-llm.py` | The whole agent |
| `personalities/clara.txt` | Who she is (system prompt for every call) |
| `prompt.txt` | Her first input on startup |
| `values.txt` | Lasting traits, fixed |
| `goals.txt` | Seed goals, concrete; she revises them herself |
| `reset_memory.py` | Wipes her memory and state (keeps `workspace/` unless `--files`) |
| `.env.example` | Every setting, documented |

Runtime state (memory store, goals she's written, her workspace folder) is
created next to the code and is gitignored.
