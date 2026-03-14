# AGENTS.md

## Project

LangGraph agent-search workflow using Exa Python SDK (`exa-py`).

Core behavior:
- Route simple questions to a fast retrieval path.
- Route complex/comparison queries to agentic decomposition.
- Generate initial answer, validate coverage, run one refinement loop if needed.
- Compare initial vs refined answer and return final cited output.

Reference: [README.md](./README.md)

## Runtime and Entry Point

- Graph config: `langgraph.json`
- Graph entry: `./src/agent_search/graph.py:make_graph`
- Main builder: `build_graph()` in `src/agent_search/graph.py`

## Node Flow (must stay aligned)

1. `prepare_tool_input`
2. `initial_tool_choice`
3. `call_tool` (simple path) OR `start_agent_search` (agentic path)
4. `generate_sub_answers_subgraph` + `retrieve_orig_question_docs_subgraph_wrapper`
5. `generate_initial_answer`
6. `validate_initial_answer`
7. `extract_entity_term`
8. `decide_refinement_need`
9. `create_refined_sub_questions` (conditional)
10. `answer_refined_question_subgraphs`
11. `ingest_refined_sub_answers`
12. `generate_validate_refined_answer`
13. `compare_answers`
14. `logging_node`

## State Contracts

Primary state type: `AgentSearchState` in `src/agent_search/state.py`.

Do not rename/remove keys without updating:
- graph nodes
- tests
- README samples

Important keys:
- `question`, `normalized_question`, `query_type`, `complexity`
- `initial_subquestions`, `initial_results`, `initial_answer`
- `coverage_gaps`, `needs_refinement`, `entity_terms`
- `refined_subquestions`, `refined_results`, `refined_results_dedup`, `refined_answer`
- `final_answer`, `citations`, `tool_trace`, `errors`, `run_metadata`

## Retrieval Layer

File: `src/agent_search/exa_client.py`

- Retriever class: `ExaSDKRetriever`
- Uses `AsyncExa.search(...)`
- Retrieval profiles:
  - `general` -> `exa_search_web`
  - `code` -> `exa_search_code`
  - `hybrid` -> both
- Output must normalize into `RetrievedEvidence` schema.

## Config and Env

File: `src/agent_search/config.py`

Required/important env vars:
- `EXA_API_KEY`
- `OPENAI_API_KEY` (optional for LLM synthesis; fallback mode works without it)
- `LANGSMITH_API_KEY` and `LANGCHAIN_API_KEY` (optional, for tracing)

LangGraph env loading is configured in `langgraph.json` via:
- `"env": ".env"`

## Commands

Install:
```bash
uv sync --extra dev
```

Run server:
```bash
uv run langgraph dev --config langgraph.json
```

Test:
```bash
uv run pytest -q
```

Live Exa smoke tests:
```bash
RUN_LIVE_EXA_TESTS=1 uv run pytest -q tests/test_smoke_live.py
```

## Testing Expectations

Keep these tests passing:
- `tests/test_routing.py`
- `tests/test_refinement.py`
- `tests/test_synthesis.py`

Live tests are opt-in and skipped unless `RUN_LIVE_EXA_TESTS=1`.

## Guardrails for Future Changes

- Keep import style package-absolute in graph modules (e.g. `from agent_search...`) to avoid LangGraph file-loader import errors.
- Preserve final output shape (`FinalAnswer`) and citations format.
- If adding new node/state fields, update:
  - `schemas.py`
  - `state.py`
  - README examples
  - tests
- Prefer additive changes; avoid breaking existing node names/edges unless intentionally refactoring with test updates.
