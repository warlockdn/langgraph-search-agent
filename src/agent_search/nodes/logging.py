from __future__ import annotations

from agent_search.state import (
    AgentSearchStateInput,
    AgentSearchStateUpdateDict,
    dump_state_update,
    load_state_update,
)


class LoggingMixin:
    async def logging_node(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        state = load_state_update(state).model_dump()
        final_answer = dict(state.get("final_answer") or {})
        if not final_answer:
            return {}
        return dump_state_update({
            "final_answer": {
                "answer": final_answer.get("answer", "No answer generated."),
                "citations": final_answer.get("citations", []),
            }
        })
