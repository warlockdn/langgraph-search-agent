from __future__ import annotations

from agent_search.state import (
    AgentSearchStateInput,
    AgentSearchStateUpdateDict,
    dump_state_update,
)


class LoggingMixin:
    async def logging_node(
        self, state: AgentSearchStateInput
    ) -> AgentSearchStateUpdateDict:
        final_answer = state.get("final_answer", {}).get("answer", None)
        if not final_answer:
            return {}
        return dump_state_update({
            "final_answer": {
                "answer": state["final_answer"]["answer"],
                "citations": state["final_answer"]["citations"],
            }
        })
