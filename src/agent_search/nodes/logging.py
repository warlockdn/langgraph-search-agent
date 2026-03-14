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
        self._emit_progress(
            "answer_ready",
            used_refinement=bool(state.get("final_answer", {}).get("used_refinement")),
            citation_count=len(state.get("final_answer", {}).get("citations", [])),
        )
        return dump_state_update({
            "final_answer": state["final_answer"],
            "messages": [self._assistant_message(state.get("final_answer"))],
        })
