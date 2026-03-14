from __future__ import annotations

from typing import Any


class LoggingMixin:
    async def logging_node(self, state: dict[str, Any]) -> dict[str, Any]:
        final_answer = dict(state.get("final_answer") or {})
        if not final_answer:
            return {"final_answer": {}}
        return {
            "final_answer": {
                "answer": final_answer.get("answer", "No answer generated."),
                "citations": final_answer.get("citations", []),
            }
        }
