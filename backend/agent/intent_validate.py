"""
薄校验层（兼容）：委托给 turn_resolver.resolve_turn。
"""

from __future__ import annotations

from .intent import RetrievalIntent
from .turn_resolver import resolve_turn

# 向后兼容 re-export
from .turn_resolver import TurnState  # noqa: F401


def validate_intent(
    intent: RetrievalIntent,
    query: str,
    session_paper_ids: list[str],
    *,
    raw_query: str = "",
) -> dict:
    return resolve_turn(
        intent,
        query,
        session_paper_ids,
        raw_query=raw_query,
    )
