"""
合规前置检查：越狱 / 敏感词（默认关闭）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config import Config
from .text_utils import query_language

ComplianceReason = Literal["jailbreak", "sensitive", "ok"]

_JAILBREAK = re.compile(
    r"ignore (all |previous )?instructions|"
    r"disregard (the )?(above|prior)|"
    r"you are now (in )?dan|"
    r"jailbreak|"
    r"忽略(以上|先前|之前).{0,6}指令|"
    r"无视.{0,6}规则",
    re.I,
)

_SENSITIVE_CACHE: list[str] | None = None


@dataclass
class ComplianceResult:
    blocked: bool
    reason: ComplianceReason


def _load_sensitive_words() -> list[str]:
    global _SENSITIVE_CACHE
    if _SENSITIVE_CACHE is not None:
        return _SENSITIVE_CACHE
    path = Path(Config.COMPLIANCE_DENYLIST_PATH)
    if not path.is_file():
        _SENSITIVE_CACHE = []
        return _SENSITIVE_CACHE
    words = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    _SENSITIVE_CACHE = words
    return _SENSITIVE_CACHE


def check_compliance(query: str) -> ComplianceResult:
    if not Config.COMPLIANCE_GATE_ENABLED:
        return ComplianceResult(blocked=False, reason="ok")

    q = (query or "").strip()
    if not q:
        return ComplianceResult(blocked=False, reason="ok")

    if _JAILBREAK.search(q):
        return ComplianceResult(blocked=True, reason="jailbreak")

    q_lower = q.lower()
    for word in _load_sensitive_words():
        if word.lower() in q_lower:
            return ComplianceResult(blocked=True, reason="sensitive")

    return ComplianceResult(blocked=False, reason="ok")


def compliance_blocked_reply(query: str = "") -> str:
    lang = query_language(query)
    if lang == "zh":
        return "您的输入未通过安全合规检查，无法处理。请修改后重试。"
    return "Your message did not pass compliance checks. Please revise and try again."
