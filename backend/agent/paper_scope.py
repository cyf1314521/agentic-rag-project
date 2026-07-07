"""
多 PDF 会话下的论文 scope 解析。

在检索前判断用户是否指向某一篇论文；无法消歧时标记 needs_clarification。
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from .text_utils import COMPLEX_HINTS, query_language

ScopeMode = Literal[
    "no_session",
    "single_session",
    "explicit",
    "session_wide",
    "ambiguous",
]

_ABSTRACT_HINTS = re.compile(r"摘要|abstract", re.I)
_SINGLE_PAPER_QUERY_HINTS = re.compile(
    r"这篇|该论文|该篇|此文|本文|this paper|the paper|摘要里|摘要中|"
    r"abstract says|in the abstract",
    re.I,
)
_BARE_ABSTRACT_QUERY = re.compile(
    r"^(摘要|abstract)\s*(说了什么|讲了什么|内容|是什么|如何|怎样)?[？?]?\s*$",
    re.I,
)
_SINGLE_PAPER_INTENT = re.compile(
    r"说了什么|讲了什么|主要内容|what does .+ say|summarize the abstract",
    re.I,
)
_EVAL_NUM = re.compile(r"\beval[_\-\s]?0?(\d{1,2})\b", re.I)

# 主题词 → eval paper_id（用于「这篇海洋酸化论文」类无显式 id 的收窄）
_TOPIC_ALIASES: dict[str, list[str]] = {
    "eval_01_solar_pv": ["太阳能", "光伏", "薄膜", "碲化镉", "solar", "photovoltaic"],
    "eval_02_ocean_ph": ["海洋", "酸化", "太平洋", "北太平洋", "ocean", "acidification"],
    "eval_03_blockchain_scm": ["区块链", "联盟链", "冷链", "blockchain", "traceability"],
    "eval_04_mri_glioma": ["胶质瘤", "mri", "glioma", "肿瘤", "影像"],
    "eval_05_wind_turbine": ["风电", "风力", "叶片", "风机", "wind", "turbine", "offshore"],
    "eval_06_nlp_sentiment": ["情感", "sentiment", "nlp", "评论"],
    "eval_07_battery_solid": ["固态", "电池", "锂离子", "solid", "battery", "electrolyte"],
    "eval_08_urban_traffic": ["交通", "拥堵", "urban", "traffic"],
    "eval_09_protein_folding": ["蛋白质", "折叠", "protein", "folding", "alphafold"],
    "eval_10_quantum_error": ["量子", "纠错", "quantum", "qubit", "error correction"],
}


class PaperScopeResult(TypedDict):
    scope_mode: ScopeMode
    focus_paper_ids: list[str]
    needs_clarification: bool
    clarification_message: str
    match_reason: str
    candidate_paper_ids: list[str]


def _match_by_paper_id_substring(query: str, session_paper_ids: list[str]) -> list[str]:
    q_norm = query.lower().replace("-", "_")
    matched: list[str] = []
    for pid in session_paper_ids:
        pid_norm = pid.lower().replace("-", "_")
        if pid_norm in q_norm:
            matched.append(pid)
            continue
        # paper_id 下划线 token 较长时（如 wind_turbine）也尝试匹配
        for token in pid_norm.split("_"):
            if len(token) >= 5 and token in q_norm:
                matched.append(pid)
                break
    return _unique_preserve_order(matched)


def _match_by_eval_number(query: str, session_paper_ids: list[str]) -> list[str]:
    matched: list[str] = []
    for m in _EVAL_NUM.finditer(query):
        num = m.group(1)
        if len(num) == 1:
            num = f"0{num}"
        prefix = f"eval_{num}"
        matched.extend(p for p in session_paper_ids if p.lower().startswith(prefix))
    return _unique_preserve_order(matched)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


_SCOPE_NOISE = re.compile(
    r"这篇|该论文|该篇|此文|本文|this paper|the paper|摘要|abstract|"
    r"说了什么|讲了什么|主要内容|是什么|如何|怎样|多少|哪|什么|"
    r"what does|what is|summarize|about|the|a|an|in|of|for|to",
    re.I,
)


def _strip_scope_noise(query: str) -> str:
    text = _SCOPE_NOISE.sub(" ", query)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has_topic_anchor(query: str, session_paper_ids: list[str]) -> bool:
    """问题里是否带有足够长的主题描述，或能唯一匹配某篇论文。"""
    substantive = _strip_scope_noise(query)
    if len(substantive) >= 8:
        return True
    return len(_match_by_topic_anchor(query, session_paper_ids)) == 1


def _match_by_topic_anchor(query: str, session_paper_ids: list[str]) -> list[str]:
    """用主题词 / paper_id 英文 token 匹配唯一论文。"""
    q_lower = query.lower()
    scores: dict[str, int] = {}
    for pid in session_paper_ids:
        score = 0
        for alias in _TOPIC_ALIASES.get(pid, []):
            al = alias.lower()
            if al in q_lower or alias in query:
                score += max(len(alias), 2)
        for token in pid.lower().replace("-", "_").split("_"):
            if len(token) >= 4 and token in q_lower:
                score += len(token)
        if score > 0:
            scores[pid] = score
    if not scores:
        return []
    best = max(scores.values())
    winners = [p for p, s in scores.items() if s == best]
    return winners if len(winners) == 1 and best >= 3 else []


def _is_single_paper_intent(query: str, session_paper_ids: list[str]) -> bool:
    """多 PDF 下是否像「只问某一篇文章」但缺少可解析的目标。"""
    q = query.strip()
    if COMPLEX_HINTS.search(q):
        return False
    if _has_topic_anchor(q, session_paper_ids):
        return False
    if _BARE_ABSTRACT_QUERY.match(q):
        return True
    substantive = _strip_scope_noise(q)
    if _SINGLE_PAPER_QUERY_HINTS.search(q) and len(substantive) < 6:
        return True
    if _SINGLE_PAPER_INTENT.search(q) and _ABSTRACT_HINTS.search(q) and len(substantive) < 6:
        return True
    return False


def is_paper_id_only_query(query: str, session_paper_ids: list[str]) -> str | None:
    """若用户输入实质上只是 paper_id（澄清后的追问），返回该 id。"""
    q = (query or "").strip()
    if not q or not session_paper_ids:
        return None
    matched = _unique_preserve_order(
        _match_by_paper_id_substring(q, session_paper_ids)
        + _match_by_eval_number(q, session_paper_ids)
    )
    if len(matched) != 1:
        return None
    pid = matched[0]
    q_norm = q.lower().replace("-", "_")
    pid_norm = pid.lower().replace("-", "_")
    remainder = q_norm.replace(pid_norm, "")
    remainder = re.sub(r"eval[_\-\s]?0?\d{1,2}([_\w]*)?", "", remainder, flags=re.I)
    remainder = re.sub(r"[^\w\u4e00-\u9fff]+", "", remainder)
    if len(remainder) <= 2:
        return pid
    return None


def format_clarification_message(
    query: str,
    session_paper_ids: list[str],
    *,
    reason: str = "",
) -> str:
    """生成让用户选择论文的提示文案。"""
    lang = query_language(query)
    papers = "\n".join(f"- `{pid}`" for pid in session_paper_ids)
    if lang == "zh":
        head = (
            "当前会话绑定了多篇论文，无法确定您指的是哪一篇。"
            if not reason
            else f"当前会话有多篇论文可能匹配（{reason}），请指明要查询的论文。"
        )
        hint = (
            "请在问题中写明论文标识，例如：\n"
            "• `eval_05` 或完整 `paper_id`\n"
            "• 上传时的 PDF 文件名（不含扩展名）\n\n"
            f"本会话可用论文：\n{papers}\n\n"
            "示例：「eval_05 的摘要说了什么？」"
        )
        return f"{head}\n\n{hint}"
    head = (
        "This chat session has multiple papers; please specify which one you mean."
        if not reason
        else f"Multiple papers may match ({reason}); please specify one."
    )
    hint = (
        "Include a paper id or uploaded filename stem, for example:\n"
        f"`eval_05` or one of:\n{papers}\n\n"
        "Example: \"What does the abstract of eval_05 say?\""
    )
    return f"{head}\n\n{hint}"


def effective_retrieval_paper_ids(
    session_paper_ids: list[str],
    focus_paper_ids: list[str],
) -> list[str] | None:
    """子图 retrieve 使用的 Milvus paper_id_filter。"""
    if focus_paper_ids:
        return focus_paper_ids
    if session_paper_ids:
        return session_paper_ids
    return None


def boost_docs_for_focus_paper(
    docs: list,
    focus_paper_id: str,
) -> list:
    """将目标 paper 的检索结果前置（同分时优先）。"""
    if not focus_paper_id or not docs:
        return docs
    focused = [d for d in docs if (d.metadata or {}).get("paper_id") == focus_paper_id]
    others = [d for d in docs if (d.metadata or {}).get("paper_id") != focus_paper_id]
    return focused + others


def resolve_paper_scope(query: str, session_paper_ids: list[str]) -> PaperScopeResult:
    """
    解析本轮检索应聚焦的 paper_id 列表。

    - 0 篇：全库检索（无 filter）
    - 1 篇：自动聚焦
    - 多篇 + 显式指名：收窄到匹配论文
    - 多篇 + 单篇意图但无指名：needs_clarification
    - 多篇 + 对比/泛问：session_wide（检索全会话）
    """
    empty: PaperScopeResult = {
        "scope_mode": "no_session",
        "focus_paper_ids": [],
        "needs_clarification": False,
        "clarification_message": "",
        "match_reason": "",
        "candidate_paper_ids": [],
    }
    if not session_paper_ids:
        return empty

    if len(session_paper_ids) == 1:
        return {
            "scope_mode": "single_session",
            "focus_paper_ids": list(session_paper_ids),
            "needs_clarification": False,
            "clarification_message": "",
            "match_reason": "single_paper_session",
            "candidate_paper_ids": [],
        }

    explicit = _unique_preserve_order(
        _match_by_paper_id_substring(query, session_paper_ids)
        + _match_by_eval_number(query, session_paper_ids)
    )

    if len(explicit) == 1:
        return {
            "scope_mode": "explicit",
            "focus_paper_ids": explicit,
            "needs_clarification": False,
            "clarification_message": "",
            "match_reason": "explicit_paper_reference",
            "candidate_paper_ids": [],
        }

    if len(explicit) > 1:
        if COMPLEX_HINTS.search(query):
            return {
                "scope_mode": "session_wide",
                "focus_paper_ids": [],
                "needs_clarification": False,
                "clarification_message": "",
                "match_reason": "multi_paper_compare_query",
                "candidate_paper_ids": explicit,
            }
        msg = format_clarification_message(
            query, explicit, reason="多个 paper_id 同时匹配"
        )
        return {
            "scope_mode": "ambiguous",
            "focus_paper_ids": [],
            "needs_clarification": True,
            "clarification_message": msg,
            "match_reason": "multiple_explicit_matches",
            "candidate_paper_ids": explicit,
        }

    topic_focus = _match_by_topic_anchor(query, session_paper_ids)
    if len(topic_focus) == 1:
        return {
            "scope_mode": "explicit",
            "focus_paper_ids": topic_focus,
            "needs_clarification": False,
            "clarification_message": "",
            "match_reason": "topic_anchor",
            "candidate_paper_ids": [],
        }

    if _is_single_paper_intent(query, session_paper_ids):
        msg = format_clarification_message(query, session_paper_ids)
        return {
            "scope_mode": "ambiguous",
            "focus_paper_ids": [],
            "needs_clarification": True,
            "clarification_message": msg,
            "match_reason": "single_paper_intent_without_target",
            "candidate_paper_ids": list(session_paper_ids),
        }

    return {
        "scope_mode": "session_wide",
        "focus_paper_ids": [],
        "needs_clarification": False,
        "clarification_message": "",
        "match_reason": "multi_paper_general_query",
        "candidate_paper_ids": [],
    }
