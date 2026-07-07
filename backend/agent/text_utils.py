"""共享文本工具（避免 nodes / paper_scope 重复定义）。"""

from __future__ import annotations

import re

COMPLEX_HINTS = re.compile(
    r"对比|比较|分别|差异|以及.+和|vs\.?|compare|respectively|both .+ and",
    re.I,
)


def query_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text or "") else "en"
