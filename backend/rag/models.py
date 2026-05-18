"""
RAG 数据模型：PDF 解析后的语义节点 PaperNode。

每个节点对应 Docling 解析出的一个结构单元（段落、表格、图、公式等），
后续会转为 LangChain Document 并分块写入 Milvus。
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Any

# 支持的节点类型，与 node_generator 工厂一一对应
NodeType = Literal[
    "section_header",
    "paragraph",
    "table",
    "figure",
    "caption",
    "formula"
]


@dataclass
class PaperNode:
    """
    单篇论文中的一个语义节点。

    Attributes:
        node_id: 全局唯一 ID
        paper_id: 论文标识（通常为上传文件名 stem）
        node_type: 节点类型
        text: 用于检索的最终文本（由 NodeContentGenerator 生成）
        page_num: 页码（1-based）
        order: 阅读顺序序号
        section_path: 章节路径栈，如 ["1 Introduction", "1.1 Motivation"]
        bbox: 版面坐标 (l, t, r, b)，图/表/图注用于裁剪与关联
        parent_id: 预留父子关系
        related_ids: 正文引用 Figure/Table 时建立的关联 node_id
        image_path: 图表裁剪 PNG 路径（仅 figure）
        metadata: 扩展字段，如 section_type、vlm_description、原始 Docling item
    """

    node_id: str
    paper_id: str
    node_type: NodeType

    text: str
    page_num: int
    order: int

    section_path: list[str] = field(default_factory=list)
    bbox: Optional[tuple[float, float, float, float]] = None

    parent_id: Optional[str] = None
    related_ids: list[str] = field(default_factory=list)

    image_path: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)
