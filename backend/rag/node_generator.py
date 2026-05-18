"""
节点文本生成工厂：按 node_type 将原始解析内容转为适合检索的 text。

策略示例：
- 段落：附带 section_path 上下文
- 表格：线性化为 Row N: col=val, ...
- 图：合并 caption 与邻近描述
"""

from abc import ABC, abstractmethod
from .models import PaperNode, NodeType


class NodeContentGenerator(ABC):
    """各类型节点内容生成器的抽象基类。"""

    @abstractmethod
    def generate_text(
        self,
        node: PaperNode,
        raw_content: str,
        context: dict
    ) -> str:
        """
        生成写入 PaperNode.text 的检索用文本。

        Args:
            node: 当前节点
            raw_content: Docling 提取的原始文本
            context: 附加上下文（caption、线性化表格等）
        """
        pass


class SectionHeaderGenerator(NodeContentGenerator):
    def generate_text(self, _node: PaperNode, raw_content: str, _context: dict) -> str:
        return f"Section: {raw_content}"


class ParagraphGenerator(NodeContentGenerator):
    def generate_text(self, node: PaperNode, raw_content: str, _context: dict) -> str:
        if node.section_path:
            section_path = " > ".join(node.section_path)
            return f"Section: {section_path}\n\nParagraph:\n{raw_content}"
        return f"Paragraph:\n{raw_content}"


class CaptionGenerator(NodeContentGenerator):
    def generate_text(self, _node: PaperNode, raw_content: str, _context: dict) -> str:
        return f"Caption: {raw_content}"


class FigureGenerator(NodeContentGenerator):
    def generate_text(self, _node: PaperNode, _raw_content: str, context: dict) -> str:
        caption_text = context.get("caption_text", "")
        nearby_context = context.get("nearby_context", "")
        parts = []
        if caption_text:
            parts.append(f"Figure: {caption_text}")
        if nearby_context:
            parts.append(f"\nRelated description:\n{nearby_context}")
        return "\n".join(parts) if parts else "Figure"


class TableGenerator(NodeContentGenerator):
    def generate_text(self, _node: PaperNode, _raw_content: str, context: dict) -> str:
        caption_text = context.get("caption_text", "")
        linearized_table = context.get("linearized_table", "")
        parts = []
        if caption_text:
            parts.append(f"Table: {caption_text}")
        if linearized_table:
            parts.append(f"\nTable content:\n{linearized_table}")
        return "\n".join(parts) if parts else "Table"

    @staticmethod
    def linearize_table(headers: list[str], rows: list[list[str]]) -> str:
        """将表格转为「Row i: h1=v1, h2=v2」每行一条，便于 BM25/向量检索。"""
        lines = []
        for i, row in enumerate(rows, 1):
            pairs = [f"{h}={v}" for h, v in zip(headers, row)]
            lines.append(f"Row {i}: {', '.join(pairs)}")
        return "\n".join(lines)


class FormulaGenerator(NodeContentGenerator):
    def generate_text(self, node: PaperNode, raw_content: str, _context: dict) -> str:
        if node.section_path:
            section_path = " > ".join(node.section_path)
            return f"Section: {section_path}\n\nFormula:\n{raw_content}"
        return f"Formula:\n{raw_content}"


class NodeContentGeneratorFactory:
    """根据 node_type 返回对应生成器单例。"""

    _generators = {
        "section_header": SectionHeaderGenerator(),
        "paragraph": ParagraphGenerator(),
        "caption": CaptionGenerator(),
        "figure": FigureGenerator(),
        "table": TableGenerator(),
        "formula": FormulaGenerator(),
    }

    @classmethod
    def get_generator(cls, node_type: NodeType) -> NodeContentGenerator:
        generator = cls._generators.get(node_type)
        if generator is None:
            raise ValueError(f"Unsupported node type: {node_type}")
        return generator
