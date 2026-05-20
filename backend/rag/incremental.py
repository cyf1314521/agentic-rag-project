"""
Milvus 增量更新：按 paper_id / content_hash 删除或替换向量。

文件上传、删除 API 通过 Retriever.get_updater() 获取本类实例。
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_milvus import Milvus

from .factory import ensure_milvus_orm_connection


class IncrementalUpdater:
    """对父子两个 Milvus collection 执行按论文维度的增删改。"""

    def __init__(self, parent_store: Milvus, child_store: Milvus):
        self.parent_store = parent_store
        self.child_store = child_store

    def _ensure_connections(self) -> None:
        ensure_milvus_orm_connection(self.parent_store)
        ensure_milvus_orm_connection(self.child_store)

    def delete_paper(self, paper_id: str) -> bool:
        """删除指定 paper_id 在 parent/child 集合中的全部 chunk。"""
        self._ensure_connections()
        try:
            expr = f'paper_id == "{paper_id.replace(chr(34), "")}"'
            self.parent_store.delete(expr=expr)
            self.child_store.delete(expr=expr)
            return True
        except Exception:
            return False

    def has_content_hash(self, content_hash: str) -> Optional[str]:
        """
        上传去重：查询是否已有相同 SHA256 内容。

        Returns:
            已存在的 paper_id，或 None
        """
        self._ensure_connections()
        try:
            col = self.child_store.col
            if col is None:
                return None
            safe_hash = content_hash.replace('"', "")
            results = col.query(
                expr=f'content_hash == "{safe_hash}"',
                output_fields=["paper_id"],
                limit=1,
            )
            if results:
                return results[0].get("paper_id")
            return None
        except Exception:
            return None

    def update_paper(
        self,
        paper_id: str,
        parents: List[Document],
        children: List[Document],
    ) -> bool:
        """先删后插，实现单篇论文的完整替换。"""
        self.delete_paper(paper_id)
        self._ensure_connections()
        try:
            self.parent_store.add_documents(parents)
            self.child_store.add_documents(children)
            return True
        except Exception:
            return False
