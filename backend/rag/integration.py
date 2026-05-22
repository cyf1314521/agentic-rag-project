"""
PDF 解析与 RAG 入库集成。

主要组件：TextCleaner、PDFParser、RAGIntegration。
"""

from __future__ import annotations

import logging  # 标准库：记录解析、入库过程中的日志与异常
import time
from pathlib import Path  # 面向对象的路径操作，用于图片保存目录
import uuid  # 生成全局唯一的 node_id、chunk_id
import re  # 正则：清洗文本、识别页码/图注/章节编号等
from typing import TYPE_CHECKING, Optional, Any  # 类型标注：可选类型与 Docling 动态 item 类型
from docling.document_converter import DocumentConverter  # Docling：PDF → 结构化 document
from langchain_core.documents import Document  # LangChain 文档，供分块与 Milvus 入库
from langchain_core.messages import SystemMessage, HumanMessage  # LLM 章节分类用的消息格式
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符切分，生成子块
from langchain_milvus import Milvus, BM25BuiltInFunction  # Milvus 向量库封装 + 内置 BM25 稀疏向量
from pydantic import BaseModel, Field  # 结构化 LLM 输出：章节标题 → section_type 映射
from .models import PaperNode, NodeType  # 项目内论文节点模型与节点类型字面量
from .node_generator import NodeContentGeneratorFactory, TableGenerator  # 按节点类型生成可读文本
from .factory import EmbeddingService  # 嵌入模型工厂，与检索侧共用同一套向量模型
from config import Config

if TYPE_CHECKING:
    from .parse_artifact import ParseRecorder

FIGURE_SAVE_DIR = Path("./data/figures")  # 从 PDF 裁出的图片默认保存根目录

logger = logging.getLogger(__name__)  # 模块级 logger，便于在日志中定位 integration


class SectionClassification(BaseModel):
    """LLM 结构化输出的 schema：多条 {title, type}。"""
    classifications: list[dict[str, str]] = Field(
        description="List of {title, type} mappings"
    )  # 每项含章节标题与分类（method/experiment/background/conclusion 等）


# 发给 LLM 的系统提示：规定四类 section_type 及输出格式
SECTION_CLASSIFIER_PROMPT = """Classify each paper section title into one category:
- method: describes approach/model/algorithm/architecture/framework
- experiment: presents results/evaluation/analysis/performance/ablation
- background: introduction/related work/motivation/literature review
- conclusion: conclusion/discussion/future work/limitations

Output format: for each title, return {"title": "original title", "type": "category"}"""


class TextCleaner:
    """PDF 文本清洗与版式噪声（页眉页脚、页码）检测。"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove extra whitespace and normalize text."""
        text = re.sub(r'-\n', '', text)  # 去掉行尾连字符断行（如 "inter-\n national" → "international"）
        text = re.sub(r'\n+', '\n', text)  # 多个连续换行合并为一个
        text = re.sub(r' +', ' ', text)  # 多个空格合并为一个
        return text.strip()  # 去掉首尾空白后返回

    @staticmethod
    def is_header_footer(text: str, page_height: float, bbox: Optional[tuple]) -> bool:
        """Detect if text is likely header/footer based on position."""
        if not bbox or len(bbox) != 4:  # 无边界框或格式不对则无法按位置判断
            return False
        _, top, _, bottom = bbox  # bbox 为 (left, top, right, bottom)，Docling 坐标系
        margin_threshold = page_height * 0.1  # 页高上下各 10% 视为页眉页脚区域
        # 文本块顶部在页面上方 10% 或底部在页面下方 10% 则判为页眉/页脚
        return top < margin_threshold or bottom > (page_height - margin_threshold)

    @staticmethod
    def is_page_number(text: str) -> bool:
        """Detect if text is a page number."""
        return bool(re.match(r'^\d+$', text.strip()))  # 纯数字（如 "12"）视为页码


class PDFParser:
    """
    将 PDF 解析为 PaperNode 列表。

    流程：Docling convert → 过滤/排序 → 逐 item 转节点 → 关联 caption → 裁图 → LLM section_type 分类
    """

    def __init__(self, figure_save_dir: Optional[Path] = None, llm=None):
        self.cleaner = TextCleaner()  # 文本清洗与噪声检测工具
        self._converter_cache = {}  # 按 use_ocr True/False 缓存 DocumentConverter，避免重复创建
        self.figure_save_dir = figure_save_dir or FIGURE_SAVE_DIR  # 图片输出目录
        self.llm = llm  # 可选 LLM，用于 _classify_sections；为 None 则跳过分类
        self._recorder: ParseRecorder | None = None

    def parse(
        self,
        pdf_path: str,
        paper_id: str,
        recorder: ParseRecorder | None = None,
    ) -> list[PaperNode]:
        """Parse PDF file into list of PaperNodes."""
        self._recorder = recorder
        if recorder:
            recorder.stage("parse_start", {"pdf_path": pdf_path, "paper_id": paper_id})

        t0 = time.perf_counter()
        nodes = self._parse_with_ocr(pdf_path, paper_id, use_ocr=False)
        total_text = sum(len(n.text) for n in nodes)
        page_count = max((n.page_num for n in nodes), default=1)

        if recorder:
            recorder.stage(
                "docling_pass",
                {
                    "use_ocr": False,
                    "total_text_chars": total_text,
                    "page_count_est": page_count,
                    "node_count": len(nodes),
                },
                duration_sec=time.perf_counter() - t0,
            )

        if total_text < 1000 or total_text / page_count < 200:
            msg = f"Low text ({total_text} chars, {page_count} pages), retrying with OCR"
            logger.info(msg)
            if recorder:
                recorder.warn(msg)
            t1 = time.perf_counter()
            nodes = self._parse_with_ocr(pdf_path, paper_id, use_ocr=True)
            if recorder:
                recorder.stage(
                    "docling_pass",
                    {
                        "use_ocr": True,
                        "total_text_chars": sum(len(n.text) for n in nodes),
                        "node_count": len(nodes),
                    },
                    duration_sec=time.perf_counter() - t1,
                )

        self._recorder = None
        return nodes

    @staticmethod
    def _build_pipeline_options(use_ocr: bool):
        """构建 Docling PDF 管道选项；低内存模式可缓解 preprocess 阶段 std::bad_alloc。"""
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        opts = PdfPipelineOptions()
        opts.do_ocr = use_ocr
        if Config.DOCLING_LOW_MEMORY:
            opts.layout_batch_size = 1
            opts.table_batch_size = 1
            opts.ocr_batch_size = 1
            opts.do_chart_extraction = False
            opts.do_picture_classification = False
            opts.do_picture_description = False
            opts.generate_picture_images = False
            opts.generate_table_images = False
            opts.images_scale = 1.0
        return opts

    def _parse_with_ocr(self, pdf_path: str, paper_id: str, use_ocr: bool) -> list[PaperNode]:
        """Internal parse with OCR option."""
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # PDF 管道配置（含 do_ocr）
        from docling.document_converter import PdfFormatOption  # 为 PDF 格式绑定 pipeline
        from docling.datamodel.base_models import InputFormat  # 输入格式枚举，如 InputFormat.PDF

        if use_ocr not in self._converter_cache:  # 该 OCR 模式尚未创建过 converter
            pipeline_options = self._build_pipeline_options(use_ocr)
            self._converter_cache[use_ocr] = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

        converter = self._converter_cache[use_ocr]  # 取出对应 OCR 开关的转换器
        rec = self._recorder
        t_convert = time.perf_counter()
        result = converter.convert(pdf_path)  # 执行 Docling 转换
        doc = result.document  # 结构化文档对象，可 iterate_items
        if rec:
            rec.stage(
                "docling_convert",
                {"use_ocr": use_ocr, "low_memory": Config.DOCLING_LOW_MEMORY},
                duration_sec=time.perf_counter() - t_convert,
            )
        self._fitz_doc = None  # 每次 parse 重置，避免跨文件复用 PyMuPDF 句柄

        page_height = self._get_page_height(doc)  # 用于页眉页脚过滤的参考页高

        raw_items = [item for item, _ in doc.iterate_items()]  # 扁平化所有版面元素
        filtered_items = self._filter_items(raw_items, page_height)  # 去掉页码、页眉页脚等
        sorted_items = self._sort_reading_order(filtered_items)  # 按阅读顺序（行→列）排序
        top_level_x = self._compute_top_level_x(sorted_items)  # 一级标题左边界 x，用于章节栈
        if rec:
            rec.stage(
                "postprocess_layout",
                {
                    "raw_items": len(raw_items),
                    "after_filter": len(filtered_items),
                    "after_sort": len(sorted_items),
                    "page_height_pt": page_height,
                },
            )

        nodes = []  # 最终 PaperNode 列表
        order = 0  # 文档内全局顺序号
        section_stack = []  # 当前章节路径（本实现主要保留顶层 section 标题）

        for item in sorted_items:  # 按阅读顺序逐个处理
            node = self._process_item(
                item, paper_id, order, section_stack, top_level_x
            )
            if node:  # 无法映射类型的 item 返回 None，跳过
                nodes.append(node)
                order += 1  # 仅成功建节点时递增 order

        self._link_captions_to_figures_tables(nodes)  # 图/表与 caption 文本合并进 node.text
        tables_with_content = sum(
            1 for n in nodes if n.node_type == "table" and "Table content:" in n.text
        )
        if rec:
            rec.stage("link_captions_tables", {"tables_with_linearized_content": tables_with_content})

        self._link_text_references(nodes)  # 段落中 "Figure 3" 等与图/表节点双向 related_ids
        figures_before = sum(1 for n in nodes if n.image_path)
        self._extract_figure_images(pdf_path, doc, nodes)  # PyMuPDF 按 bbox 裁图保存
        figures_saved = sum(1 for n in nodes if n.image_path) - figures_before
        if rec:
            rec.stage(
                "extract_figure_images",
                {"figures_with_image_path": sum(1 for n in nodes if n.image_path), "newly_saved": figures_saved},
            )

        self._classify_sections(nodes)  # LLM 给 section_header 打 section_type 并继承到子节点

        return nodes

    def _get_page_height(self, doc) -> float:
        """Extract page height from document."""
        if hasattr(doc, 'pages') and doc.pages:  # Docling 文档带 pages 信息
            try:
                if isinstance(doc.pages, dict):  # pages 可能是 page_no → page 的字典
                    first_page = next(iter(doc.pages.values()))
                else:
                    first_page = doc.pages[0]  # 或列表，取第一页

                if hasattr(first_page, 'size') and hasattr(first_page.size, 'height'):
                    return first_page.size.height  # 返回第一页高度（点/point）
            except (KeyError, IndexError, StopIteration):
                pass  # 解析失败则用默认值
        return 792.0  # 美式 Letter 约 11×72=792 pt，作为兜底页高

    def _filter_items(self, items: list[Any], page_height: float) -> list[Any]:
        """Filter out headers, footers, and page numbers."""
        filtered = []

        for item in items:
            if not hasattr(item, 'text'):  # 图、表等可能无 text 字段，保留
                filtered.append(item)
                continue

            text = item.text.strip()
            if not text:  # 主 text 为空
                orig = getattr(item, 'orig', None)  # 部分 item 用 orig 存原始字符串
                if orig and orig.strip():
                    filtered.append(item)  # 有 orig 则仍保留
                continue  # 完全无文本则丢弃

            bbox = self._extract_bbox(item)  # 取边界框供位置过滤

            if self.cleaner.is_page_number(text):  # 纯数字页码
                continue

            if bbox and self.cleaner.is_header_footer(text, page_height, bbox):
                continue  # 位于页眉页脚区域则丢弃

            filtered.append(item)

        return filtered

    def _sort_reading_order(self, items: list[Any]) -> list[Any]:
        """Sort items by reading order using bbox coordinates.

        PDF coordinate system: y increases upward (larger y = higher on page).
        Strategy: group items at the same vertical band (row), sort rows top-to-bottom,
        sort items within each row left-to-right. This handles both single and multi-column.
        """
        pages = {}  # page_no → 该页 item 列表
        for item in items:
            page_num = item.prov[0].page_no if item.prov and len(item.prov) > 0 else 0
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(item)

        sorted_items = []  # 全文档按页、按阅读顺序的 item 列表
        for page_num in sorted(pages.keys()):  # 按页码从小到大
            page_items = pages[page_num]

            items_with_bbox = []
            for item in page_items:
                bbox = self._extract_bbox(item)
                items_with_bbox.append((item, bbox))

            valid = [(item, bbox) for item, bbox in items_with_bbox if bbox]  # 有 bbox 的
            no_bbox = [(item, bbox) for item, bbox in items_with_bbox if not bbox]  # 无 bbox 的放最后

            if not valid:  # 整页都没有 bbox，保持原顺序
                sorted_items.extend([item for item, _ in items_with_bbox])
                continue

            row_tolerance = self._estimate_row_tolerance(valid)  # 同一“行”的垂直容差
            rows = self._group_into_rows(valid, row_tolerance)  # 按 y 聚类成行

            for row in rows:
                row.sort(key=lambda x: x[1][0])  # 行内按左边界 x 从左到右

            for row in rows:
                sorted_items.extend([item for item, _ in row])  # 自上而下追加各行
            sorted_items.extend([item for item, _ in no_bbox])  # 无 bbox 项接在该页末尾

        return sorted_items

    def _estimate_row_tolerance(self, items_with_bbox: list[tuple]) -> float:
        """Estimate vertical tolerance for grouping items into rows."""
        heights = [abs(bbox[1] - bbox[3]) for _, bbox in items_with_bbox if bbox]  # 各块高度
        if not heights:
            return 10.0
        avg_height = sum(heights) / len(heights)
        return max(avg_height * 0.6, 5.0)  # 容差约为平均块高的 60%，至少 5pt

    def _group_into_rows(self, items_with_bbox: list[tuple], tolerance: float) -> list[list[tuple]]:
        """Group items into rows based on vertical proximity."""
        sorted_by_y = sorted(items_with_bbox, key=lambda x: -x[1][1])  # y 大在上，先处理上方块

        rows = []  # 每行是 [(item, bbox), ...]
        for item, bbox in sorted_by_y:
            placed = False
            for row in rows:
                row_y = row[0][1][1]  # 以该行第一个块的 top y 代表行高
                if abs(bbox[1] - row_y) <= tolerance:  # 垂直距离在容差内视为同一行
                    row.append((item, bbox))
                    placed = True
                    break
            if not placed:
                rows.append([(item, bbox)])  # 新开一行

        rows.sort(key=lambda r: -r[0][1][1])  # 行按从上到下排序（y 从大到小）
        return rows

    def _process_item(
        self,
        item,
        paper_id: str,
        order: int,
        section_stack: list[str],
        top_level_x: float,
    ) -> Optional[PaperNode]:
        """Process a single document item into a PaperNode."""
        item_type = type(item).__name__  # Docling 类型名，如 TextItem、TableItem
        raw_text = item.text if hasattr(item, 'text') else ""
        if not raw_text and hasattr(item, 'orig') and item.orig:
            raw_text = item.orig
        raw_text = self.cleaner.clean_text(raw_text)

        if self._is_caption_text(raw_text):  # 匹配 "Figure 1" 等模式则强制为 caption
            node_type = "caption"
        else:
            node_type = self._map_item_type(item_type)  # Docling 类型 → 项目 node_type

        if not node_type:  # 未映射类型不建节点
            return None

        node_id = str(uuid.uuid4())  # 节点唯一 ID
        page_num = item.prov[0].page_no if item.prov and len(item.prov) > 0 else 0
        # 图/表/caption 需要 bbox 用于裁图、找 caption
        bbox = self._extract_bbox(item) if node_type in ["figure", "table", "caption"] else None

        if node_type == "section_header":
            item_bbox = self._extract_bbox(item)
            self._update_section_stack(section_stack, raw_text, item_bbox, top_level_x)

        generator = NodeContentGeneratorFactory.get_generator(node_type)  # 按类型选文本生成器

        node = PaperNode(
            node_id=node_id,
            paper_id=paper_id,
            node_type=node_type,
            text="",  # 下面由 generator 填充
            page_num=page_num,
            order=order,
            section_path=section_stack.copy(),  # 快照当前章节路径
            bbox=bbox,
        )

        if node_type == "table":
            node.metadata["item"] = item  # 保留原始 TableItem，供后续线性化表格

        context = {"raw_text": raw_text, "item": item}  # 传入生成器的上下文
        node.text = generator.generate_text(node, raw_text, context)

        return node

    def _map_item_type(self, item_type: str) -> Optional[NodeType]:
        """Map Docling item type to NodeType."""
        mapping: dict[str, NodeType] = {
            "SectionHeaderItem": "section_header",
            "TextItem": "paragraph",
            "ListItem": "paragraph",
            "TableItem": "table",
            "PictureItem": "figure",
            "FormulaItem": "formula",
        }
        return mapping.get(item_type)  # 未知类型返回 None

    def _is_caption_text(self, text: str) -> bool:
        """Check if text is a caption based on pattern."""
        if not text:
            return False
        text = text.strip()
        return bool(
            re.match(r'^(Caption:\s*)?(Figure|Table|Fig\.|Tab\.)\s+\d+', text, re.IGNORECASE)
        )

    def _extract_bbox(self, item) -> Optional[tuple[float, float, float, float]]:
        """Extract bounding box from item."""
        if hasattr(item, 'self_ref') and item.self_ref:
            ref = item.self_ref
            if hasattr(ref, 'bbox') and ref.bbox:
                bbox = ref.bbox
                return (bbox.l, bbox.t, bbox.r, bbox.b)

        if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
            prov = item.prov[0]  # provenance：版面来源，常含 bbox
            if hasattr(prov, 'bbox') and prov.bbox:
                bbox = prov.bbox
                return (bbox.l, bbox.t, bbox.r, bbox.b)

        if hasattr(item, 'bbox') and item.bbox:
            bbox = item.bbox
            return (bbox.l, bbox.t, bbox.r, bbox.b)

        return None

    def _compute_top_level_x(self, items: list[Any]) -> float:
        """Compute minimum left-edge x of all section headers as top-level baseline."""
        xs = []
        for item in items:
            if type(item).__name__ == 'SectionHeaderItem':
                bbox = self._extract_bbox(item)
                if bbox:
                    xs.append(bbox[0])  # 收集所有一级标题左边界 x
        return min(xs) if xs else 0.0

    def _is_top_level_section(self, header_text: str, bbox, top_level_x: float) -> bool:
        """Determine if a section header is top-level via numeric prefix or x position."""
        num_match = re.match(r'^(\d+(?:\.\d+)*)\s+', header_text)  # 如 "1.2 Introduction"
        if num_match:
            dots = len(num_match.group(1).split('.'))  # "1.2" → 2 级，非顶层
            if dots > 1:
                return False
            if bbox:
                return bbox[0] <= top_level_x + 10  # 左缩进接近最左标题则视为顶层
            return True
        return True  # 无编号时默认当作可更新栈的 section

    def _update_section_stack(self, section_stack: list[str], header_text: str, bbox, top_level_x: float):
        """Update section stack, keeping only the top-level section."""
        if self._is_top_level_section(header_text, bbox, top_level_x):
            section_stack[:] = [header_text]  # 原地替换为仅含当前顶层标题

    def _linearize_table(self, item) -> str:
        """Linearize table into key-value format."""
        if not hasattr(item, 'data') or not item.data:
            return ""

        data = item.data
        if not hasattr(data, 'table_cells') or not data.table_cells or len(data.table_cells) == 0:
            return ""

        try:
            cells = data.table_cells
            rows_dict = {}  # row_idx → {col_idx: text}
            for cell in cells:
                row_idx = cell.start_row_offset_idx if hasattr(cell, 'start_row_offset_idx') else 0
                col_idx = cell.start_col_offset_idx if hasattr(cell, 'start_col_offset_idx') else 0
                text = cell.text if hasattr(cell, 'text') else str(cell)

                if row_idx not in rows_dict:
                    rows_dict[row_idx] = {}
                rows_dict[row_idx][col_idx] = text

            if not rows_dict:
                return ""

            sorted_rows = sorted(rows_dict.items())  # 按行号排序
            # 第一行作表头
            headers = (
                [sorted_rows[0][1].get(i, "") for i in range(max(sorted_rows[0][1].keys()) + 1)]
                if sorted_rows
                else []
            )
            # 其余行为数据行
            rows = [
                [row_data.get(i, "") for i in range(max(row_data.keys()) + 1)]
                for _, row_data in sorted_rows[1:]
            ]

            if not headers:
                return ""

        except (AttributeError, IndexError, TypeError, KeyError):
            return ""

        return TableGenerator.linearize_table(headers, rows)  # 转为 "列名: 值" 可读字符串

    def _link_captions_to_figures_tables(self, nodes: list[PaperNode]):
        """Link captions to figures/tables; always linearize tables (not only when caption exists)."""
        for node in nodes:
            if node.node_type not in ("figure", "table"):
                continue
            caption_type = "Figure" if node.node_type == "figure" else "Table"
            caption_text = self._find_caption_for_node(node, nodes, caption_type)
            generator = NodeContentGeneratorFactory.get_generator(node.node_type)
            context: dict = {}
            if caption_text:
                context["caption_text"] = caption_text
            if node.node_type == "table" and "item" in node.metadata:
                linearized = self._linearize_table(node.metadata["item"])
                if linearized:
                    context["linearized_table"] = linearized
            if context:
                node.text = generator.generate_text(node, "", context)

    def _find_caption_for_node(self, node: PaperNode, nodes: list[PaperNode], caption_type: str) -> str:
        """Find caption for a figure/table node."""
        if not node.bbox:
            return ""

        best_caption = ""
        min_distance = float('inf')

        for other in nodes:
            if other.node_type == "caption" and other.page_num == node.page_num:
                if caption_type.lower() in other.text.lower():  # caption 文本含 Figure/Table
                    if other.bbox:
                        distance = abs(other.bbox[1] - node.bbox[1])  # 垂直距离（同页）
                        if distance < min_distance and distance < 300:  # 300pt 内最近的 caption
                            min_distance = distance
                            best_caption = other.text

        return best_caption

    def _link_text_references(self, nodes: list[PaperNode]):
        """Link paragraphs to figures/tables they reference in text."""
        fig_table_index: dict[tuple[str, str], PaperNode] = {}  # ("figure","3") → 对应节点
        for node in nodes:
            if node.node_type in ["figure", "table", "caption"]:
                match = re.search(r'(Figure|Table)\s+(\d+)', node.text, re.IGNORECASE)
                if match:
                    key = (match.group(1).lower(), match.group(2))
                    if key not in fig_table_index:
                        fig_table_index[key] = node

        for node in nodes:
            if node.node_type not in ["paragraph", "formula"]:
                continue
            for match in re.finditer(r'(Figure|Table)\s+(\d+)', node.text, re.IGNORECASE):
                key = (match.group(1).lower(), match.group(2))
                target = fig_table_index.get(key)
                if target and target.node_id not in node.related_ids:
                    node.related_ids.append(target.node_id)  # 段落 → 图/表
                    if node.node_id not in target.related_ids:
                        target.related_ids.append(node.node_id)  # 双向关联

    def _extract_figure_images(self, pdf_path: str, doc, nodes: list[PaperNode]):
        """Crop and save figure images using pymupdf based on bbox.

        PDF coordinate system (pymupdf): origin top-left, y increases downward.
        Docling bbox: (l, t, r, b) where t > b (origin bottom-left).
        We convert via: fitz_y = page_height - docling_y.
        """
        try:
            import fitz  # pymupdf
        except ImportError:
            print("pymupdf not installed, skipping figure image extraction.")
            return

        figure_nodes = [n for n in nodes if n.node_type == "figure" and n.bbox and n.page_num]
        if not figure_nodes:
            return

        try:
            fitz_doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Failed to open PDF with pymupdf: {e}")
            return

        paper_id = figure_nodes[0].paper_id
        save_dir = self.figure_save_dir / paper_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Build page height map from fitz (points)
        page_heights = {i: fitz_doc[i].rect.height for i in range(len(fitz_doc))}

        for node in figure_nodes:
            page_idx = node.page_num - 1  # fitz 页码从 0 开始
            if page_idx < 0 or page_idx >= len(fitz_doc):
                continue

            page = fitz_doc[page_idx]
            ph = page_heights[page_idx]

            # Docling bbox: (l, t, r, b) in PDF points, origin bottom-left
            bbox = node.bbox
            if not bbox:
                continue
            l, t, r, b = bbox
            # Convert to fitz (origin top-left)
            fitz_rect = fitz.Rect(l, ph - t, r, ph - b)
            # Add small padding
            fitz_rect = fitz_rect + fitz.Rect(-4, -4, 4, 4)
            fitz_rect = fitz_rect & page.rect  # clamp to page

            if fitz_rect.is_empty or fitz_rect.is_infinite:
                continue

            try:
                mat = fitz.Matrix(2.0, 2.0)  # 2x DPI for clarity
                clip = page.get_pixmap(matrix=mat, clip=fitz_rect)
                img_path = save_dir / f"page{node.page_num}_order{node.order}.png"
                clip.save(str(img_path))
                node.image_path = str(img_path)  # 写入节点，后续可进 metadata
            except Exception as e:
                print(f"Failed to crop figure {node.node_id}: {e}")

        fitz_doc.close()

    def _classify_sections(self, nodes: list[PaperNode]):
        """Classify section headers using LLM."""
        rec = self._recorder
        if not self.llm:
            if rec:
                rec.set_section_classification("skipped", {"reason": "no_llm_configured"})
            return

        section_nodes = [n for n in nodes if n.node_type == "section_header"]
        if not section_nodes:
            if rec:
                rec.set_section_classification("skipped", {"reason": "no_section_headers"})
            return

        titles = [n.text.replace("Section: ", "").strip() for n in section_nodes]
        titles_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))

        try:
            t0 = time.perf_counter()
            structured_llm = self.llm.with_structured_output(SectionClassification)
            result = structured_llm.invoke([
                SystemMessage(content=SECTION_CLASSIFIER_PROMPT),
                HumanMessage(content=f"Section titles:\n{titles_str}"),
            ])

            title_to_type = {c["title"]: c["type"] for c in result.classifications}

            for node in section_nodes:
                title = node.text.replace("Section: ", "").strip()
                section_type = title_to_type.get(title, "other")
                node.metadata["section_type"] = section_type

            for node in nodes:
                if node.node_type != "section_header" and node.section_path:
                    for section_node in section_nodes:
                        section_title = section_node.text.replace("Section: ", "").strip()
                        if section_title in node.section_path:
                            node.metadata["section_type"] = section_node.metadata.get(
                                "section_type", "other"
                            )
                            break
            if rec:
                rec.set_section_classification(
                    "ok",
                    {
                        "section_count": len(section_nodes),
                        "duration_sec": round(time.perf_counter() - t0, 3),
                        "mapping": title_to_type,
                    },
                )
        except Exception as e:
            logger.warning("Section classification failed: %s", e)
            if rec:
                rec.set_section_classification("failed", {"error": str(e)})


class RAGIntegration:
    """
    将 PaperNode 转为 LangChain Document 并执行父子分块。

    父块：完整语义单元；子块：长段落再切 500 字（重叠 50），表格/图/标题等不再切分。
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "papers",
    ):
        self.embeddings = EmbeddingService.get_embeddings(embedding_model)
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 子块目标长度（字符）
            chunk_overlap=50,  # 相邻子块重叠，避免句意被截断
            separators=["\n\n", "\n", ".", "?", "!", ";", " "],  # 优先在段落/句/空格处切
        )

    def nodes_to_documents(self, nodes: list[PaperNode], content_hash: str = "") -> list[Document]:
        """Convert PaperNodes to LangChain Documents."""
        docs = []
        for node in nodes:
            if not node.text.strip():  # 空文本节点跳过
                continue

            # 不把 Docling 原始 item 对象写入 metadata（不可序列化）
            metadata = {k: v for k, v in node.metadata.items() if k != "item"}
            metadata.update({
                "node_id": node.node_id,
                "paper_id": node.paper_id,
                "node_type": node.node_type,
                "page_num": node.page_num,
                "order": node.order,
                "section_path": " > ".join(node.section_path) if node.section_path else "",
                "section_type": node.metadata.get("section_type", "other"),  # 供检索过滤
                "content_hash": content_hash,  # 论文内容哈希，便于增量更新
                "bbox": str(node.bbox) if node.bbox else "",
                "node_parent_id": node.parent_id if node.parent_id else "",
                "image_path": node.image_path if node.image_path else "",
                "vlm_description": node.metadata.get("vlm_description", ""),
            })

            docs.append(Document(page_content=node.text, metadata=metadata))
        return docs

    def create_chunks(self, docs: list[Document]) -> tuple[list[Document], list[Document]]:
        """Create parent and child chunks for retrieval."""
        parents = []
        children = []

        no_split_types = {"table", "figure", "section_header", "caption"}  # 这些类型整段作一块

        for doc in docs:
            chunk_parent_id = str(uuid.uuid4())  # 父块 ID，子块通过 chunk_parent_id 回溯
            doc.metadata["chunk_id"] = chunk_parent_id
            if "bbox" not in doc.metadata:
                doc.metadata["bbox"] = ""
            if "image_path" not in doc.metadata:
                doc.metadata["image_path"] = ""
            if "vlm_description" not in doc.metadata:
                doc.metadata["vlm_description"] = ""
            parents.append(doc)  # 每个逻辑节点对应一个父 Document

            node_type = doc.metadata.get("node_type", "")
            should_split = node_type not in no_split_types and len(doc.page_content) > 500

            if should_split:
                splits = self.splitter.split_documents([doc])
                for i, split in enumerate(splits):
                    split.metadata["chunk_parent_id"] = chunk_parent_id
                    split.metadata["chunk_id"] = f"{chunk_parent_id}_child_{i}"
                    if "bbox" not in split.metadata:
                        split.metadata["bbox"] = ""
                    if "image_path" not in split.metadata:
                        split.metadata["image_path"] = ""
                    if "vlm_description" not in split.metadata:
                        split.metadata["vlm_description"] = ""
                    children.append(split)
            else:
                # 不切分：子块与父块正文相同，仅 chunk_id 不同
                child = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "chunk_parent_id": chunk_parent_id,
                        "chunk_id": f"{chunk_parent_id}_child_0",
                    },
                )
                if "bbox" not in child.metadata:
                    child.metadata["bbox"] = ""
                if "image_path" not in child.metadata:
                    child.metadata["image_path"] = ""
                if "vlm_description" not in child.metadata:
                    child.metadata["vlm_description"] = ""
                children.append(child)

        return parents, children

    def store_in_milvus(self, parents: list[Document], children: list[Document]) -> bool:
        """Store documents in Milvus with hybrid index (dense + BM25)."""
        if not parents or not children:
            return False

        # Milvus 内置 BM25：对 text 字段生成 sparse 向量，与 dense 一起做混合检索
        bm25 = BM25BuiltInFunction(input_field_names="text", output_field_names="sparse")

        try:
            from .factory import ensure_milvus_orm_connection

            child_store = Milvus(
                embedding_function=self.embeddings,
                builtin_function=bm25,
                vector_field=["dense", "sparse"],  # 双字段：嵌入 + BM25
                collection_name=f"{self.collection_name}_children",  # 检索主用子 collection
                connection_args={"uri": self.milvus_uri},
            )
            ensure_milvus_orm_connection(child_store)
            child_store.add_documents(children)

            parent_store = Milvus(
                embedding_function=self.embeddings,
                builtin_function=bm25,
                vector_field=["dense", "sparse"],
                collection_name=f"{self.collection_name}_parents",  # 子块命中后回溯父块
                connection_args={"uri": self.milvus_uri},
            )
            ensure_milvus_orm_connection(parent_store)
            parent_store.add_documents(parents)
            return True
        except Exception as e:
            logger.exception(f"Failed to store in Milvus: {e}")
            raise
