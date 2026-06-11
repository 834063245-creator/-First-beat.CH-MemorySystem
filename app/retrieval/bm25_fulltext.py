"""BM25 全文检索 — 对 ChromaDB 全部 document 建索引。

内存索引，适合 benchmark 数据量（< 10K 条）。生产环境可扩展为磁盘索引。
"""
import logging
from rank_bm25 import BM25Okapi
from app.brain.semantic import tokenize as _sem_tokenize

logger = logging.getLogger(__name__)


class BM25FullTextIndex:
    """内存 BM25 索引，覆盖 ChromaDB collection 全部 document 全文。

    使用方式：
        index = BM25FullTextIndex(chroma_service)
        ids = index.search("用户的查询", top_k=20)
        # → [memory_id, ...] 按 BM25 分数降序

    新增记忆后索引自动检测 rebuild（对比 doc_count），带 30s 冷却防抖。
    """

    def __init__(self, chroma_service):
        self._chroma = chroma_service
        self._doc_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._doc_count: int = 0
        self._last_rebuild: float = 0.0
        self._rebuild_cooldown: float = 30.0  # 最少间隔 30 秒
        self._build()

    def _build(self):
        """从 ChromaDB 全量构建 BM25 索引。"""
        try:
            result = self._chroma._collection.get(
                include=["documents", "metadatas"])
        except Exception:
            logger.warning("BM25 索引构建失败：无法读取 ChromaDB")
            return

        ids = result.get("ids", [])
        docs = result.get("documents", []) or [None] * len(ids)
        metas = result.get("metadatas", []) or [None] * len(ids)

        corpus = []
        doc_ids = []
        for i, mid in enumerate(ids):
            doc = docs[i] if i < len(docs) and docs[i] else ""
            if not doc:
                # 尝试从 metadata 取 summary 作为 fallback
                meta = metas[i] if i < len(metas) and metas[i] else {}
                doc = (meta or {}).get("summary", "")
            if not doc:
                continue
            tokens = _sem_tokenize(doc)
            corpus.append(tokens)
            doc_ids.append(mid)

        self._doc_ids = doc_ids
        self._doc_count = len(doc_ids)
        if corpus:
            self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 全文索引构建完成: %d 条文档", len(corpus))

    def _maybe_rebuild(self):
        """如果 ChromaDB 有新增文档且超过冷却期，重建索引。"""
        import time as _time
        try:
            current_count = self._chroma.count()
        except Exception:
            return
        if current_count != self._doc_count:
            now = _time.monotonic()
            if now - self._last_rebuild < self._rebuild_cooldown:
                return  # 冷却期内跳过，防抖
            logger.info("BM25 检测到文档数变化 (%d → %d)，重建索引",
                        self._doc_count, current_count)
            self._last_rebuild = now
            self._build()

    def search(self, query: str, top_k: int = 20) -> list[str]:
        """返回匹配的 memory_id 列表（按 BM25 分数降序）。

        每次搜索前自动检测 ChromaDB 文档数变化并增量重建。
        """
        self._maybe_rebuild()
        if not self._bm25 or not self._doc_ids:
            return []
        tokens = _sem_tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [self._doc_ids[i] for i, score in ranked if score > 0]

    def rebuild(self):
        """强制全量重建索引（外部可在注入完成后调用）。"""
        self._doc_ids = []
        self._bm25 = None
        self._doc_count = 0
        self._build()

    def clear(self):
        """清空索引（benchmark reset 用）。"""
        self._doc_ids = []
        self._bm25 = None
        self._doc_count = 0
        logger.info("BM25 全文索引已清空")
