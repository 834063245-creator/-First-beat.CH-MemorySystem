"""测试 app/retrieval/bm25_fulltext.py — BM25 全文检索索引。

覆盖：BM25FullTextIndex 的构建、搜索、自动重建、手动重建、清空。
"""
import pytest
from unittest.mock import MagicMock, patch


class FakeChromaService:
    """ChromaService 的最小模拟，暴露 _collection API。"""
    def __init__(self, ids=None, docs=None, metas=None):
        self._ids = ids or []
        self._docs = docs or []
        self._metas = metas or []
        self._collection = MagicMock()

    def count(self):
        return len(self._ids)

    def setup_collection_mock(self):
        self._collection.get.return_value = {
            "ids": self._ids,
            "documents": self._docs,
            "metadatas": self._metas,
        }


class TestBM25FullTextIndex:
    def test_build_empty_ok(self):
        """空 collection 不会崩溃。"""
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService()
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        assert idx.search("任何查询") == []

    def test_build_and_search(self):
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService(
            ids=["m1", "m2", "m3"],
            docs=["今天天气很好适合出去散步", "Python代码写完了非常开心", "最近压力好大项目快要崩了"],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        results = idx.search("Python代码")
        assert "m2" in results

    def test_search_returns_empty_for_no_match(self):
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService(
            ids=["m1", "m2"],
            docs=["文本一内容", "文本二内容"],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        results = idx.search("完全不相关的查询词")
        assert results == []

    def test_search_empty_query(self):
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService(
            ids=["m1"],
            docs=["一些文本"],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        results = idx.search("")
        assert results == []

    def test_top_k_respected(self):
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService(
            ids=[f"m{i}" for i in range(50)],
            docs=[f"文档内容编号{i}" for i in range(50)],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        results = idx.search("文档", top_k=5)
        assert len(results) <= 5

    def test_auto_rebuild_on_count_change(self):
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService(
            ids=["m1", "m2"],
            docs=["文档一", "文档二"],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        assert idx._doc_count == 2
        # 模拟新增文档
        chroma._ids = ["m1", "m2", "m3"]
        chroma._docs = ["文档一", "文档二", "新文档三"]
        # 重置 mock 以返回新数据
        chroma.setup_collection_mock()
        results = idx.search("新文档")
        assert "m3" in results
        assert idx._doc_count == 3

    def test_rebuild_manual(self):
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService(
            ids=["m1"],
            docs=["旧数据"],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        # 修改数据
        chroma._ids = ["m1", "m2"]
        chroma._docs = ["更新后一", "更新后二"]
        chroma.setup_collection_mock()
        idx.rebuild()
        assert idx._doc_count == 2

    def test_clear_resets_index(self):
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService(
            ids=["m1", "m2"],
            docs=["文档一", "文档二"],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        assert idx._doc_count == 2
        idx.clear()
        assert idx._doc_count == 0
        assert idx._bm25 is None
        assert idx.search("文档") == []

    def test_handles_missing_documents(self):
        """部分文档缺失时用 metadata.summary 兜底。"""
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        # BM25 需要 ≥3 文档才有非零 IDF（rank_bm25 的 IDF 公式特性）
        chroma = FakeChromaService(
            ids=["m1", "m2", "m3"],
            docs=["", "", "独立文档内容"],
            metas=[
                {"summary": "咖啡是黑色的饮料"},
                {"summary": "茶是绿色的饮料"},
                {"summary": ""},
            ],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        results = idx.search("咖啡")
        assert "m1" in results

    def test_handles_build_failure_gracefully(self):
        """ChromaDB 不可用时构建失败不应崩溃。"""
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        chroma = FakeChromaService()
        chroma._collection.get.side_effect = Exception("ChromaDB down")
        idx = BM25FullTextIndex(chroma)
        assert idx._doc_count == 0
        assert idx.search("任何") == []

    def test_count_failure_skips_rebuild(self):
        """count() 失败时不应 crash，使用已有索引。"""
        from app.retrieval.bm25_fulltext import BM25FullTextIndex
        # BM25 需要 ≥3 文档才有非零 IDF
        chroma = FakeChromaService(
            ids=["m1", "m2", "m3"],
            docs=["文档一", "无关文档二", "额外文档三"],
        )
        chroma.setup_collection_mock()
        idx = BM25FullTextIndex(chroma)
        assert idx._doc_count == 3
        # 验证现有索引可用
        results_before = idx.search("文档一")
        assert "m1" in results_before
        # 模拟 count 失败 → _maybe_rebuild 不崩溃，仍用旧索引
        chroma.count = MagicMock(side_effect=Exception("count failed"))
        results_after = idx.search("文档一")
        assert "m1" in results_after
