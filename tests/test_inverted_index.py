"""测试 app/memory/inverted.py 的 InvertedIndex。

覆盖：构建、查询、增量更新、OR退化、删除、标签索引、多线程并发。
"""
import sys
sys.path.insert(0, ".")

import threading
import pytest
from app.memory.inverted import InvertedIndex


class TestInvertedIndex:

    # ── 基础构建与查询 ──

    def test_build_and_query(self):
        idx = InvertedIndex()
        idx.build([
            ("mem1", "最近压力好大项目快要崩了"),
            ("mem2", "Python代码写完了非常开心"),
            ("mem3", "项目上线了好累需要休息"),
        ])
        # "项目" 出现在 mem1 和 mem3
        results = idx.query(["项目"], min_match=1)
        assert "mem1" in results
        assert "mem3" in results

    def test_query_multi_keyword_and(self):
        """多关键词 AND 查询。"""
        idx = InvertedIndex()
        idx.build([
            ("mem1", "最近压力好大项目崩了"),
            ("mem2", "天气不错适合出去走走"),
        ])
        # "压力" + "项目" 只在 mem1
        results = idx.query(["压力", "项目"], min_match=2)
        assert "mem1" in results

    def test_build_empty_returns_empty(self):
        idx = InvertedIndex()
        idx.build([])
        assert idx.query(["压力"]) == []

    # ── 增量更新 ──

    def test_add_incremental(self):
        idx = InvertedIndex()
        idx.build([("mem1", "最近压力好大")])
        idx.add("mem2", "今天写代码")
        results = idx.query(["今天", "代码"], min_match=2)
        assert "mem2" in results

    # ── 删除 ──

    def test_remove(self):
        idx = InvertedIndex()
        idx.build([
            ("mem1", "今天天气真好"),
            ("mem2", "今天写代码"),
        ])
        idx.remove("mem1")
        results = idx.query(["今天"], min_match=1)
        assert "mem1" not in results
        assert "mem2" in results

    def test_remove_nonexistent(self):
        """删除不存在的ID不抛异常。"""
        idx = InvertedIndex()
        idx.build([("mem1", "今天天气真好")])
        idx.remove("nonexistent")  # should not raise

    # ── OR退化逻辑 ──

    def test_or_fallback(self):
        """AND结果<3条时退化为OR + 按匹配数排序。"""
        idx = InvertedIndex()
        idx.build([
            ("memA", "今天下雨压力大"),
            ("memB", "今天天气不错"),
            ("memC", "下雨天真好"),
            ("memD", "今天写代码"),
        ])
        # "今天" 匹配 memA, memB, memD；"下雨" 匹配 memA, memC
        # AND = memA（仅1条<3）→ OR退化
        results = idx.query(["今天", "下雨"], min_match=2)
        assert "memA" in results  # 匹配2个词

    # ── 标签索引 ──

    def test_build_tags(self):
        idx = InvertedIndex()
        idx._tag_index = {}
        idx.build_tags([
            ("mem1", "技术,编程,Python"),
            ("mem2", "生活,旅行"),
        ])
        results = idx.query_tags(["技术"])
        assert "mem1" in results
        assert "mem2" not in results

    def test_query_tags_empty(self):
        idx = InvertedIndex()
        assert idx.query_tags([]) == set()

    def test_query_tags_no_match(self):
        idx = InvertedIndex()
        idx.build_tags([("mem1", "技术,编程")])
        assert idx.query_tags(["音乐"]) == set()

    # ── get_exact ──

    def test_get_exact(self):
        idx = InvertedIndex()
        idx.build([("mem1", "最近压力好大需要休息")])
        result = idx.get_exact("最近")
        assert "mem1" in result

    def test_get_exact_no_match(self):
        idx = InvertedIndex()
        idx.build([("mem1", "今天天气真好")])
        assert idx.get_exact("不存在的词") == set()

    # ── 空查询 ──

    def test_empty_query(self):
        idx = InvertedIndex()
        idx.build([("mem1", "任意内容")])
        assert idx.query([]) == []

    # ── 多线程并发 ──

    def test_concurrent_query_and_add(self):
        idx = InvertedIndex()
        idx.build([
            ("mem1", "今天天气真好"),
            ("mem2", "Python写代码"),
        ])
        errors = []

        def worker_query():
            for _ in range(20):
                try:
                    r = idx.query(["今天"], min_match=1)
                    assert isinstance(r, list)
                except Exception as e:
                    errors.append(f"query: {e}")

        def worker_add():
            for i in range(10):
                try:
                    idx.add(f"mem_new_{i}", f"新的内容{i}")
                except Exception as e:
                    errors.append(f"add: {e}")

        threads = [
            threading.Thread(target=worker_query) for _ in range(5)
        ] + [
            threading.Thread(target=worker_add) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发错误: {errors}"
        # 验证新增的ID可查询
        for i in range(10):
            results = idx.query(["内容"], min_match=1)
            assert f"mem_new_{i}" in results, f"mem_new_{i} 未找到"
