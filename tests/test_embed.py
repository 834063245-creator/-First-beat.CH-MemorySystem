# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 78a8913e

"""embed.py 测试 — 缓存命中、向量归一化、批量嵌入。

v3: 从 bge-m3 (Ollama HTTP) 切换到 qwen_embed（纯 Python+numpy）。
    去掉 n-gram 签名、合并器测试；维度 1024→3584。
"""
import math
import threading
import time

# v3: qwen_embed 是纯 Python，不再需要 Ollama
# 移除 pytestmark = pytest.mark.real_embed


# ═══════════════════════════════════════════════════════════════
# 缓存命中
# ═══════════════════════════════════════════════════════════════

class TestEmbedCache:
    """精确缓存命中。"""

    def test_exact_cache_hit(self):
        """同一文本两次调用，第二次应命中精确缓存（<1ms）。"""
        from app.llm.embed import local_embed

        r1 = local_embed("缓存命中测试文本")
        assert r1 is not None and len(r1) == 3584

        t0 = time.perf_counter()
        r2 = local_embed("缓存命中测试文本")
        cache_ms = (time.perf_counter() - t0) * 1000

        assert r2 is not None
        assert cache_ms < 5, f"缓存命中太慢: {cache_ms:.0f}ms"
        # 同一文本两次结果完全一致
        assert r1 == r2

    def test_different_texts_different_vectors(self):
        """不同文本产生不同向量。"""
        from app.llm.embed import local_embed

        r1 = local_embed("你好世界")
        r2 = local_embed("今天天气真好适合出去走走")

        assert r1 is not None and r2 is not None
        # 两者不应完全相同
        assert r1 != r2

    def test_empty_text_returns_none(self):
        """空文本 / 纯空白返回 None。"""
        from app.llm.embed import local_embed

        assert local_embed("") is None
        assert local_embed("   ") is None

    def test_cache_lru_eviction(self):
        """缓存超过上限时淘汰最旧的条目。"""
        from app.llm.embed import local_embed, _embed_cache, _EMBED_CACHE_MAX, _embed_cache_lock

        # 填满缓存
        for i in range(_EMBED_CACHE_MAX + 10):
            local_embed(f"LRU填充文本_{i:04d}")

        with _embed_cache_lock:
            size = len(_embed_cache)
        # 不超过上限
        assert size <= _EMBED_CACHE_MAX, f"缓存溢出: {size} > {_EMBED_CACHE_MAX}"


# ═══════════════════════════════════════════════════════════════
# 向量归一化
# ═══════════════════════════════════════════════════════════════

class TestVectorNormalization:
    """返回向量应为单位向量。"""

    def test_single_vector_normalized(self):
        from app.llm.embed import local_embed
        r = local_embed("归一化测试")
        assert r is not None
        norm = math.sqrt(sum(v * v for v in r))
        assert abs(norm - 1.0) < 0.01, f"向量未归一化: norm={norm:.4f}"

    def test_batch_vectors_normalized(self):
        from app.llm.embed import local_embed_batch
        results = local_embed_batch(["文本A", "文本B", "文本C"])
        for i, r in enumerate(results):
            assert r is not None, f"batch[{i}] 为 None"
            norm = math.sqrt(sum(v * v for v in r))
            assert abs(norm - 1.0) < 0.01, f"batch[{i}] 未归一化: norm={norm:.4f}"

    def test_vector_dimension(self):
        """qwen_embed 输出 3584 维。"""
        from app.llm.embed import local_embed, local_embed_batch

        assert len(local_embed("维度测试")) == 3584
        for r in local_embed_batch(["a", "b"]):
            assert len(r) == 3584


# ═══════════════════════════════════════════════════════════════
# 批量嵌入
# ═══════════════════════════════════════════════════════════════

class TestBatchEmbed:
    """local_embed_batch 行为。"""

    def test_batch_empty(self):
        from app.llm.embed import local_embed_batch
        assert local_embed_batch([]) == []

    def test_batch_single(self):
        from app.llm.embed import local_embed_batch
        results = local_embed_batch(["单条"])
        assert len(results) == 1
        assert len(results[0]) == 3584

    def test_batch_all_succeed(self):
        from app.llm.embed import local_embed_batch
        texts = [f"批量测试_{i}" for i in range(10)]
        results = local_embed_batch(texts)
        assert len(results) == 10
        for i, r in enumerate(results):
            assert r is not None, f"batch[{i}] 失败"
            assert len(r) == 3584

    def test_batch_with_duplicates(self):
        """重复文本在 batch 中走缓存，不重复 embed。"""
        from app.llm.embed import local_embed_batch

        # 先嵌入一次写入缓存
        local_embed_batch(["去重测试文本"])

        # 再 batch 包含同一文本
        results = local_embed_batch(["去重测试文本", "新文本_A", "新文本_B", "去重测试文本"])
        assert len(results) == 4
        assert all(r is not None for r in results)
        # 重复的两条结果应相同
        assert results[0] == results[3]


# ═══════════════════════════════════════════════════════════════
# 线程安全
# ═══════════════════════════════════════════════════════════════

class TestThreadSafety:
    """并发环境下 embed 模块不崩溃、不丢数据。"""

    def test_concurrent_read_write_cache(self):
        """并发读写缓存不抛异常。"""
        from app.llm.embed import local_embed

        errors = []

        def worker(start, count):
            for i in range(start, start + count):
                try:
                    local_embed(f"线程安全测试_{i:04d}")
                except Exception as e:
                    errors.append(str(e))

        N = 4
        threads = [
            threading.Thread(target=worker, args=(i * 25, 25))
            for i in range(N)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发异常: {errors}"

    def test_mixed_single_and_batch(self):
        """单条和批量混合调用，彼此不干扰。"""
        from app.llm.embed import local_embed, local_embed_batch

        errors = []

        def single_worker():
            for i in range(10):
                try:
                    local_embed(f"混合单条_{i}")
                except Exception as e:
                    errors.append(("single", str(e)))

        def batch_worker():
            for i in range(5):
                try:
                    local_embed_batch([f"混合批量_{i}_a", f"混合批量_{i}_b"])
                except Exception as e:
                    errors.append(("batch", str(e)))

        t1 = threading.Thread(target=single_worker)
        t2 = threading.Thread(target=batch_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"混合并发异常: {errors}"
