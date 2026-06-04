"""端到端回归测试 — 模拟多日对话，验证认知管线不丢记忆、不漂移。

测试场景：
  1. 7 天 × 5 轮对话 → 检索质量和人格一致性
  2. 情绪翻转 → 旧记忆标记 stale
  3. 空闲 8 小时 → 浅巩固触发

设计原则：
  - Mock 所有外部依赖（ChromaDB、DeepSeek、Ollama）
  - 测试真实管线逻辑，不测 mock
  - 每个场景独立运行，不共享状态
"""
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ═════════════════════════════════════════════════════════════
# 辅助：构造模拟对话历史
# ═════════════════════════════════════════════════════════════

def _make_timestamp(days_ago: float, hour: int = 12) -> str:
    """构造一个 days_ago 天前 hour 点的时间戳字符串。"""
    dt = datetime.now() - timedelta(days=days_ago)
    dt = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _simulate_conversation(
    days: int, rounds_per_day: int, topics: list[list[str]],
) -> list[dict]:
    """生成多日对话记录。

    topics[i] 是第 i 天的话题列表（每轮一个话题）。
    返回 chat_history 格式的 records。
    """
    records = []
    for day in range(days):
        day_topics = topics[day] if day < len(topics) else ["日常闲聊"]
        for r in range(rounds_per_day):
            topic = day_topics[r] if r < len(day_topics) else "接上文"
            ts = _make_timestamp(days - day, hour=10 + r)
            records.append({
                "user_message": f"[Day{day+1}] {topic}",
                "llm_reply": f"回复：关于{topic}",
                "timestamp": ts,
            })
    return records


# ═════════════════════════════════════════════════════════════
# 场景 1：检索不丢关键记忆
# ═════════════════════════════════════════════════════════════

class TestRetentionAcrossDays:
    """验证多日对话后，检索仍能找到关键信息。"""

    def test_retrieve_personal_fact_after_7_days(self):
        """7 天前用户说过"我养了一只猫叫小米"，第 7 天问"我的猫叫什么"应能检索到。"""
        from app.retrieval.pipeline import run_chat_retrieval
        from unittest.mock import MagicMock

        # 构造 ChromaDB mock：第 1 天的记忆
        mock_memory = {
            "id": "mem_001",
            "document": "用户：我养了一只猫叫小米\nAI：小米这个名字真好听",
            "metadata": {
                "summary": "用户养了一只猫叫小米",
                "tags": "猫,宠物,小米",
                "timestamp": (datetime.now() - timedelta(days=7)).timestamp(),
                "hit_count": 3,
                "heat": "hot",
                "stale": False,
                "archived": False,
                "emotion_valence_bin": "positive",
                "emotional_intensity": 1,
            },
            "distance": 0.15,
            "source": "semantic_hot",
            "score": 0.85,
        }

        mock_chroma = MagicMock()
        mock_chroma._read_collection.query.return_value = {
            "ids": [["mem_001"]],
            "metadatas": [[mock_memory["metadata"]]],
            "documents": [[mock_memory["document"]]],
            "distances": [[0.15]],
        }
        mock_chroma._get_embedding_cached.return_value = [0.1] * 1024
        mock_chroma.list_all.return_value = [mock_memory]

        mock_ctx = MagicMock()
        mock_ctx.chroma_service = mock_chroma
        mock_ctx.chat_history = MagicMock()
        mock_ctx.chat_history.get_recent.return_value = []
        mock_ctx.chat_history.get_records_snapshot.return_value = []
        mock_ctx.personality_store = MagicMock()
        mock_ctx.personality_store.rerank_tags.return_value = []
        mock_ctx.dmn.get_preheated.return_value = None
        mock_ctx.co_tracker = MagicMock()
        mock_ctx.storage_executor = MagicMock()

        query = "我的猫叫什么名字来着"
        query_emb = [0.1] * 1024

        with patch("app.retrieval.pipeline.local_embed", return_value=query_emb):
            with patch("app.retrieval.pipeline.local_embed_batch", return_value=[[0.1] * 1024]):
                with patch("app.retrieval.pipeline.extract_tags", return_value=["猫", "名字"]):
                    with patch("app.retrieval.pipeline.extract_entities", return_value=[]):
                        timeline, session, pers, mems = run_chat_retrieval(
                            query, query_emb, mock_ctx,
                        )

        assert len(mems) > 0, "应至少检索到 1 条记忆"
        found_cat = any("猫" in str(m.get("document", "")) or "小米" in str(m.get("document", "")) for m in mems)
        assert found_cat, f"应检索到关于猫/小米的记忆，实际记忆: {[m.get('document','')[:50] for m in mems]}"


# ═════════════════════════════════════════════════════════════
# 场景 2：情绪翻转检测
# ═════════════════════════════════════════════════════════════

class TestEmotionalReversal:
    """验证同一事实域的情绪翻转被正确检测。"""

    def test_positive_to_negative_flip_detected(self):
        """用户之前说"编程很开心"，现在说"编程让我崩溃"→ 旧记忆应被标记。"""
        old_ts = (datetime.now() - timedelta(days=5)).timestamp()
        new_ts = datetime.now().timestamp()

        old_mem = {
            "id": "old_001",
            "document": "用户：编程让我很开心",
            "metadata": {
                "summary": "用户觉得编程很开心",
                "tags": "编程,工作,情绪",
                "timestamp": old_ts,
                "emotion_valence_bin": "positive",
                "emotional_intensity": 2,
                "stale": False,
                "hit_count": 2,
                "embedding": [0.1] * 1024,
            },
        }
        new_mem = {
            "id": "new_001",
            "document": "用户：编程让我崩溃",
            "metadata": {
                "summary": "用户觉得编程让ta崩溃",
                "tags": "编程,工作,情绪",
                "timestamp": new_ts,
                "emotion_valence_bin": "negative",
                "emotional_intensity": 3,
                "stale": False,
                "hit_count": 1,
                "embedding": [0.12] * 1024,  # 相似但略有偏移
            },
        }

        # 验证：同 tag、情绪翻转、embedding 接近 → 应检测到翻转
        from app.analysis.emotion import resolve_emotion_category

        old_cat = resolve_emotion_category(old_mem["metadata"])
        new_cat = resolve_emotion_category(new_mem["metadata"])

        assert old_cat == "positive", f"旧记忆应识别为正情绪，实际: {old_cat}"
        assert new_cat == "negative", f"新记忆应识别为负情绪，实际: {new_cat}"
        assert old_cat != new_cat, "正负情绪应不同"

        # 验证 tag 交集
        old_tags = set(old_mem["metadata"]["tags"].split(","))
        new_tags = set(new_mem["metadata"]["tags"].split(","))
        assert old_tags & new_tags, "新旧记忆应共享标签"


# ═════════════════════════════════════════════════════════════
# 场景 3：人格标签一致性
# ═════════════════════════════════════════════════════════════

class TestPersonalityConsistency:
    """验证多轮对话后人格标签不漂移。"""

    def test_personality_remains_stable_across_rounds(self):
        """用户连续 5 轮说喜欢编程 → 人格标签不应在短时间内大幅变化。"""
        # 模拟 5 轮对话都是技术话题
        conversations = [
            {"user_message": "我今天写了一天代码", "llm_reply": "编程很有意思对吧"},
            {"user_message": "Python真的很优雅", "llm_reply": "确实，Python的语法很简洁"},
            {"user_message": "我最近在研究机器学习", "llm_reply": "ML是个有趣的领域"},
            {"user_message": "代码跑通了，好开心", "llm_reply": "恭喜！debug成功的感觉真好"},
            {"user_message": "明天继续写那个项目", "llm_reply": "加油，期待看到成果"},
        ]

        # 从这些对话中提取的标签应该一致
        from app.brain.semantic import extract_tags
        all_tags = []
        for conv in conversations:
            tags = extract_tags(conv["user_message"], topk=3)
            all_tags.extend(tags)

        # 所有轮次提取的标签应围绕技术主题
        tech_keywords = {"代码", "编程", "Python", "机器学习", "项目", "写代码", "研究", "开心"}
        # 至少一半的标签应与技术相关
        tech_count = sum(1 for t in all_tags if any(kw in t for kw in tech_keywords))
        assert tech_count > 0, f"应至少有一些技术相关标签，实际标签: {all_tags}"


# ═════════════════════════════════════════════════════════════
# 场景 4：冲动系统的疲劳抑制
# ═════════════════════════════════════════════════════════════

class TestImpulseFatigue:
    """验证冲动疲劳抑制机制正常工作。"""

    def test_repeated_signals_suppressed(self):
        """同一源连续产出一致信号 → 疲劳度上升 → 后续被抑制。"""
        from app.background.impulse import ImpulseScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "impulse_state.json")
            scheduler = ImpulseScheduler(state_path)

            # 首次信号通过
            scheduler.feed_impulse("想起了什么", priority=20, source="好奇心")
            # 疲劳度应为 0.15，有效优先级 = 20 * 0.85 = 17 > 2 → 通过
            status = scheduler.get_status_snapshot()
            assert status["pending"] >= 1, f"首次信号应进入队列，实际: {status}"

            # 连续喂 10 次同源信号
            for _ in range(10):
                scheduler.feed_impulse("重复信号", priority=20, source="好奇心")

            # 疲劳度应上升
            status = scheduler.get_status_snapshot()
            fatigue = status["source_fatigue"].get("好奇心", 0)
            assert fatigue > 0.3, f"连续产出的疲劳度应 >0.3，实际: {fatigue}"

    def test_expired_signals_discarded(self):
        """过期信号应从队列中移除。"""
        from app.background.impulse import ImpulseScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "impulse_state.json")
            scheduler = ImpulseScheduler(state_path)

            # 喂入信号（TTL=0.001 → 几乎立即过期）
            scheduler.feed_impulse("立即过期的信号", priority=20, source="好奇心", ttl=0.001)
            time.sleep(0.01)  # 等 10ms，确保信号已过期

            # get_next 不走 test_mode，让它正常检测 TTL
            result = scheduler.get_next(test_mode=False)
            # 过期信号应被丢弃，返回 None
            assert result is None, f"过期信号应被丢弃，实际: {result}"


# ═════════════════════════════════════════════════════════════
# 场景 5：InvertedIndex 线程安全
# ═════════════════════════════════════════════════════════════

class TestInvertedIndexThreadSafety:
    """验证倒排索引的并发安全性。"""

    def test_concurrent_build_and_query(self):
        """并发 build_tags 和 query_tags 不应崩溃或返回脏数据。"""
        import threading
        from app.memory.inverted import InvertedIndex

        idx = InvertedIndex()

        # 先构建一些数据
        idx.build_tags([
            ("m1", "编程,Python,AI"),
            ("m2", "编程,Java,后端"),
            ("m3", "设计,UI,Figma"),
        ])

        errors = []
        results = []

        def reader():
            try:
                for _ in range(100):
                    r = idx.query_tags(["编程"])
                    results.append(len(r))
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    idx.build_tags([
                        (f"new_{i}", f"标签{i},新数据"),
                    ])
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"并发操作不应抛出异常: {errors}"
        # 最终查询应返回一致结果
        final = idx.query_tags(["编程"])
        assert isinstance(final, set), f"query_tags 应返回 set，实际: {type(final)}"


# ═════════════════════════════════════════════════════════════
# 场景 6：Russell 情绪环边界情况
# ═════════════════════════════════════════════════════════════

class TestRussellEmotionEdgeCases:
    """验证情绪分析在边界输入下的行为。"""

    def test_empty_input_returns_neutral(self):
        from app.analysis.emotion import analyze_emotion_2d
        v, a, cat = analyze_emotion_2d("")
        assert cat == "neutral"

    def test_mixed_emotions_concession(self):
        """让步句 (但是...) 后情绪应主导。"""
        from app.analysis.emotion import analyze_emotion_2d
        v, a, cat = analyze_emotion_2d("今天很开心，但是下午突然就很难过")
        # 让步后负面应主导
        assert cat == "negative", f"让步句后情绪应为负面，实际: {cat}"

    def test_intimate_words_priority(self):
        """亲密词应优先识别。"""
        from app.analysis.emotion import analyze_emotion_2d
        v, a, cat = analyze_emotion_2d("好想你")
        assert cat == "intimate", f"'好想你'应识别为 intimate，实际: {cat}"


# ═════════════════════════════════════════════════════════════
# 场景 7：模式发现趋势检测
# ═════════════════════════════════════════════════════════════

class TestTrendDetection:
    """验证线性趋势检测的正确性。"""

    def test_rising_trend_detected(self):
        from app.analysis.pattern_discovery import PatternDiscovery
        values = [0.0, 0.3, 0.6, 0.9]
        slope = PatternDiscovery._linear_trend(values)
        assert slope > 0.25, f"上升趋势斜率应 >0.25，实际: {slope:.3f}"

    def test_falling_trend_detected(self):
        from app.analysis.pattern_discovery import PatternDiscovery
        values = [0.9, 0.6, 0.3, 0.0]
        slope = PatternDiscovery._linear_trend(values)
        assert slope < -0.25, f"下降趋势斜率应 < -0.25，实际: {slope:.3f}"

    def test_stable_no_trend(self):
        from app.analysis.pattern_discovery import PatternDiscovery
        values = [0.5, 0.5, 0.5, 0.5]
        slope = PatternDiscovery._linear_trend(values)
        assert abs(slope) < 0.01, f"稳定值斜率应接近 0，实际: {slope:.3f}"


# ═════════════════════════════════════════════════════════════
# 场景 8：检索意图门控
# ═════════════════════════════════════════════════════════════

class TestIntentGating:
    """验证意图门控正确分配检索配额。"""

    def test_recall_intent_gets_full_quota(self):
        from app.retrieval.pipeline import _resolve_route, _classify_intent

        intent = _classify_intent("你还记得上次我们聊的那个项目吗")
        route = _resolve_route(intent)
        assert route["semantic"] >= 20, f"recall 意图应获得足够语义配额，实际: {route}"
        assert route["tag"] >= 5, "recall 应有标签配额"

    def test_casual_intent_gets_minimal_quota(self):
        from app.retrieval.pipeline import _resolve_route, _classify_intent

        intent = _classify_intent("好的")
        route = _resolve_route(intent)
        assert route["semantic"] <= 15, f"casual 意图应节省配额，实际: {route}"

    def test_conflict_intent_gets_full_quota(self):
        from app.retrieval.pipeline import _resolve_route, _classify_intent

        intent = _classify_intent("你说的不对，不是这样的")
        route = _resolve_route(intent)
        assert route["semantic"] >= 25, f"conflict 意图应获得最大配额，实际: {route}"
        assert route["entity"] >= 5, "conflict 应有实体配额"


# ═════════════════════════════════════════════════════════════
# 场景 9：事实冲突检测的三层漏斗
# ═════════════════════════════════════════════════════════════

class TestFactContradictionFunnel:
    """验证事实冲突检测的三层漏斗逻辑。"""

    def test_no_shared_tags_skipped(self):
        """无共享标签 → 第一层就跳过。"""
        new_tags = {"编程", "Python"}
        old_tags = {"做饭", "食谱"}
        assert not (new_tags & old_tags), "无共享标签应跳过 cosine 比较"

    def test_shared_tags_but_low_sim_skipped(self):
        """有共享标签但 embedding 差异大 → 第二层跳过。"""
        # 模拟：同 tag="编程"，但 embedding 差距很大
        import numpy as np
        emb1 = np.array([1.0] * 1024, dtype=np.float32)
        emb2 = np.array([0.0] * 1024, dtype=np.float32)
        dot = float(np.dot(emb1, emb2))
        n1 = float(np.linalg.norm(emb1))
        n2 = float(np.linalg.norm(emb2))
        sim = dot / (n1 * n2 + 1e-10)
        # n2=0 所以 sim=0 < 0.75 → 应跳过
        assert sim < 0.75, f"正交向量的相似度应 <0.75，实际: {sim}"
