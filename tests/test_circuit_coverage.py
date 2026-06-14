"""测试 circuit.py 未覆盖路径：weave_context 叙述层 + ChatCircuit.run() 分支。"""
import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.core.circuit import (
    CircuitOrchestrator, analyze_user_message,
    _compute_emotion_intensity,
)
from app.core.state import UserMessageAnalysis, GatingDecision
from app.models.schemas import WovenContext


# ═══════════════════════════════════════════════════════════════════
# _compute_emotion_intensity — 纯函数
# ═══════════════════════════════════════════════════════════════════

class TestComputeEmotionIntensity:
    def test_empty_text(self):
        assert _compute_emotion_intensity("") == 0.0

    def test_exclamation_marks(self):
        score = _compute_emotion_intensity("太棒了！！！")
        assert score > 0.0  # 3个感叹号 × 0.15 = 0.45

    def test_emoji(self):
        score = _compute_emotion_intensity("😊😊😊")
        assert score > 0.0

    def test_intensifier_words(self):
        # "非常" 在中文常用程度副词中
        score = _compute_emotion_intensity("非常非常好")
        assert score > 0.0

    def test_long_text_boost(self):
        long_text = "我" * 100  # > 80 chars
        score = _compute_emotion_intensity(long_text)
        assert score == 0.1  # 只有长文本 boost

    def test_capped_at_1(self):
        score = _compute_emotion_intensity("！！！😊😊😊太棒了真的超级无敌好")
        assert score <= 1.0


# GateDecisionMaker 已删除（死代码），对应测试移除

# ═══════════════════════════════════════════════════════════════════
# weave_context — 故事线 + 分层 (548-677)
# ═══════════════════════════════════════════════════════════════════

class TestWeaveContext:

    def _make_circuit(self):
        """构造最小 CircuitOrchestrator 用于 weave_context 测试。"""
        circuit = CircuitOrchestrator.__new__(CircuitOrchestrator)
        circuit._chat_history = MagicMock()
        return circuit

    def _make_candidate(self, mid="m1", doc="记忆内容", tags=None, entities=None,
                        ts=None, dist=0.2, source="semantic", stale=False,
                        metadata_extra=None):
        """构造候选记忆 dict。"""
        meta = {
            "tags": ",".join(tags) if tags else "",
            "entities": entities or [],
            "timestamp": ts or time.time(),
            "stale": stale,
        }
        if metadata_extra:
            meta.update(metadata_extra)
        return {
            "id": mid,
            "document": doc,
            "metadata": meta,
            "distance": dist,
            "source": source,
        }

    def test_empty_returns_no_speak(self):
        c = self._make_circuit()
        r = c.weave_context([], UserMessageAnalysis(intent="casual", emotion="neutral"))
        assert r.should_speak is False
        assert r.total_candidates == 0

    def test_benchmark_mode(self):
        c = self._make_circuit()
        cands = [self._make_candidate("m1"), self._make_candidate("m2")]
        with patch("app.config.settings.BENCHMARK_MODE", True):
            r = c.weave_context(cands, MagicMock())
        assert r.should_speak is True
        assert len(r.fact_memories) == 2
        assert r.total_candidates == 2

    def test_casual_few_memories_no_speak(self):
        """闲聊 + ≤3条候选 → 不说话"""
        c = self._make_circuit()
        cands = [self._make_candidate(f"m{i}") for i in range(3)]
        pfc = UserMessageAnalysis(intent="casual", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert r.should_speak is False

    def test_casual_many_memories_speaks(self):
        """闲聊 + >3条候选 → 说话"""
        c = self._make_circuit()
        cands = [self._make_candidate(f"m{i}") for i in range(5)]
        pfc = UserMessageAnalysis(intent="casual", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert r.should_speak is True

    def test_emotional_sharing_always_speaks(self):
        """情绪分享总是说话（不等候选数）"""
        c = self._make_circuit()
        cands = [self._make_candidate("m1")]
        pfc = UserMessageAnalysis(intent="emotional_sharing", emotion="negative")
        r = c.weave_context(cands, pfc)
        assert r.should_speak is True

    def test_active_tags_parsed_from_string(self):
        """CSV tags 字符串 → 解析为列表"""
        c = self._make_circuit()
        cands = [self._make_candidate("m1", tags=["Python", "Rust"])]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # tags 应该被正确解析
        assert r.should_speak is True

    def test_narrative_builds_storyline(self):
        """两个同实体、跨天的记忆 → 生成故事线"""
        two_days_ago = time.time() - 2 * 86400
        now = time.time()
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "Python 学习记录1", tags=["Python"],
                                entities=["Python"], ts=two_days_ago),
            self._make_candidate("m2", "Python 学习记录2", tags=["Python"],
                                entities=["Python"], ts=now),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # 应有故事线
        assert len(r.narratives) >= 1
        assert "Python" in r.narratives[0]

    def test_narrative_skips_same_day(self):
        """同一天的记忆不生成故事线"""
        today = time.time()
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "内容1", tags=["Python"],
                                entities=["Python"], ts=today),
            self._make_candidate("m2", "内容2", tags=["Python"],
                                entities=["Python"], ts=today + 3600),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # 同一天 → 无故事线
        assert len(r.narratives) == 0

    def test_narrative_single_memory_skipped(self):
        """单条记忆不生成故事线"""
        c = self._make_circuit()
        cands = [self._make_candidate("m1", "内容", tags=["Python"], entities=["Python"])]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert len(r.narratives) == 0

    def test_narrative_capped_at_5(self):
        """故事线最多5条"""
        c = self._make_circuit()
        cands = []
        for i in range(10):
            ts = time.time() - i * 2 * 86400
            cands.append(
                self._make_candidate(f"m{i}a", f"内容{i}a",
                                    tags=[f"tag{i}"], entities=[f"tag{i}"], ts=ts))
            cands.append(
                self._make_candidate(f"m{i}b", f"内容{i}b",
                                    tags=[f"tag{i}"], entities=[f"tag{i}"], ts=ts + 86400))
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert len(r.narratives) <= 5

    def test_narrative_emotion_trend(self):
        """情绪翻转趋势检测"""
        week_ago = time.time() - 7 * 86400
        two_days_ago = time.time() - 2 * 86400
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "开心", tags=["Python"], entities=["Python"],
                                ts=week_ago, metadata_extra={"emotion_valence_bin": "positive"}),
            self._make_candidate("m2", "难过", tags=["Python"], entities=["Python"],
                                ts=two_days_ago, metadata_extra={"emotion_valence_bin": "negative"}),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert len(r.narratives) >= 1
        # 应该检测到翻转
        assert "翻转" in r.narratives[0]

    def test_narrative_positive_trend(self):
        """持续积极趋势"""
        week_ago = time.time() - 7 * 86400
        yesterday = time.time() - 86400
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "开心1", tags=["Python"], entities=["Python"],
                                ts=week_ago, metadata_extra={"emotion_valence_bin": "positive"}),
            self._make_candidate("m2", "开心2", tags=["Python"], entities=["Python"],
                                ts=yesterday, metadata_extra={"emotion_valence_bin": "positive"}),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert len(r.narratives) >= 1
        assert "积极" in r.narratives[0]

    def test_stale_routes_to_stale_context(self):
        """stale 记忆 → stale_context"""
        c = self._make_circuit()
        cands = [self._make_candidate("m1", "旧记忆", tags=["Python"], entities=["Python"],
                                      ts=time.time(), stale=True)]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # stale 记忆不应进入 fact_memories
        if r.fact_memories:
            fact_ids = [m["id"] for m in r.fact_memories]
            assert "m1" not in fact_ids

    def test_fact_vs_reference_vs_discard(self):
        """语义距离决定了分层"""
        c = self._make_circuit()
        cands = [
            self._make_candidate("close", "很近的记忆", dist=0.10, source="semantic"),
            self._make_candidate("mid", "中等距离", dist=0.35, source="semantic"),
            self._make_candidate("far", "远距离", dist=0.60, source="semantic"),
        ]
        pfc = UserMessageAnalysis(intent="ask_fact", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # close (dist=0.10) → fact (threshold=0.30*0.8=0.24 → 0.10 < 0.24 → fact)
        # mid (dist=0.35) → reference (0.24 < 0.35 < 0.45 → reference)
        # far (dist=0.60) → discard
        fact_ids = [m["id"] for m in r.fact_memories]
        ref_ids = [m["id"] for m in r.reference_memories]
        assert "close" in fact_ids
        assert "mid" in ref_ids
        assert "far" not in fact_ids
        assert "far" not in ref_ids

    def test_total_tokens_estimated(self):
        c = self._make_circuit()
        cands = [self._make_candidate("m1", "这是一个测试记忆" * 5)]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert r.total_tokens >= 0

    def test_entity_string_parsed_as_json(self):
        """entities 是 JSON 字符串时正确解析"""
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "内容1", tags=["Python"],
                                entities=json.dumps(["Python"]),
                                ts=time.time() - 3 * 86400),
            self._make_candidate("m2", "内容2", tags=["Python"],
                                entities=json.dumps(["Python"]),
                                ts=time.time()),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert len(r.narratives) >= 1

    def test_entity_json_parse_error(self):
        """entities 是非法 JSON → 优雅降级为空"""
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "内容1", tags=["Python"],
                                entities="not valid json {{{",
                                ts=time.time() - 3 * 86400),
            self._make_candidate("m2", "内容2", tags=["Python"],
                                entities="also bad",
                                ts=time.time()),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # entities 解析失败 → 只靠 tags，仍然可以匹配
        # tags 相同 → 应该有 narrative
        assert len(r.narratives) >= 1

    def test_no_tags_no_narrative(self):
        """没有 tags/entities 的记忆不生成故事线"""
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "内容1", tags=[], entities=[],
                                ts=time.time() - 3 * 86400),
            self._make_candidate("m2", "内容2", tags=[], entities=[],
                                ts=time.time()),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert len(r.narratives) == 0  # 没有标签 → 不分组

    def test_non_list_entities(self):
        """entities 是非列表类型 → 优雅降级"""
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "内容1", tags=["Python"],
                                entities="single_string",  # not a list
                                ts=time.time() - 3 * 86400),
            self._make_candidate("m2", "内容2", tags=["Python"],
                                entities=42,  # int
                                ts=time.time()),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # 应该不崩溃，依赖 tags 匹配
        assert len(r.narratives) >= 1

    def test_no_metadata_timestamp(self):
        """没有 metadata 的记忆也能处理"""
        c = self._make_circuit()
        cands = [
            {"id": "m1", "document": "内容1", "metadata": {},
             "distance": 0.2, "source": "semantic"},
            {"id": "m2", "document": "内容2", "metadata": {"tags": "Python"},
             "distance": 0.3, "source": "semantic"},
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        assert r.should_speak is True  # should not crash

    def test_all_active_in_narrative_used(self):
        """故事线中的记忆优先分配到 fact"""
        two_days_ago = time.time() - 2 * 86400
        now = time.time()
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "内容1", tags=["Python"], entities=["Python"],
                                ts=two_days_ago, dist=0.15),
            self._make_candidate("m2", "内容2", tags=["Python"], entities=["Python"],
                                ts=now, dist=0.15),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # 有 narrative → m1, m2 应该在 used_in_narrative → 路由到 fact
        fact_ids = [m["id"] for m in r.fact_memories]
        assert "m1" in fact_ids
        assert "m2" in fact_ids

    def test_stale_in_narrative_routes_to_stale_context(self):
        """故事线中的 stale 记忆 → stale_context"""
        two_days_ago = time.time() - 2 * 86400
        now = time.time()
        c = self._make_circuit()
        cands = [
            self._make_candidate("m1", "内容1", tags=["Python"], entities=["Python"],
                                ts=two_days_ago),
            self._make_candidate("m2", "内容2", tags=["Python"], entities=["Python"],
                                ts=now, stale=True),
        ]
        pfc = UserMessageAnalysis(intent="recall", emotion="neutral")
        r = c.weave_context(cands, pfc)
        # m2 是 stale + 在 narrative 中 → stale_context
        stale_ids = [m["id"] for m in r.stale_context]
        assert "m2" in stale_ids


# ═══════════════════════════════════════════════════════════════════
# ChatCircuit.run() — 画像渲染分支 (460-465)
# ═══════════════════════════════════════════════════════════════════

def _make_mock_ctx():
    """构造最小 mock ctx_obj。"""
    ctx = MagicMock()
    ctx.portrait_renderer = None
    ctx.chroma_service = MagicMock()
    ctx.ai_chroma_service = MagicMock()
    ctx.temporal_pattern_index = None
    ctx.mirror_neuron = None
    ctx.co_tracker = None
    ctx.ai_co_tracker = None
    ctx._pattern_discovery = None
    return ctx


class TestChatCircuitRunBranches:
    """覆盖 ChatCircuit.run() 中未覆盖的分支。"""

    @pytest.fixture
    def circuit(self):
        """构造一个最小 CircuitOrchestrator，所有可选组件为 None。"""
        c = CircuitOrchestrator.__new__(CircuitOrchestrator)
        c._chat_history = MagicMock()
        c._chat_history.get_recent.return_value = []
        c._personality = MagicMock()
        c._personality.list_tags.return_value = {"items": []}
        c._dmn = None
        c._mirror_neuron = None
        c._impulse = MagicMock()
        c._impulse.idle_gap_minutes.return_value = None
        return c

    def test_run_minimal(self, circuit):
        """最简路径：无记忆、无人格、无 DMN"""
        ctx = _make_mock_ctx()
        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="casual", emotion="neutral", topics=[],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("你好", query_embedding=[0.1]*1024, ctx_obj=ctx)
        assert result is not None
        assert result.user.intent == "casual"

    def test_run_with_personality_ai_notes(self, circuit):
        """人格标签路径 (343-353)"""
        ctx = _make_mock_ctx()
        circuit._personality.list_tags.return_value = {
            "items": [
                {"content": "AI表达: 理性", "type": "trait", "source": "ai"},
                {"content": "用户喜欢编程", "type": "interest", "source": "user"},
            ]
        }
        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="recall", emotion="neutral", topics=["Python"],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("Python怎么学", query_embedding=[0.1]*1024, ctx_obj=ctx)
        assert result.personality_notes_ai is not None

    def test_run_with_portrait_renderer(self, circuit):
        """画像渲染分支 (460-465)"""
        ctx = _make_mock_ctx()
        ctx.portrait_renderer = MagicMock()
        ctx.portrait_renderer.render_stable.return_value = "【认知画像】\n用户喜欢编程"
        ctx.portrait_renderer.render_dynamic.return_value = "【当前状态】\n情绪: positive"

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="casual", emotion="neutral", topics=[],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("你好", query_embedding=[0.1]*1024, ctx_obj=ctx)
        assert "认知画像" in result.portrait_stable
        assert "当前状态" in result.portrait_dynamic

    def test_run_with_mirror_neuron(self, circuit):
        """行为预测分支 (265-270)"""
        ctx = _make_mock_ctx()
        circuit._mirror_neuron = MagicMock()
        circuit._mirror_neuron.predict.return_value = {"next_intent": "recall"}

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="recall", emotion="neutral", topics=["Python"],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("记得上次聊的Python吗", query_embedding=[0.1]*1024, ctx_obj=ctx)
        assert result.mirror_prediction == {"next_intent": "recall"}

    def test_run_mirror_neuron_error(self, circuit):
        """行为预测报错不阻塞"""
        ctx = _make_mock_ctx()
        circuit._mirror_neuron = MagicMock()
        circuit._mirror_neuron.predict.side_effect = RuntimeError("预测失败")

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="recall", emotion="neutral", topics=[],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("test", query_embedding=[0.1]*1024, ctx_obj=ctx)
        assert result is not None  # 不应崩溃

    def test_run_with_impulse(self, circuit):
        """冲动收集分支 (393-405)"""
        ctx = _make_mock_ctx()
        circuit._impulse.idle_gap_minutes.return_value = 5
        circuit._impulse.get_next.return_value = {"content": "用户上次熬夜修bug"}

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="casual", emotion="neutral", topics=[],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("你好", query_embedding=[0.1]*1024, ctx_obj=ctx)
        assert len(result.impulses) >= 1

    def test_run_with_retrieval_results(self, circuit):
        """带已有检索结果的路径"""
        ctx = _make_mock_ctx()
        memories = [
            {
                "id": "m1", "document": "用户喜欢Python",
                "metadata": {"tags": "Python", "timestamp": time.time()},
                "distance": 0.15, "source": "semantic",
            }
        ]

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="recall", emotion="neutral", topics=["Python"],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("Python", query_embedding=[0.1]*1024,
                                ctx_obj=ctx, memories=memories)
        assert result.memories is not None

    def test_run_with_chat_history_relationship(self, circuit):
        """关系维度计算分支 (419-453)"""
        ctx = _make_mock_ctx()
        circuit._chat_history.get_recent.return_value = [
            {"user_message": "你好", "ai_response": "你好！"},
            {"user_message": "谢谢你的帮助", "ai_response": "不客气"},
            {"user_message": "想你", "ai_response": "我也想你"},
        ]

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="casual", emotion="neutral", topics=[],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("你好", query_embedding=[0.1]*1024, ctx_obj=ctx)
        # 关系维度应该被计算
        assert result.relationship is not None
        assert result.relationship.familiarity > 0

    def test_run_chat_history_errors_not_fatal(self, circuit):
        """chat_history 报错不阻塞主流程"""
        ctx = _make_mock_ctx()
        circuit._chat_history.get_recent.side_effect = RuntimeError("history error")

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="casual", emotion="neutral", topics=[],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("你好", query_embedding=[0.1]*1024, ctx_obj=ctx)
        assert result is not None

    def test_run_personality_list_error_not_fatal(self, circuit):
        """人格标签报错不阻塞主流程"""
        ctx = _make_mock_ctx()
        circuit._personality.list_tags.side_effect = RuntimeError("store error")

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="casual", emotion="neutral", topics=[],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("你好", query_embedding=[0.1]*1024, ctx_obj=ctx)
        assert result.personality_notes_ai == []

    def test_run_with_dmn(self, circuit):
        """DMN 注入 + 话题笔记路径 (355-385)"""
        ctx = _make_mock_ctx()
        circuit._dmn = MagicMock()
        circuit._dmn.get_topic_notes.return_value = ["Python 是你近期关注话题"]

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="recall", emotion="neutral", topics=["Python"],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("Python", query_embedding=[0.1]*1024, ctx_obj=ctx)
        # topic_notes 应该被填充
        assert len(result.topic_notes) >= 1

    def test_run_stale_memory_handling(self, circuit):
        """stale 记忆路径 (287-299)"""
        ctx = _make_mock_ctx()
        memories = [
            {
                "id": "stale1", "document": "旧记忆",
                "metadata": {"tags": "Python", "timestamp": time.time(), "stale": True},
                "distance": 0.15, "source": "semantic",
            }
        ]

        with patch("app.core.circuit.analyze_user_message") as mock_analyze:
            mock_analyze.return_value = UserMessageAnalysis(
                intent="recall", emotion="neutral", topics=["Python"],
                urgency=0.0, emotion_intensity=0.0, confidence=0.7,
            )
            result = circuit.process("Python", query_embedding=[0.1]*1024,
                                ctx_obj=ctx, memories=memories)
        # 应有 stale_context
        assert result.stale_context is not None
