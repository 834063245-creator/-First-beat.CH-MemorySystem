"""回路调度 + 行为预测器 测试。"""
import sys
sys.path.insert(0, ".")

import pytest
from unittest.mock import MagicMock, patch


class TestAnalyzeUserMessage:
    """回路①：用户消息分析测试。"""
    pytestmark = pytest.mark.real_embed  # 需要真实 Ollama embedding 做语义分类

    def test_emotional_sharing_intimate(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("我昨晚梦到你了")
        assert r.intent in ("emotional_sharing", "recall", "casual")
        assert r.emotion in ("intimate", "positive", "neutral")
        assert r.raw_text == "我昨晚梦到你了"

    def test_recall(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("还记得我们第一次聊了什么吗")
        assert r.intent == "recall"

    def test_casual(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("你好")
        assert r.intent == "casual"

    def test_ask_fact(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("今天的天气怎么样")
        assert r.intent in ("ask_fact", "casual")  # Ollama不可用时降级为casual

    def test_conflict(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("不对，你搞错了")
        assert r.intent == "conflict"

    def test_emotional_sharing_negative(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("今天心情很差，感觉很累")
        assert r.intent == "emotional_sharing"
        assert r.emotion == "negative"

    def test_empty_message(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("")
        assert r.intent == "casual"
        assert r.emotion == "neutral"

    def test_urgency_high(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("急! 快帮我看看这个bug!")
        assert r.urgency > 0.5

    def test_topics_extracted(self):
        from app.core.circuit import analyze_user_message
        r = analyze_user_message("我昨天写了个Python爬虫")
        assert len(r.topics) >= 1

    # ── M4 验证 user_mood / affective_context 映射 ──

    def test_user_mood_positive(self):
        from app.core.circuit import analyze_user_message
        pfc = analyze_user_message("好开心啊今天")
        mood_map = {"positive": "positive", "negative": "negative",
                    "frustrated": "negative", "intimate": "positive"}
        assert mood_map.get(pfc.emotion, "neutral") == "positive"

    def test_user_mood_negative(self):
        from app.core.circuit import analyze_user_message
        pfc = analyze_user_message("今天好难过，压力太大了")
        mood_map = {"positive": "positive", "negative": "negative",
                    "frustrated": "negative", "intimate": "positive"}
        assert mood_map.get(pfc.emotion, "neutral") == "negative"

    def test_affective_context_intimate(self):
        from app.core.circuit import analyze_user_message
        pfc = analyze_user_message("好想你啊，昨晚梦到你了")
        ctx_map = {"conflict": "conflict", "emotional_sharing": "casual_chat",
                   "recall": "casual_chat", "ask_fact": "focused_work",
                   "request": "focused_work", "meta": "casual_chat"}
        ctx = ctx_map.get(pfc.intent, "casual_chat")
        if pfc.intent == "emotional_sharing" and pfc.emotion in ("negative", "intimate", "frustrated"):
            ctx = "intimate"
        # Ollama可用时ctx=intimate，不可用时降级为casual_chat
        assert ctx in ("intimate", "casual_chat")

    def test_affective_context_focused_work(self):
        from app.core.circuit import analyze_user_message
        pfc = analyze_user_message("帮我查一下这个bug怎么修")
        ctx_map = {"conflict": "conflict", "emotional_sharing": "casual_chat",
                   "recall": "casual_chat", "ask_fact": "focused_work",
                   "request": "focused_work", "meta": "casual_chat"}
        ctx = ctx_map.get(pfc.intent, "casual_chat")
        assert ctx == "focused_work"


class TestBasalGangliaGate:
    """回路④：响应门控测试。"""
    pytestmark = pytest.mark.real_embed

    def test_intimate_emotional_sharing(self):
        from app.core.circuit import analyze_user_message, basal_ganglia_gate
        pfc = analyze_user_message("我昨晚梦到你了")
        gate = basal_ganglia_gate(pfc, [], [], [])
        # Ollama可用时tone=caring/soothe，不可用时降级为warm/auto
        assert gate.tone in ("caring", "warm")
        assert gate.response_mode in ("soothe", "auto", "question_first")

    def test_conflict(self):
        from app.core.circuit import analyze_user_message, basal_ganglia_gate
        pfc = analyze_user_message("不对，你搞错了")
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.tone in ("soft", "warm")
        assert gate.response_mode in ("confirm", "auto")

    def test_ask_fact(self):
        from app.core.circuit import analyze_user_message, basal_ganglia_gate
        pfc = analyze_user_message("Python怎么读文件")
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.tone in ("direct", "warm")
        assert gate.response_mode in ("direct_answer", "auto")

    def test_casual(self):
        from app.core.circuit import analyze_user_message, basal_ganglia_gate
        pfc = analyze_user_message("你好")
        gate = basal_ganglia_gate(pfc, [], [], [])
        assert gate.tone == "warm"

    def test_intimate_suppresses_work_impulses(self):
        from app.core.circuit import analyze_user_message, basal_ganglia_gate
        from app.core.state import ImpulseDirective
        pfc = analyze_user_message("好想你啊，你今天在干嘛呢")
        work_impulse = ImpulseDirective(
            intent="share_observation",
            target_concept="用户上次熬夜修bug",
            emotional_tone="neutral",
        )
        gate = basal_ganglia_gate(pfc, [], [work_impulse], [])
        # Ollama不可用时可能不会触发压制
        assert len(gate.impulses_to_show) <= 1
        if pfc.intent == "emotional_sharing":
            assert len(gate.suppression_reasons) >= 1


class TestBehaviorPredictor:
    """行为预测器测试。"""

    def test_learn_and_predict(self, tmp_path):
        from app.analysis.predictor import BehaviorPredictor
        data_dir = str(tmp_path)
        mn = BehaviorPredictor(data_dir)

        # 学习
        records = [
            {"user_message": "好想你啊", "llm_reply": "我也想你"},
            {"user_message": "上次那个bug后来怎么样了", "llm_reply": "查了一下已经修复了"},
            {"user_message": "今天天气真好", "llm_reply": "是啊，适合出去走走"},
            {"user_message": "周末要不要一起", "llm_reply": "好啊"},
        ]
        mn.learn_from(records)
        assert mn._table["total_sequences"] == 4

        # 预测
        pred = mn.predict("emotional_sharing", ["想念", "心情"])
        assert isinstance(pred, dict)

    def test_cold_start_returns_empty(self, tmp_path):
        from app.analysis.predictor import BehaviorPredictor
        mn = BehaviorPredictor(str(tmp_path))
        pred = mn.predict("casual", [])
        assert pred == {}

    def test_persist_across_instances(self, tmp_path):
        from app.analysis.predictor import BehaviorPredictor
        data_dir = str(tmp_path)

        mn1 = BehaviorPredictor(data_dir)
        records = [
            {"user_message": "A", "llm_reply": "a"},
            {"user_message": "B", "llm_reply": "b"},
            {"user_message": "C", "llm_reply": "c"},
            {"user_message": "D", "llm_reply": "d"},
        ]
        mn1.learn_from(records)

        mn2 = BehaviorPredictor(data_dir)
        assert mn2._table["total_sequences"] == 4


class TestUtteranceSpec:
    """UtteranceSpec 数据结构测试。"""

    def test_mirror_prediction_field(self):
        from app.core.state import UtteranceSpec
        spec = UtteranceSpec(mirror_prediction={"next_intent": "recall"})
        assert spec.mirror_prediction["next_intent"] == "recall"

    def test_gate_impulses_to_show(self):
        from app.core.state import GatingDecision, ImpulseDirective
        imp = ImpulseDirective(intent="recall", target_concept="test")
        gate = GatingDecision(impulses_to_show=[imp])
        assert len(gate.impulses_to_show) == 1

    def test_gate_memories_to_show(self):
        from app.core.state import GatingDecision
        gate = GatingDecision(memories_to_show=[{"id": "1"}])
        assert len(gate.memories_to_show) == 1

    # ── M3 验证 reference_memories ──

    def test_reference_memories_field(self):
        from app.core.state import UtteranceSpec, MemoryDirective
        ref = MemoryDirective(memory_id="ref1", summary="test reference", role="reference")
        spec = UtteranceSpec(reference_memories=[ref])
        assert len(spec.reference_memories) == 1
        assert spec.reference_memories[0].memory_id == "ref1"
        assert spec.reference_memories[0].role == "reference"


# ═══════════════════════════════════════════════════════════════
# weave_context 引擎编织
# ═══════════════════════════════════════════════════════════════

class TestWeaveContext:
    def test_empty_candidates(self):
        from app.core.circuit import CircuitOrchestrator
        orch = CircuitOrchestrator(
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), MagicMock(), MagicMock(),
        )
        result = orch.weave_context([], MagicMock())
        assert result.total_candidates == 0
        assert result.should_speak is False

    @patch("app.config.settings.BENCHMARK_MODE", True)
    def test_benchmark_mode_all_candidates_fact(self):
        from app.core.circuit import CircuitOrchestrator
        orch = CircuitOrchestrator(
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), MagicMock(), MagicMock(),
        )
        candidates = [
            {"id": "1", "document": "test doc", "metadata": {"summary": "summary1", "tags": "test"}},
            {"id": "2", "document": "test doc 2", "metadata": {"summary": "summary2", "tags": "test"}},
        ]
        result = orch.weave_context(candidates, MagicMock())
        assert result.should_speak is True
        assert len(result.fact_memories) == 2

    @patch("app.config.settings.BENCHMARK_MODE", False)
    def test_with_candidates_and_tags(self):
        from app.core.circuit import CircuitOrchestrator
        orch = CircuitOrchestrator(
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), MagicMock(), MagicMock(),
        )
        cog = MagicMock()
        cog.intent = "casual"
        candidates = [
            {"id": "1", "document": "doc1", "metadata": {"summary": "s1", "tags": "Python,编程", "timestamp": 1000000}},
        ]
        result = orch.weave_context(candidates, cog)
        assert result.total_candidates == 1


# ═══════════════════════════════════════════════════════════════
# CircuitOrchestrator.process
# ═══════════════════════════════════════════════════════════════

class TestCircuitOrchestratorProcess:
    @patch("app.config.settings.BENCHMARK_MODE", True)
    def test_process_with_empty_memories(self):
        from app.core.circuit import CircuitOrchestrator
        from unittest.mock import MagicMock
        orch = CircuitOrchestrator(
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), MagicMock(), MagicMock(),
        )
        result = orch.process(
            "你好", [0.1] * 1024, MagicMock(),
            timeline_recent=[], session_context="",
            personalities=[], memories=[],
        )
        assert result is not None
        assert hasattr(result, 'user')

    @patch("app.config.settings.BENCHMARK_MODE", True)
    def test_process_with_memories(self):
        from app.core.circuit import CircuitOrchestrator
        from unittest.mock import MagicMock
        orch = CircuitOrchestrator(
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), MagicMock(), MagicMock(),
        )
        memories = [
            {"id": "1", "document": "关于Python的讨论", "metadata": {"tags": "Python", "summary": "Python学习"}}
        ]
        result = orch.process(
            "Python学习", [0.1] * 1024, MagicMock(),
            timeline_recent=[], session_context="",
            personalities=[], memories=memories,
        )
        assert result is not None

    @patch("app.config.settings.BENCHMARK_MODE", True)
    def test_process_with_timeline(self):
        from app.core.circuit import CircuitOrchestrator
        from unittest.mock import MagicMock
        orch = CircuitOrchestrator(
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), MagicMock(), MagicMock(),
        )
        timeline = [
            {"user_message": "你好", "llm_reply": "你好！", "timestamp": "2026-06-01"}
        ]
        result = orch.process(
            "继续", [0.1] * 1024, MagicMock(),
            timeline_recent=timeline, session_context="之前聊过天",
            personalities=[], memories=[],
        )
        assert result is not None

