"""单轮对话集成测试 — CircuitOrchestrator.process()。

通过 mock 外部依赖（ChromaDB、DeepSeek、embedding），验证整条管线不崩溃。
"""
import pytest

from app.core.circuit import CircuitOrchestrator, analyze_user_message, basal_ganglia_gate
from app.core.state import UserMessageAnalysis, UtteranceSpec, GatingDecision, ImpulseDirective


class TestCircuitOrchestratorProcess:
    """验证 CircuitOrchestrator.process() 在 mock 环境下正确工作。"""

    @pytest.fixture
    def mock_services(self, monkeypatch):
        """创建一组 mock 服务，避免真实 ChromaDB/DeepSeek/Ollama 依赖。"""

        class MockChroma:
            """mock ChromaService，只实现 process() 用到的接口。"""
            def list_all(self):
                return []

        class MockPersonality:
            def list_tags(self, page=1, page_size=5):
                return {"items": [
                    {"content": "用户喜欢技术讨论", "source": "user", "type": "interest"},
                    {"content": "AI回复偏理性分析", "source": "ai", "type": "style"},
                ]}

        class MockImpulse:
            def __init__(self):
                self._queue = []
            def feed_impulse(self, *a, **kw):
                pass
            def get_next(self):
                return None
            def idle_seconds(self, _):
                return None

        class MockDMN:
            def apply_to_cognitive_state(self, state):
                pass

        class MockChatHistory:
            def __init__(self):
                self.records = []
            def get_recent(self, n=5):
                return []

        class MockCoTracker:
            pass

        return {
            "chroma": MockChroma(),
            "personality": MockPersonality(),
            "impulse": MockImpulse(),
            "dmn": MockDMN(),
            "chat_history": MockChatHistory(),
            "co_tracker": MockCoTracker(),
        }

    def test_process_with_prefilled_memories(self, mock_services):
        """传入预制记忆，验证记忆分层正确、门控输出合理。"""
        orchestrator = CircuitOrchestrator(
            chroma_service=mock_services["chroma"],
            personality_store=mock_services["personality"],
            impulse_scheduler=mock_services["impulse"],
            dmn_engine=mock_services["dmn"],
            chat_history=mock_services["chat_history"],
            co_tracker=mock_services["co_tracker"],
        )

        # 预制记忆（模拟检索管线返回的数据）
        prefilled_memories = [
            {
                "id": "mem_fact_1",
                "summary": "用户上次聊到在做一个AI项目",
                "metadata": {"hit_count": 5, "stale": False, "timestamp": 1717000000},
                "source": "semantic",
                "distance": 0.2,
            },
            {
                "id": "mem_ref_1",
                "summary": "用户喜欢喝咖啡",
                "metadata": {"hit_count": 2, "stale": False, "timestamp": 1716900000},
                "source": "kw_match",
                "distance": 0.5,
            },
            {
                "id": "mem_bg_1",
                "summary": "模糊的时间片段",
                "metadata": {"hit_count": 0, "stale": False, "timestamp": 1716800000},
                "source": "time_rhythm",
                "distance": 0.8,
            },
            {
                "id": "mem_stale",
                "summary": "过时记忆",
                "metadata": {"hit_count": 1, "stale": True, "timestamp": 1716000000},
                "source": "semantic",
                "distance": 0.3,
            },
        ]

        # 传入 query_embedding=None（用纯规则路径）
        spec = orchestrator.process(
            user_message="最近压力好大",
            query_embedding=None,
            ctx_obj=None,
            timeline_recent=[],
            session_context="",
            personalities=[],
            memories=prefilled_memories,
        )

        assert isinstance(spec, UtteranceSpec)
        assert spec.user.intent == "emotional_sharing"
        assert spec.user.emotion == "negative"

        # 记忆分层验证
        fact_ids = [m.memory_id for m in spec.memories]
        ref_ids = [m.memory_id for m in spec.reference_memories]
        assert "mem_fact_1" in fact_ids, "高置信度记忆应被分到 fact"
        assert "mem_ref_1" in ref_ids, "中等置信度记忆应被分到 reference"
        # stale 记忆不应出现
        assert "mem_stale" not in fact_ids
        assert "mem_stale" not in ref_ids

        # 门控
        assert spec.gate.tone == "caring"
        assert spec.gate.response_mode == "soothe"

    @pytest.mark.parametrize("msg,exp_tone,exp_mode", [
        ("帮我查一下数据库", "direct", "direct_answer"),
        ("不对，你错了", "soft", "confirm"),
        ("你好", "warm", "auto"),
    ])
    def test_process_various_intents(self, mock_services, msg, exp_tone, exp_mode):
        orchestrator = CircuitOrchestrator(
            chroma_service=mock_services["chroma"],
            personality_store=mock_services["personality"],
            impulse_scheduler=mock_services["impulse"],
            dmn_engine=mock_services["dmn"],
            chat_history=mock_services["chat_history"],
            co_tracker=mock_services["co_tracker"],
        )
        spec = orchestrator.process(
            user_message=msg,
            query_embedding=None,
            ctx_obj=None,
            timeline_recent=[],
            session_context="",
            personalities=[],
            memories=[],
        )
        assert spec.gate.tone == exp_tone, f"{msg}: tone={spec.gate.tone}"
        assert spec.gate.response_mode == exp_mode, f"{msg}: mode={spec.gate.response_mode}"

    def test_process_with_personality_notes(self, mock_services):
        """验证 personality_notes 被正确注入 spec。"""
        orchestrator = CircuitOrchestrator(
            chroma_service=mock_services["chroma"],
            personality_store=mock_services["personality"],
            impulse_scheduler=mock_services["impulse"],
            dmn_engine=mock_services["dmn"],
            chat_history=mock_services["chat_history"],
            co_tracker=mock_services["co_tracker"],
        )
        spec = orchestrator.process(
            user_message="写个Python脚本",
            query_embedding=None,
            ctx_obj=None,
            timeline_recent=[],
            session_context="",
            personalities=[{"content": "用户正在学习Python", "source": "user", "type": "interest"}],
            memories=[],
        )
        assert len(spec.personality_notes) >= 1
        notes_text = " ".join(n.get("content", "") for n in spec.personality_notes)
        assert "Python" in notes_text or "学习" in notes_text

    def test_impulse_suppression_in_emotional_context(self, mock_services):
        """情绪场景下工作类冲动被压制。"""
        orchestrator = CircuitOrchestrator(
            chroma_service=mock_services["chroma"],
            personality_store=mock_services["personality"],
            impulse_scheduler=mock_services["impulse"],
            dmn_engine=mock_services["dmn"],
            chat_history=mock_services["chat_history"],
            co_tracker=mock_services["co_tracker"],
        )
        spec = orchestrator.process(
            user_message="最近压力好大",
            query_embedding=None,
            ctx_obj=None,
            timeline_recent=[],
            session_context="",
            personalities=[],
            memories=[],
        )
        # emotional_sharing + negative → tone=caring, 工作冲动被压制
        assert spec.gate.tone == "caring"
        assert spec.gate.response_mode == "soothe"


class TestAnalyzeUserMessageWithBrain:
    """analyze_user_message 在 brain 可用时的行为验证。"""

    def test_brain_model_path_returns_model_result(self, monkeypatch):
        """mock ChuchuCNN 返回固定结果，验证 brain 路径生效。"""
        from app.brain.models import IntentClassifier, EmotionAnalyzer
        from app.core.circuit import analyze_user_message

        # mock 意图分类器固定返回 emotional_sharing
        monkeypatch.setattr(
            IntentClassifier, "_chuchu_predict",
            lambda self, text: __import__("app.brain.models", fromlist=["IntentResult"]).IntentResult(
                intent="emotional_sharing", confidence=0.95, source="model",
            )
        )
        # mock 情绪分类器固定返回 negative
        monkeypatch.setattr(
            EmotionAnalyzer, "_chuchu_analyze",
            lambda self, text: __import__("app.brain.models", fromlist=["EmotionResult"]).EmotionResult(
                primary="negative", valence=-0.5, arousal=0.4,
                intensity=0.5, confidence=0.9, source="model",
            )
        )

        # 需要一个可用的 brain 实例
        from app.brain.models import ChuchenBrain
        brain = ChuchenBrain()
        brain.intent_classifier._chuchu_ok = True
        brain.emotion_analyzer._chuchu_ok = True

        r = analyze_user_message("最近压力好大", brain=brain)
        assert r.intent == "emotional_sharing"
        assert r.emotion == "negative"
        assert r.confidence >= 0.6
