"""测试 app/llm/deepseek.py — _build_prompt / _build_stable_system_prompt。

覆盖：记忆区格式化、置信度标签、人格区、核心规则注入、stale 处理。
"""
from unittest.mock import MagicMock, patch
import pytest


# ═══════════════════════════════════════════════════════
# _build_prompt — 记忆 + 人格 + 规则 拼接
# ═══════════════════════════════════════════════════════

class TestBuildPrompt:
    @patch("app.llm.deepseek.load_system_prompt", return_value="【系统提示词】")
    def test_no_memories(self, mock_load):
        from app.llm.deepseek import LLMClient
        prompt = LLMClient._build_prompt([])
        assert "【记忆】" in prompt
        assert "没有找到" in prompt

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    def test_with_memories(self, mock_load):
        from app.llm.deepseek import LLMClient
        mem = {
            "id": "m1",
            "document": "用户喜欢喝咖啡",
            "metadata": {"timestamp": 1700000000, "hit_count": 80},
            "source": "semantic",
            "display_source": "语义检索",
        }
        prompt = LLMClient._build_prompt([mem])
        assert "【记忆】" in prompt
        assert "高置信" in prompt  # hit_count=80 → 高
        assert "用户喜欢喝咖啡" in prompt

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    def test_stale_memory_tagged(self, mock_load):
        from app.llm.deepseek import LLMClient
        mem = {
            "id": "m1",
            "document": "旧信息",
            "metadata": {"timestamp": 1700000000, "stale": True, "hit_count": 5},
            "source": "semantic",
        }
        prompt = LLMClient._build_prompt([mem])
        assert "已更新" in prompt

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    def test_with_personality_notes(self, mock_load):
        from app.llm.deepseek import LLMClient
        p_notes = ["用户喜欢安静", {"content": "偏好早睡", "type": "偏好模式", "confidence": "高"}]
        prompt = LLMClient._build_prompt([], personalities=p_notes)
        assert "我对你的了解" in prompt
        assert "用户喜欢安静" in prompt
        assert "偏好早睡" in prompt

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    def test_emotional_intensity_tag(self, mock_load):
        from app.llm.deepseek import LLMClient
        mem = {
            "id": "m1",
            "document": "情绪消息",
            "metadata": {
                "timestamp": 1700000000, "hit_count": 20,
                "emotional_intensity": 3, "emotion_valence_bin": "negative",
            },
            "source": "semantic",
        }
        prompt = LLMClient._build_prompt([mem])
        assert "情绪·负向" in prompt

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    def test_summary_only_memory(self, mock_load):
        from app.llm.deepseek import LLMClient
        mem = {
            "id": "m1",
            "summary": "简短摘要",
            "document": "完整文档内容很长",
            "metadata": {"timestamp": 1700000000, "hit_count": 5},
            "summary_only": True,
        }
        prompt = LLMClient._build_prompt([mem])
        assert "简短摘要" in prompt

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    def test_with_session_context(self, mock_load):
        from app.llm.deepseek import LLMClient
        prompt = LLMClient._build_prompt([], session_context="对话脉络：用户提到咖啡")
        assert "咖啡" in prompt

    @patch("app.llm.deepseek.load_system_prompt", return_value="SYS")
    def test_indirect_source_low_confidence(self, mock_load):
        from app.llm.deepseek import LLMClient
        mem = {
            "id": "m1",
            "document": "间接来源记忆",
            "metadata": {"timestamp": 1700000000, "hit_count": 200},
            "source": "co_occurrence",
        }
        prompt = LLMClient._build_prompt([mem])
        assert "低置信" in prompt  # 间接来源 → 低


# ═══════════════════════════════════════════════════════
# _build_stable_system_prompt
# ═══════════════════════════════════════════════════════

class TestBuildStableSystemPrompt:
    def test_includes_core_rules(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.personality_notes = []
        mock_state.personality_notes_ai = []
        prompt = LLMClient._build_stable_system_prompt(mock_state)
        assert "记忆使用核心规则" in prompt

    @patch("app.llm.deepseek.BENCHMARK_MODE", True)
    def test_benchmark_rule_included(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.personality_notes = []
        mock_state.personality_notes_ai = []
        prompt = LLMClient._build_stable_system_prompt(mock_state)
        assert "知识更新冲突解决" in prompt

    def test_personality_notes_formatted(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.personality_notes = [
            {"type": "行为模式", "content": "用户喜欢安静"},
            {"type": "偏好模式", "content": "早睡早起"},
            "纯文本标签",
        ]
        mock_state.personality_notes_ai = []
        prompt = LLMClient._build_stable_system_prompt(mock_state)
        assert "我对你的了解" in prompt
        assert "行为" in prompt  # type_tag 缩写
        assert "纯文本标签" in prompt

    def test_ai_notes_formatted(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.personality_notes = []
        mock_state.personality_notes_ai = [
            {"type": "沟通模式", "content": "简洁回答"},
        ]
        prompt = LLMClient._build_stable_system_prompt(mock_state)
        assert "我自己的表达习惯" in prompt
        assert "简洁回答" in prompt


# ═══════════════════════════════════════════════════════
# _build_execute_directive
# ═══════════════════════════════════════════════════════

class TestBuildExecuteDirective:
    def test_basic_directive(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.user.intent = "emotional_sharing"
        mock_state.user.raw_text = "我今天很难过"
        mock_state.gate.tone = "caring"
        mock_state.gate.response_mode = "soothe"
        mock_state.mirror_prediction = None
        mock_state.relationship = None
        mock_state.emotional_reversals = []
        result = LLMClient._build_execute_directive(mock_state)
        assert "先共情" in result or "soothe" in result

    def test_with_mirror_prediction(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.user.intent = "recall"
        mock_state.user.raw_text = "我之前说过"
        mock_state.gate.tone = "warm"
        mock_state.gate.response_mode = "auto"
        mock_state.mirror_prediction = {"next_intents": ["追问", "澄清"]}
        mock_state.relationship = None
        mock_state.emotional_reversals = []
        result = LLMClient._build_execute_directive(mock_state)
        assert "准备方向" in result

    def test_with_relationship(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.user.intent = "casual"
        mock_state.user.raw_text = "你好"
        mock_state.gate.tone = "warm"
        mock_state.gate.response_mode = "auto"
        mock_state.mirror_prediction = None
        mock_state.emotional_reversals = []
        mock_rel = MagicMock()
        mock_rel.familiarity = 0.8
        mock_rel.trust = 0.7
        mock_rel.closeness = 0.6
        mock_rel.interaction_mode = "conversation"
        mock_state.relationship = mock_rel
        result = LLMClient._build_execute_directive(mock_state)
        assert "【关系状态】" in result
        assert "高" in result

    def test_no_raw_text(self):
        from app.llm.deepseek import LLMClient
        mock_state = MagicMock()
        mock_state.user.raw_text = ""
        result = LLMClient._build_execute_directive(mock_state)
        assert result == ""
