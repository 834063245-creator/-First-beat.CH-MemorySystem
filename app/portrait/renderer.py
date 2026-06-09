"""PortraitRenderer — PORTRAIT.md 渲染为 LLM prompt 片段。

渲染规则（纯规则，不调 LLM）:
  1. 过滤: 去掉 pending / cooling / 低 confidence 条目
  2. 剥离: 去掉 backtick 元数据、状态标记、"待验证"
  3. 分配: 按稳定/动态分入 stable message 和 dynamic message

输出:
  - render_stable() → stable system message (8 维度，可缓存)
  - render_dynamic() → dynamic system message (4 维度，每轮更新)
"""

from app.portrait.manager import PortraitManager, DIM_LABELS
from app.portrait.state import EntryStatus

# Stable dimensions: go into message[0] (DeepSeek prefix cache hits >95%)
STABLE_DIMS = ["usr1", "usr3", "usr5", "usr6", "ai1", "ai3", "ai5", "ai6"]

# Dynamic dimensions: go into message[N+1] (changes every round)
DYNAMIC_DIMS = ["usr2", "ai2", "usr4", "ai4"]

# 维度中文标签
DIM_CN_LABELS = {
    "usr1": "核心特征",
    "usr2": "当前情绪",
    "usr3": "行为节律",
    "usr4": "关系状态",
    "usr5": "兴趣关注",
    "usr6": "情绪模式",
    "ai1": "核心表达特征",
    "ai2": "当前表达色调",
    "ai3": "行为节律",
    "ai4": "关系认知",
    "ai5": "知识积累域",
    "ai6": "表达风格",
}


class PortraitRenderer:
    """将 PortraitManager 的条目渲染为 LLM prompt 片段。"""

    def __init__(self, manager: PortraitManager):
        self._manager = manager

    def render_stable(self) -> str:
        """渲染稳定画像 (8 维度) → 注入 message[0] system prompt。

        过滤: 只包含 ACTIVE + confidence >= 0.40 的条目
        """
        sections = []
        active = self._manager.get_all_active()

        # 用户 4 维度
        user_dims = ["usr1", "usr3", "usr5", "usr6"]
        user_entries = []
        for dim in user_dims:
            if dim in active:
                for entry in active[dim]:
                    user_entries.append(f"  - {self._strip_entry(entry.text)}")
        if user_entries:
            sections.append("  用户")
            sections.extend(user_entries)

        # AI 4 维度
        ai_dims = ["ai1", "ai3", "ai5", "ai6"]
        ai_entries = []
        for dim in ai_dims:
            if dim in active:
                for entry in active[dim]:
                    ai_entries.append(f"  - {self._strip_entry(entry.text)}")
        if ai_entries:
            sections.append("  AI")
            sections.extend(ai_entries)

        if not sections:
            return ""

        sections.insert(0, "【认知画像】")
        return "\n".join(sections)

    def render_dynamic(self) -> str:
        """渲染动态画像 (4 维度) → 注入 message[N+1] system prompt。

        包含: 用户.2(当前状态), AI.2(当前状态), 用户.4(关系), AI.4(关系)
        """
        sections = []
        active = self._manager.get_all_active()

        # 用户当前状态 + 关系
        user_parts = []
        for dim in ["usr2", "usr4"]:
            if dim in active:
                for entry in active[dim]:
                    user_parts.append(self._strip_entry(entry.text))
        if user_parts:
            sections.append("用户 · " + " · ".join(user_parts))

        # AI 当前状态 + 关系
        ai_parts = []
        for dim in ["ai2", "ai4"]:
            if dim in active:
                for entry in active[dim]:
                    ai_parts.append(self._strip_entry(entry.text))
        if ai_parts:
            sections.append("AI · " + " · ".join(ai_parts))

        if not sections:
            return ""

        sections.insert(0, "【当前状态】")
        return "\n".join(sections)

    def render_full(self) -> str:
        """渲染全量画像（供 API 返回，不过滤低置信度）。"""
        sections = []
        active = self._manager.get_all_active()

        sections.append("# 认知画像")
        sections.append("")

        # 用户画像
        sections.append("## 用户画像")
        for dim in ["usr1", "usr2", "usr3", "usr4", "usr5", "usr6"]:
            label = DIM_LABELS.get(dim, dim)
            sections.append(f"\n### {label}")
            if dim in active:
                for entry in active[dim]:
                    # full mode 保留置信度信息
                    conf_str = f"[{entry.confidence:.0%}]" if entry.confidence < 1.0 else ""
                    sections.append(f"- {self._strip_entry(entry.text)} {conf_str}")
            else:
                sections.append("<!-- 暂无数据 -->")

        # AI 画像
        sections.append("\n## AI 画像")
        for dim in ["ai1", "ai2", "ai3", "ai4", "ai5", "ai6"]:
            label = DIM_LABELS.get(dim, dim)
            sections.append(f"\n### {label}")
            if dim in active:
                for entry in active[dim]:
                    conf_str = f"[{entry.confidence:.0%}]" if entry.confidence < 1.0 else ""
                    sections.append(f"- {self._strip_entry(entry.text)} {conf_str}")
            else:
                sections.append("<!-- 暂无数据 -->")

        return "\n".join(sections)

    @staticmethod
    def _strip_entry(text: str) -> str:
        """剥离条目中的元数据标记。

        移除: '（待验证）' 前缀/后缀、状态标记。
        保留: 认知描述本身。
        """
        import re
        # 移除 "（待验证 · ...）" / "（待验证）" / "（cooling）"
        text = re.sub(r"[（(]待验证[^)）]*[)）]\s*", "", text)
        text = re.sub(r"[（(]cooling[)）]\s*", "", text)
        text = re.sub(r"[（(]warm[)）]\s*", "", text)
        text = re.sub(r"[（(]hot[)）]\s*", "", text)

        # 移除行尾 backtick 元数据
        text = re.sub(r"\s*`[^`]*`\s*$", "", text)

        # 清理多余空格
        text = re.sub(r"\s+", " ", text).strip()
        return text
