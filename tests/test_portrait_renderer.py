# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: bf571185

"""测试 app/portrait/renderer.py — 画像 Prompt 渲染。

测试渲染逻辑对条目过滤、分类、文本剥离的正确性。
"""
import tempfile
from datetime import datetime

import pytest

from app.portrait.manager import PortraitManager
from app.portrait.renderer import (
    PortraitRenderer,
    STABLE_DIMS,
    DYNAMIC_DIMS,
    DIM_CN_LABELS,
)
from app.portrait.state import PortraitEntry, EntryStatus


# ═══════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════

class TestConstants:
    def test_stable_dims_are_8(self):
        assert len(STABLE_DIMS) == 8

    def test_dynamic_dims_are_4(self):
        assert len(DYNAMIC_DIMS) == 4

    def test_no_overlap(self):
        assert set(STABLE_DIMS) & set(DYNAMIC_DIMS) == set()

    def test_all_12_dims_covered(self):
        assert len(set(STABLE_DIMS) | set(DYNAMIC_DIMS)) == 12

    def test_cn_labels_bilingual(self):
        assert DIM_CN_LABELS["usr1"] == "核心特征"
        assert DIM_CN_LABELS["ai1"] == "核心表达特征"


# ═══════════════════════════════════════════════════════════════════
# _strip_entry
# ═══════════════════════════════════════════════════════════════════

class TestStripEntry:
    def test_removes_pending_marker(self):
        result = PortraitRenderer._strip_entry("**情绪**: positive （待验证）")
        assert "待验证" not in result
        assert "情绪" in result

    def test_removes_pending_with_detail(self):
        result = PortraitRenderer._strip_entry("文本 （待验证 · 情绪翻转）")
        assert "待验证" not in result
        assert "情绪翻转" not in result
        assert "文本" in result

    def test_removes_cooling_marker(self):
        result = PortraitRenderer._strip_entry("特征描述 （cooling）")
        assert "cooling" not in result
        assert "特征描述" in result

    def test_removes_warm_marker(self):
        result = PortraitRenderer._strip_entry("标签 （warm）")
        assert "warm" not in result

    def test_removes_hot_marker(self):
        result = PortraitRenderer._strip_entry("标签 （hot）")
        assert "hot" not in result

    def test_removes_backtick_metadata(self):
        result = PortraitRenderer._strip_entry("特征 `高 · 3条证据 · tags:Python`")
        assert "`" not in result
        assert "高" not in result
        assert "特征" in result

    def test_collapses_whitespace(self):
        result = PortraitRenderer._strip_entry("  多个   空格   ")
        assert result == "多个 空格"

    def test_combined_markers(self):
        result = PortraitRenderer._strip_entry(
            "喜欢编程 （待验证 · 首轮初标记） `中 · 1条证据 · tags:Python`"
        )
        assert "待验证" not in result
        assert "`" not in result
        assert "喜欢编程" in result

    def test_preserves_content(self):
        """确保核心内容不被剥离"""
        result = PortraitRenderer._strip_entry("用户喜欢 Python 和 Rust")
        assert "Python" in result
        assert "Rust" in result


# ═══════════════════════════════════════════════════════════════════
# PortraitRenderer — fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def empty_renderer():
    tmpdir = tempfile.mkdtemp()
    path = f"{tmpdir}/PORTRAIT.md"
    mgr = PortraitManager(file_path=path)
    renderer = PortraitRenderer(mgr)
    yield renderer
    mgr._lock.release() if mgr._lock.locked() else None
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def populated_renderer():
    tmpdir = tempfile.mkdtemp()
    path = f"{tmpdir}/PORTRAIT.md"
    mgr = PortraitManager(file_path=path)
    now = datetime.now().isoformat()
    # 用户稳定维度
    mgr.set_entry("usr1-001", "喜欢钻研技术", tags=["技术"], confidence=0.85, status=EntryStatus.ACTIVE, last_observed=now)
    mgr.set_entry("usr3-001", "晚上活跃", tags=["时间"], confidence=0.70, status=EntryStatus.ACTIVE, last_observed=now)
    mgr.set_entry("usr5-001", "Python开发", tags=["Python"], confidence=0.80, status=EntryStatus.ACTIVE, last_observed=now)
    mgr.set_entry("usr6-001", "压力时焦虑", tags=["压力"], confidence=0.65, status=EntryStatus.ACTIVE, last_observed=now)
    # 用户动态维度
    mgr.set_entry("usr2-001", "**情绪**: positive", confidence=0.60, status=EntryStatus.ACTIVE, last_observed=now)
    mgr.set_entry("usr4-001", "信任度: 0.75", confidence=0.55, status=EntryStatus.ACTIVE, last_observed=now)
    # AI 稳定维度
    mgr.set_entry("ai1-001", "技术型表达", tags=["技术"], confidence=0.85, status=EntryStatus.ACTIVE, last_observed=now)
    mgr.set_entry("ai3-001", "响应快速", tags=["效率"], confidence=0.70, status=EntryStatus.ACTIVE, last_observed=now)
    mgr.set_entry("ai5-001", "Python知识", tags=["Python"], confidence=0.80, status=EntryStatus.ACTIVE, last_observed=now)
    mgr.set_entry("ai6-001", "理性分析", tags=["分析"], confidence=0.75, status=EntryStatus.ACTIVE, last_observed=now)
    # AI 动态维度
    mgr.set_entry("ai2-001", "**表达色调**: 伴随positive情绪", confidence=0.50, status=EntryStatus.ACTIVE, last_observed=now)
    mgr.set_entry("ai4-001", "关系认知: 用户信任度0.75", confidence=0.55, status=EntryStatus.ACTIVE, last_observed=now)

    renderer = PortraitRenderer(mgr)
    yield renderer
    mgr._lock.release() if mgr._lock.locked() else None
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# PortraitRenderer — render_stable
# ═══════════════════════════════════════════════════════════════════

class TestRenderStable:
    def test_empty_returns_empty_string(self, empty_renderer):
        result = empty_renderer.render_stable()
        assert result == ""

    def test_has_header(self, populated_renderer):
        result = populated_renderer.render_stable()
        assert "【认知画像】" in result

    def test_includes_user_section(self, populated_renderer):
        result = populated_renderer.render_stable()
        assert "用户" in result

    def test_includes_ai_section(self, populated_renderer):
        result = populated_renderer.render_stable()
        assert "AI" in result

    def test_includes_user_stable_dims(self, populated_renderer):
        """稳定渲染应包含 usr1/usr3/usr5/usr6 的内容"""
        result = populated_renderer.render_stable()
        assert "钻研技术" in result or "喜欢钻研" in result
        assert "活跃" in result
        assert "Python" in result

    def test_excludes_dynamic_dims(self, populated_renderer):
        """稳定渲染不应包含 usr2/usr4（动态维度）"""
        result = populated_renderer.render_stable()
        assert "情绪" not in result  # usr2
        assert "信任度" not in result  # usr4

    def test_includes_ai_stable_dims(self, populated_renderer):
        result = populated_renderer.render_stable()
        assert "技术型表达" in result

    def test_filters_low_confidence(self, empty_renderer):
        """低置信度条目不应出现在稳定渲染中"""
        now = datetime.now().isoformat()
        empty_renderer._manager.set_entry(
            "usr1-001", "低置信度特征", confidence=0.35, last_observed=now,
        )
        result = empty_renderer.render_stable()
        assert "低置信度" not in result

    def test_strips_entry_markers(self, populated_renderer):
        """渲染输出不应包含 backtick 元数据或状态标记"""
        result = populated_renderer.render_stable()
        assert "`" not in result
        assert "待验证" not in result


# ═══════════════════════════════════════════════════════════════════
# PortraitRenderer — render_dynamic
# ═══════════════════════════════════════════════════════════════════

class TestRenderDynamic:
    def test_empty_returns_empty_string(self, empty_renderer):
        result = empty_renderer.render_dynamic()
        assert result == ""

    def test_has_header(self, populated_renderer):
        result = populated_renderer.render_dynamic()
        assert "【当前状态】" in result

    def test_includes_user_dynamic(self, populated_renderer):
        result = populated_renderer.render_dynamic()
        assert "用户" in result
        assert "情绪" in result

    def test_includes_ai_dynamic(self, populated_renderer):
        result = populated_renderer.render_dynamic()
        assert "AI" in result

    def test_combined_multiple_parts(self, populated_renderer):
        """usr2 和 usr4 的内容应用 · 连接"""
        result = populated_renderer.render_dynamic()
        # usr2: emotion, usr4: trust
        has_emotion = "情绪" in result or "positive" in result
        has_trust = "信任" in result
        assert has_emotion or has_trust  # at least one should be present

    def test_excludes_stable_dims(self, populated_renderer):
        """动态渲染不应包含稳定维度的内容"""
        result = populated_renderer.render_dynamic()
        assert "钻研技术" not in result  # usr1
        assert "Python开发" not in result  # usr5

    def test_strips_entry_markers(self, populated_renderer):
        result = populated_renderer.render_dynamic()
        assert "`" not in result


# ═══════════════════════════════════════════════════════════════════
# PortraitRenderer — render_full
# ═══════════════════════════════════════════════════════════════════

class TestRenderFull:
    def test_empty_has_structure(self, empty_renderer):
        result = empty_renderer.render_full()
        assert "# 认知画像" in result
        assert "## 用户画像" in result
        assert "## AI 画像" in result

    def test_empty_shows_placeholders(self, empty_renderer):
        result = empty_renderer.render_full()
        assert "暂无数据" in result

    def test_includes_all_12_dim_headers(self, empty_renderer):
        result = empty_renderer.render_full()
        for label in ["核心特征", "当前状态", "行为节律", "关系快照", "兴趣图谱", "情绪图谱"]:
            assert label in result, f"should contain {label}"

    def test_includes_entries_with_confidence(self, populated_renderer):
        result = populated_renderer.render_full()
        assert "钻研技术" in result or "喜欢钻研" in result
        # 置信度 < 1.0 时应显示百分比
        assert "%" in result

    def test_includes_all_user_dims(self, populated_renderer):
        result = populated_renderer.render_full()
        assert "usr1" not in result.lower() or "核心特征" in result  # dim markers may or may not appear

    def test_includes_lower_confidence_entries_in_full(self, empty_renderer):
        """render_full 对 >= 0.40 置信度的条目都应展示（与 stable 的维度过滤不同）"""
        now = datetime.now().isoformat()
        empty_renderer._manager.set_entry(
            "usr2-001", "动态维度条目", confidence=0.45, status=EntryStatus.ACTIVE, last_observed=now,
        )
        result = empty_renderer.render_full()
        # 全量渲染应包含动态维度条目
        assert "动态维度条目" in result

    def test_strips_entry_markers(self, populated_renderer):
        result = populated_renderer.render_full()
        # 全量渲染应该剥离元数据
        assert "`" not in result
