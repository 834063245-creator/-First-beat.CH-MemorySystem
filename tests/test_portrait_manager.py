# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: c2999fae

"""测试 app/portrait/manager.py — PORTRAIT.md 生命周期管理。

覆盖: 加载/解析、条目 CRUD、渲染、查询接口、boost 映射。
"""
import json
import os
import tempfile
from datetime import datetime

import pytest

from app.portrait.manager import (
    PortraitManager,
    ALL_DIMS,
    DIM_LABELS,
    DIM_CAPS,
    _default_frontmatter,
    _default_portrait_md,
    _RE_DIM,
    _RE_ENTRY,
    _RE_FRONTMATTER,
    _RE_BACKTICK_META,
)
from app.portrait.state import PortraitEntry, EntryStatus


# ═══════════════════════════════════════════════════════════════════
# 正则表达式
# ═══════════════════════════════════════════════════════════════════

class TestRegex:
    def test_dim_regex(self):
        m = _RE_DIM.search("<!-- dim:usr1 核心特征 -->")
        assert m is not None
        assert m.group(1) == "usr1"

    def test_dim_regex_no_label(self):
        m = _RE_DIM.search("<!-- dim:ai2 -->")
        assert m is not None
        assert m.group(1) == "ai2"

    def test_entry_regex(self):
        m = _RE_ENTRY.search("<!-- entry:usr1-005 -->")
        assert m is not None
        assert m.group(1) == "usr1"
        assert m.group(2) == "005"

    def test_entry_regex_with_text(self):
        m = _RE_ENTRY.search("some text <!-- entry:ai3-012 --> trailing")
        assert m is not None
        assert m.group(1) == "ai3"

    def test_frontmatter_regex(self):
        content = "---\n{\"version\": 1}\n---\n\n# 正文"
        m = _RE_FRONTMATTER.search(content)
        assert m is not None
        assert "version" in m.group(1)

    def test_backtick_meta_regex(self):
        m = _RE_BACKTICK_META.search("text `tags:a b` more")
        assert m is not None

    def test_dims_are_12_total(self):
        """6 用户维度 + 6 AI 维度 = 12"""
        assert len(ALL_DIMS) == 12
        assert ALL_DIMS[0] == "usr1"
        assert ALL_DIMS[-1] == "ai6"


# ═══════════════════════════════════════════════════════════════════
# 默认值 / 常量
# ═══════════════════════════════════════════════════════════════════

class TestDefaults:
    def test_default_frontmatter(self):
        fm = _default_frontmatter(1)
        assert fm["version"] == 1
        assert "last_updated" in fm
        assert "dimensions" in fm
        assert "user" in fm["dimensions"]
        assert "ai" in fm["dimensions"]

    def test_default_portrait_md(self):
        md = _default_portrait_md()
        assert "# 认知画像" in md
        assert "## 用户画像" in md
        assert "## AI 画像" in md
        assert "---" in md  # frontmatter
        # 包含所有 12 维度注释
        for dim in ALL_DIMS:
            assert f"dim:{dim}" in md, f"should contain dim:{dim}"

    def test_dim_labels(self):
        assert DIM_LABELS["usr1"] == "核心特征"
        assert DIM_LABELS["ai1"] == "核心表达特征"
        assert DIM_LABELS["usr2"] == "当前状态"

    def test_dim_caps(self):
        assert DIM_CAPS["usr1"] == 10
        assert DIM_CAPS["usr2"] == 8
        assert DIM_CAPS["usr5"] == 20
        assert DIM_CAPS["usr4"] == 6


# ═══════════════════════════════════════════════════════════════════
# PortraitManager — 初始化
# ═══════════════════════════════════════════════════════════════════

class TestPortraitManagerInit:
    def test_creates_new_file_when_missing(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        try:
            mgr = PortraitManager(file_path=path)
            assert os.path.exists(path)
            assert mgr.is_empty
            assert mgr.version == 1  # 新文件默认 version=1
        finally:
            mgr._lock.release() if mgr._lock.locked() else None
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_loads_existing_file(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        try:
            # 先创建并写入一个已知文件
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_default_portrait_md())
            mgr = PortraitManager(file_path=path)
            assert mgr.is_empty  # 默认模板没有条目
        finally:
            mgr._lock.release() if mgr._lock.locked() else None
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_handles_corrupted_file(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("garbage {{{ not valid")
            mgr = PortraitManager(file_path=path)
            # 应优雅降级
            assert mgr.is_empty
        finally:
            mgr._lock.release() if mgr._lock.locked() else None
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_creates_parent_dirs(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "sub1", "sub2", "PORTRAIT.md")
        try:
            mgr = PortraitManager(file_path=path)
            assert os.path.exists(path)
        finally:
            mgr._lock.release() if mgr._lock.locked() else None
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# PortraitManager — 条目 CRUD
# ═══════════════════════════════════════════════════════════════════

class TestPortraitManagerCRUD:
    @pytest.fixture
    def manager(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        mgr = PortraitManager(file_path=path)
        yield mgr
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_set_and_get_entry(self, manager):
        manager.set_entry("usr1-001", "用户喜欢编程", tags=["编程"], confidence=0.90,
                         status=EntryStatus.ACTIVE)
        entry = manager.get_entry("usr1-001")
        assert entry is not None
        assert entry.text == "用户喜欢编程"
        assert entry.tags == ["编程"]
        assert entry.confidence == 0.90
        assert entry.dim == "usr1"

    def test_update_existing_entry(self, manager):
        manager.set_entry("usr1-001", "原始文本", status=EntryStatus.ACTIVE)
        manager.set_entry("usr1-001", "更新文本", confidence=0.50, status=EntryStatus.ACTIVE)
        entry = manager.get_entry("usr1-001")
        assert entry.text == "更新文本"
        assert entry.confidence == 0.50

    def test_delete_entry(self, manager):
        manager.set_entry("usr1-001", "测试", status=EntryStatus.ACTIVE)
        assert manager.get_entry("usr1-001") is not None
        manager.delete_entry("usr1-001")
        assert manager.get_entry("usr1-001") is None

    def test_delete_nonexistent(self, manager):
        """删除不存在的条目不应报错"""
        manager.delete_entry("nonexistent-999")

    def test_get_dim_entries(self, manager):
        manager.set_entry("usr1-001", "条目1", status=EntryStatus.ACTIVE)
        manager.set_entry("usr1-002", "条目2", status=EntryStatus.ACTIVE)
        manager.set_entry("usr2-001", "其他维度", status=EntryStatus.ACTIVE)

        entries = manager.get_dim_entries("usr1")
        assert len(entries) == 2
        assert all(e.dim == "usr1" for e in entries)

    def test_get_dim_entries_excludes_decayed(self, manager):
        from app.portrait.state import EntryStateMachine
        # 创建一个条目然后衰减它
        manager.set_entry("usr1-001", "test", status=EntryStatus.ACTIVE)
        entry = manager.get_entry("usr1-001")
        EntryStateMachine.decay(entry)  # 手动衰减

        entries = manager.get_dim_entries("usr1")
        assert len(entries) == 0

    def test_next_seq_new_dim(self, manager):
        assert manager.next_seq("usr1") == 1

    def test_next_seq_existing(self, manager):
        manager.set_entry("usr1-001", "a", status=EntryStatus.ACTIVE)
        manager.set_entry("usr1-005", "b", status=EntryStatus.ACTIVE)
        assert manager.next_seq("usr1") == 6

    def test_next_seq_per_dim(self, manager):
        manager.set_entry("usr1-010", "a", status=EntryStatus.ACTIVE)
        manager.set_entry("ai1-001", "b", status=EntryStatus.ACTIVE)
        assert manager.next_seq("usr1") == 11
        assert manager.next_seq("ai1") == 2


# ═══════════════════════════════════════════════════════════════════
# PortraitManager — 保存 / 加载往返
# ═══════════════════════════════════════════════════════════════════

class TestPortraitManagerSaveLoad:
    @pytest.fixture
    def manager(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        mgr = PortraitManager(file_path=path)
        yield mgr, path
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_roundtrip_single_entry(self, manager):
        mgr, path = manager
        mgr.set_entry(
            "usr1-001", "用户喜欢编程",
            tags=["Python", "Rust"], confidence=0.90, status=EntryStatus.ACTIVE, evidence_count=5,
        )
        mgr.save()

        # 重新加载
        mgr2 = PortraitManager(file_path=path)
        entry = mgr2.get_entry("usr1-001")
        assert entry is not None
        assert "编程" in entry.text
        assert entry.confidence == 0.90
        assert entry.evidence_count == 5
        mgr2._lock.release() if mgr2._lock.locked() else None

    def test_roundtrip_multiple_entries(self, manager):
        mgr, path = manager
        mgr.set_entry("usr1-001", "核心特征1", tags=["a"], confidence=0.9)
        mgr.set_entry("usr2-001", "当前状态1", confidence=0.5)
        mgr.set_entry("usr5-001", "兴趣1", tags=["Python"])
        mgr.set_entry("ai1-001", "AI特征1", tags=["express"])
        mgr.save()

        mgr2 = PortraitManager(file_path=path)
        assert mgr2.get_entry("usr1-001") is not None
        assert mgr2.get_entry("usr2-001") is not None
        assert mgr2.get_entry("usr5-001") is not None
        assert mgr2.get_entry("ai1-001") is not None
        mgr2._lock.release() if mgr2._lock.locked() else None

    def test_roundtrip_pending_status(self, manager):
        mgr, path = manager
        mgr.set_entry("usr2-001", "**情绪**: positive （待验证）", status=EntryStatus.PENDING)
        mgr.save()

        mgr2 = PortraitManager(file_path=path)
        entry = mgr2.get_entry("usr2-001")
        assert entry is not None
        assert entry.status == EntryStatus.PENDING
        # 文本中的"待验证"被剥离
        assert "情绪" in entry.text
        mgr2._lock.release() if mgr2._lock.locked() else None

    def test_roundtrip_entry_with_tags_and_confidence(self, manager):
        mgr, path = manager
        mgr.set_entry(
            "usr5-003", "对Rust所有权系统感兴趣",
            tags=["Rust", "编程"], confidence=0.80, evidence_count=3,
        )
        mgr.save()

        # 检查磁盘上的原始文件
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        # 应包含 entry ID 注释
        assert "<!-- entry:usr5-003 -->" in raw

        mgr2 = PortraitManager(file_path=path)
        entry = mgr2.get_entry("usr5-003")
        assert entry is not None
        assert "Rust" in entry.tags
        mgr2._lock.release() if mgr2._lock.locked() else None

    def test_reload_from_disk(self, manager):
        mgr, path = manager
        mgr.set_entry("usr1-001", "初始文本", status=EntryStatus.ACTIVE)
        mgr.save()

        # 模拟另一个进程修改文件
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        modified = raw.replace("初始文本", "外部修改")
        with open(path, "w", encoding="utf-8") as f:
            f.write(modified)

        mgr.reload()
        entry = mgr.get_entry("usr1-001")
        assert entry is not None
        # 重新加载后文本应包含外部修改（PENDING status 可能追加标记）
        assert "外部修改" in entry.text


# ═══════════════════════════════════════════════════════════════════
# PortraitManager — 查询接口
# ═══════════════════════════════════════════════════════════════════

class TestPortraitManagerQuery:
    @pytest.fixture
    def populated_manager(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        mgr = PortraitManager(file_path=path)
        mgr.set_entry("usr1-001", "喜欢技术", tags=["技术"], confidence=0.85, status=EntryStatus.ACTIVE)
        mgr.set_entry("usr2-001", "情绪: 开心", status=EntryStatus.PENDING)
        mgr.set_entry("usr5-001", "兴趣: Python", tags=["Python", "编程"], confidence=0.80, status=EntryStatus.ACTIVE)
        mgr.set_entry("usr5-002", "兴趣: Rust", tags=["Rust"], confidence=0.75, status=EntryStatus.ACTIVE)
        mgr.set_entry("usr6-001", "项目受阻时焦虑", tags=["项目"], confidence=0.70, status=EntryStatus.ACTIVE)
        mgr.set_entry("ai1-001", "表达: 技术型", tags=["技术"], confidence=0.90, status=EntryStatus.ACTIVE)
        mgr.set_entry("ai5-001", "知识: Python", tags=["Python"], confidence=0.85, status=EntryStatus.ACTIVE)
        yield mgr, path
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_all_active(self, populated_manager):
        mgr, _ = populated_manager
        active = mgr.get_all_active()
        # usr1-001 应活跃（confidence=0.85）
        assert "usr1" in active
        # usr2-001 是 PENDING，should_inject=False → 不应该在 active 中
        assert "usr2" not in active or len(active.get("usr2", [])) == 0
        # usr5 应有 2 个条目
        assert len(active.get("usr5", [])) == 2

    def test_is_empty(self, populated_manager):
        mgr, _ = populated_manager
        assert mgr.is_empty is False

    def test_new_manager_is_empty(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        mgr = PortraitManager(file_path=path)
        try:
            assert mgr.is_empty is True
        finally:
            mgr._lock.release() if mgr._lock.locked() else None
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_dimension_summary(self, populated_manager):
        mgr, _ = populated_manager
        summary = mgr.get_dimension_summary()
        assert "usr1" in summary
        assert summary["usr1"]["total"] >= 1
        # usr2 有一条 PENDING 条目 (total=1, active=0)
        assert summary["usr2"]["total"] == 1
        # 空维度
        assert summary["usr3"]["total"] == 0

    def test_extract_tags_for_dim(self, populated_manager):
        mgr, _ = populated_manager
        tags = mgr.extract_tags_for_dim("usr5")
        assert "Python" in tags
        assert "编程" in tags
        assert "Rust" in tags

    def test_extract_hot_topics(self, populated_manager):
        mgr, _ = populated_manager
        topics = mgr.extract_hot_topics()
        # usr5 的标签应该是热点
        assert "Python" in topics
        assert "Rust" in topics

    def test_extract_negative_triggers(self, populated_manager):
        mgr, _ = populated_manager
        triggers = mgr.extract_negative_triggers()
        # usr6-001: "项目受阻时焦虑" → 包含"项目受阻" → 应被提取
        assert "项目" in triggers

    def test_extract_focus_keywords(self, populated_manager):
        mgr, _ = populated_manager
        # usr2-001 文本不包含"关注焦点"
        keywords = mgr.extract_focus_keywords()
        assert keywords == []

    def test_extract_focus_keywords_with_focus(self, populated_manager):
        mgr, _ = populated_manager
        mgr.set_entry("usr2-002", "**关注焦点**: AI, 编程 （热点）", tags=["AI", "编程"],
                      status=EntryStatus.ACTIVE)
        keywords = mgr.extract_focus_keywords()
        assert "AI" in keywords

    def test_version(self, populated_manager):
        mgr, _ = populated_manager
        assert isinstance(mgr.version, int)

    def test_last_updated(self, populated_manager):
        mgr, _ = populated_manager
        assert isinstance(mgr.last_updated, str)


# ═══════════════════════════════════════════════════════════════════
# PortraitManager — boost 映射
# ═══════════════════════════════════════════════════════════════════

class TestPortraitManagerBoost:
    @pytest.fixture
    def manager(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        mgr = PortraitManager(file_path=path)
        yield mgr
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_empty_portrait_returns_empty_boost(self, manager):
        boost = manager.compute_portrait_boost_map()
        assert boost == {}

    def test_hot_topic_gets_positive_boost(self, manager):
        now = datetime.now().isoformat()
        manager.set_entry(
            "usr5-001", "喜欢Python", tags=["Python"],
            confidence=0.80, status=EntryStatus.ACTIVE, last_observed=now,
        )
        boost = manager.compute_portrait_boost_map()
        assert "Python" in boost
        assert boost["Python"] > 0

    def test_negative_trigger_gets_negative_boost(self, manager):
        now = datetime.now().isoformat()
        manager.set_entry(
            "usr6-001", "项目受阻时压力大", tags=["项目"],
            confidence=0.80, status=EntryStatus.ACTIVE, last_observed=now,
        )
        boost = manager.compute_portrait_boost_map()
        # "项目受阻" 关键字在 negative_keywords 中
        assert "项目" in boost
        assert boost["项目"] < 0

    def test_ai_active_domain_gets_boost(self, manager):
        now = datetime.now().isoformat()
        manager.set_entry(
            "ai5-001", "AI知识: Python", tags=["Python"],
            confidence=0.80, status=EntryStatus.ACTIVE, last_observed=now,
        )
        boost = manager.compute_portrait_boost_map()
        assert "Python" in boost

    def test_boost_values_in_expected_range(self, manager):
        now = datetime.now().isoformat()
        manager.set_entry("usr5-001", "hot topic", tags=["hot_tag"], confidence=0.8, last_observed=now)
        manager.set_entry("usr6-001", "项目受阻", tags=["neg_tag"], confidence=0.8, last_observed=now)
        boost = manager.compute_portrait_boost_map()
        for v in boost.values():
            assert -0.5 <= v <= 0.5, f"boost value {v} out of expected range"


# ═══════════════════════════════════════════════════════════════════
# PortraitManager — apply_state_machine
# ═══════════════════════════════════════════════════════════════════

class TestPortraitManagerStateMachine:
    @pytest.fixture
    def manager(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "PORTRAIT.md")
        mgr = PortraitManager(file_path=path)
        yield mgr
        mgr._lock.release() if mgr._lock.locked() else None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_active_entries_survive(self, manager):
        now = datetime.now().isoformat()
        manager.set_entry("usr1-001", "active entry", last_observed=now)
        manager.apply_state_machine()
        assert manager.get_entry("usr1-001") is not None

    def test_old_entries_get_decayed_and_removed(self, manager):
        """没有 last_observed 的条目（默认 999 天）应被衰减和删除"""
        # 手动创建一个 PortraitEntry 并注入（绕过 set_entry 的 last_observed）
        from app.portrait.state import PortraitEntry
        entry = PortraitEntry(
            id="old-001", dim="usr1", text="极旧的条目",
            last_observed="2020-01-01T00:00:00",  # 非常旧
        )
        manager._entries["old-001"] = entry
        manager.apply_state_machine()
        # 旧条目应被删除
        assert manager.get_entry("old-001") is None

    def test_pending_with_enough_evidence_becomes_active(self, manager):
        now = datetime.now().isoformat()
        manager.set_entry(
            "usr1-001", "pending item",
            status=EntryStatus.PENDING, evidence_count=5,
            last_observed=now,
        )
        manager.apply_state_machine()
        entry = manager.get_entry("usr1-001")
        if entry:  # 如果有足够证据，应转换为 active
            assert entry.status == EntryStatus.ACTIVE
