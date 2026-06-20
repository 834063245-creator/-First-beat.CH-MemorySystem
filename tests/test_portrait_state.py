# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: fc9cb468

"""测试 app/portrait/state.py — 画像条目状态机。

纯逻辑，无 I/O 依赖。
"""
import time
from datetime import datetime, timedelta

import pytest

from app.portrait.state import (
    PortraitEntry,
    EntryStatus,
    EntryStateMachine,
)


# ═══════════════════════════════════════════════════════════════════
# PortraitEntry
# ═══════════════════════════════════════════════════════════════════

class TestPortraitEntry:
    def test_default_values(self):
        entry = PortraitEntry(id="usr1-001", dim="usr1", text="用户喜欢编程")
        assert entry.id == "usr1-001"
        assert entry.dim == "usr1"
        assert entry.text == "用户喜欢编程"
        assert entry.tags == []
        assert entry.confidence == 1.0
        assert entry.status == EntryStatus.ACTIVE
        assert entry.evidence_count == 0
        assert entry.first_observed is None
        assert entry.last_observed is None

    def test_custom_values(self):
        entry = PortraitEntry(
            id="usr5-003",
            dim="usr5",
            text="对Rust感兴趣",
            tags=["Rust", "编程"],
            confidence=0.85,
            status=EntryStatus.PENDING,
            evidence_count=3,
            first_observed="2026-06-01",
            last_observed="2026-06-08",
        )
        assert entry.id == "usr5-003"
        assert entry.tags == ["Rust", "编程"]
        assert entry.confidence == 0.85
        assert entry.status == EntryStatus.PENDING
        assert entry.evidence_count == 3

    def test_days_since_last_observed_no_timestamp(self):
        entry = PortraitEntry(id="test-001", dim="usr1", text="test")
        assert entry.days_since_last_observed == 999.0

    def test_days_since_last_observed_recent(self):
        now = datetime.now()
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            last_observed=now.isoformat(),
        )
        assert entry.days_since_last_observed < 0.1

    def test_days_since_last_observed_week_old(self):
        week_ago = datetime.now() - timedelta(days=7)
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            last_observed=week_ago.isoformat(),
        )
        assert 6.9 <= entry.days_since_last_observed <= 7.1

    def test_days_since_last_observed_invalid_iso(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            last_observed="not-a-date",
        )
        assert entry.days_since_last_observed == 999.0

    def test_should_inject_active_high_confidence(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE, confidence=0.80,
        )
        assert entry.should_inject is True

    def test_should_inject_active_low_confidence(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE, confidence=0.35,
        )
        assert entry.should_inject is False

    def test_should_inject_pending(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.PENDING, confidence=0.90,
        )
        assert entry.should_inject is False

    def test_should_inject_cooling(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.COOLING, confidence=0.90,
        )
        assert entry.should_inject is False

    def test_should_inject_boundary_confidence(self):
        """恰好 0.40 confidence 应允许注入"""
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE, confidence=0.40,
        )
        assert entry.should_inject is True


# ═══════════════════════════════════════════════════════════════════
# EntryStatus enum
# ═══════════════════════════════════════════════════════════════════

class TestEntryStatus:
    def test_values(self):
        assert EntryStatus.PENDING.value == "pending"
        assert EntryStatus.ACTIVE.value == "active"
        assert EntryStatus.COOLING.value == "cooling"
        assert EntryStatus.DECAYED.value == "decayed"


# ═══════════════════════════════════════════════════════════════════
# EntryStateMachine.transition
# ═══════════════════════════════════════════════════════════════════

class TestEntryStateMachineTransition:
    def test_recent_entry_stays_active(self):
        """最近观察的 active 条目保持 active"""
        now = datetime.now()
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE,
            last_observed=now.isoformat(),
        )
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.ACTIVE

    def test_old_entry_goes_cooling(self):
        """>14 天未观察 → cooling"""
        old = datetime.now() - timedelta(days=20)
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE,
            last_observed=old.isoformat(),
        )
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.COOLING

    def test_older_entry_goes_decayed(self):
        """>30 天未观察 → decayed"""
        old = datetime.now() - timedelta(days=35)
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE,
            last_observed=old.isoformat(),
        )
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.DECAYED

    def test_very_old_entry_decayed(self):
        """>60 天 → decayed（大于 DELETE_THRESHOLD）"""
        old = datetime.now() - timedelta(days=65)
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE,
            last_observed=old.isoformat(),
        )
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.DECAYED

    def test_pending_to_active_with_enough_evidence(self):
        """pending + evidence ≥ 3 → active"""
        now = datetime.now()
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.PENDING, evidence_count=5,
            last_observed=now.isoformat(),
        )
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.ACTIVE

    def test_pending_stays_pending_with_insufficient_evidence(self):
        """pending + evidence < 3 + 最近观察 → 保持 pending"""
        now = datetime.now()
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.PENDING, evidence_count=1,
            last_observed=now.isoformat(),
        )
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.PENDING

    def test_cooling_reactivates_when_recent(self):
        """cooling + 最近观察 → 重新激活为 active"""
        now = datetime.now()
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.COOLING,
            last_observed=now.isoformat(),
        )
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.ACTIVE

    def test_cooling_persists_when_still_old(self):
        """cooling + 仍在14-30天窗口 → 保持 cooling"""
        old = datetime.now() - timedelta(days=20)
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.COOLING,
            last_observed=old.isoformat(),
        )
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.COOLING

    def test_no_last_observed_goes_decayed(self):
        """没有 last_observed → 999天 → decayed"""
        entry = PortraitEntry(id="test-001", dim="usr1", text="test")
        result = EntryStateMachine.transition(entry)
        assert result.status == EntryStatus.DECAYED

    def test_returns_same_entry_object(self):
        entry = PortraitEntry(id="test-001", dim="usr1", text="test")
        result = EntryStateMachine.transition(entry)
        assert result is entry


# ═══════════════════════════════════════════════════════════════════
# EntryStateMachine.confirm_pending
# ═══════════════════════════════════════════════════════════════════

class TestEntryStateMachineConfirmPending:
    def test_confirms_pending_to_active(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.PENDING, confidence=0.50,
        )
        result = EntryStateMachine.confirm_pending(entry)
        assert result.status == EntryStatus.ACTIVE
        assert result.confidence == 0.60  # max(0.50, 0.60) = 0.60

    def test_preserves_higher_confidence(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.PENDING, confidence=0.85,
        )
        result = EntryStateMachine.confirm_pending(entry)
        assert result.confidence == 0.85  # max(0.85, 0.60) = 0.85

    def test_noop_on_active_entry(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE, confidence=0.50,
        )
        result = EntryStateMachine.confirm_pending(entry)
        assert result.status == EntryStatus.ACTIVE
        assert result.confidence == 0.50

    def test_noop_on_cooling_entry(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.COOLING, confidence=0.50,
        )
        result = EntryStateMachine.confirm_pending(entry)
        assert result.status == EntryStatus.COOLING


# ═══════════════════════════════════════════════════════════════════
# EntryStateMachine.mark_cooling
# ═══════════════════════════════════════════════════════════════════

class TestEntryStateMachineMarkCooling:
    def test_marks_cooling(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE, confidence=1.0,
        )
        result = EntryStateMachine.mark_cooling(entry)
        assert result.status == EntryStatus.COOLING
        assert result.confidence == 0.70  # 1.0 * 0.70

    def test_reduces_confidence(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.PENDING, confidence=0.50,
        )
        result = EntryStateMachine.mark_cooling(entry)
        assert result.confidence == 0.35  # 0.50 * 0.70


# ═══════════════════════════════════════════════════════════════════
# EntryStateMachine.decay
# ═══════════════════════════════════════════════════════════════════

class TestEntryStateMachineDecay:
    def test_marks_decayed(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.ACTIVE, confidence=0.80,
        )
        result = EntryStateMachine.decay(entry)
        assert result.status == EntryStatus.DECAYED
        assert result.confidence == 0.0

    def test_zeros_high_confidence(self):
        entry = PortraitEntry(
            id="test-001", dim="usr1", text="test",
            status=EntryStatus.PENDING, confidence=0.95,
        )
        result = EntryStateMachine.decay(entry)
        assert result.confidence == 0.0
