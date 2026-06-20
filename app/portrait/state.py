# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: df1c02e1

"""画像条目状态机 — 生命周期管理。

状态转换路径:
  pending  → active → cooling → decayed
     ↑          ↓         ↓
     └──────────┴─────────┘ (重新激活)

规则:
  - pending: 实时层标记"待验证"，等待浅巩固确认
  - active: 确认成立，注入 LLM prompt
  - cooling: 超过14天未观察，不注入 prompt
  - decayed: 超过30天未观察，下次写入时物理删除
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EntryStatus(Enum):
    PENDING = "pending"       # 待验证
    ACTIVE = "active"         # 活跃确认
    COOLING = "cooling"       # 冷却中（>14天）
    DECAYED = "decayed"       # 已衰减（>30天→删除）


@dataclass
class PortraitEntry:
    """画像条目的内存表示。"""
    id: str                                    # e.g. "usr5-001"
    dim: str                                   # e.g. "usr5"
    text: str                                  # 条目正文（纯认知描述，不含元数据）
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    status: EntryStatus = EntryStatus.ACTIVE
    evidence_count: int = 0
    first_observed: str | None = None       # ISO date
    last_observed: str | None = None        # ISO date

    @property
    def days_since_last_observed(self) -> float:
        """自上次观察到现在的天数。"""
        if not self.last_observed:
            return 999.0
        try:
            dt = datetime.fromisoformat(self.last_observed)
            return (datetime.now() - dt).total_seconds() / 86400.0
        except (ValueError, OSError):
            return 999.0

    @property
    def should_inject(self) -> bool:
        """是否应注入 LLM prompt。"""
        return self.status == EntryStatus.ACTIVE and self.confidence >= 0.40


class EntryStateMachine:
    """画像条目状态机 — 驱动 pending → active → cooling → decayed 转换。"""

    COOLING_THRESHOLD_DAYS: float = 14.0
    DECAY_THRESHOLD_DAYS: float = 30.0
    DELETE_THRESHOLD_DAYS: float = 60.0

    @staticmethod
    def transition(entry: PortraitEntry) -> PortraitEntry:
        """根据距上次观察时间更新状态。"""
        days = entry.days_since_last_observed

        if days > EntryStateMachine.DELETE_THRESHOLD_DAYS or days > EntryStateMachine.DECAY_THRESHOLD_DAYS:
            entry.status = EntryStatus.DECAYED
        elif days > EntryStateMachine.COOLING_THRESHOLD_DAYS:
            entry.status = EntryStatus.COOLING
        elif entry.status == EntryStatus.PENDING and entry.evidence_count >= 3:
            entry.status = EntryStatus.ACTIVE
        elif entry.status == EntryStatus.COOLING and days <= EntryStateMachine.COOLING_THRESHOLD_DAYS:
            entry.status = EntryStatus.ACTIVE  # 重新激活

        return entry

    @staticmethod
    def confirm_pending(entry: PortraitEntry) -> PortraitEntry:
        """确认一个 pending 条目。"""
        if entry.status == EntryStatus.PENDING:
            entry.status = EntryStatus.ACTIVE
            entry.confidence = max(entry.confidence, 0.60)
        return entry

    @staticmethod
    def mark_cooling(entry: PortraitEntry) -> PortraitEntry:
        """标记为冷却。"""
        entry.status = EntryStatus.COOLING
        entry.confidence *= 0.70  # 降低置信度
        return entry

    @staticmethod
    def decay(entry: PortraitEntry) -> PortraitEntry:
        """标记为衰减。"""
        entry.status = EntryStatus.DECAYED
        entry.confidence = 0.0
        return entry
