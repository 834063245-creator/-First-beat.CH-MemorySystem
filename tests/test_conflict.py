# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 932e4254

"""测试 app/core/conflict.py — 记忆冲突用户消解。

覆盖：check_resolution 的确认信号检测、消解有效期检查、stale 标记执行。
"""
import json
import os
import time
import tempfile
from unittest.mock import MagicMock, patch
from app.core.conflict import check_resolution, _CONFIRM_SIGNALS, _RESOLUTION_TTL


class TestCheckResolution:

    def test_returns_none_when_no_pending_conflicts(self):
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        result = check_resolution("我记错了", [], mock_chroma, mock_co)
        assert result is None

    def test_returns_none_when_no_signal(self):
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        conflicts = [{
            "shared_tags": ["咖啡"],
            "old_id_full": "mem_old_001",
            "new_id_full": "mem_new_001",
            "old_summary": "喜欢喝咖啡",
            "new_summary": "咖啡戒了",
            "detected_at": time.time(),
        }]
        result = check_resolution("今天天气不错", conflicts, mock_chroma, mock_co)
        assert result is None

    def test_returns_none_when_tag_not_in_message(self):
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        conflicts = [{
            "shared_tags": ["咖啡"],
            "old_id_full": "mem_old_001",
            "new_id_full": "mem_new_001",
            "old_summary": "喜欢喝咖啡",
            "new_summary": "咖啡戒了",
            "detected_at": time.time(),
        }]
        # 有确认信号但话题不匹配
        result = check_resolution("对了，我今天要写代码", conflicts, mock_chroma, mock_co)
        assert result is None

    def test_resolves_when_signal_and_tag_match(self):
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        conflicts = [{
            "shared_tags": ["咖啡"],
            "old_id_full": "mem_old_001",
            "new_id_full": "mem_new_001",
            "old_summary": "喜欢喝咖啡",
            "new_summary": "咖啡戒了",
            "detected_at": time.time(),
        }]
        result = check_resolution("对，我确实把咖啡戒了", conflicts, mock_chroma, mock_co)
        assert result is not None
        assert result["tag"] == "咖啡"
        assert result["old_id"] == "mem_old_001"
        # 验证 chroma 被调用了 update
        mock_chroma._collection.update.assert_called_once()
        call_args = mock_chroma._collection.update.call_args
        assert call_args[1]["ids"] == ["mem_old_001"]
        assert call_args[1]["metadatas"] == [{"stale": True}]
        # 验证 co_tracker.remove 被调用
        mock_co.remove.assert_called_once_with("mem_old_001")

    def test_skips_expired_conflicts(self):
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        old_ts = time.time() - _RESOLUTION_TTL - 100  # 已过期
        conflicts = [{
            "shared_tags": ["咖啡"],
            "old_id_full": "mem_old_001",
            "new_id_full": "mem_new_001",
            "old_summary": "喜欢喝咖啡",
            "new_summary": "咖啡戒了",
            "detected_at": old_ts,
        }]
        result = check_resolution("我确实把咖啡戒了", conflicts, mock_chroma, mock_co)
        assert result is None

    def test_skips_when_no_shared_tags(self):
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        conflicts = [{
            "shared_tags": [],
            "old_id_full": "mem_old_001",
            "new_id_full": "mem_new_001",
            "detected_at": time.time(),
        }]
        result = check_resolution("对了，我改了", conflicts, mock_chroma, mock_co)
        assert result is None

    def test_uses_old_id_fallback(self):
        """当没有 old_id_full 时回退到 old_id。"""
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        conflicts = [{
            "shared_tags": ["运动"],
            "old_id": "mem_old_002",
            "new_id": "mem_new_002",
            "old_summary": "每天跑步",
            "new_summary": "改为游泳",
            "detected_at": time.time(),
        }]
        result = check_resolution("没错，我现在不跑步改游泳了，运动习惯变了", conflicts, mock_chroma, mock_co)
        assert result is not None
        mock_chroma._collection.update.assert_called_once_with(
            ids=["mem_old_002"],
            metadatas=[{"stale": True}],
        )

    def test_returns_none_when_no_old_id(self):
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        conflicts = [{
            "shared_tags": ["运动"],
            "old_summary": "每天跑步",
            "detected_at": time.time(),
        }]
        result = check_resolution("没错，我改了", conflicts, mock_chroma, mock_co)
        assert result is None

    def test_chroma_update_failure_continues(self):
        """Chroma 写入失败不应抛异常，继续处理下一个冲突。"""
        mock_chroma = MagicMock()
        mock_chroma._collection.update.side_effect = Exception("DB error")
        mock_co = MagicMock()
        conflicts = [{
            "shared_tags": ["咖啡"],
            "old_id_full": "mem_old_001",
            "new_id_full": "mem_new_001",
            "old_summary": "喜欢喝咖啡",
            "new_summary": "咖啡戒了",
            "detected_at": time.time(),
        }]
        # 不应抛异常
        result = check_resolution("对，咖啡戒了", conflicts, mock_chroma, mock_co)
        # Chroma 写入失败，仍尝试了 co_tracker.remove
        # 因为 continue 跳过 return，最终无匹配会返回 None
        # 但如果只有这一个 conflict 且 chroma 失败，会 return None
        assert result is None

    def test_multiple_conflicts_resolves_first_match_only(self):
        """只消解第一个匹配的冲突。"""
        mock_chroma = MagicMock()
        mock_co = MagicMock()
        now = time.time()
        conflicts = [
            {
                "shared_tags": ["咖啡"],
                "old_id_full": "mem_old_001",
                "new_id_full": "mem_new_001",
                "old_summary": "喜欢喝咖啡",
                "new_summary": "咖啡戒了",
                "detected_at": now,
            },
            {
                "shared_tags": ["运动"],
                "old_id_full": "mem_old_002",
                "new_id_full": "mem_new_002",
                "old_summary": "跑步",
                "new_summary": "游泳",
                "detected_at": now,
            },
        ]
        # 消息同时包含两个话题
        result = check_resolution("对了，咖啡我戒了，运动也换成游泳了", conflicts, mock_chroma, mock_co)
        assert result is not None
        # 只消解了第一个
        assert result["tag"] == "咖啡"
        assert result["old_id"] == "mem_old_001"
        mock_chroma._collection.update.assert_called_once()
