"""链路 3：对话历史流转集成测试。

验证 ChatHistory 组件的文件 IO + 内存操作闭环。
无需 mock（ChatHistory 纯文件 IO，不涉及 LLM）。
"""
import os
import pytest


class TestIntHistoryFlow:
    """验证：对话历史 写入 → 读取 → 上下文关联 → 删除。"""

    def test_append_then_get_recent(self, isolated_env):
        """写入 3 轮对话 → get_recent(3) 返回 3 条。"""
        ctx = isolated_env
        for i in range(3):
            ctx.chat_history.append(
                f"用户消息 {i}", f"AI回复 {i}",
                f"2026-06-01 10:0{i}:00"
            )
        recent = ctx.chat_history.get_recent(n=3)
        assert len(recent) == 3, f"应返回 3 条，实际 {len(recent)}"

    def test_context_by_timestamp(self, isolated_env):
        """写入 5 轮 → 按第 3 轮时间戳查上下文 → 应返回非 None。"""
        ctx = isolated_env
        timestamps = []
        for i in range(5):
            ts = f"2026-06-01 10:0{i}:00"
            ctx.chat_history.append(f"msg_{i}", f"reply_{i}", ts)
            timestamps.append(ts)

        result = ctx.chat_history.get_context_by_timestamp(
            timestamps[2], before=2, after=2
        )
        assert result is not None, "按时间戳应能查到上下文"

    def test_history_persists_to_jsonl(self, isolated_env):
        """写入 1 条 → JSONL 文件存在且有内容。"""
        ctx = isolated_env
        ctx.chat_history.append("测试持久化", "回复", "2026-06-01 10:00:00")

        history_path = os.path.join(ctx.data_dir, "chat_history.jsonl")
        assert os.path.exists(history_path), "chat_history.jsonl 应存在"
        with open(history_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 1, f"JSONL 应至少 1 行，实际 {len(lines)}"

    def test_delete_by_timestamp(self, isolated_env):
        """写入 → 删除 → 查最近不含该条。"""
        ctx = isolated_env
        ts = "2026-06-01 10:00:00"
        ctx.chat_history.append("要删除的消息", "回复", ts)

        deleted = ctx.chat_history.delete_by_timestamp(ts)
        assert deleted is True, "删除应返回 True"

        recent = ctx.chat_history.get_recent(n=10)
        assert all(
            r.get("timestamp") != ts for r in recent
        ), "删除后不应在最近记录中出现"

    def test_update_chroma_id(self, isolated_env):
        """写入 → update_chroma_id → snapshot 中该条含 chroma_id。"""
        ctx = isolated_env
        ts = "2026-06-01 10:00:00"
        ctx.chat_history.append("测试更新", "回复", ts)
        ctx.chat_history.update_chroma_id(ts, "abc123")

        snapshot = ctx.chat_history.get_records_snapshot()
        matching = [r for r in snapshot if r.get("timestamp") == ts]
        assert len(matching) >= 1, "应找到对应记录"
        assert matching[0].get("chroma_id") == "abc123", (
            f"chroma_id 应为 'abc123'，实际: {matching[0].get('chroma_id')}"
        )
