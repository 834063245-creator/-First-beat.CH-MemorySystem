"""链路 4：巩固流水线集成测试。

验证：巩固引擎在真实 ChromaDB 数据上的行为。
BENCHMARK_MODE=true 下 dmn=None，因此手动构建 ConsolidationEngine。
"""
import sys
sys.path.insert(0, ".")

import os
import time
import pytest
from unittest.mock import patch
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# Mock 桩
# ═══════════════════════════════════════════════════════════════

def _mock_extract_tags(text: str, topk: int = 5) -> list[str]:
    tag_map = {
        "Rust": ["Rust", "编程"], "Python": ["Python", "编程"],
        "猫": ["宠物", "猫"], "橘猫": ["宠物", "猫"],
        "东京": ["旅行", "东京"], "大阪": ["旅行", "大阪"],
        "工作": ["工作", "职场"], "压力": ["工作", "压力"],
        "健身": ["健身", "运动"], "跑步": ["健身", "跑步"],
        "妈妈": ["家庭", "妈妈"], "郑州": ["家庭", "郑州"],
        "周杰伦": ["音乐", "周杰伦"], "边牧": ["宠物", "边牧"],
        "微服务": ["微服务", "Docker"], "Docker": ["微服务", "Docker"],
        "年终奖": ["工作", "年终奖"], "失眠": ["生活", "失眠"],
    }
    tags = []
    for kw, tl in tag_map.items():
        if kw in text:
            tags.extend(tl)
    return list(set(tags))[:topk] if tags else ["通用"]


@pytest.fixture
def consolidation_env(isolated_env):
    """构建含 12 条种子记忆 + 手动 ConsolidationEngine 的测试环境。

    BENCHMARK_MODE=true 下 ctx.dmn 为 None，因此手动构建。
    返回 (ctx, dmn, memory_ids)。
    """
    ctx = isolated_env

    # 用 mock 标签写入种子记忆
    with patch("app.brain.semantic.extract_tags", side_effect=_mock_extract_tags):
        seed_data = [
            ("我在学习 Rust 编程", "Rust 很棒", "2026-06-01 10:00:00"),
            ("橘猫又尿闭了", "要注意饮食", "2026-06-01 11:00:00"),
            ("去了东京旅行", "东京很好玩", "2026-06-01 12:00:00"),
            ("工作压力好大", "注意休息", "2026-06-01 13:00:00"),
            ("每周去健身跑步", "坚持运动", "2026-06-01 14:00:00"),
            ("在读黑客与画家", "经典好书", "2026-06-01 15:00:00"),
            ("妈妈要来郑州看我", "家人真好", "2026-06-01 16:00:00"),
            ("周杰伦新专辑", "还是老歌经典", "2026-06-01 17:00:00"),
            ("边牧学会按门铃", "聪明的狗", "2026-06-01 18:00:00"),
            ("在学微服务和Docker", "很强大", "2026-06-01 19:00:00"),
            ("年终奖会被砍吗", "焦虑", "2026-06-01 20:00:00"),
            ("下周去大阪旅行", "大阪美食多", "2026-06-01 21:00:00"),
        ]
        for user, ai, ts in seed_data:
            ctx._store_conversation(user, ai, ts)
            time.sleep(0.05)
        time.sleep(1.0)  # 等待全部入库

    # 手动构建 ConsolidationEngine
    from app.background.consolidation import ConsolidationEngine
    dmn = ConsolidationEngine(
        chroma_service=ctx.chroma_service,
        personality_store=ctx.personality_store,
        behavior_store=ctx.behavior_store,
        chat_history=ctx.chat_history,
        co_tracker=ctx.co_tracker,
        state_path=os.path.join(ctx.data_dir, "dmn_state.json"),
        notes_path=os.path.join(ctx.data_dir, "topic_notes.json"),
        temporal_pattern_index=getattr(ctx, 'temporal_pattern_index', None),
        topic_affinity=getattr(ctx, 'topic_affinity', None),
    )

    all_ids = [m["id"] for m in ctx.chroma_service.list_all()]
    return ctx, dmn, all_ids


# ═══════════════════════════════════════════════════════════════
# 测试类
# ═══════════════════════════════════════════════════════════════

class TestIntConsolidation:
    """验证：巩固引擎在真实数据上的行为。"""

    def test_shallow_consolidation_no_error(self, consolidation_env):
        """12 条种子 → 浅巩固不报错。"""
        ctx, dmn, ids = consolidation_env
        assert len(ids) >= 12, f"应有 ≥12 条种子，实际 {len(ids)}"
        dmn.consolidate_shallow()  # 不应抛异常

    def test_deep_consolidation_no_error(self, consolidation_env):
        """12 条种子 → 深巩固不报错。"""
        ctx, dmn, ids = consolidation_env
        dmn.consolidate_deep()  # 不应抛异常

    def test_consolidation_state_dict(self, consolidation_env):
        """浅巩固后 → get_state_update() 返回 dict 含 topics 键。"""
        ctx, dmn, ids = consolidation_env
        dmn.consolidate_shallow()
        state = dmn.get_state_update()
        assert isinstance(state, dict), "状态应为字典"
        assert "topics" in state, f"状态应含 'topics' 键，实际键: {list(state.keys())}"

    def test_consolidation_writes_state_file(self, consolidation_env):
        """浅巩固后 → dmn_state.json 文件存在且有内容。"""
        ctx, dmn, ids = consolidation_env
        dmn.consolidate_shallow()
        state_path = dmn._state_path
        assert os.path.exists(state_path), f"状态文件应存在: {state_path}"
        import json
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        assert isinstance(state, dict), "状态文件应为 JSON 字典"
        assert len(state) >= 1, "状态文件不应为空"
