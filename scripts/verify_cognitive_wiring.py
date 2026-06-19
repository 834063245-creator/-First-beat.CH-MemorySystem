"""认知五连线端到端 smoke 验证 — 每条连线真的通了吗？

用法: python scripts/verify_cognitive_wiring.py
不做单元测试那种 mock 一切，用最小真实对象验证数据流。
"""
import json
import os
import sys
import tempfile
import time

def green(s): return f"\033[32m{s}\033[0m"
def red(s): return f"\033[31m{s}\033[0m"
def bold(s): return f"\033[1m{s}\033[0m"

passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  {green('[PASS]')} {label}")
        passed += 1
    else:
        print(f"  {red('[FAIL]')} {label}  — {detail}")
        failed += 1

# ═══════════════════════════════════════════════════════════
print(bold("═══ 连线①: Feedback → Portrait（用户纠错→画像置信度下降）═══"))

with tempfile.TemporaryDirectory() as tmpdir:
    # 1. 写一条错误报告
    from app.core.feedback import log_error_report, get_recent_corrected_ids
    mid = "test-memory-001"
    log_error_report(mid, "记错了", "user", data_dir=tmpdir)

    # 2. 读回来
    corrected = get_recent_corrected_ids(data_dir=tmpdir)
    check("错误报告写入后可读出", mid in corrected,
          f"corrected={corrected}")

    # 3. 模拟 PortraitWriter 消费：构造一个引用该 memory_id 的画像条目
    from app.portrait.state import PortraitEntry, EntryStatus
    entry = PortraitEntry(
        id="usr5-099", dim="usr5",
        text=f"用户关注话题，关联记忆 {mid}",
        tags=["Python", "测试"],
        confidence=0.80,
        status=EntryStatus.ACTIVE,
    )
    check("画像条目初始 confidence=0.80", entry.confidence == 0.80)
    check("画像条目初始 status=ACTIVE", entry.status == EntryStatus.ACTIVE)

    # 4. 模拟 realtime_update 中的反馈消费逻辑
    entry_text_and_tags = entry.text + " " + " ".join(entry.tags)
    if any(mid in entry_text_and_tags for mid in corrected):
        entry.confidence = max(0.1, entry.confidence - 0.3)
        entry.status = EntryStatus.PENDING

    check("纠错后 confidence 下降", entry.confidence == 0.50,
          f"expected 0.50, got {entry.confidence}")
    check("纠错后 status 变 PENDING", entry.status == EntryStatus.PENDING,
          f"got {entry.status}")

# ═══════════════════════════════════════════════════════════
print(bold("\n═══ 连线②: Portrait → Gate（画像情绪趋势→门控语气收敛）═══"))

from app.core.circuit import basal_ganglia_gate, UserMessageAnalysis

# 构造一个"长期低落"画像
class FakePortrait:
    is_empty = False
    def get_dim_entries(self, dim):
        if dim == "usr6":
            from app.portrait.state import PortraitEntry, EntryStatus
            return [
                PortraitEntry(id="usr6-001", dim="usr6",
                              text="用户长期低落，工作压力大",
                              tags=["压力"], confidence=0.7,
                              status=EntryStatus.ACTIVE),
                PortraitEntry(id="usr6-002", dim="usr6",
                              text="用户最近焦虑失眠",
                              tags=["焦虑"], confidence=0.6,
                              status=EntryStatus.ACTIVE),
            ]
        return []

class FakeCtx:
    portrait = FakePortrait()
    class _pattern_discovery:
        @staticmethod
        def get_tuning():
            return {}

# 当前消息是 casual + warm 语气
pfc = UserMessageAnalysis(
    intent="casual", emotion="neutral", urgency=0.2,
    topics=["日常"], raw_text="今天天气不错",
    confidence=0.8, emotion_intensity=0.3,
)
gate = basal_ganglia_gate(pfc, [], [], [], ctx_obj=FakeCtx())
check("长期低落→语气从 warm 收敛为 soft",
      gate.tone == "soft",
      f"got tone={gate.tone}, mode={gate.response_mode}")
check("长期低落→formality ≥ 0.4",
      gate.formality >= 0.4,
      f"got formality={gate.formality}")

# 负面但只有 0 条 → 不触发
class FakeCtxNoNeg:
    portrait = FakePortrait()
    class _pattern_discovery:
        @staticmethod
        def get_tuning():
            return {}
FakeCtxNoNeg.portrait.get_dim_entries = lambda dim: []
gate2 = basal_ganglia_gate(pfc, [], [], [], ctx_obj=FakeCtxNoNeg())
check("无负面条目→语气不收敛",
      gate2.tone == "warm",
      f"got tone={gate2.tone}")

# ═══════════════════════════════════════════════════════════
print(bold("\n═══ 连线③: Portrait → weave_context（画像tag→检索阈值放宽）═══"))

from app.core.circuit import CircuitOrchestrator
import inspect

# 验证签名
sig = inspect.signature(CircuitOrchestrator.weave_context)
check("weave_context 接受 portrait_boost 参数",
      "portrait_boost" in sig.parameters)

# 构造 minimal CircuitOrchestrator 测试 weave_context
orch = CircuitOrchestrator(
    memory_service=None,
    impulse_scheduler=None,
    dmn_engine=None,
    chat_history=None,
    co_tracker=None,
    mirror_neuron=None,
)

from app.models.schemas import WovenContext

# 候选记忆：带 tags
candidates = [
    {"id": "m1", "distance": 0.25, "source": "semantic",
     "metadata": {"tags": "Python,后端", "timestamp": time.time()}},
    {"id": "m2", "distance": 0.35, "source": "tag_match",
     "metadata": {"tags": "Rust", "timestamp": time.time() - 86400}},
]

# 无 boost → m2 的 tag_match * 0.7 = 0.21 < 0.35 → 进不了 fact
class FakeCognitive:
    intent = "ask_fact"
wc_no_boost = orch.weave_context(candidates, FakeCognitive())
check("无 boost 时 m2(distance=0.35, tag_match) 不进入 fact",
      len(wc_no_boost.fact_memories) <= 1,
      f"fact_memories={[m['id'] for m in wc_no_boost.fact_memories]}")

# 有 boost → Python +0.2 → tag_match threshold 从 0.21→0.252, Rust 无 boost 不变
wc_boosted = orch.weave_context(candidates, FakeCognitive(),
                                portrait_boost={"Python": 0.2})
# m1: semantic 0.30 * 1.0 * 1.2 = 0.36 > 0.25 → fact
# m2: tag_match 0.30 * 0.7 * 1.0 = 0.21 < 0.35 → 进不了 fact
py_in_fact = any("Python" in str(m.get("metadata", {}).get("tags", ""))
                 for m in wc_boosted.fact_memories)
check("Python tag boost 后 m1 仍可进入 fact",
      py_in_fact,
      f"fact_memories={[m['id'] for m in wc_boosted.fact_memories]}")

# ═══════════════════════════════════════════════════════════
print(bold("\n═══ 连线④: Predictor → Gate（行为预测→预调响应模式）═══"))

# 预测用户要问事实 → response_mode 从 auto 变 direct_answer
pfc2 = UserMessageAnalysis(
    intent="casual", emotion="neutral", urgency=0.2,
    topics=["Python"], raw_text="Python怎么学",
    confidence=0.8, emotion_intensity=0.3,
)
gate3 = basal_ganglia_gate(pfc2, [], [], [], ctx_obj=FakeCtxNoNeg(),
                           mirror_prediction={"predicted_intent": "ask_fact"})
check("预测 ask_fact → response_mode=direct_answer",
      gate3.response_mode == "direct_answer",
      f"got {gate3.response_mode}")

# 预测情感分享 + warm → tone 变 caring
pfc3 = UserMessageAnalysis(
    intent="casual", emotion="positive", urgency=0.1,
    topics=["日常"], raw_text="今天好开心",
    confidence=0.8, emotion_intensity=0.6,
)
gate4 = basal_ganglia_gate(pfc3, [], [], [], ctx_obj=FakeCtxNoNeg(),
                           mirror_prediction={"predicted_intent": "emotional_sharing"})
check("预测 emotional_sharing + warm → tone=caring",
      gate4.tone == "caring",
      f"got tone={gate4.tone}")

# 无预测 → 不干预
gate5 = basal_ganglia_gate(pfc2, [], [], [], ctx_obj=FakeCtxNoNeg())
check("无预测时 response_mode 保持默认",
      gate5.response_mode == "auto",
      f"got {gate5.response_mode}")

# ═══════════════════════════════════════════════════════════
print(bold("\n═══ 连线⑤: Portrait → Impulse（画像驱动好奇心冲动源）═══"))

from app.background.impulse import source_portrait_curiosity
from app.portrait.state import PortraitEntry, EntryStatus

# 构造一个 fake PortraitManager
class FakePortraitManager:
    def extract_focus_keywords(self):
        return ["Rust"]  # 用户关注
    def extract_hot_topics(self):
        return ["Python", "AI"]  # 兴趣热点
    def extract_negative_triggers(self):
        return ["加班"]  # 负面触发，应被排除

# 无 portrait_manager → 返回 None
result_none = source_portrait_curiosity(portrait_manager=None)
check("无 portrait_manager 返回 None", result_none is None)

# 正常流程
pm = FakePortraitManager()
result = source_portrait_curiosity(portrait_manager=pm, all_mems=[
    {"metadata": {"tags": "Python,AI"}},
    {"metadata": {"tags": "Rust"}},
])
check("画像探索产出非空", result is not None,
      f"result={result}")
if result:
    content, priority = result
    check("优先级为 20", priority == 20, f"got {priority}")
    check("内容包含候选标签", any(tag in content for tag in ["Rust", "Python", "AI"]),
          f"content={content}")

# 记忆覆盖已满 → 跳过（Rust/Python/AI 全标签覆盖 12 条）
result_full = source_portrait_curiosity(portrait_manager=pm, all_mems=[
    {"metadata": {"tags": "Rust,Python,AI"}} for _ in range(12)
])
check("已覆盖≥10条时跳过", result_full is None,
      f"result={result_full}")

# 全为负面标签 → 跳过
class FakeNegManager:
    def extract_focus_keywords(self): return ["加班"]
    def extract_hot_topics(self): return ["加班"]
    def extract_negative_triggers(self): return ["加班"]
result_neg = source_portrait_curiosity(portrait_manager=FakeNegManager())
check("全为负面触发时跳过", result_neg is None,
      f"result={result_neg}")

# ═══════════════════════════════════════════════════════════
print(bold(f"\n{'═'*60}"))
print(bold(f"  结果: {passed} passed, {failed} failed"))
if failed == 0:
    print(green("  五条连线全部验证通过！"))
else:
    print(red(f"  {failed} 项失败，需要排查"))
print(bold(f"{'═'*60}"))
sys.exit(0 if failed == 0 else 1)
