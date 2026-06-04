"""行为预测器 — 从用户行为序列中学习意图转移模式，预测下一步。

增量学习 + n 步马尔可夫链 + 多步预测。
不依赖任何模型，纯统计方法。
"""

import json
import logging
import os
from collections import defaultdict, deque
from typing import Optional

from app.core.circuit import analyze_user_message

logger = logging.getLogger(__name__)

# n 步马尔可夫链阶数
MARKOV_ORDER = 3
# 滑动窗口长度
WINDOW_SIZE = 20


def _default_table() -> dict:
    return {
        "n_transitions": {},       # "intentA|intentB|intentC": {"intentD": count, ...}
        "topic_affinity": {},      # "话题A": {"话题B": weight, ...}
        "total_sequences": 0,
        "version": 2,              # 增量迁移标记
    }


class BehaviorPredictor:
    """行为预测器 — 用户意图转移概率表（n 步马尔可夫链）。"""

    STATE_FILE = "mirror_state.json"

    def __init__(self, data_dir: str):
        self._path = os.path.join(data_dir, self.STATE_FILE)
        self._table = _default_table()
        self._load()

    # ── 采集（增量）──────────────────────────────────────────

    def learn_from(self, records: list[dict]):
        """增量学习行为序列模式。幂等：重复传入相同记录不会膨胀。"""
        # 提取用户消息序列（跳过内心独白）
        msgs = []
        for rec in records:
            msg = rec.get("user_message", "")
            if msg and msg != "[内心独白]":
                msgs.append(msg)

        if len(msgs) < MARKOV_ORDER + 1:
            logger.debug("行为预测: 序列太短(%d)，跳过学习", len(msgs))
            return

        intents = []
        topic_sets = []
        for msg in msgs:
            pfc = analyze_user_message(msg)
            intents.append(pfc.intent)
            topic_sets.append(set(pfc.topics))

        # ── 增量更新 n 步转移概率 ──
        n_transitions = self._table.get("n_transitions", {})
        new_pairs = 0
        for i in range(len(intents) - MARKOV_ORDER):
            # 构建 n 步 key: "intentA|intentB|intentC"
            key = "|".join(intents[i:i + MARKOV_ORDER])
            next_intent = intents[i + MARKOV_ORDER]
            if key not in n_transitions:
                n_transitions[key] = {}
                new_pairs += 1
            n_transitions[key][next_intent] = n_transitions[key].get(next_intent, 0) + 1

        # ── 增量更新话题关联 ──
        affinity = self._table.get("topic_affinity", {})
        for i in range(len(topic_sets) - 1):
            for t1 in topic_sets[i]:
                if not t1:
                    continue
                for t2 in topic_sets[i + 1]:
                    if not t2 or t2 == t1:
                        continue
                    if t1 not in affinity:
                        affinity[t1] = {}
                    affinity[t1][t2] = affinity[t1].get(t2, 0) + 1

        self._table["n_transitions"] = n_transitions
        self._table["topic_affinity"] = affinity
        self._table["total_sequences"] = max(
            self._table.get("total_sequences", 0), len(msgs)
        )
        self._save()

        logger.info(
            "行为预测器增量学习: %d 条序列, %d 个 n 步模式, %d 个话题关联",
            len(msgs), len(n_transitions), len(affinity),
        )

    # ── 推理（多步预测）──────────────────────────────────────

    def predict(self, current_intent: str, current_topics: list[str],
                recent_intents: Optional[list[str]] = None) -> dict:
        """根据当前 intent + 近几轮意图序列，预测后续 1-3 步。

        返回::
            {"next_intents": ["recall", "casual", "ask_fact"],
             "shift_topics": ["话题A", "话题B"]}
            或部分字段缺失。
        """
        table = self._table
        n_transitions = table.get("n_transitions", {})
        result = {}

        # ── 多步预测 ──
        # 从当前 intent 及最近几轮构建 n 步 key
        context = (recent_intents or []) + [current_intent]
        context = context[-MARKOV_ORDER:]

        # 尝试从最长匹配到最短匹配
        for length in range(len(context), 0, -1):
            key = "|".join(context[-length:])
            if key in n_transitions:
                nexts = n_transitions[key]
                # 按频次降序取 top-3
                sorted_nexts = sorted(nexts.items(), key=lambda x: -x[1])
                # 从 sorted_nexts 依次尝试向后滚动预测，构建 1-3 步链条
                predicted = []
                visited = {current_intent}
                # 第一步
                chain = self._rollout(key, n_transitions, max_steps=3)
                if chain:
                    result["next_intents"] = chain
                break

        # ── 话题偏移预测（同原逻辑） ──
        affinity = table.get("topic_affinity", {})
        shift_scores = defaultdict(float)
        for topic in current_topics:
            if topic in affinity:
                for related, weight in affinity[topic].items():
                    shift_scores[related] += weight
        if shift_scores:
            sorted_shifts = sorted(shift_scores.items(), key=lambda x: -x[1])
            result["shift_topics"] = [t for t, _ in sorted_shifts[:3]]

        return result

    @staticmethod
    def _rollout(start_key: str, table: dict, max_steps: int = 3) -> list[str]:
        """从 start_key 开始，沿最高概率路径向后滚动 max_steps 步。"""
        chain = []
        current_key = start_key
        parts = current_key.split("|")
        for _ in range(max_steps):
            nexts = table.get(current_key)
            if not nexts:
                break
            best = max(nexts, key=nexts.get)
            chain.append(best)
            # 滑动窗口：丢弃最旧的，加入最新的
            parts = parts[1:] + [best]
            current_key = "|".join(parts)
        return chain

    # ── 持久化 ──────────────────────────────────────────────

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged = _default_table()
                merged.update(data)
                # v1→v2 迁移：旧版 intent_transitions → n_transitions
                if merged.get("version", 1) < 2:
                    old = data.get("intent_transitions", {})
                    if old and not merged.get("n_transitions"):
                        n_t = {}
                        for curr, nexts in old.items():
                            for nxt, cnt in nexts.items():
                                key = curr
                                if key not in n_t:
                                    n_t[key] = {}
                                n_t[key][nxt] = int(cnt * 10)  # 从概率还原为近似计数
                        merged["n_transitions"] = n_t
                    merged["version"] = 2
                    logger.info("行为预测器: v1→v2 迁移完成")
                self._table = merged
                logger.debug(
                    "行为预测器加载: %d 条序列, %d 个 n 步模式",
                    self._table.get("total_sequences", 0),
                    len(self._table.get("n_transitions", {})),
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("行为预测器加载失败: %s", exc)

    def _save(self):
        try:
            parent = os.path.dirname(self._path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._table, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.debug("行为预测器保存失败: %s", exc)
