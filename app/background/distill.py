"""蒸馏引擎 — 算法级模式提取，不调用 LLM。

从记忆的标签/时间/情绪/内容中提取用户画像标签。
纯统计方法：标签共现聚类 → 时间模式检测 → 情绪关联分析 → 趋势分析。
"""

import json
import logging
import math
import os
import threading
from collections import Counter, defaultdict
from datetime import datetime

from app.tools.atomic import atomic_write

logger = logging.getLogger(__name__)


def _read_state(state_path: str) -> dict:
    path = state_path
    if not os.path.exists(path):
        return {"last_distill_timestamp": None, "total_distill_runs": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"last_distill_timestamp": None, "total_distill_runs": 0}


def _write_state(state: dict, state_path: str):
    atomic_write(state_path, state)


# 时段名映射
_PERIOD_NAMES = {
    0: "深夜", 1: "深夜", 2: "深夜", 3: "深夜", 4: "深夜", 5: "深夜",
    6: "早晨", 7: "早晨", 8: "早晨",
    9: "上午", 10: "上午", 11: "上午",
    12: "中午", 13: "中午",
    14: "下午", 15: "下午", 16: "下午", 17: "下午",
    18: "傍晚", 19: "傍晚", 20: "傍晚",
    21: "晚上", 22: "晚上", 23: "晚上",
}

_WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# ── 关键词缓存（避免重复 jieba） ──
_keyword_cache: dict[str, list[str]] = {}
_kw_cache_lock = threading.Lock()


def _extract_keywords(text: str, topk: int = 5) -> list[str]:
    """从文本中提取关键词，带缓存。"""
    if not text:
        return []
    with _kw_cache_lock:
        if text in _keyword_cache:
            return _keyword_cache[text]
    try:
        from app.brain.semantic import extract_tags
        kw = extract_tags(text, topk=topk)
        with _kw_cache_lock:
            _keyword_cache[text] = kw
        return kw
    except Exception:
        return []


def _recency_score(last_ts: float, now_ts: float) -> float:
    """距今越近分数越高。7 天内 1.0，30 天以上 0.0，中间线性衰减。"""
    days_ago = (now_ts - last_ts) / 86400
    if days_ago <= 7:
        return 1.0
    if days_ago >= 30:
        return 0.0
    return 1.0 - (days_ago - 7) / 23


def _compute_confidence(count: int, days_span: int, recency: float,
                        has_emotion: bool, kw_diversity: int) -> tuple[float, str]:
    """连续置信度计算，映射到三档。"""
    norm_count = min(count / 10, 1.0)
    norm_days = min(days_span / 14, 1.0)
    score = (
        norm_count * 0.30
        + norm_days * 0.25
        + recency * 0.20
        + (0.15 if has_emotion else 0.0)
        + min(kw_diversity / 5, 1.0) * 0.10
    )
    score = max(0.0, min(1.0, score))
    if score >= 0.70:
        label = "高"
    elif score >= 0.40:
        label = "中"
    else:
        label = "低"
    return score, label


def _generate_content(pattern_type: str, tag: str, period: str, days_span: int,
                     count: int, valence_dist: dict, top_kws: list[str],
                     who_prefix: str = "用户") -> str:
    """根据模式类型和特征生成可读的内容文本。"""
    if pattern_type == "周期性行为":
        extra = ""
        if top_kws:
            extra = f"，常涉及{'、'.join(top_kws[:3])}"
        return f"{who_prefix}习惯在{period}聊{tag}{extra}"
    elif pattern_type == "情绪波动":
        pos = valence_dist.get("positive", 0)
        neg = valence_dist.get("negative", 0)
        return f"{who_prefix}聊{tag}时情绪波动较大（正向{pos}次/负向{neg}次）"
    elif pattern_type == "临时热点":
        return f"{who_prefix}近期密集关注{tag}（{count}条/{days_span}天）"
    elif pattern_type == "稳定兴趣":
        extra = ""
        if top_kws:
            extra = f"，关联{'、'.join(top_kws[:3])}"
        return f"{who_prefix}长期关注{tag}（{days_span}天/{count}条）{extra}"
    elif pattern_type == "情绪关联":
        dominant = max(valence_dist, key=valence_dist.get) if valence_dist else "中性"
        return f"{who_prefix}聊{tag}时情绪偏{dominant}"
    else:
        return f"{who_prefix}多次提到{tag}"


# ── 新增标签聚合参数 ──
_MIN_OCCURRENCES = 2
"""单个标签最少出现次数才进入模式提取。"""

_MAX_PATTERNS = 15


def _extract_patterns(memories: list[dict], source: str = "user") -> list[dict]:
    """从记忆中提取行为/偏好/情绪/兴趣模式，不调用任何 LLM。

    改进策略（vs V3 纯 tag 统计）：
      1. 多维特征提取：时间分布 + 情绪分布 + 参与度 + 关键词签名
      2. 跨标签关联分析：共现标签组合形成复合模式
      3. 趋势检测：话题是增长还是衰减
      4. 连续置信度：综合多维度打分
    """
    if not memories:
        return []

    # 按标签聚合记忆
    tag_groups: dict[str, list[dict]] = defaultdict(list)
    for m in memories:
        meta = m.get("metadata") or {}
        tags_str = meta.get("tags", "") or ""
        for t in tags_str.split(","):
            t = t.strip()
            if len(t) >= 2:
                tag_groups[t].append(m)

    # 标签共现矩阵
    co_occur: dict[str, Counter] = defaultdict(Counter)
    for m in memories:
        tags = [t.strip() for t in (m.get("metadata") or {}).get("tags", "").split(",") if len(t.strip()) >= 2]
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                co_occur[t1][t2] += 1
                co_occur[t2][t1] += 1

    now_ts = datetime.now().timestamp()
    patterns = []

    for tag, group in tag_groups.items():
        if len(group) < _MIN_OCCURRENCES:
            continue

        # ── 多维特征提取 ──
        hours = []
        weekdays = []
        valences = Counter()
        emotional_count = 0
        timestamps = []
        summaries = []

        for m in group:
            meta = m.get("metadata") or {}
            ts = meta.get("timestamp", 0)
            if ts:
                try:
                    dt = datetime.fromtimestamp(ts)
                    hours.append(dt.hour)
                    weekdays.append(dt.weekday())
                    timestamps.append(ts)
                except (OSError, ValueError):
                    pass
            valence = meta.get("emotion_valence_bin", "") or ""
            if valence in ("positive", "negative"):
                valences[valence] += 1
            if (meta.get("emotional_intensity", 0) or 0) >= 1:
                emotional_count += 1
            summary = meta.get("summary", "") or ""
            if summary:
                summaries.append(summary)

        if not timestamps:
            continue

        # 时间分布
        period_counts = Counter()
        for h in hours:
            period_counts[_PERIOD_NAMES.get(h, "其他")] += 1
        dominant_period, period_freq = period_counts.most_common(1)[0]

        # 跨天跨度
        days = set()
        for ts in timestamps:
            try:
                days.add(datetime.fromtimestamp(ts).date().isoformat())
            except (OSError, ValueError):
                pass
        day_span = len(days)

        # 周几分布（看是否有周期性）
        weekday_dist = Counter()
        for wd in weekdays:
            weekday_dist[_WEEKDAY_NAMES[wd]] += 1

        # 近期活跃度
        last_ts = max(timestamps)
        recency = _recency_score(last_ts, now_ts)

        # 关键词签名
        all_text = " ".join(summaries)
        top_kws = _extract_keywords(all_text, topk=5) if all_text else []

        # 密度（条数/天，越高越可能是热点）
        density = len(group) / max(day_span, 1)

        # 情绪关联强度
        dominant_valence = max(valences, key=valences.get) if valences else None
        has_strong_emotion = emotional_count >= 3 and dominant_valence is not None
        has_volatility = valences.get("positive", 0) >= 2 and valences.get("negative", 0) >= 2

        # ── 模式分类 ──
        pattern_type = "偏好模式"
        content = f"用户多次提到{tag}"

        # 检查：是否是"周期性行为"（集中在某时段 + 跨多天）
        is_periodic = day_span >= 2 and period_freq / max(len(hours), 1) > 0.4
        # 检查：是否是"稳定兴趣"（跨多天 + 持续出现）
        is_sustained = day_span >= 4 and recency > 0.5
        # 检查：是否是"临时热点"（密度高 + 近期活跃）
        is_hotspot = density >= 0.8 and recency >= 0.8
        # 检查：是否有"情绪波动"
        is_volatile = has_volatility
        # 检查：是否有"情绪关联"
        is_emotional = has_strong_emotion

        if is_volatile:
            pattern_type = "情绪波动"
        elif is_hotspot:
            pattern_type = "临时热点"
        elif is_periodic and day_span >= 3:
            pattern_type = "周期性行为"
        elif is_sustained:
            pattern_type = "稳定兴趣"
        elif is_emotional:
            pattern_type = "情绪关联"
        elif day_span >= 2:
            pattern_type = "偏好模式"

        who_prefix = "AI" if source == "ai" else "用户"
        content = _generate_content(
            pattern_type, tag, dominant_period, day_span, len(group),
            dict(valences), top_kws, who_prefix=who_prefix,
        )

        conf_score, conf_label = _compute_confidence(
            len(group), day_span, recency,
            has_emotion=(has_strong_emotion or has_volatility),
            kw_diversity=len(top_kws),
        )

        patterns.append({
            "content": content,
            "type": pattern_type,
            "confidence": conf_label,
            "confidence_score": round(conf_score, 3),
            "status": "新增",
            "evidence": {
                "count": len(group),
                "days_span": day_span,
                "recency_days": round((now_ts - last_ts) / 86400, 1),
                "peak_period": dominant_period,
                "keywords": top_kws,
                "density": round(density, 2),
            },
        })

    # ── 跨标签复合模式 ──
    # 找出两两共现 >= 3 次且各自都达到阈值的标签对
    compound_patterns = []
    compound_seen = set()
    for t1, counter in co_occur.items():
        for t2, count in counter.items():
            pair_key = "|".join(sorted([t1, t2]))
            if pair_key in compound_seen:
                continue
            if count < 5:
                continue
            g1 = tag_groups.get(t1, [])
            g2 = tag_groups.get(t2, [])
            if len(g1) < 2 or len(g2) < 2:
                continue
            # 合并两组时间戳
            all_ts = []
            for g in g1 + g2:
                ts = (g.get("metadata") or {}).get("timestamp", 0)
                if ts:
                    all_ts.append(ts)
            if not all_ts:
                continue
            day_span = len(set(
                datetime.fromtimestamp(ts).date().isoformat()
                for ts in all_ts if ts
            ))
            conf_score, conf_label = _compute_confidence(
                len(g1) + len(g2), day_span,
                _recency_score(max(all_ts), now_ts),
                has_emotion=False, kw_diversity=3,
            )
            content = f"用户经常同时聊{t1}和{t2}（{count}次共现）"
            compound_patterns.append({
                "content": content,
                "type": "兴趣领域",
                "confidence": conf_label,
                "confidence_score": round(conf_score, 3),
                "status": "新增",
                "evidence": {
                    "co_occur_count": count,
                    "tags": [t1, t2],
                },
            })
            compound_seen.add(pair_key)

    patterns.extend(compound_patterns)

    # 排序+截断
    patterns.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
    return patterns[:_MAX_PATTERNS]


class DistillEngine:
    """蒸馏引擎：从记忆中提取画像标签，纯算法，不调用 LLM。"""

    @staticmethod
    def _cleanup_junk_patterns(store):
        """清理所有'用户经常同时聊'开头的共现类人格标签。"""
        try:
            all_tags = store.list_all() or []
            junk_count = 0
            for tag in all_tags:
                meta = tag.get("metadata") or {}
                content = meta.get("content", "")
                if content.startswith("用户经常同时聊"):
                    store.delete(tag["id"])
                    junk_count += 1
            if junk_count:
                logger.info("已清理 %d 条共现类人格标签", junk_count)
        except Exception as exc:
            logger.warning("共现类标签清理失败: %s", exc)

    def __init__(self, personality_store, chroma_service, behavior_store=None, *,
                 state_path: str, source: str = "user"):
        self._state_path = state_path
        self._store = personality_store
        self._behavior_store = behavior_store
        self._chroma = chroma_service
        self._source = source

    def run_distill(self, force_all: bool = False, existing_tags: list[dict] = None) -> dict:
        """完整蒸馏流程：增量记忆 → 算法提取 → 去重写入人格库。

        不再调用任何 LLM。纯统计方法。
        """
        try:
            state = _read_state(self._state_path)
            last_ts = state.get("last_distill_timestamp")

            all_memories = self._chroma.list_all_cached()
            if force_all or not last_ts:
                if force_all:
                    self._cleanup_junk_patterns(self._store)
                pending = all_memories
            else:
                try:
                    last_dt = datetime.fromisoformat(last_ts)
                    last_float = last_dt.timestamp()
                except (ValueError, TypeError):
                    last_float = 0
                pending = [
                    m for m in all_memories
                    if m.get("metadata", {}).get("timestamp", 0) > last_float
                ]

            if not pending:
                return {"status": "skipped", "reason": "no_new_memories"}

            logger.info("蒸馏: %d 条待处理记忆（算法模式提取）", len(pending))

            tags = _extract_patterns(pending, source=self._source)

            if not tags:
                return {"status": "skipped", "reason": "no_patterns_detected"}

            new_count, updated_count = self._merge_to_store(tags)

            state["last_distill_timestamp"] = datetime.now().isoformat()
            state["total_distill_runs"] = state.get("total_distill_runs", 0) + 1
            _write_state(state, self._state_path)

            logger.info(
                "蒸馏完成: 新增=%d 更新=%d, 最高置信度=%.3f",
                new_count, updated_count,
                tags[0].get("confidence_score", 0) if tags else 0,
            )
            return {"status": "done", "new_tags": new_count, "updated_tags": updated_count}
        except Exception as exc:
            logger.error("蒸馏流程失败: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

    def _merge_to_store(self, tags: list[dict]) -> tuple[int, int]:
        """将蒸馏结果合并到人格库。返回 (new_count, updated_count)。"""
        new_count = 0
        updated_count = 0
        from app.llm.embed import local_embed

        for tag in tags:
            content = tag.get("content", "").strip()
            if not content:
                continue
            tag_type = tag.get("type", "行为模式")
            confidence = tag.get("confidence", "低")

            embedding = local_embed(content)
            if embedding is None:
                continue

            # 去重检查
            existing = self._store.search(embedding, top_k=1)
            if existing and (1 - existing[0].get("distance", 1)) >= 0.85:
                self._store.increment_hit(existing[0]["id"])
                updated_count += 1
                logger.debug("蒸馏去重命中: %s → %s", content[:30], existing[0].get("content", "")[:30])
            else:
                self._store.store_tag(content, embedding,
                    tag_type=tag_type, confidence=confidence,
                    source=self._source)
                new_count += 1

            # 行为模式同步写入独立库
            if tag_type == "行为模式" and self._behavior_store is not None:
                try:
                    self._behavior_store.store(content, confidence=confidence)
                except Exception as exc:
                    logger.warning("行为模式写入独立库失败: %s", exc)

        return new_count, updated_count
