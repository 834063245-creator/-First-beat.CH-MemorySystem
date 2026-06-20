# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 31e0d0c3

"""自主触发冲动系统 — 记忆系统主动敲门。

每个冲动源是一个函数，返回 (content, priority) 或 None。
调度器管理队列 + 速率限制，由 main.py 中的后台线程驱动。
"""
import json
import logging
import os
import queue
import random
import threading
import time
from datetime import datetime, date

from app.tools.atomic import atomic_write

from app.config.settings import (
    IMPULSE_MAX_PER_HOUR,
    IMPULSE_MIN_INTERVAL,
    IMPULSE_IDLE_MINUTES,
    IMPULSE_HEARTBEAT_IDLE,
    IMPULSE_TTL,
)

logger = logging.getLogger(__name__)


# ── 状态持久化 ──────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "impulse_count_today": 0,
        "last_impulse_date": "",
        "last_impulse_time": 0,
        "history": [],
    }


def _load_state(state_path: str) -> dict:
    path = state_path
    if not os.path.exists(path):
        return _default_state()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        merged = _default_state()
        merged.update(data)
        return merged
    except (json.JSONDecodeError, OSError):
        return _default_state()


def _save_state(state: dict, state_path: str):
    atomic_write(state_path, state)


# ── 冲动源 ──────────────────────────────────────────────────

def source_time_rhythm(memory_service=None, temporal_pattern_index=None, all_mems=None) -> tuple | None:
    """检查当前时间段是否有话题模式——上线就能触发。

    不再扫描全库查日期，而是查 TemporalPatternIndex。
    TemporalPatternIndex 由 DMN 浅巩固每4h增量维护，无需冲动源操心。
    产出该模式下的一条具体记忆内容，而非"模式触发"元描述。
    """
    if temporal_pattern_index is None:
        logger.info("  time_rhythm 跳过: 无时间模式索引")
        return None
    try:
        patterns = temporal_pattern_index.query()
        if not patterns:
            logger.info("  time_rhythm 跳过: 当前时段无活跃模式")
            return None
        # 取优先级最高的模式
        tag, priority, gran = patterns[0]

        # 用 tag 找一条具体记忆，产出实际内容
        if memory_service or all_mems:
            try:
                mems = all_mems if all_mems is not None else memory_service.list_all()
                tagged = [
                    m for m in mems
                    if tag in ((m.get("metadata") or {}).get("tags", "") or "")
                ]
                if tagged:
                    recent = sorted(
                        tagged,
                        key=lambda m: -(m.get("metadata") or {}).get("timestamp", 0)
                    )[:3]
                    for m in recent:
                        meta = m.get("metadata") or {}
                        content = meta.get("summary", "") or meta.get("user_message", "") or ""
                        if len(content) >= 10:
                            return (content[:200], priority)
            except Exception as exc:
                logger.debug("impulse exception: %s", exc)

        gran_labels = {
            "month": "这个月份", "day_of_week": "每周这天",
            "season": "这个季节", "period": "这个时段",
        }
        label = gran_labels.get(gran, "最近")
        return (f"最近常聊到{tag}", priority)
    except Exception as exc:
        logger.debug("time_rhythm 源异常: %s", exc)
    return None


def source_emotion_trend(memory_service, all_mems=None) -> tuple | None:
    """检查今天情绪强度波动。产出具体情绪相关的内容片段。"""
    try:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        mem_pool = all_mems if all_mems is not None else memory_service.list_all()
        if not mem_pool:
            logger.info("  emotion_trend 跳过: 记忆库为空")
            return None
        today_mems = [
            m for m in mem_pool
            if (m.get("metadata") or {}).get("timestamp", 0) >= today_start
        ]
        if len(today_mems) < 2:
            logger.info("  emotion_trend 跳过: 今天仅 %d 条（需要 2+）", len(today_mems))
            return None
        emotional = [
            m for m in today_mems
            if (m.get("metadata") or {}).get("emotional_intensity", 0) >= 2
        ]
        ratio = len(emotional) / len(today_mems) if today_mems else 0

        def _pick_content(candidates, min_score=0):
            """从候选记忆里挑一条有内容的。"""
            for m in candidates:
                meta = m.get("metadata") or {}
                s = 0
                if meta.get("emotional_intensity", 0) >= 2: s += 2
                if meta.get("entities"): s += 1
                if s < min_score:
                    continue
                content = meta.get("summary", "") or meta.get("user_message", "") or ""
                if len(content) >= 10:
                    return content[:200]
            return None

        if ratio > 0.4 and len(emotional) >= 2:
            excerpt = _pick_content(emotional, min_score=1)
            if excerpt:
                return (excerpt, 50)
            return ("今天的对话里，用户情绪起伏比平时明显", 50)
        elif len(today_mems) >= 5 and ratio < 0.15:
            # 异常平静时，挑一条今天的内容
            excerpt = _pick_content(today_mems)
            if excerpt:
                return (excerpt, 30)
            return ("今天的对话里，用户异常平静", 30)
        logger.info("  emotion_trend 跳过: 情绪比率 %.2f 未达阈值", ratio)
    except Exception as exc:
        logger.debug("emotion_trend 源异常: %s", exc)
    return None


def source_random_roam(memory_service, all_mems=None) -> tuple | None:
    """翻一条有情绪或有实体的旧记忆，产出具体记忆内容。"""
    try:
        mem_pool = all_mems if all_mems is not None else memory_service.list_all()
        if not mem_pool:
            logger.info("  random_roam 跳过: 记忆库为空")
            return None
        pool = [
            m for m in mem_pool
            if (m.get("metadata") or {}).get("timestamp", 0) < time.time() - 3600
        ]
        if len(pool) < 2:
            logger.info("  random_roam 跳过: 近期记忆仅 %d 条", len(pool))
            return None

        def _score(m):
            meta = m.get("metadata") or {}
            s = 0
            if meta.get("emotional_intensity", 0) >= 1: s += 3
            if meta.get("entities"): s += 2
            um = meta.get("user_message", "") or ""
            if "?" in um or "？" in um: s += 1
            return s

        pool.sort(key=_score, reverse=True)
        top = pool[:max(5, len(pool) // 3)]
        picked = random.choice(top)
        meta = picked.get("metadata") or {}
        score = _score(picked)

        # 提取具体内容
        content = meta.get("summary", "") or meta.get("user_message", "") or ""
        if len(content) < 10:
            doc = picked.get("document", "") or ""
            if doc:
                content = doc[:200]
        if len(content) < 10:
            content = ""

        if content:
            if score >= 3:
                return (content, 18)
            elif score >= 2:
                return (content, 15)
            return (content, 5)

        # 底线 fallback（没有可用的具体内容时）
        runtime_candidates = [m for m in mem_pool
                              if (m.get("metadata") or {}).get("timestamp", 0) < time.time() - 3600]
        if len(runtime_candidates) >= 3:
            return ("脑海里闪过一段以前的对话", 5)

        logger.info("  random_roam 跳过: 最高分记忆仅 %d 分", score)
    except Exception as exc:
        logger.debug("random_roam 源异常: %s", exc)
    return None


def source_curiosity(memory_service, all_mems=None) -> tuple | None:
    """好奇心源：翻出几乎从未被提起过的记忆。

    选择 hit_count <= 2 且超过 1 小时前的记忆，
    按"低命中+久远"加权随机选取。
    产出具体记忆内容，去掉元描述前缀。
    """
    try:
        mem_pool = all_mems if all_mems is not None else memory_service.list_all()
        if not mem_pool:
            return None

        now = time.time()
        # 候选：hit_count <= 2 且不是刚发生的
        candidates = []
        for m in mem_pool:
            meta = m.get("metadata") or {}
            hc = meta.get("hit_count", 0) or 0
            ts = meta.get("timestamp", 0)
            if hc > 2:
                continue
            if not ts or ts > now - 3600:
                continue
            # 提取内容
            content = meta.get("summary", "") or meta.get("user_message", "") or ""
            doc = m.get("document", "") or ""
            if len(content) < 10 and doc:
                content = doc[:200]
            if len(content) < 10:
                continue
            # 权重：hit_count 越低越好、越久远越好
            curiosity_weight = (3 - hc) * 0.6 + min((now - ts) / 86400 / 7, 1.0) * 0.4
            candidates.append((content[:200], curiosity_weight))

        if len(candidates) < 2:
            logger.info("  curiosity 跳过: 候选记忆仅 %d 条", len(candidates))
            return None

        # 加权随机
        total_weight = sum(w for _, w in candidates)
        r = random.uniform(0, total_weight)
        cumulative = 0
        picked = candidates[-1][0]  # fallback
        for content, weight in candidates:
            cumulative += weight
            if r <= cumulative:
                picked = content
                break

        return (picked, 15)
    except Exception as exc:
        logger.debug("curiosity 源异常: %s", exc)
    return None


def source_portrait_curiosity(portrait_manager=None, all_mems=None, **kwargs) -> tuple | None:
    """画像驱动的好奇心源：对用户关注但引擎了解不足的话题主动探索。

    数据来源：
      - usr2（当前状态/关注焦点） → extract_focus_keywords()
      - usr5（兴趣图谱）           → extract_hot_topics()
      - usr6（情绪图谱）           → 排除负向触发，避免踩雷

    优先级 20（介于随机漫游 18 和好奇心 15 之间，画像引导比随机更有价值）。
    """
    if portrait_manager is None:
        return None
    try:
        focus_tags = portrait_manager.extract_focus_keywords()
        hot_tags = portrait_manager.extract_hot_topics()
        neg_tags = set(portrait_manager.extract_negative_triggers())

        # 合并候选：关注焦点优先
        candidates = []
        for tag in focus_tags:
            if tag not in neg_tags:
                candidates.append(tag)
        for tag in hot_tags:
            if tag not in neg_tags and tag not in candidates:
                candidates.append(tag)

        if not candidates:
            logger.debug("portrait_curiosity 跳过: 无可用候选标签")
            return None

        # 加权随机：关注焦点（前几个）权重更高
        weights = [1.5 if i < len(focus_tags) else 1.0 for i in range(len(candidates))]
        total_w = sum(weights)
        r = random.uniform(0, total_w)
        cumulative = 0.0
        picked_tag = candidates[-1]
        for tag, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                picked_tag = tag
                break

        # 检查记忆库中该 tag 的覆盖深度：少则探索，多则跳过
        if all_mems:
            tagged_count = sum(
                1 for m in all_mems
                if picked_tag in ((m.get("metadata") or {}).get("tags", "") or "")
            )
            if tagged_count >= 10:
                logger.debug("portrait_curiosity 跳过: tag '%s' 已覆盖 %d 条",
                             picked_tag, tagged_count)
                return None

        return (f"我注意到你最近常提到「{picked_tag}」，想多聊聊这个话题吗？", 20)
    except Exception as exc:
        logger.debug("portrait_curiosity 源异常: %s", exc)
    return None


# ── 调度器 ──────────────────────────────────────────────────

class ImpulseScheduler:
    """冲动调度器 — 接收冲动源产出，管理队列 + 速率限制。

    每个冲动源运行在独立的泊松线程中，按各自的平均间隔随机触发。
    所有源共享队列和速率限制，消费端（get_next / 前端轮询）不受影响。
    """

    MAX_HISTORY = 30

    # 每个冲动源的泊松平均间隔（秒），λ = 1/interval
    SOURCE_CONFIG = [
        ("情绪趋势", source_emotion_trend, 600),    # 平均每 10 分钟
        ("时间节律", source_time_rhythm, 1800),     # 平均每 30 分钟（模式索引更稳，不用频繁查）
        ("随机漫游", source_random_roam, 600),       # 平均每 10 分钟
        ("好奇心", source_curiosity, 1200),          # 平均每 20 分钟（探索从未提起的记忆）
        ("画像探索", source_portrait_curiosity, 900),  # 平均每 15 分钟
    ]

    def __init__(self, state_path: str, temporal_pattern_index=None,
                 portrait_manager=None):
        self._state_path = state_path
        self._temporal_index = temporal_pattern_index
        self._portrait_manager = portrait_manager
        self._pq = queue.PriorityQueue()  # 无界：泊松源发射频率由 SOURCE_CONFIG 限制，排队项 < MAX_HISTORY
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._last_fingerprints: dict[str, str] = {}
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        # 内抑制 — 各源的疲劳度和发射记录
        self._source_fatigue: dict[str, float] = {}
        self._source_fire_times: dict[str, list[float]] = {}
        # 全量记忆缓存（各源共享，避免重复 list_all）
        self._all_mems_cache: list[dict] | None = None
        self._all_mems_cache_time: float = 0
        self._ALL_MEMS_CACHE_TTL = 60
        self._load_state()

    def _get_all_mems(self, memory_service):
        """各冲动源共享的全量记忆缓存，委托给记忆存储层统一缓存。"""
        return memory_service.list_all_cached()

    def _load_state(self):
        state = _load_state(self._state_path)
        today_str = date.today().isoformat()
        if state.get("last_impulse_date") != today_str:
            state["impulse_count_today"] = 0
            state["last_impulse_date"] = today_str
        self._history = state.get("history", []) or []
        self._state = state

    def _persist(self):
        self._state["history"] = self._history[-self.MAX_HISTORY:]
        _save_state(self._state, self._state_path)

    def _record_history(self, entry: dict):
        self._history.append(entry)
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]

    def get_history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    def feed_impulse(self, content: str, priority: float, source: str, ttl: float = IMPULSE_TTL):
        # 内抑制：该源疲劳度上升
        now = time.time()
        with self._lock:
            self._source_fatigue[source] = min(1.0,
                self._source_fatigue.get(source, 0.0) + 0.15)
            self._source_fire_times.setdefault(source, []).append(now)
            # 有效优先级 = 基础优先级 × (1 - 疲劳度)，低于阈值则丢弃
            effective_priority = priority * (1.0 - self._source_fatigue.get(source, 0.0))
        if effective_priority < 2:
            self._record_history({
                "event": "suppressed",
                "content": content,
                "source": source,
                "priority": priority,
                "effective_priority": round(effective_priority, 1),
                "fatigue": round(self._source_fatigue[source], 2),
                "created_at": now,
            })
            return

        self._pq.put((-effective_priority, now, {
            "content": content,
            "source": source,
            "created_at": now,
            "ttl": ttl,
            "expired": False,
        }))
        self._record_history({
            "event": "generated",
            "content": content,
            "source": source,
            "priority": priority,
            "effective_priority": round(effective_priority, 1),
            "fatigue": round(self._source_fatigue[source], 2),
            "created_at": now,
        })

    def _decay_fatigue(self):
        """所有源的疲劳度随时间自然衰减。半衰期约 30 分钟。"""
        now = time.time()
        for source in list(self._source_fatigue.keys()):
            times = self._source_fire_times.get(source)
            if not times:
                self._source_fatigue[source] = 0.0
                continue
            last_fire = max(times)
            mins_since = (now - last_fire) / 60
            self._source_fatigue[source] = max(0.0,
                min(1.0, self._source_fatigue[source] * (0.5 ** (mins_since / 15))))
            # 清除 2 小时前的发射记录
            self._source_fire_times[source] = [t for t in times if now - t < 7200]
            if self._source_fatigue[source] < 0.01:
                self._source_fatigue[source] = 0.0

    def get_next(self, test_mode: bool = False) -> dict | None:
        if test_mode:
            try:
                item = self._pq.get_nowait()
                return item[2]
            except queue.Empty:
                return None
        today_str = date.today().isoformat()
        with self._lock:
            self._decay_fatigue()
            if self._state.get("last_impulse_date") != today_str:
                self._state["impulse_count_today"] = 0
                self._state["last_impulse_date"] = today_str
            if self._state["impulse_count_today"] >= IMPULSE_MAX_PER_HOUR:
                return None
            if time.time() - self._state.get("last_impulse_time", 0) < IMPULSE_MIN_INTERVAL:
                return None

        while not self._pq.empty():
            try:
                item = self._pq.get_nowait()
            except queue.Empty:
                return None
            impulse = item[2]
            age = time.time() - impulse["created_at"]
            if age > impulse.get("ttl", IMPULSE_TTL) or impulse.get("expired"):
                logger.info("  冲动过期: [%s] %s (存活 %.1fs)", impulse.get("source", "?"), impulse.get("content", "")[:40], age)
                self._record_history({
                    "event": "expired",
                    "content": impulse.get("content", ""),
                    "source": impulse.get("source", ""),
                    "created_at": impulse.get("created_at", 0),
                })
                continue
            self._record_history({
                "event": "delivered",
                "content": impulse.get("content", ""),
                "source": impulse.get("source", ""),
                "created_at": impulse.get("created_at", 0),
                "delivered_at": time.time(),
            })
            with self._lock:
                self._state["impulse_count_today"] += 1
                self._state["last_impulse_time"] = time.time()
                self._state["history"] = self._history[-self.MAX_HISTORY:]
                _save_state(self._state, self._state_path)
            return impulse
        return None

    # ── 泊松调度 ──────────────────────────────────────────

    def start_source_workers(self, memory_service=None, chat_history=None):
        """为每个冲动源启动独立泊松线程。"""
        self._stop_event.clear()
        self._workers.clear()

        kwargs_map = {
            "情绪趋势": {"memory_service": memory_service},
            "时间节律": {"memory_service": memory_service, "temporal_pattern_index": self._temporal_index},
            "随机漫游": {"memory_service": memory_service},
            "好奇心": {"memory_service": memory_service},
            "画像探索": {"memory_service": memory_service, "portrait_manager": self._portrait_manager},
        }

        for name, source_fn, avg_interval in self.SOURCE_CONFIG:
            kwargs = kwargs_map.get(name, {})
            t = threading.Thread(
                target=self._source_loop,
                args=(name, source_fn, avg_interval, kwargs),
                daemon=True,
                name=f"impulse_{name}",
            )
            t.start()
            self._workers.append(t)

        logger.info("冲动源泊松线程已启动: %s",
                     ", ".join(f"{name}({avg_interval}s)" for name, _, avg_interval in self.SOURCE_CONFIG))

    def stop(self):
        """停止所有冲动源泊松线程。"""
        self._stop_event.set()
        self._workers.clear()
        logger.info("冲动源泊松线程已停止")

    def _source_loop(self, name, source_fn, avg_interval, kwargs):
        """独立泊松循环：执行 → 指数等待 → 执行。"""
        # 启动冷却期：首个 120 秒不执行任何冲动源，等系统预热完
        COOLDOWN = 120
        logger.debug("冲动源 '%s' 冷却 %ds...", name, COOLDOWN)
        if self._stop_event.wait(COOLDOWN):
            return
        # 随机初始偏移，防止冷却后所有源扎堆
        initial_delay = random.uniform(0, avg_interval * 0.5)
        if initial_delay > 0 and self._stop_event.wait(initial_delay):
            return

        while not self._stop_event.is_set():
            try:
                # 注入共享的全量记忆缓存，避免各源重复 list_all
                loop_kwargs = dict(kwargs)
                if "memory_service" in loop_kwargs and loop_kwargs["memory_service"] is not None:
                    loop_kwargs["all_mems"] = self._get_all_mems(loop_kwargs["memory_service"])
                # 只传非 None 的参数
                clean_kwargs = {k: v for k, v in loop_kwargs.items() if v is not None}
                result = source_fn(**clean_kwargs)
                if result is not None:
                    content, priority = result
                    with self._lock:
                        last_fp = self._last_fingerprints.get(name)
                        if content == last_fp:
                            continue
                        self._last_fingerprints[name] = content
                    self.feed_impulse(content, priority, name)
                    logger.info("冲动源 '%s' 产出: %s (优先级=%s)", name, content[:40], priority)
                else:
                    logger.debug("冲动源 '%s' 本轮无产出", name)
            except Exception as exc:
                logger.warning("冲动源 '%s' 异常已隔离: %s", name, exc)

            # 泊松间隔：指数分布，上限 1 小时防止线程饿死
            interval = min(random.expovariate(1.0 / avg_interval), 3600)
            if self._stop_event.wait(interval):
                return

    def idle_seconds(self, chat_history) -> float | None:
        if not chat_history or not chat_history.records:
            return None
        last = chat_history.records[-1]
        if not last.get("user_message"):
            return None
        try:
            last_ts = datetime.strptime(last["timestamp"], "%Y-%m-%d %H:%M:%S")
            return (datetime.now() - last_ts).total_seconds()
        except (ValueError, KeyError):
            return None

    def idle_gap_minutes(self, chat_history) -> float | None:
        """计算最近两条消息之间的间隔（分钟），用于聊天注入路径。

        Path A（聊天注入）被调用时，用户刚发了一条消息，records[-1] 就是当前消息。
        检查 records[-2]→records[-1] 的间隙，如果间隙够长说明用户离开后回来，
        此时注入冲动更自然。
        """
        if not chat_history or len(chat_history.records) < 2:
            return None
        try:
            prev = chat_history.records[-2]
            last = chat_history.records[-1]
            prev_ts = datetime.strptime(prev["timestamp"], "%Y-%m-%d %H:%M:%S")
            last_ts = datetime.strptime(last["timestamp"], "%Y-%m-%d %H:%M:%S")
            gap = (last_ts - prev_ts).total_seconds() / 60
            return gap
        except (ValueError, KeyError):
            return None

    def get_status_snapshot(self) -> dict:
        """线程安全的状态快照（替代外部直接读 _state / _pq）。"""
        with self._lock:
            return {
                "pending": self._pq.qsize(),
                "delivered_today": self._state.get("impulse_count_today", 0),
                "last_delivered": self._state.get("last_impulse_time", 0),
                "source_fatigue": dict(self._source_fatigue),
            }

    def should_deliver(self, chat_history, last_heartbeat_time: float | None) -> bool:
        """完整空闲检查，用于前端轮询（Path B）。"""
        idle = self.idle_seconds(chat_history)
        if idle is None:
            return False
        if idle < IMPULSE_IDLE_MINUTES * 60:
            return False
        if last_heartbeat_time and time.time() - last_heartbeat_time < IMPULSE_HEARTBEAT_IDLE:
            return False
        return True
