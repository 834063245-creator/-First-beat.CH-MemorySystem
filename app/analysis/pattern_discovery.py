"""模式发现层 — 从已有索引和对话历史中提取可读的模式观察。

零 LLM 调用，不入库，纯缓存层。由 background worker 周期性触发。
"""
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

from .emotion import analyze_emotion_2d
from app.tools.atomic import atomic_write

logger = logging.getLogger(__name__)

# ── 引擎调参 inject 判定规则 ──────────────────────────────────
_INJECT_RULES = {
    "深夜情绪话题多": True,
    "早上工作话题多": True,
    "连续多轮深度讨论": False,
    "发现用户焦虑话题": False,
}

# ── 默认调参模板 ──────────────────────────────────────────────
_DEFAULT_TUNING = {
    "emotional_dampening": False,
    "formality_shift": 0,
    "proactive_suppression": False,
}


def _merge_tuning(a: dict, b: dict) -> dict:
    """合并两个 tuning dict：bool/数值取非默认值优先。"""
    merged = {}
    _D = {"emotional_dampening": False, "formality_shift": 0, "proactive_suppression": False}
    for k in set(a) | set(b):
        va, vb = a.get(k, _D.get(k)), b.get(k, _D.get(k))
        merged[k] = vb if va == _D.get(k) else va
    return merged


class PatternDiscovery:
    """从现有索引和 ChatHistory 提取模式观察 + 引擎调参。"""

    # 运行间隔（秒）
    RUN_INTERVAL = 21600  # 6h
    # 观察上限
    MAX_OBSERVATIONS = 3
    # 最少出现次数才认为是模式
    MIN_OBSERVATIONS = 3
    # 观察窗口
    RECENT_DAYS = 7

    def __init__(self, data_dir: str, temporal_index=None, affinity=None,
                 chat_history_path: str = None):
        self._cache_path = os.path.join(data_dir, "cache", "pattern_cache.json")
        self._temporal = temporal_index
        self._affinity = affinity
        self._chat_history_path = chat_history_path or os.path.join(data_dir, "chat_history.jsonl")
        self._observations: list[dict] = []
        self._tuning: dict = dict(_DEFAULT_TUNING)

    # ---- 公开接口 ----

    def run(self):
        """增量运行（被 worker 周期性调用）。"""
        try:
            obs = []
            tuning = dict(_DEFAULT_TUNING)
            for detector in [self._detect_temporal_patterns,
                             self._detect_emotional_anchors,
                             self._detect_topic_drift,
                             self._detect_interaction_rhythm]:
                result = detector()
                for item in result:
                    obs.append(item)
                    if "tuning" in item:
                        tuning = _merge_tuning(tuning, item["tuning"])
            obs = self._dedup_and_filter(obs)
            # 趋势检测：从调参历史中找长期变化
            trend_obs = self.detect_trends()
            obs.extend(trend_obs)
            self._observations = obs[:self.MAX_OBSERVATIONS]
            self._tuning = tuning
            self._save()
            logger.info("模式发现完成：%d 条观察, tuning=%s",
                        len(self._observations), self._tuning)
        except Exception:
            logger.exception("模式发现运行异常")

    def get_observations(self) -> list[str]:
        """返回格式化的观察文本列表，供 prompt 注入。"""
        if not self._observations:
            return []
        result = []
        for o in self._observations:
            inj = o.get("inject", True)
            if inj:
                result.append(o.get("text", ""))
        return result

    def get_tuning(self) -> dict:
        """返回当前引擎调参。"""
        return dict(self._tuning)

    # ---- 四种模式检测 ----

    def _detect_temporal_patterns(self) -> list[dict]:
        """1. 时间节律模式。从 TemporalPatternIndex 读当前活跃话题。"""
        if not self._temporal:
            return []
        try:
            now = datetime.now()
            patterns = self._temporal.query(now)
            if not patterns:
                return []
            lines = []
            gran_names = {
                "day_of_week": "星期", "month": "月",
                "season": "季", "period": "时段",
            }
            for tag, priority, gran in patterns[:3]:
                gran_label = gran_names.get(gran, gran)
                lines.append(
                    f'"{tag}"在{gran_label}上频率偏高（优先级{priority}）'
                )
            if lines:
                # 深夜情绪话题多 → 开 emotional_dampening
                is_night = now.hour < 6 or now.hour >= 22
                tuning = {}
                name = None
                if is_night:
                    tuning = {"emotional_dampening": True, "formality_shift": -1}
                    name = "深夜情绪话题多"
                elif now.hour >= 7 and now.hour <= 10:
                    tuning = {"formality_shift": 1}
                    name = "早上工作话题多"
                text = f"[模式观察] 当前时段话题模式：{'；'.join(lines)}"
                result = {"type": "temporal", "text": text}
                if tuning:
                    result["tuning"] = tuning
                    result["name"] = name
                    result["inject"] = _INJECT_RULES.get(name, True)
                return [result]
        except Exception:
            logger.exception("时间节律检测异常")
        return []

    def _detect_emotional_anchors(self) -> list[dict]:
        """2. 情绪锚点。用 Russell 连续坐标替换离散统计。

        每个话题计算 valence/arousal 均值，高唤醒高负效价 → 负面锚点。
        """
        try:
            recent = self._load_recent_chat()
            if len(recent) < self.MIN_OBSERVATIONS:
                return []
            # {tag: {"vals": [...], "aros": [...], "total": int}}
            topic_emotions: dict[str, dict] = defaultdict(
                lambda: {"vals": [], "aros": [], "total": 0}
            )
            for entry in recent:
                text = entry.get("user_message", "")
                if not text:
                    continue
                v, a, _ = analyze_emotion_2d(text)
                if v == 0.0 and a == 0.0:
                    continue  # 中性跳过，不影响统计
                tags = self._extract_tags(text)
                for tag in tags:
                    te = topic_emotions[tag]
                    te["vals"].append(v)
                    te["aros"].append(a)
                    te["total"] += 1
            obs = []
            found_anxiety = False
            for tag, stats in topic_emotions.items():
                if stats["total"] < self.MIN_OBSERVATIONS:
                    continue
                avg_v = sum(stats["vals"]) / len(stats["vals"])
                avg_a = sum(stats["aros"]) / len(stats["aros"])
                # 高唤醒高负效价 → 负面情绪锚点
                if avg_v <= -0.3:
                    obs.append({
                        "type": "emotion", "name": None,
                        "text": f'[模式观察] 你提到"{tag}"时情绪偏负面',
                    })
                    # 高唤醒负面 → 焦虑信号
                    if avg_a >= 0.6 or tag in ("焦虑", "压力", "烦", "累"):
                        found_anxiety = True
                elif avg_v >= 0.5:
                    obs.append({
                        "type": "emotion", "name": None,
                        "text": f'[模式观察] 你提到"{tag}"时情绪偏积极',
                    })
            if found_anxiety:
                name = "发现用户焦虑话题"
                obs.append({
                    "type": "emotion", "name": name,
                    "text": "",  # 静默不注入
                    "tuning": {"emotional_dampening": True},
                    "inject": _INJECT_RULES.get(name, False),
                })
            return obs[:3]
        except Exception:
            logger.exception("情绪锚点检测异常")
        return []

    def _detect_topic_drift(self) -> list[dict]:
        """3. 话题漂移。比较近 N 轮话题分布与历史基线。"""
        try:
            recent = self._load_recent_chat(limit=100)
            if len(recent) < self.MIN_OBSERVATIONS:
                return []
            # 分两半：前半作基线，后半作当前
            mid = len(recent) // 2
            history = recent[:mid]
            current = recent[mid:]
            hist_tags = self._tag_frequency(history)
            curr_tags = self._tag_frequency(current)
            if not hist_tags or not curr_tags:
                return []
            # 找当前话题中偏离基线最大的
            drift = []
            for tag, curr_count in curr_tags.items():
                hist_count = hist_tags.get(tag, 0)
                if curr_count > hist_count * 2 and curr_count >= self.MIN_OBSERVATIONS:
                    drift.append((tag, curr_count, "上升"))
                elif hist_count > 0 and curr_count == 0:
                    if hist_count >= self.MIN_OBSERVATIONS:
                        drift.append((tag, 0, "消失"))
            if drift:
                rising = [t for t, _, s in drift[:2] if s == "上升"]
                gone = [t for t, _, s in drift[:2] if s == "消失"]
                parts = []
                if rising:
                    parts.append(f"话题「{'」「'.join(rising)}」出现频率上升")
                if gone:
                    parts.append(f"「{'」「'.join(gone)}」近期未出现")
                if parts:
                    return [{
                        "type": "drift",
                        "text": f"[模式观察] {'，'.join(parts)}",
                    }]
        except Exception:
            logger.exception("话题漂移检测异常")
        return []

    def _detect_interaction_rhythm(self) -> list[dict]:
        """4. 交互节奏。会话长度、间隔、时段偏好。"""
        try:
            recent = self._load_recent_chat(limit=200)
            if len(recent) < 10:
                return []
            timestamps = []
            for entry in recent:
                ts = entry.get("timestamp")
                if ts:
                    try:
                        timestamps.append(float(ts))
                    except (ValueError, TypeError):
                        pass
            if len(timestamps) < 5:
                return []
            # 会话间隔
            intervals = []
            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i - 1]
                if 60 < gap < 86400 * 3:  # 1 分钟 ~ 3 天
                    intervals.append(gap)
            if not intervals:
                return []
            avg_interval = sum(intervals) / len(intervals)
            # 当前会话长度（以 timestamps 中最近的连续段为准）
            current_session_len = 0
            for i in range(len(timestamps) - 1, -1, -1):
                if i == len(timestamps) - 1 or timestamps[i + 1] - timestamps[i] < 1800:
                    current_session_len += 1
                else:
                    break
            obs = []
            if current_session_len > 20:
                name = "连续多轮深度讨论"
                obs.append({
                    "type": "rhythm", "name": name,
                    "text": f"[模式观察] 当前会话已持续{current_session_len}轮，超出日常节奏",
                    "tuning": {"proactive_suppression": True},
                    "inject": _INJECT_RULES.get(name, False),
                })
            recent_gap = intervals[-1] if intervals else 0
            if recent_gap < 300 and avg_interval > 3600:
                obs.append({
                    "type": "rhythm",
                    "text": f"[模式观察] 你回复很快（间隔<5分钟），通常间隔约{int(avg_interval / 60)}分钟",
                })
            return obs[:1]
        except Exception:
            logger.exception("交互节奏检测异常")
        return []

    # ---- 辅助方法 ----

    def _load_recent_chat(self, limit: int = 100) -> list[dict]:
        """从 ChatHistory JSONL 加载最近的记录。"""
        path = self._chat_history_path
        if not path or not os.path.exists(path):
            return []
        try:
            entries = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return entries[-limit:]
        except Exception:
            return []

    def _extract_tags(self, text: str) -> list[str]:
        """从文本提取标签（基于 jieba TF-IDF）。"""
        try:
            import jieba.analyse
            tags = jieba.analyse.extract_tags(text, topK=3, withWeight=False)
            return [t for t in tags if len(t) >= 2]
        except Exception:
            return []

    def _tag_frequency(self, entries: list[dict]) -> dict[str, int]:
        """统计一堆聊天记录的话题频率。"""
        freq: dict[str, int] = defaultdict(int)
        for entry in entries:
            text = entry.get("user_message", "")
            if text:
                tags = self._extract_tags(text)
                for t in tags:
                    freq[t] += 1
        return dict(freq)

    def _dedup_and_filter(self, obs: list[dict]) -> list[dict]:
        """去重和去噪。"""
        seen_texts = set()
        result = []
        for o in obs:
            text = o.get("text", "")
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            result.append(o)
        return result

    def _save(self):
        """写出缓存文件（v3：追加轨迹，保留最近 30 条调参历史）。"""
        try:
            hist = []
            if os.path.exists(self._cache_path):
                try:
                    with open(self._cache_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                        hist = old.get("trajectory", [])
                except (json.JSONDecodeError, OSError):
                    pass
            # 追加当前快照
            hist.append({
                "time": time.time(),
                "tuning": dict(self._tuning),
                "obs_count": len(self._observations),
            })
            hist = hist[-30:]  # 最近 30 条（30 × 6h ≈ 一周）
            data = {
                "version": 3,
                "updated_at": time.time(),
                "tuning": self._tuning,
                "observations": self._observations,
                "trajectory": hist,
            }
            atomic_write(self._cache_path, data)
        except Exception:
            logger.exception("模式缓存写入失败")

    def load_cache(self):
        """从磁盘加载缓存（启动时恢复）。兼容 v1/v2/v3。"""
        try:
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ver = data.get("version", 1)
                self._observations = data.get("observations", [])
                if ver >= 2:
                    self._tuning = data.get("tuning", dict(_DEFAULT_TUNING))
                # v3+：轨迹信息在 trajectory 字段中，启动时不加载为 observations
                # 缓存超过 1 小时清空 observations，等下次 run() 刷新
                updated = data.get("updated_at", 0)
                if time.time() - updated > 3600:
                    self._observations = []
        except Exception:
            self._observations = []

    @staticmethod
    def _linear_trend(values: list[float]) -> float:
        """简单线性回归斜率。x 为等间距索引 0..n-1。"""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0

    def detect_trends(self) -> list[dict]:
        """从调参轨迹中检测趋势变化，返回观察列表。"""
        try:
            hist = []
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    hist = data.get("trajectory", [])
            if len(hist) < 4:
                return []  # 至少 4 个数据点（24h）才做趋势

            trends = []

            # 检测 formality_shift 趋势
            f_vals = [h["tuning"].get("formality_shift", 0) for h in hist[-10:]]
            if len(set(f_vals)) >= 2:
                slope = self._linear_trend(f_vals)
                if abs(slope) >= 0.3:
                    direction = "上升" if slope > 0 else "下降"
                    trends.append({
                        "text": (
                            f"[趋势观察] 近 24h 对话正式度持续{direction}，"
                            f"可能你更适应{'直接' if slope > 0 else '轻松'}的沟通了"
                        ),
                        "type": "trend",
                        "inject": True,
                    })

            # 检测 emotional_dampening 频率
            damp_count = sum(
                1 for h in hist[-8:]
                if h["tuning"].get("emotional_dampening", False)
            )
            damp_ratio = damp_count / min(len(hist[-8:]), 8)
            if damp_ratio >= 0.5:
                trends.append({
                    "text": "[趋势观察] 近期情绪压制信号持续偏高，值得留意",
                    "type": "trend",
                    "inject": True,
                })

            return trends
        except Exception:
            logger.debug("趋势检测异常")
            return []
