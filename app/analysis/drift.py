"""偏移率追踪 — 检测用户行为倾向的偏移方向 (纯规则, 零 LLM)。

spend (愿投, +) ←→ frugal (省钱, λ) ←→ drift (放弃, -)
                                               ├─ 放弃 (深)
                                               ├─ 妥协 (中)
                                               └─ 烦躁 (浅)

每条用户消息检测一次, EMA 合并连续同向信号。
状态通过 JSONL 持久化, 重启后回读最近状态。
"""
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 信号词表 ────────────────────────────────────────────────

_SPEND_WORDS = [
    "花钱", "付费", "值得投资", "效率优先", "买",
    "氪金", "值得买", "省时间", "花钱买", "氪了",
]

_FRUGAL_WORDS = [
    "免费", "省钱", "自己搞", "开源", "性价比",
    "白嫖", "不花钱", "太贵", "贵了", "不划算",
]

_DRIFT_放弃 = [
    "不管了", "随便", "放弃", "不做了", "不搞了",
    "算了不", "懒得", "不学了", "不玩了", "退出",
]

_DRIFT_妥协 = [
    "算了", "将就", "凑合", "能用就行", "差不多",
    "就这样吧", "认了", "没办法", "行吧",
]

_DRIFT_烦躁 = [
    "烦死了", "劝退", "坑爹", "垃圾", "有毒",
    "恶心", "离谱", "受不了", "崩溃", "疯了",
]

# ── 检测优先级: drift > spend > frugal ──

_SIGNAL_GROUPS = [
    ("drift_放弃", _DRIFT_放弃, -60),
    ("drift_烦躁", _DRIFT_烦躁, -50),
    ("drift_妥协", _DRIFT_妥协, -30),
    ("spend", _SPEND_WORDS, 40),
    ("frugal", _FRUGAL_WORDS, 20),
]


class DriftTracker:
    """偏移率追踪器 — 纯规则检测用户行为倾向偏移。

    每次对话后调用 detect(), EMA 合并连续同向信号。
    状态通过 JSONL 持久化, 重启后回读最近状态。
    """

    def __init__(self, log_path: str):
        self._log_path = log_path
        self._alpha = 0.7  # EMA 平滑系数
        self._current: dict = {
            "direction": "neutral", "offset": 0.0, "signal": "",
        }
        self._consecutive = 0
        self._restore()

    # ── 公共 API ──────────────────────────────────────────

    def detect(self, text: str) -> dict:
        """检测单条消息的偏移信号。

        Returns:
            {"direction": str, "offset": float, "signal": str}
            direction: "spend" | "frugal" | "drift_*" | "neutral"
            offset: -100~+100 (负=drift方向, 正=spend方向)
            signal: 人类可读摘要
        """
        if not text:
            return dict(self._current)

        # 按优先级匹配
        matched = None
        for direction, words, raw_offset in _SIGNAL_GROUPS:
            for w in words:
                if w in text:
                    matched = (direction, raw_offset)
                    break
            if matched:
                break

        if matched is None:
            # 无信号: 连续计数衰减
            self._consecutive = max(0, self._consecutive - 1)
            return dict(self._current)

        direction, raw_offset = matched

        # EMA 合并
        prev_offset = self._current.get("offset", 0.0)
        if self._current.get("direction") == direction:
            self._consecutive += 1
            merged = self._alpha * raw_offset + (1 - self._alpha) * prev_offset
        else:
            self._consecutive = 1
            merged = float(raw_offset)

        result = {
            "direction": direction,
            "offset": round(merged, 1),
            "signal": f"{direction}({merged:+.0f}%)",
        }
        self._current = result
        self._log(result)
        return result

    def comprehensive_offset(self) -> dict:
        """返回当前综合偏移状态, 供浅巩固/画像层读取。"""
        return dict(self._current)

    @property
    def current_direction(self) -> str:
        return self._current.get("direction", "neutral")

    @property
    def current_offset(self) -> float:
        return self._current.get("offset", 0.0)

    @property
    def consecutive(self) -> int:
        return self._consecutive

    def render_for_prompt(self) -> str:
        """生成注入 prompt 的偏移片段。

        Returns:
            "偏移: frugal(+25%) 连续3轮节省倾向" 或 ""
        """
        d = self._current
        direction = d.get("direction", "neutral")
        if direction == "neutral":
            return ""
        offset = d.get("offset", 0.0)
        labels = {
            "spend": "愿投",
            "frugal": "省钱",
            "drift_放弃": "放弃倾向",
            "drift_妥协": "妥协倾向",
            "drift_烦躁": "烦躁倾向",
        }
        label = labels.get(direction, direction)
        parts = [f"偏移: {label}({offset:+.0f}%)"]
        if self._consecutive >= 2:
            parts.append(f"连续{self._consecutive}轮")
        return " ".join(parts)

    # ── 持久化 ────────────────────────────────────────────

    def _log(self, result: dict):
        """追加一条偏移检测记录到 JSONL。"""
        try:
            parent = os.path.dirname(self._log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.now().isoformat(),
                    **result,
                }, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug("偏移日志写入失败: %s", exc)

    def _restore(self):
        """从 JSONL 恢复最后一次偏移状态。"""
        try:
            if not os.path.exists(self._log_path):
                return
            with open(self._log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return
            last = json.loads(lines[-1])
            self._current = {
                "direction": last.get("direction", "neutral"),
                "offset": last.get("offset", 0.0),
                "signal": last.get("signal", ""),
            }
            logger.debug("偏移状态恢复: %s", self._current.get("signal"))
        except Exception:
            pass
