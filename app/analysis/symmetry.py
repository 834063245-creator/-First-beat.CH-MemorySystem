"""人格对称性 — 双共现矩阵差分，发现 AI 对用户兴趣的理解盲区。

从 CoOccurrenceTracker.export_for_symmetry() 获取实时共现数据（SQLite），
比较用户和 AI 两套矩阵中的关联分布差异。
零新存储，纯缓存重建。
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class PersonaSymmetry:
    """双共现矩阵差分分析。

    支持两种输入模式：
    - 文件路径（向后兼容旧 JSON）: PersonaSymmetry("user.json", "ai.json")
    - 直接传入 dict 数据（推荐，从 CoOccurrenceTracker.export_for_symmetry() 获取）:
      PersonaSymmetry(user_dict, ai_dict, from_dicts=True)
    """

    MIN_GAP = 0.3          # 差异度阈值
    MAX_BLIND_SPOTS = 3    # 每次最多产出的盲区数

    def __init__(self, user_source, ai_source, from_dicts: bool = False):
        if from_dicts:
            self._user_data = user_source
            self._ai_data = ai_source
            self._user_path = None
            self._ai_path = None
        else:
            self._user_path = user_source
            self._ai_path = ai_source
            self._user_data = None
            self._ai_data = None
        self._blind_spots: list[dict] = []

    def analyze(self) -> list[dict]:
        """计算盲区，返回 [{tag, gap, user_related, ai_related}, ...]."""
        if self._user_data is not None:
            user = self._user_data
            ai = self._ai_data
        else:
            user = self._load(self._user_path)
            ai = self._load(self._ai_path)
        if not user or not ai:
            logger.debug("人格对称性: 缺少共现数据，跳过")
            return []

        shared = set(user.keys()) & set(ai.keys())
        if not shared:
            logger.debug("人格对称性: 无共享标签，跳过")
            return []

        spots = []
        for tag in shared:
            uv = user[tag]  # {related_tag: count}
            av = ai.get(tag, {})
            gap = self._distribution_gap(uv, av)
            if gap >= self.MIN_GAP:
                user_top = sorted(uv.items(), key=lambda x: -x[1])[:3]
                ai_top = sorted(av.items(), key=lambda x: -x[1])[:3]
                spots.append({
                    "tag": tag,
                    "gap": round(gap, 3),
                    "user_related": [t for t, _ in user_top],
                    "ai_related": [t for t, _ in ai_top],
                })

        spots.sort(key=lambda x: -x["gap"])
        self._blind_spots = spots[:self.MAX_BLIND_SPOTS]
        if self._blind_spots:
            logger.info(
                "人格对称性分析: %d 个盲区 %s",
                len(self._blind_spots),
                [(s["tag"], s["gap"]) for s in self._blind_spots],
            )
        return self._blind_spots

    def get_observations(self) -> list[str]:
        """格式化为观察文本列表。"""
        lines = []
        for s in self._blind_spots:
            user_tags = "、".join(s["user_related"][:2])
            ai_tags = "、".join(s["ai_related"][:2])
            if user_tags and ai_tags:
                lines.append(
                    f"[模式观察] 你提到「{s['tag']}」时更关联 {user_tags}，"
                    f"我之前更多联想到 {ai_tags}"
                )
            elif user_tags:
                lines.append(
                    f"[模式观察] 你提到「{s['tag']}」时关联 {user_tags}，"
                    f"我之前没太注意到这点"
                )
        return lines

    @property
    def blind_spots(self) -> list[dict]:
        return self._blind_spots

    def _distribution_gap(self, d1: dict, d2: dict) -> float:
        """计算两个标签关联分布的余弦距离。

        d1: 用户共现 {related_tag: count}
        d2: AI 共现 {related_tag: count}
        """
        all_keys = set(d1.keys()) | set(d2.keys())
        if not all_keys:
            return 0.0
        v1 = [d1.get(k, 0) for k in all_keys]
        v2 = [d2.get(k, 0) for k in all_keys]
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = (sum(a * a for a in v1) or 1) ** 0.5
        n2 = (sum(b * b for b in v2) or 1) ** 0.5
        return 1.0 - dot / (n1 * n2)

    @staticmethod
    def _load(path: str) -> dict:
        try:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("人格对称性: 无法加载 %s: %s", path, exc)
            return {}
