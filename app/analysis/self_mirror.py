# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: d583900f

"""AI 自我镜像生成器 — 检索 AI 在相似情绪下的历史回应, 组装为 prompt 上下文。

纯读操作, 零 LLM 调用, 不落盘。

管线:
  当前用户情绪 → 查 AI 记忆库 (按 valence 范围) → 取用户下一轮反应
  → analyze_emotion_2d() 标记有效性 → 分散采样 → 渲染为 prompt 段
"""
import logging
from app.analysis.emotion import analyze_emotion_2d

logger = logging.getLogger(__name__)


class SelfMirror:
    """AI 自我镜像生成器。

    查 AI 过去在相似情绪下的回应,
    匹配用户下一轮反应的有效性,
    组装为 prompt 片段供 LLM 参考。
    """

    # 情绪类别中文映射
    _EMO_LABELS_CN = {
        "positive": "正向", "negative": "负向",
        "neutral": "中性", "intimate": "亲密",
    }

    def build_mirror(
        self,
        user_emotion: dict,
        ai_memory,
        chat_history,
        *,
        limit: int = 3,
    ) -> str:
        """输入当前用户情绪, 输出自我镜像 prompt 段, 或空字符串。

        Args:
            user_emotion: {"valence": float, "arousal": float, "category": str}
            ai_memory: QdrantService (AI 记忆库)
            chat_history: ChatHistory 实例
            limit: 最多几条

        Returns:
            格式化的自我镜像字符串, 或 ""
        """
        if not ai_memory or not chat_history:
            return ""

        valence = user_emotion.get("valence", 0)

        # 1. 从 AI 记忆库中按情绪 valence 范围筛选
        try:
            all_ai = ai_memory.list_all()
        except Exception as exc:
            logger.debug("SelfMirror: AI 记忆库读取失败: %s", exc)
            return ""

        if not all_ai:
            return ""

        # Python 侧按 valence 接近度排序 (AI 记忆量小, 无需后端 where)
        valence_range = 0.3
        candidates = []
        for mem in all_ai:
            meta = mem.get("metadata") or {}
            mv = meta.get("emotion_valence")
            if mv is None:
                continue
            try:
                mv = float(mv)
            except (ValueError, TypeError):
                continue
            if abs(mv - valence) <= valence_range:
                candidates.append((abs(mv - valence), mem))

        if not candidates:
            return ""

        # 按最接近排序, 取前 10 条候选
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:10]

        # 2. 对每条 AI 记忆, 查用户下一轮反应
        episodes = []
        for _, mem in candidates:
            meta = mem.get("metadata") or {}
            ts = meta.get("timestamp")
            if not ts:
                continue

            # 时间戳标准化为字符串
            try:
                if isinstance(ts, (int, float)):
                    from datetime import datetime as _dt
                    ts_str = _dt.fromtimestamp(float(ts)).strftime(
                        "%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(ts)
            except (ValueError, OSError):
                continue

            # 查用户下一轮反应 (before=0, after=1)
            try:
                ctx = chat_history.get_context_by_timestamp(
                    ts_str, before=0, after=1)
            except Exception:
                continue

            if not ctx or not ctx.get("context_after"):
                continue

            user_reaction_item = ctx["context_after"][0]
            # 实际键是 "user" 而非 SPEC 中的 "user_message"
            user_reaction = user_reaction_item.get("user", "")
            if not user_reaction:
                continue

            # 3. 情绪分析
            try:
                e_valence, e_arousal, e_cat = analyze_emotion_2d(user_reaction)
            except Exception:
                e_valence, e_arousal, e_cat = 0.0, 0.0, "neutral"

            doc = mem.get("document", "") or ""
            episodes.append({
                "when": str(ts)[:10],
                "ai_response": doc[:120],
                "user_reaction": user_reaction[:80],
                "reaction_valence": e_valence,
                "effective": e_valence > 0.3,
            })

            # 收集足够候选后提前退出
            if len(episodes) >= limit * 2:
                break

        if not episodes:
            return ""

        # 4. 分散采样: 优先覆盖不同 valence 区间
        sampled = self._diverse_sample(episodes, limit)

        return self._render_mirror(sampled, user_emotion)

    # ── 内部 ──────────────────────────────────────────────

    @staticmethod
    def _diverse_sample(episodes: list[dict], limit: int) -> list[dict]:
        """按 reaction_valence 分层采样, 优先覆盖 pos/neg/mid 各一。"""
        episodes_sorted = sorted(episodes, key=lambda e: e["reaction_valence"])
        sampled = []
        seen_signs = set()
        for ep in episodes_sorted:
            rv = ep["reaction_valence"]
            sign = "pos" if rv > 0.2 else "neg" if rv < -0.2 else "mid"
            if sign not in seen_signs:
                sampled.append(ep)
                seen_signs.add(sign)
            if len(sampled) >= limit:
                break
        # 不够则从剩余中补充
        for ep in episodes_sorted:
            if ep not in sampled and len(sampled) < limit:
                sampled.append(ep)
        return sampled

    @classmethod
    def _render_mirror(cls, episodes: list[dict], current_emotion: dict) -> str:
        """组装为 LLM 可读的自我镜像段落。"""
        emo_label = current_emotion.get("category", "neutral")
        emo_cn = cls._EMO_LABELS_CN.get(emo_label, emo_label)
        lines = [f"【自我镜像 — 面对{emo_cn}情绪的你】"]
        for i, ep in enumerate(episodes, 1):
            tag = "✓ 用户转向正向" if ep["effective"] else "✗ 用户情绪未改善"
            ai_text = ep["ai_response"][:80]
            user_text = ep["user_reaction"][:50]
            lines.append(
                f"{i}. {ep['when']} | 你: {ai_text}...\n"
                f"   用户反应: {user_text}... ({tag})"
            )
        return "\n".join(lines)
