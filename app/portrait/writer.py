"""PortraitWriter — 画像写入引擎。

三层更新:
  1. 实时层（每轮对话后，<100ms，不调 LLM）→ 更新 dim 2/4
  2. 浅巩固层（idle ≥ 4h，调 LLM 合成）→ 更新 dim 3/5/6
  3. 深巩固层（idle ≥ 24h，调 LLM 重述）→ 更新 dim 1 + 一致性审查

核心原则: 引擎做删除判断和分类，LLM 做文本合成。
"""

import logging
from datetime import datetime
from collections import Counter
from typing import Any, Optional

from app.portrait.manager import PortraitManager, DIM_LABELS
from app.portrait.state import PortraitEntry, EntryStatus

logger = logging.getLogger(__name__)


class PortraitWriter:
    """画像写入引擎 — 按节律从各模块拉数据、分类、调度 LLM 合成。"""

    def __init__(self, manager: PortraitManager):
        self._manager = manager
        self._last_shallow_update: float = 0.0
        self._last_deep_update: float = 0.0
        self._turns_since_last_deep: int = 0

    # ── 实时层更新（每轮对话后） ──────────────────────────

    def realtime_update_user(
        self,
        utterance_spec: Any,  # UtteranceSpec
        relationship: Any,    # RelationshipState
    ):
        """每轮对话后更新用户侧 dim 2/4。

        不改死结论，只标记"待验证"或直接改行。
        引擎直接操作，不调 LLM。
        """
        user = utterance_spec.user  # UserMessageAnalysis

        # ── 用户.2 当前状态: 情绪 ──
        current_emotion = getattr(user, "emotion", "neutral")
        prev_usr2_001 = self._manager.get_entry("usr2-001")
        if prev_usr2_001:
            prev_emotion = self._extract_emotion_from_text(prev_usr2_001.text)
            if prev_emotion and prev_emotion != current_emotion:
                from app.portrait.extractors import detect_emotion_flip
                flipped = detect_emotion_flip(prev_emotion, current_emotion)
                status = "待验证 · 情绪翻转" if flipped else "待验证"
                self._manager.set_entry(
                    "usr2-001",
                    f"**情绪**: {current_emotion} （{status}）",
                    status=EntryStatus.PENDING if flipped else EntryStatus.ACTIVE,
                    last_observed=datetime.now().isoformat(),
                )
        else:
            self._manager.set_entry(
                "usr2-001",
                f"**情绪**: {current_emotion} （待验证 · 首轮初标记）",
                status=EntryStatus.PENDING,
                first_observed=datetime.now().isoformat(),
                last_observed=datetime.now().isoformat(),
            )

        # ── 用户.2 当前状态: 关注焦点 ──
        topics = getattr(user, "topics", []) or []
        if topics:
            focus_text = f"**关注焦点**: {', '.join(topics[:3])} （热点）"
            self._manager.set_entry(
                "usr2-002",
                focus_text,
                tags=list(topics[:3]),
                status=EntryStatus.ACTIVE,
                last_observed=datetime.now().isoformat(),
            )

        # ── 用户.4 关系快照 ──
        trust = getattr(relationship, "trust", None)
        closeness = getattr(relationship, "closeness", None)
        familiarity = getattr(relationship, "familiarity", None)
        interaction_mode = getattr(relationship, "interaction_mode", None)

        if trust is not None:
            self._manager.set_entry(
                "usr4-001",
                f"信任度: {trust:.2f}",
                confidence=min(1.0, max(0.3, trust)),
                last_observed=datetime.now().isoformat(),
            )
        if closeness is not None:
            self._manager.set_entry(
                "usr4-002",
                f"亲密度: {closeness:.2f}",
                confidence=min(1.0, max(0.3, closeness)),
                last_observed=datetime.now().isoformat(),
            )
        if familiarity is not None:
            self._manager.set_entry(
                "usr4-003",
                f"熟悉度: {familiarity:.2f}",
                last_observed=datetime.now().isoformat(),
            )
        if interaction_mode:
            self._manager.set_entry(
                "usr4-004",
                f"互动模式: {interaction_mode}",
                last_observed=datetime.now().isoformat(),
            )

    def realtime_update_ai(
        self,
        utterance_spec: Any,
        relationship: Any,
    ):
        """每轮对话后更新 AI 侧 dim 2/4。

        与用户侧完全镜像。
        """
        # ── AI.2 当前状态 ──
        user_emotion = getattr(utterance_spec.user, "emotion", "neutral")
        # AI 的表达色调跟随用户情绪做标记（简化版，完整版在浅巩固 LLM 合成）
        self._manager.set_entry(
            "ai2-001",
            f"**表达色调**: 伴随用户{user_emotion}情绪，自动调节表达密度 （待验证）",
            status=EntryStatus.PENDING,
            last_observed=datetime.now().isoformat(),
        )

        # ── AI.4 关系快照（与用户同源关系数据，AI 视角） ──
        interaction_mode = getattr(relationship, "interaction_mode", None)
        trust = getattr(relationship, "trust", None)

        if trust is not None:
            self._manager.set_entry(
                "ai4-001",
                f"关系认知: 用户信任度{trust:.2f}，将用户视为思考伙伴",
                last_observed=datetime.now().isoformat(),
            )
        if interaction_mode:
            stage_map = {
                "collaborator": "深度合作",
                "partner": "伙伴关系",
                "casual": "日常交流",
            }
            stage_cn = stage_map.get(interaction_mode, interaction_mode)
            self._manager.set_entry(
                "ai4-003",
                f"关系阶段感知: {stage_cn}",
                last_observed=datetime.now().isoformat(),
            )

    def realtime_update(self, utterance_spec: Any, relationship: Any):
        """实时层更新入口 — 用户 + AI 两侧。"""
        try:
            self.realtime_update_user(utterance_spec, relationship)
        except Exception as exc:
            logger.warning("用户画像实时更新失败: %s", exc)
        try:
            self.realtime_update_ai(utterance_spec, relationship)
        except Exception as exc:
            logger.warning("AI 画像实时更新失败: %s", exc)
        # ── 反馈消费：用户"记错了" → 关联画像条目 confidence 下降 ──
        try:
            from app.core.feedback import get_recent_corrected_ids
            corrected = get_recent_corrected_ids()
            if corrected:
                for entry_id, entry in list(self._manager._entries.items()):
                    # 检查条目标签或文本是否引用了被纠正的 memory_id
                    entry_text_and_tags = entry.text + " " + " ".join(entry.tags)
                    if any(mid in entry_text_and_tags for mid in corrected):
                        entry.confidence = max(0.1, entry.confidence - 0.3)
                        entry.status = EntryStatus.PENDING
                        logger.info("画像条目 %s confidence 降至 %.2f（反馈纠正）",
                                    entry_id, entry.confidence)
        except Exception:
            pass  # 反馈消费失败不影响主链路

        self._turns_since_last_deep += 1
        self._manager.save()

    # ── 提取辅助 ──────────────────────────────────────────

    @staticmethod
    def _extract_emotion_from_text(text: str) -> str | None:
        """从条目标题中提取情绪值。

        匹配格式: **情绪**: xxx
        """
        import re
        match = re.search(r"\*\*情绪\*\*:\s*(\S+)", text)
        if match:
            emotion = match.group(1)
            # 过滤中文
            if emotion in ("positive", "negative", "neutral", "frustrated",
                          "intimate", "sad", "angry", "anxious", "happy", "excited"):
                return emotion
            # 中文情绪映射
            cn_map = {
                "低落": "negative", "焦虑": "negative", "开心": "positive",
                "兴奋": "positive", "平静": "neutral", "沮丧": "negative",
            }
            for cn, en in cn_map.items():
                if cn in emotion:
                    return en
        return None

    # ── 浅巩固层（idle ≥ 4h，LLM 合成） ─────────────────

    def shallow_update(self, ctx_obj: Any):
        """浅巩固画像更新 — dim 3/5/6 (用户) + AI 镜像。

        用户侧和 AI 侧并行执行（独立 Qdrant collection，无竞态）。
        """
        import time as _time
        now = _time.time()
        from app.config.settings import PORTRAIT_SHALLOW_HOURS
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if now - self._last_shallow_update < PORTRAIT_SHALLOW_HOURS * 3600:
            return
        self._last_shallow_update = now

        # 用户侧 + AI 侧并行
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self._shallow_update_user, ctx_obj): "user",
                pool.submit(self._shallow_update_ai, ctx_obj): "ai",
            }
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    logger.warning("%s画像浅巩固失败: %s", label, exc)

        # 应用状态机转换
        self._manager.apply_state_machine()
        self._manager.save()
        logger.info("浅巩固画像更新完成 (dim 3/5/6)")

    def _shallow_update_user(self, ctx: Any):
        """用户侧浅巩固: dim 3(行为节律), dim 5(兴趣图谱), dim 6(情绪图谱)。"""
        # Step 1: 拉数据
        tag_stats = self._pull_tag_stats(ctx.memory_service)
        temporal_data = self._pull_temporal(ctx)

        # Step 2-3: 四态分类 + Step 4: LLM 合成
        # dim 5: 兴趣图谱
        self._update_interest_graph(ctx, tag_stats, "usr5", "user")
        # dim 3: 行为节律
        if temporal_data:
            self._update_dim_via_llm(ctx, "usr3", temporal_data)
        # dim 6: 情绪图谱
        emotion_data = self._pull_emotion_data(ctx)
        if emotion_data:
            self._update_dim_via_llm(ctx, "usr6", emotion_data)

    def _shallow_update_ai(self, ctx: Any):
        """AI 侧浅巩固: dim 3(行为节律), dim 5(知识图谱), dim 6(情绪/表达图谱)。"""
        tag_stats = self._pull_tag_stats(ctx.ai_memory_service)
        # dim 5: 知识图谱
        self._update_interest_graph(ctx, tag_stats, "ai5", "ai")

    def _pull_tag_stats(self, memory_service) -> dict:
        """从 Qdrant 拉取标签分布统计。

        Returns:
            {tag: {count, last_seen_days, ids}}
        """
        stats: dict[str, dict] = {}
        try:
            all_data = memory_service.list_all_cached()
            now = datetime.now()
            for mem in all_data:
                meta = mem.get("metadata") or {}
                tags_str = meta.get("tags", "") or ""
                ts = meta.get("timestamp", 0) or 0
                age_days = (now.timestamp() - ts) / 86400.0 if ts else 999

                for tag in tags_str.split(","):
                    tag = tag.strip()
                    if not tag:
                        continue
                    if tag not in stats:
                        stats[tag] = {"count": 0, "last_seen_days": 999, "ids": []}
                    stats[tag]["count"] += 1
                    stats[tag]["last_seen_days"] = min(stats[tag]["last_seen_days"], age_days)
                    stats[tag]["ids"].append(mem.get("id", ""))
        except Exception as exc:
            logger.debug("标签统计拉取失败: %s", exc)
        return stats

    def _pull_temporal(self, ctx: Any) -> dict:
        """拉取时间模式数据供画像 dim 3 使用。"""
        data = {}
        # 时间分布
        try:
            if hasattr(ctx, "temporal_pattern_index") and ctx.temporal_pattern_index:
                now = datetime.now()
                current_patterns = ctx.temporal_pattern_index.query(
                    now.month, now.weekday(), self._get_time_period(now.hour))
                if current_patterns:
                    data["current_patterns"] = current_patterns
        except Exception:
            pass

        # 行为预测转移概率
        try:
            if hasattr(ctx, "mirror_neuron") and ctx.mirror_neuron:
                data["predictions"] = getattr(ctx.mirror_neuron, "_table", {})
        except Exception:
            pass

        return data

    def _pull_emotion_data(self, ctx: Any) -> dict:
        """拉取情绪相关数据供画像 dim 6 使用。"""
        data = {}
        try:
            all_data = ctx.memory_service.list_all_cached()
            emotions = []
            for mem in all_data:
                meta = mem.get("metadata") or {}
                valence = meta.get("emotion_valence", 0) or 0
                category = meta.get("emotion_valence_bin", "neutral")
                intensity = meta.get("emotional_intensity", 0) or 0
                tags = meta.get("tags", "") or ""
                emotions.append({
                    "valence": valence, "category": category,
                    "intensity": intensity, "tags": tags,
                })

            # 统计正向/负向触发
            pos_triggers = Counter()
            neg_triggers = Counter()
            for e in emotions:
                tags = [t.strip() for t in e["tags"].split(",") if t.strip()]
                if e["valence"] > 0.2:
                    for t in tags:
                        pos_triggers[t] += 1
                elif e["valence"] < -0.2:
                    for t in tags:
                        neg_triggers[t] += 1

            data["positive_triggers"] = pos_triggers.most_common(5)
            data["negative_triggers"] = neg_triggers.most_common(5)
        except Exception:
            pass
        return data

    def _update_interest_graph(self, ctx: Any, tag_stats: dict, dim: str, source: str):
        """更新兴趣/知识图谱维度（usr5 或 ai5）。

        引擎四态分类 + LLM 合成。
        """
        # 现有条目
        existing = self._manager.get_dim_entries(dim)
        now = datetime.now()

        # 四态分类
        keep_ids = []
        delete_ids = []
        new_tags = []

        for entry in existing:
            entry_tags = entry.tags
            has_activity = False
            for tag in entry_tags:
                if tag in tag_stats:
                    stats = tag_stats[tag]
                    if stats["last_seen_days"] <= 14:
                        has_activity = True
                        break
            if has_activity:
                keep_ids.append(entry.id)
            elif entry.days_since_last_observed > 30:
                delete_ids.append(entry.id)
            else:
                keep_ids.append(entry.id)  # still within window, keep

        # 找新标签（高密度但不在现有条目中）
        existing_tags = set()
        for e in existing:
            existing_tags.update(e.tags)

        for tag, stats in tag_stats.items():
            if tag not in existing_tags and stats["count"] >= 3 and stats["last_seen_days"] <= 7:
                density = stats["count"] / max(stats["last_seen_days"], 1)
                if density >= 0.5:
                    new_tags.append({"tag": tag, "count": stats["count"],
                                    "days": stats["last_seen_days"]})

        # 有变化时才调 LLM
        if not delete_ids and not new_tags:
            return

        # 构建 LLM prompt
        existing_texts = []
        for e in existing:
            heat = "hot" if e.days_since_last_observed <= 3 else ("warm" if e.days_since_last_observed <= 7 else "cooling")
            existing_texts.append(f"[{e.id}] {heat} {e.text}")

        prompt = f"""你正在更新{source}画像的"{DIM_LABELS.get(dim, dim)}"维度。

现有条目:
{chr(10).join(existing_texts) if existing_texts else '(空)'}

引擎指令:
- 保留: {', '.join(keep_ids) if keep_ids else '无'}
- 删除: {', '.join(delete_ids) if delete_ids else '无'}
- 新增: {new_tags}

规则:
1. 保留的条目: 更新 evidence 数值，保留原有 <!-- entry:XXX-NNN --> 注释
2. 删除的条目: 删除该行及注释
3. 新增条目: 分配新 entry ID (格式<!-- entry:{dim}-NNN -->, NNN 使用当前最大序号+1)
4. 不要修改任何 entry ID
5. 只输出该维度的 Markdown 内容

请合成更新后的维度内容:"""
        result = self._call_local_llm(prompt)
        if result:
            self._apply_llm_dim_update(dim, result)

    def _update_dim_via_llm(self, ctx: Any, dim: str, data: dict):
        """通用维度 LLM 合成更新。

        用于 dim 3(行为节律), dim 6(情绪图谱) 等整维度更新的场景。
        """
        existing = self._manager.get_dim_entries(dim)
        existing_texts = []
        for e in existing:
            existing_texts.append(f"[{e.id}] {e.text}")

        label = DIM_LABELS.get(dim, dim)
        prompt = f"""你正在更新用户画像的"{label}"维度。

现有内容:
{chr(10).join(existing_texts) if existing_texts else '(空)'}

新观察数据:
{data}

规则:
1. 保留仍成立的条目，更新描述
2. 删除已不成立的条目
3. 为新的观察数据创建新条目（分配新 entry ID: <!-- entry:{dim}-NNN -->）
4. 不要修改保留条目的 entry ID
5. 只输出该维度的 Markdown 内容

请合成更新后的维度内容:"""
        result = self._call_local_llm(prompt)
        if result:
            self._apply_llm_dim_update(dim, result)

    def _apply_llm_dim_update(self, dim: str, llm_output: str):
        """将 LLM 合成的维度内容写回 PORTRAIT.md。

        解析 LLM 输出中的 entry ID，替换/新增条目。
        """
        import re
        # 提取所有 entry ID
        entry_pattern = re.compile(r"<!--\s*entry:(\w+-\d{3})\s*-->")
        new_ids = entry_pattern.findall(llm_output)

        if not new_ids:
            logger.warning("LLM 合成为 %s 未生成有效 entry ID，跳过应用", dim)
            return

        # 删除该维度旧条目（LLM 输出包含了最终的完整维度内容）
        for old_entry in list(self._manager.get_dim_entries(dim)):
            self._manager.delete_entry(old_entry.id)

        # 从 LLM 输出中提取每个条目
        lines = llm_output.split("\n")
        current_id = None
        current_text_lines = []
        all_entries: list[tuple[str, str]] = []

        for line in lines:
            m = entry_pattern.search(line)
            if m:
                # 保存上一个条目
                if current_id and current_text_lines:
                    all_entries.append((current_id, "\n".join(current_text_lines).strip()))
                current_id = m.group(1)
                # 提取同行文本
                remaining = line[m.end():].strip()
                current_text_lines = [remaining] if remaining else []
            elif current_id:
                current_text_lines.append(line)

        # 保存最后一个
        if current_id and current_text_lines:
            all_entries.append((current_id, "\n".join(current_text_lines).strip()))

        # 写入条目
        now_iso = datetime.now().isoformat()
        for entry_id, text in all_entries:
            clean_text = text
            if clean_text.startswith("- "):
                clean_text = clean_text[2:]
            # 剥离 backtick 元数据
            import re as _re_mod
            clean_text = _re_mod.sub(r"\s*`[^`]*`\s*$", "", clean_text).strip()
            if clean_text:
                self._manager.set_entry(
                    entry_id, clean_text,
                    last_observed=now_iso,
                    status=EntryStatus.ACTIVE,
                )

    @staticmethod
    def _call_local_llm(prompt: str, timeout: int = 30) -> str | None:
        """调用本地小模型（qwen2.5:7b / Ollama）进行文本合成。

        失败时返回 None，调用方应优雅降级（跳过本次更新）。
        """
        try:
            from app.llm.local import LocalLLM
            llm = LocalLLM()
            result = llm.generate(prompt, max_tokens=1024)
            return result if result else None
        except Exception as exc:
            logger.warning("LocalLLM 调用失败: %s", exc)
            return None

    @staticmethod
    def _get_time_period(hour: int) -> str:
        """将小时映射到时间段标签。"""
        from app.config.settings import TIME_PERIOD_MAP
        for (lo, hi), name in TIME_PERIOD_MAP.items():
            if lo <= hour <= hi:
                return name
        return "晚上"

    # ── 深巩固层（idle ≥ 24h，LLM 重述） ──────────────

    def deep_update(self, ctx_obj: Any):
        """深巩固画像更新 — dim 1 (用户 + AI 核心特征) + 一致性审查。

        触发条件:
          - 至少 20 轮新对话（避免数据不足）
          - 距上次深巩固 ≥ 24h
        """
        import time as _time
        now = _time.time()
        from app.config.settings import PORTRAIT_DEEP_HOURS, PORTRAIT_DEEP_MIN_TURNS

        if self._turns_since_last_deep < PORTRAIT_DEEP_MIN_TURNS:
            return
        if now - self._last_deep_update < PORTRAIT_DEEP_HOURS * 3600:
            return
        self._last_deep_update = now
        self._turns_since_last_deep = 0

        # ── 用户 dim 1: 核心特征 ──
        try:
            self._deep_update_user_core(ctx_obj)
        except Exception as exc:
            logger.warning("用户核心特征深巩固失败: %s", exc)

        # ── AI dim 1: 核心表达特征 ──
        try:
            self._deep_update_ai_core(ctx_obj)
        except Exception as exc:
            logger.warning("AI 核心特征深巩固失败: %s", exc)

        # ── 一致性审查 ──
        try:
            self._cross_dimension_review()
        except Exception as exc:
            logger.warning("画像一致性审查失败: %s", exc)

        self._manager.apply_state_machine()
        self._manager.save()
        logger.info("深巩固画像更新完成 (dim 1 + 一致性审查)")

    def _deep_update_user_core(self, ctx: Any):
        """用户核心特征深巩固 (dim usr1)。

        数据来源: Distill 模式 + PatternDiscovery + PersonaSymmetry + 长期记忆统计。
        """
        evidence_summary = []

        # 从模式发现层读取
        try:
            if hasattr(ctx, "_pattern_discovery") and ctx._pattern_discovery:
                pd = ctx._pattern_discovery
                obs = getattr(pd, "_observations", [])
                if obs:
                    evidence_summary.append(f"模式发现: {len(obs)} 条观察")
                    for o in obs[-10:]:
                        evidence_summary.append(f"  - {o}")
        except Exception:
            pass

        # 从人格对称分析读取
        try:
            from app.analysis.symmetry import PersonaSymmetry
            if hasattr(ctx, "co_tracker") and hasattr(ctx, "ai_co_tracker"):
                user_data = ctx.co_tracker.export_for_symmetry()
                ai_data = ctx.ai_co_tracker.export_for_symmetry()
                symmetry = PersonaSymmetry(user_data, ai_data, from_dicts=True)
                blind_spots = symmetry.analyze()
                if blind_spots:
                    evidence_summary.append(f"认知盲区: {len(blind_spots)} 处")
        except Exception:
            pass

        # 从 Qdrant 长期统计读取
        try:
            all_data = ctx.memory_service.list_all_cached()
            total_mems = len(all_data)
            if total_mems > 0:
                days_span = "N/A"
                try:
                    timestamps = [m.get("metadata", {}).get("timestamp", 0) or 0 for m in all_data]
                    valid_ts = [t for t in timestamps if t > 0]
                    if valid_ts:
                        days_span = f"{int((max(valid_ts) - min(valid_ts)) / 86400)}天"
                except Exception:
                    pass
                evidence_summary.append(f"记忆库: {total_mems} 条 / 跨度 {days_span}")
        except Exception:
            pass

        if not evidence_summary:
            return

        existing = self._manager.get_dim_entries("usr1")
        existing_texts = [f"[{e.id}] {e.text}" for e in existing]

        prompt = f"""你正在重述用户画像的"核心特征"维度（深巩固 24h）。

现有内容:
{chr(10).join(existing_texts) if existing_texts else '(空)'}

近期证据摘要:
{chr(10).join(evidence_summary)}

规则:
1. 重述用户的稳定核心特征（超14天未确认的降 confidence，标注 [cooling]）
2. 保留仍成立的条目的 entry ID (<!-- entry:usr1-NNN -->)
3. 为新的稳定特征创建新条目
4. 删除已不成立的特征
5. 上限 10 条
6. 只输出该维度的 Markdown 内容

请重述更新后的维度内容:"""
        result = self._call_local_llm(prompt, timeout=45)
        if result:
            self._apply_llm_dim_update("usr1", result)

    def _deep_update_ai_core(self, ctx: Any):
        """AI 核心表达特征深巩固 (dim ai1)。"""
        evidence_summary = []

        # 从 AI 记忆库统计
        try:
            all_data = ctx.ai_memory_service.list_all_cached()
            if all_data:
                evidence_summary.append(f"AI 记忆库: {len(all_data)} 条")
                # 表达色调统计
                categories = Counter(
                    (m.get("metadata") or {}).get("emotion_valence_bin", "neutral")
                    for m in all_data)
                evidence_summary.append(f"情绪分布: {dict(categories)}")
        except Exception:
            pass

        # 从 AI 共现追踪读取
        try:
            if hasattr(ctx, "ai_co_tracker"):
                evidence_summary.append("AI 共现数据可用")
        except Exception:
            pass

        if not evidence_summary:
            return

        existing = self._manager.get_dim_entries("ai1")
        existing_texts = [f"[{e.id}] {e.text}" for e in existing]

        prompt = f"""你正在重述 AI 画像的"核心表达特征"维度（深巩固 24h）。

现有内容:
{chr(10).join(existing_texts) if existing_texts else '(空)'}

近期证据摘要:
{chr(10).join(evidence_summary)}

规则:
1. 重述 AI 的稳定核心表达特征
2. 保留仍成立的条目的 entry ID (<!-- entry:ai1-NNN -->)
3. 上限 10 条
4. 只输出该维度的 Markdown 内容

请重述更新后的维度内容:"""
        result = self._call_local_llm(prompt, timeout=45)
        if result:
            self._apply_llm_dim_update("ai1", result)

    def _cross_dimension_review(self):
        """一致性审查 — 检查跨维度矛盾。

        如"用户.4.信任度=高"但"AI.4.边界意识=保持距离" → 标记审查。
        """
        all_entries = self._manager.get_all_active()
        if len(all_entries) < 4:
            return  # 数据太少，跳过审查

        # 收集所有活跃条目文本
        all_text = []
        for dim, entries in all_entries.items():
            for e in entries:
                all_text.append(f"[{e.id}] {e.text}")

        if not all_text:
            return

        prompt = f"""你正在审查一份认知画像的内部一致性。

画像内容:
{chr(10).join(all_text)}

请检查以下类型的矛盾:
1. 同一实体的不同维度描述相互矛盾
2. 用户画像和 AI 画像对同一关系的认知冲突
3. 置信度高的条目之间存在逻辑不一致

如果发现矛盾，输出格式:
  [CONFLICT] dim_a vs dim_b: 冲突描述
如果没有矛盾，输出:
  [OK] 无矛盾

请审查:"""
        result = self._call_local_llm(prompt, timeout=30)
        if result and "[CONFLICT]" in result:
            logger.info("画像一致性审查发现矛盾: %s", result[:200])
