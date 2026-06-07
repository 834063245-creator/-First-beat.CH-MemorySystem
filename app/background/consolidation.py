"""巩固引擎（Consolidation Engine）— 引擎的独立后台认知节律。

两级机制：
  空闲触发（原有）：Level 1-3，在用户离开时做回顾和预热
  独立节律（新增）：浅巩固每 4h、深巩固每 24h，用户在与不在都跑
"""

import json
import logging
import os
import time as _time
import threading
from collections import Counter
from datetime import datetime, date

from app.tools.atomic import atomic_write

from app.brain.semantic import extract_tags


from app.config.settings import (
    IDLE_PREHEAT_QUERIES,
    IDLE_LEVEL2_HOURS,
    IDLE_LEVEL3_HOURS,
    CONSOLIDATION_SHALLOW_INTERVAL,
    CONSOLIDATION_DEEP_INTERVAL,
    ARCHIVAL_THRESHOLD_DAYS,
)
from app.tools.dispatch import query_memory

logger = logging.getLogger(__name__)

from app.config.settings import STOP_WORDS


def _extract_keywords(text: str, topk: int = 10) -> list[str]:
    """用 jieba TF-IDF 提取关键词，过滤停用词和短词。"""
    if not text or not text.strip():
        return []
    words = extract_tags(text, topk=topk * 2)
    return [w for w in words if len(w) >= 2 and w.lower() not in STOP_WORDS][:topk]


# ── 状态读写 ──────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "last_idle_time": "",
        "last_review_date": "",
        "preheat_queries": [],
        "today_topics": [],
        "pending_conflicts": [],
        "level3_triggered_today": False,
        "last_preheat_time": "",
        "last_shallow_consolidation": 0,   # 时间戳，引擎独立节律
        "last_deep_consolidation": 0,       # 时间戳
        "archived_topic_count": 0,          # 累计归档话题簇数
    }


def _load_state(state_path: str) -> dict:
    path = state_path
    if not os.path.exists(path):
        return _default_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 确保所有字段存在
        merged = _default_state()
        merged.update(data)
        return merged
    except (json.JSONDecodeError, OSError):
        return _default_state()


def _save_state(state: dict, state_path: str):
    atomic_write(state_path, state)


# ── 巩固引擎 ──────────────────────────────────────────────

class ConsolidationEngine:
    """后台巩固引擎 — 利用空闲时间做记忆巩固、预热和冲突扫描。"""

    MAX_PREHEAT_CACHE = 10

    def __init__(self, chroma_service, personality_store, behavior_store, chat_history, co_tracker, *,
                 state_path: str, notes_path: str,
                 temporal_pattern_index=None, topic_affinity=None):
        self._state_path = state_path
        self._notes_path = notes_path
        self._chroma = chroma_service
        self._personality = personality_store
        self._behavior = behavior_store
        self._chat_history = chat_history
        self._co_tracker = co_tracker
        self._temporal_index = temporal_pattern_index
        self._topic_affinity = topic_affinity
        self._topic_tree = None
        self._tag_index = None
        self._preheat_cache: dict[str, list[dict]] = {}
        self._cache_lock = threading.Lock()
        self._state_lock = threading.Lock()

    def _read_state(self) -> dict:
        with self._state_lock:
            return _load_state(self._state_path)

    def _write_state(self, state: dict) -> None:
        with self._state_lock:
            _save_state(state, self._state_path)
    # ── 主入口 ────────────────────────────────────────────

    def on_idle(self, idle_hours: float):
        """按空闲时长触发不同级别的后台工作。"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state = self._read_state()
        state["last_idle_time"] = now_str

        level3_already = state.get("level3_triggered_today", False)

        try:
            if idle_hours >= IDLE_LEVEL3_HOURS and not level3_already:
                logger.info("后台巩固 Level 3: 开始日巩固 + 冲突检测")
                try:
                    cons = self._consolidate_day()
                    logger.info("后台日巩固完成: %s", cons.get("summary", ""))
                except Exception as exc:
                    logger.error("后台日巩固失败: %s", exc)
                try:
                    conflicts = self._check_conflicts()
                    logger.info("后台冲突检测完成: %d 个候选", len(conflicts))
                except Exception as exc:
                    logger.error("后台冲突检测失败: %s", exc)
                state["level3_triggered_today"] = True
        except Exception as exc:
            logger.error("后台巩固 Level 3 处理异常: %s", exc)

        try:
            if idle_hours >= IDLE_LEVEL2_HOURS:
                logger.info("后台巩固 Level 2: 开始回顾 + 预测预热")
                try:
                    review = self._review_today()
                    logger.info("后台回顾完成: %s", review.get("summary", ""))
                except Exception as exc:
                    logger.error("后台回顾失败: %s", exc)
                try:
                    self._preheat_predictions()
                    logger.info("后台预测预热完成")
                except Exception as exc:
                    logger.error("后台预测预热失败: %s", exc)
        except Exception as exc:
            logger.error("后台巩固 Level 2 处理异常: %s", exc)

        self._write_state(state)

    # ── Level 2 方法 ──────────────────────────────────────

    def _review_today(self) -> dict:
        """回顾今天的记忆，提取话题和情绪统计。"""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        all_memories = self._chroma.list_all_cached()
        today_mems = [
            m for m in all_memories
            if (m.get("metadata") or {}).get("timestamp", 0) >= today_start
        ]

        # 话题提取
        all_text = ""
        emotional_count = 0
        for m in today_mems:
            meta = m.get("metadata") or {}
            all_text += (meta.get("user_message", "") or "") + " "
            all_text += (meta.get("summary", "") or "") + " "
            if meta.get("emotional_intensity", 0) >= 2:
                emotional_count += 1

        keywords = _extract_keywords(all_text, topk=20)
        counter = Counter(keywords)
        top_topics = [w for w, _ in counter.most_common(10)]

        total = len(today_mems)
        mood_warning = False
        if total > 0 and emotional_count / total > 0.3:
            mood_warning = True

        review = {
            "total": total,
            "top_topics": top_topics,
            "emotional_count": emotional_count,
            "emotional_ratio": round(emotional_count / total, 2) if total else 0,
            "mood_warning": mood_warning,
            "summary": f"今天{total}条记忆，情绪密集{emotional_count}/{total}，话题：{'、'.join(top_topics[:5])}",
        }

        state = self._read_state()
        state["today_topics"] = top_topics
        state["last_review_date"] = date.today().isoformat()
        self._write_state(state)

        return review

    def _preheat_predictions(self):
        """基于今天话题 + 行为模式生成预测查询并预热缓存。"""
        state = self._read_state()
        today_topics = state.get("today_topics", [])

        # 收集行为模式标签
        behavior_keywords = set()
        try:
            behaviors = self._behavior.list_all()
            for b in behaviors:
                content = b.get("content", "") or ""
                behavior_keywords.update(_extract_keywords(content, topk=5))
        except Exception as exc:
            logger.debug("后台行为模式读取跳过: %s", exc)

        # 时间节奏检索：去年同期 / 上月同日
        time_titles = []
        try:
            now = datetime.now()
            all_mems = self._chroma.list_all_cached()
            for m in all_mems:
                meta = m.get("metadata") or {}
                ts = meta.get("timestamp", 0)
                if ts <= 0:
                    continue
                try:
                    mem_dt = datetime.fromtimestamp(ts)
                except (OSError, ValueError):
                    continue
                # 去年同期（月日相同，年份不同）
                if mem_dt.month == now.month and mem_dt.day == now.day and mem_dt.year != now.year:
                    summary = meta.get("summary", "") or ""
                    if summary:
                        time_titles.append(summary)
                # 上月同日
                if now.month > 1:
                    if mem_dt.month == now.month - 1 and mem_dt.day == now.day:
                        summary = meta.get("summary", "") or ""
                        if summary:
                            time_titles.append(summary)
        except Exception as exc:
            logger.debug("后台巩固 时间节奏检索跳过: %s", exc)

        # 合并去重，取 top N 作为预测 query
        combined = list(dict.fromkeys(today_topics + list(behavior_keywords)))
        for t in time_titles:
            combined.extend(_extract_keywords(t, topk=3))
        seen = set()
        unique_queries = []
        for q in combined:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        queries = unique_queries[:IDLE_PREHEAT_QUERIES]

        # 对每个预测 query 做检索预热，缓存为 chat 流程兼容格式
        collection = self._chroma._collection
        for query in queries:
            try:
                results = query_memory(collection, query=query, top_k=5)
                if results and not results[0].get("error"):
                    # 转成 chat 流程兼容格式
                    normalized = []
                    for r in results:
                        normalized.append({
                            "id": r.get("id", ""),
                            "document": r.get("_preview", ""),
                            "metadata": r,
                            "summary": r.get("summary", ""),
                            "hit_count": r.get("hit_count", 0) or 0,
                            "source": "dmn_preheat",
                            "distance": 1.0 - r.get("similarity", 0) if r.get("similarity") else 0.5,
                        })
                    if normalized:
                        with self._cache_lock:
                            if len(self._preheat_cache) >= self.MAX_PREHEAT_CACHE:
                                oldest_key = next(iter(self._preheat_cache))
                                del self._preheat_cache[oldest_key]
                            self._preheat_cache[query] = normalized
            except Exception as exc:
                logger.debug("后台巩固 预热查询 '%s' 失败: %s", query, exc)

        state["preheat_queries"] = queries
        state["last_preheat_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_state(state)

    def get_preheated(self, user_message: str) -> list[dict] | None:
        """检查用户消息是否匹配预热的 query，命中则返回缓存结果。"""
        if not user_message:
            return None
        try:
            msg_keywords = set(_extract_keywords(user_message, topk=8))
            if not msg_keywords:
                return None
            best_match = None
            best_overlap = 0
            with self._cache_lock:
                if not self._preheat_cache:
                    return None
                for query, results in list(self._preheat_cache.items()):
                    query_keywords = set(_extract_keywords(query, topk=8))
                    overlap = len(msg_keywords & query_keywords)
                    if overlap >= 1 and overlap > best_overlap:
                        best_overlap = overlap
                        best_match = results
            return best_match
        except Exception as exc:
            logger.debug("后台巩固 get_preheated 异常: %s", exc)
            return None

    # ── Level 3 方法 ──────────────────────────────────────

    def _consolidate_day(self) -> dict:
        """日巩固：统计今天话题分布和情绪状况。"""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        all_memories = self._chroma.list_all_cached()
        today_mems = [
            m for m in all_memories
            if (m.get("metadata") or {}).get("timestamp", 0) >= today_start
        ]

        all_text = ""
        emotional_count = 0
        for m in today_mems:
            meta = m.get("metadata") or {}
            all_text += (meta.get("user_message", "") or "") + " "
            all_text += (meta.get("summary", "") or "") + " "
            if meta.get("emotional_intensity", 0) >= 2:
                emotional_count += 1

        keywords = _extract_keywords(all_text, topk=30)
        topic_dist = Counter(keywords).most_common(10)

        # stale 候选检测：今天记忆的 tag 找旧记忆中同名 tag 的不同内容
        pending = []
        today_tags = set()
        for m in today_mems:
            meta = m.get("metadata") or {}
            tags_str = meta.get("tags", "") or ""
            for t in tags_str.split(","):
                t = t.strip()
                if len(t) >= 2:
                    today_tags.add(t)

        if today_tags and len(all_memories) > len(today_mems):
            try:
                for m in all_memories:
                    if len(pending) >= 20:
                        break
                    meta = m.get("metadata") or {}
                    ts = meta.get("timestamp", 0)
                    if ts >= today_start:
                        continue  # 跳过今天的
                    mem_tags = set()
                    for t in (meta.get("tags", "") or "").split(","):
                        t = t.strip()
                        if t:
                            mem_tags.add(t)
                    overlap = today_tags & mem_tags
                    if overlap:
                        age_days = (datetime.now().timestamp() - ts) / 86400
                        if age_days > 7:
                            pending.append({
                                "id": m["id"],
                                "tag": list(overlap)[0],
                                "age_days": round(age_days, 1),
                            })
            except Exception as exc:
                logger.debug("后台巩固 stale 扫描跳过: %s", exc)

        result = {
            "total": len(today_mems),
            "topic_distribution": [{"topic": w, "count": c} for w, c in topic_dist],
            "emotional_count": emotional_count,
            "emotional_ratio": round(emotional_count / len(today_mems), 2) if today_mems else 0,
            "stale_candidates": len(pending),
            "summary": f"今天{len(today_mems)}条记忆，{len(topic_dist)}个话题，{emotional_count}条情绪密集",
        }

        state = self._read_state()
        state["pending_conflicts"] = pending[:10]
        self._write_state(state)

        return result

    def _check_conflicts(self) -> list[dict]:
        """冲突预扫描：最近 7 天记忆 vs 旧记忆的关键词冲突检测。"""
        seven_days_ago = (datetime.now().timestamp() - 7 * 86400)
        all_memories = self._chroma.list_all_cached()

        # 最近 7 天的记忆
        recent = [
            m for m in all_memories
            if (m.get("metadata") or {}).get("timestamp", 0) >= seven_days_ago
        ]

        conflicts = []
        now_ts = datetime.now().timestamp()
        for m in recent[:50]:  # 控制扫描量
            try:
                meta = m.get("metadata") or {}
                tags_str = meta.get("tags", "") or ""
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                if not tags:
                    continue
                # 在旧记忆中找同 tag 不同内容
                for old in all_memories:
                    if len(conflicts) >= 10:
                        break
                    if old["id"] == m["id"]:
                        continue
                    old_meta = old.get("metadata") or {}
                    old_ts = old_meta.get("timestamp", 0)
                    if old_ts >= seven_days_ago:
                        continue  # 不是旧记忆
                    old_tags_str = old_meta.get("tags", "") or ""
                    old_tags = set(t.strip() for t in old_tags_str.split(",") if t.strip())
                    if set(tags) & old_tags:
                        conflicts.append({
                            "new_id": m["id"][:8],
                            "old_id": old["id"][:8],
                            "new_id_full": m["id"],
                            "old_id_full": old["id"],
                            "shared_tags": list(set(tags) & old_tags),
                            "new_summary": (meta.get("summary", "") or "")[:50],
                            "old_summary": (old_meta.get("summary", "") or "")[:50],
                            "detected_at": now_ts,
                        })
            except Exception:
                continue

        state = self._read_state()
        state["pending_conflicts"] = conflicts[:10]
        self._write_state(state)

        return conflicts

    # ── 引擎独立巩固节律 ─────────────────────────────────────
    # 不依赖用户空闲，按固定间隔由 consolidation_worker 触发

    def consolidate_shallow(self):
        """浅巩固：语义重复检测 + 话题簇追踪 + 时间模式/亲和图更新。

        每 4 小时由独立线程触发，与用户活跃与否无关。
        不放 LLM，纯算法。
        """
        try:
            all_mems = self._chroma.list_all_cached()
            state = self._read_state()

            # 确保 embedding cache 已构建（供下方重复检测使用）
            if not self._chroma._emb_cache:
                self._chroma._build_embedding_cache()

            # 更新话题亲和图 + 时间模式索引（只处理新记忆）
            try:
                last_shallow = state.get("last_shallow_consolidation", 0)
                new_mems = [m for m in all_mems
                            if (m.get("metadata") or {}).get("timestamp", 0) > last_shallow] if last_shallow else all_mems
                if new_mems:
                    if self._topic_affinity is not None:
                        for m in new_mems:
                            tags_str = (m.get("metadata") or {}).get("tags", "") or ""
                            tags = [t.strip() for t in tags_str.split(",") if len(t.strip()) >= 2]
                            if len(tags) >= 2:
                                self._topic_affinity.update(tags)
                    if self._temporal_index is not None:
                        self._temporal_index.update(new_mems)
                    # 标签嵌入索引：收集全库标签 → 嵌入新标签（替代话题树共现依赖）
                    if self._tag_index is not None and self._tag_index._embed_fn is not None:
                        all_tags: set[str] = set()
                        for m in new_mems:
                            tags_str = (m.get("metadata") or {}).get("tags", "") or ""
                            for t in tags_str.split(","):
                                t = t.strip()
                                if len(t) >= 2:
                                    all_tags.add(t)
                        if all_tags:
                            self._tag_index.update(list(all_tags))
            except Exception as exc:
                logger.debug("话题亲和图/时间模式/标签嵌入更新跳过: %s", exc)

            # 检测语义重复：利用 ChromaDB 内建 hnsw 索引逐条近邻查询。
            # 替代旧的双层 for 循环 O(n²)，每条记忆只查 top-3 近邻，O(n log n)。
            merged = 0
            try:
                seen_dupes: set[str] = set()
                collection = self._chroma._collection
                # 只查最近 30 天入库的记忆（旧记忆重复已在之前的浅巩固中处理）
                recent_cutoff = _time.time() - 86400 * 30
                recent_mems = [
                    m for m in all_mems
                    if (m.get("metadata") or {}).get("timestamp", 0) >= recent_cutoff
                    and not (m.get("metadata") or {}).get("stale", False)
                ]
                # 批量查询：每批 100 条，利用 ChromaDB 内部批处理
                BATCH_SIZE = 100
                for batch_start in range(0, len(recent_mems), BATCH_SIZE):
                    batch = recent_mems[batch_start:batch_start + BATCH_SIZE]
                    batch_embs = []
                    batch_ids = []
                    for m in batch:
                        emb = (m.get("metadata") or {}).get("embedding") or \
                              self._chroma._emb_cache.get(m.get("id", ""))
                        if emb:
                            batch_embs.append(emb)
                            batch_ids.append(m["id"])
                    if not batch_embs:
                        continue
                    # ChromaDB query：每条找 top-3 近邻
                    results = collection.query(
                        query_embeddings=batch_embs,
                        n_results=3,
                        include=["metadatas", "distances"],
                    )
                    for i, mid in enumerate(batch_ids):
                        if mid in seen_dupes:
                            continue
                        neighbor_ids = results.get("ids", [[]])[i] if i < len(results.get("ids", [[]])) else []
                        distances = results.get("distances", [[]])[i] if i < len(results.get("distances", [[]])) else []
                        for j, nid in enumerate(neighbor_ids):
                            if nid == mid:
                                continue  # 跳过自己
                            dist = distances[j] if j < len(distances) else 1.0
                            # ChromaDB 返回 L2 或 cosine distance；阈值 0.05 = cosine sim ~0.95
                            if dist < 0.05:
                                # 额外校验：共享 tag
                                mi_meta = (next((mm.get("metadata") for mm in batch if mm["id"] == mid), None) or {})
                                mi_tags = set((mi_meta.get("tags", "") or "").split(","))
                                nj_meta = results.get("metadatas", [[]])[i][j] if j < len(results.get("metadatas", [[]])[i]) else {}
                                nj_tags = set((nj_meta.get("tags", "") or "").split(","))
                                if not (mi_tags & nj_tags):
                                    continue
                                merged += 1
                                seen_dupes.add(mid)
                                seen_dupes.add(nid)
                                # 标记旧记忆被取代（较新的优先保留）
                                mid_ts = (next((mm.get("metadata") for mm in batch if mm["id"] == mid), None) or {}).get("timestamp", 0)
                                nj_ts = nj_meta.get("timestamp", 0)
                                if mid_ts >= nj_ts:
                                    self._chroma.supersede_memory(nid, mid, "语义重复（浅巩固检测）")
                                else:
                                    self._chroma.supersede_memory(mid, nid, "语义重复（浅巩固检测）")
                                break  # 每条记忆最多标记一次
            except Exception as exc:
                logger.debug("语义重复检测异常（回退跳过）: %s", exc)

            # 更新话题簇最后活跃时间（已有追踪，无需额外数据）
            state["last_shallow_consolidation"] = _time.time()
            self._write_state(state)

            if merged:
                logger.info("后台巩固 浅巩固完成: %d 对重复, 亲和图/%s新模式",
                           merged,
                           len(new_mems) if new_mems else 0)
            else:
                logger.debug("后台巩固 浅巩固完成: 无显著重复")

            # 冷却扫描：warm 记忆 hit_count=0 且入库 > 14 天 → cool
            try:
                now_ts = _time.time()
                cutoff = now_ts - 86400 * 14
                cooling = []
                for m in all_mems:
                    meta = m.get("metadata") or {}
                    if meta.get("heat") == "warm" and meta.get("hit_count", 0) == 0:
                        ts = meta.get("timestamp", 0)
                        if ts and ts < cutoff:
                            cooling.append(m["id"])
                if cooling:
                    # 批量更新
                    for i in range(0, len(cooling), 100):
                        batch = cooling[i:i + 100]
                        self._chroma._collection.update(
                            ids=batch,
                            metadatas=[{"heat": "cool"}] * len(batch),
                        )
                    logger.info("冷却扫描: %d 条 warm→cool", len(cooling))
            except Exception as exc:
                logger.debug("冷却扫描异常: %s", exc)

            # 话题树重建（跟冷却扫描同一时机，每 4h 浅巩固时触发）
            try:
                from app.memory.tree import TopicTree
                data_dir = os.path.dirname(self._state_path)
                topic_tree = TopicTree(data_dir)
                if self._topic_affinity:
                    topic_tree.rebuild(self._topic_affinity._matrix)
                    self._topic_tree = topic_tree
            except Exception:
                logger.debug("话题树重建异常")

            # 标签嵌入索引初始化/更新（替代话题树的冷启动依赖）
            try:
                if self._tag_index is None:
                    from app.memory.tag_index import TagEmbeddingIndex
                    data_dir = os.path.dirname(self._state_path)
                    self._tag_index = TagEmbeddingIndex(data_dir)
                    # 惰性注入 embed_fn（避免循环导入）
                    if self._tag_index._embed_fn is None:
                        from app.llm.embed import local_embed_batch
                        self._tag_index.set_embed_fn(local_embed_batch)
                # 首次启动时全量嵌入所有已知标签
                if self._tag_index.size() == 0:
                    all_tags: set[str] = set()
                    for m in all_mems:
                        tags_str = (m.get("metadata") or {}).get("tags", "") or ""
                        for t in tags_str.split(","):
                            t = t.strip()
                            if len(t) >= 2:
                                all_tags.add(t)
                    if all_tags:
                        logger.info("标签嵌入索引首次构建: %d 个标签", len(all_tags))
                        self._tag_index.update(list(all_tags))
            except Exception:
                logger.debug("标签嵌入索引初始化异常", exc_info=True)

            # 事实冲突检测（话题树重建之后，利用新鲜的分支信息）
            try:
                self._detect_fact_contradictions()
            except Exception:
                logger.debug("事实冲突检测异常", exc_info=True)

            # 人格对称性分析（双共现差分，每 4h 跟浅巩固同频）
            try:
                from app.analysis.symmetry import PersonaSymmetry
                data_dir = os.path.dirname(self._state_path)
                sym = PersonaSymmetry(
                    f"{data_dir}/co_occurrence.json",
                    f"{data_dir}/ai_co_occurrence.json",
                )
                sym.analyze()
                obs = sym.get_observations()
                if obs:
                    cache_path = os.path.join(data_dir, "cache", "blind_spots.json")
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    atomic_write(cache_path, {
                        "updated_at": _time.time(),
                        "observations": obs,
                    })
            except Exception:
                logger.debug("人格对称性分析异常", exc_info=True)
        except Exception as exc:
            logger.error("后台巩固 浅巩固失败: %s", exc)

    # ── 事实冲突检测 ──────────────────────────────────────────
    # 语义级的事实时序推理，不放 LLM，纯算法。

    # 两层过滤的阈值
    CONTRADICTION_SIM_LOW = 0.75        # 低于此值=不同事实，跳过
    CONTRADICTION_SIM_HIGH = 0.95       # 高于此值=近似重复（已有重复检测），跳过
    CONTENT_SHIFT_THRESHOLD = 0.85      # 路径B：sim低于此值=内容确实变了
    CONTRADICTION_RECENT_DAYS = 7        # "新记忆"窗口

    def _detect_fact_contradictions(self) -> int:
        """事实冲突检测：语义相关+情绪翻转 → 标记旧记忆被取代。

        三层漏斗（由粗到细，控制 O(n²) 扩散）：
        1. tag 交集粗筛 — 新旧记忆共享至少 1 个 tag 才进入下一层
        2. embedding 语义精筛 — sim ∈ [0.75, 0.95] 确认"说同一件事但细节不同"
        3. 情绪翻转 / 事实位移判定 — 标记者超旧记忆

        返回标记数。
        """
        try:
            all_mems = self._chroma.list_all_cached()
            now_ts = _time.time()
            cutoff = now_ts - self.CONTRADICTION_RECENT_DAYS * 86400

            # 分离新旧记忆（按时间戳 + 排除已 stale）
            new_mems = []
            old_mems = []
            for m in all_mems:
                meta = m.get("metadata") or {}
                if meta.get("stale", False):
                    continue
                ts = meta.get("timestamp", 0)
                if ts >= cutoff:
                    new_mems.append(m)
                else:
                    old_mems.append(m)

            if not new_mems or not old_mems:
                return 0

            # 确保 embedding cache 可用
            if not self._chroma._emb_cache:
                self._chroma._build_embedding_cache()

            # 预计算所有旧记忆的 tag set（第一层粗筛，避免每轮重复 parse）
            old_mem_tags: dict[str, set[str]] = {}
            for old_m in old_mems:
                old_tags_str = (old_m.get("metadata") or {}).get("tags", "") or ""
                old_tags = {t.strip() for t in old_tags_str.split(",") if len(t.strip()) >= 2}
                if old_tags:
                    old_mem_tags[old_m["id"]] = old_tags

            superseded = 0

            for new_m in new_mems:
                new_meta = new_m.get("metadata") or {}
                new_tags_str = new_meta.get("tags", "") or ""
                new_tags = {t.strip() for t in new_tags_str.split(",") if len(t.strip()) >= 2}
                if not new_tags:
                    continue
                new_emb = self._chroma._emb_cache.get(new_m["id"])
                if not new_emb:
                    continue

                # 话题树分支扩展（供 reason 日志使用，不作为硬门槛）
                if self._topic_tree:
                    new_branch = set(self._topic_tree.get_branch(
                        next(iter(new_tags), "")
                    ))
                else:
                    new_branch = new_tags

                for old_m in old_mems:
                    old_emb = self._chroma._emb_cache.get(old_m["id"])
                    if not old_emb or len(old_emb) != len(new_emb):
                        continue

                    # ── 第一层：tag 交集粗筛（替代已移除的 CNN） ──
                    old_tags = old_mem_tags.get(old_m["id"], set())
                    if not old_tags:
                        continue
                    shared = new_tags & old_tags
                    if not shared:
                        continue  # 无共同标签 → 几乎不可能是同一事实域

                    # ── 第二层：embedding 语义精筛 ──
                    dot = sum(a * b for a, b in zip(new_emb, old_emb))
                    n1 = (sum(a * a for a in new_emb) ** 0.5) or 1e-10
                    n2 = (sum(b * b for b in old_emb) ** 0.5) or 1e-10
                    sim = dot / (n1 * n2)

                    if sim < self.CONTRADICTION_SIM_LOW:
                        continue  # 不同话题
                    if sim > self.CONTRADICTION_SIM_HIGH:
                        continue  # 近似重复（已有重复检测处理）

                    # ── 第三层：事实变化判定（双路径） ──
                    old_meta = old_m.get("metadata") or {}
                    new_val = new_meta.get("emotion_valence_bin", "") or ""
                    old_val = old_meta.get("emotion_valence_bin", "") or ""
                    new_summary = (new_meta.get("summary", "") or "")[:40]
                    old_summary = (old_meta.get("summary", "") or "")[:40]

                    if new_val and old_val and new_val != old_val:
                        # 路径A：情绪翻转
                        flip_kind = "情绪翻转"
                        flip_detail = f"{old_val}→{new_val}"
                    elif sim < self.CONTENT_SHIFT_THRESHOLD:
                        # 路径B：语义位移（同一事实域但内容有偏差）
                        flip_kind = "事实更新"
                        flip_detail = f"语义位移 {sim:.2f}"
                    else:
                        continue  # 既没情绪翻也没内容变，不算冲突

                    # 标记冲突
                    reason = (
                        f"{flip_kind}: {flip_detail} "
                        f"sim={sim:.2f} "
                        f"tags={list(shared)[:3]}"
                    )
                    self._chroma.supersede_memory(old_m["id"], new_m["id"], reason)
                    superseded += 1

                    # 状态追踪
                    state = self._read_state()
                    conflicts = state.get("pending_conflicts", [])
                    conflicts.append({
                        "new_id": new_m["id"][:8],
                        "old_id": old_m["id"][:8],
                        "shared_tags": list(shared)[:3],
                        "new_summary": new_summary,
                        "old_summary": old_summary,
                        "detected_at": now_ts,
                        "reason": reason,
                    })
                    state["pending_conflicts"] = conflicts[-10:]
                    self._write_state(state)

                    break  # 一条新记忆只覆盖一条旧记忆

            if superseded:
                logger.info(
                    "事实冲突检测: %d 条旧记忆被取代 (扫描 %d 新 vs %d 旧)",
                    superseded, len(new_mems), len(old_mems),
                )
            return superseded

        except Exception as exc:
            logger.warning("事实冲突检测异常: %s", exc)
            return 0

    def consolidate_deep(self):
        """深巩固：归档评估 + 话题笔记生成 + 跨日模式统计 + 情绪淡化。

        每 24 小时由独立线程触发。
        """
        try:
            state = self._read_state()
            archived = self._assess_archival()
            notes_count = self._generate_topic_notes()
            # 情绪淡化：扫描 emotional_intensity>=1 且长时间未提及的记忆
            try:
                self._chroma._apply_emotional_desensitization()
            except Exception as exc:
                logger.debug("情绪淡化跳过: %s", exc)
            state["last_deep_consolidation"] = _time.time()
            state["archived_topic_count"] = state.get("archived_topic_count", 0) + archived
            self._write_state(state)
            logger.info("后台巩固 深巩固完成: %d 个话题簇归档, %d 条话题笔记", archived, notes_count)
        except Exception as exc:
            logger.error("后台巩固 深巩固失败: %s", exc)

    def _assess_archival(self) -> int:
        """评估哪些话题簇应该归档。

        按 tag 聚合记忆，取每条记忆的 last_hit_time 中位数。
        中位数超过 ARCHIVAL_THRESHOLD_DAYS 天未被关注 → 整簇归档。
        不删除，不丢失，只标记 archived=True。
        """
        try:
            all_mems = self._chroma.list_all_cached()
            from collections import defaultdict as _dd
            clusters: dict[str, list[dict]] = _dd(list)

            for m in all_mems:
                meta = m.get("metadata") or {}
                if meta.get("archived", False):
                    continue
                tags_str = (meta.get("tags") or "")
                for t in tags_str.split(","):
                    t = t.strip()
                    if t and len(t) >= 2:
                        clusters[t].append(m)

            now = _time.time()
            cutoff = now - ARCHIVAL_THRESHOLD_DAYS * 86400
            archived_count = 0

            for tag, mems in clusters.items():
                if len(mems) < 3:
                    continue  # 少于 3 条的话题不归档

                last_hits = []
                for m in mems:
                    meta = m.get("metadata") or {}
                    lh = meta.get("last_hit_time") or meta.get("timestamp", 0)
                    if lh and lh > 0:
                        try:
                            last_hits.append(float(lh))
                        except (ValueError, TypeError):
                            continue

                if len(last_hits) < 3:
                    continue

                last_hits.sort()
                median_lh = last_hits[len(last_hits) // 2]

                if median_lh < cutoff:
                    mem_ids = [m["id"] for m in mems if m.get("id")]
                    if mem_ids:
                        try:
                            self._chroma.archive_topic_cluster(tag, mem_ids)
                            archived_count += 1
                        except Exception as exc:
                            logger.debug("归档失败 tag=%s: %s", tag, exc)

            return archived_count
        except Exception as exc:
            logger.error("后台巩固 归档评估失败: %s", exc)
            return 0

    TOPIC_NOTE_MIN_MEMORIES = 5      # 少于该数量的簇不生成笔记
    TOPIC_NOTE_EXPIRE_DAYS = 30       # 超过30天无更新的笔记标记为过期
    TOPIC_NOTE_MAX_NOTES = 100        # 笔记总数上限

    def _generate_topic_notes(self) -> int:
        """为活跃话题簇生成辅助索引笔记。

        每簇取 tag + 首尾时间 + 关键词分布 + 情绪基调，产出结构化笔记，
        存于 topic_notes.json。笔记不替换原始记忆，只作为检索辅助入口。
        """
        try:
            all_mems = self._chroma.list_all_cached()
            from collections import defaultdict as _dd
            clusters: dict[str, list[dict]] = _dd(list)

            for m in all_mems:
                meta = m.get("metadata") or {}
                if meta.get("archived", False):
                    continue
                tags_str = (meta.get("tags") or "")
                for t in tags_str.split(","):
                    t = t.strip()
                    if t and len(t) >= 2:
                        clusters[t].append(m)

            notes = self._load_notes(self._notes_path)
            now = _time.time()
            new_count = 0

            for tag, mems in clusters.items():
                if len(mems) < self.TOPIC_NOTE_MIN_MEMORIES:
                    continue

                # 检查是否已有笔记且未过期
                existing = notes.get(tag, {})
                if existing:
                    last_updated = existing.get("last_updated", 0)
                    if now - last_updated < self.TOPIC_NOTE_EXPIRE_DAYS * 86400:
                        continue

                # 提取时间范围
                timestamps = []
                valences = _dd(int)
                keywords = []
                for m in mems:
                    meta = m.get("metadata") or {}
                    ts = meta.get("timestamp", 0)
                    if ts:
                        timestamps.append(ts)
                    v = meta.get("emotion_valence_bin", "") or ""
                    if v in ("positive", "negative"):
                        valences[v] += 1
                    summary = meta.get("summary", "") or ""
                    if summary:
                        keywords.extend(_extract_keywords(summary, topk=3))

                if not timestamps:
                    continue

                # 情绪基调
                dominant_valence = max(valences, key=valences.get) if valences else "neutral"
                emotional_ratio = round(valences.get("negative", 0) / max(len(mems), 1), 2)

                # 关键词统计
                from collections import Counter
                kw_counter = Counter(keywords)
                top_kws = [w for w, _ in kw_counter.most_common(5)]

                time_min = datetime.fromtimestamp(min(timestamps)).strftime("%Y-%m-%d")
                time_max = datetime.fromtimestamp(max(timestamps)).strftime("%Y-%m-%d")

                notes[tag] = {
                    "tag": tag,
                    "memory_count": len(mems),
                    "time_range": f"{time_min} ~ {time_max}",
                    "top_keywords": top_kws,
                    "dominant_valence": dominant_valence,
                    "emotional_ratio": emotional_ratio,
                    "last_updated": now,
                    "created_at": existing.get("created_at", now),
                }
                new_count += 1

            # 总量控制
            if len(notes) > self.TOPIC_NOTE_MAX_NOTES:
                sorted_notes = sorted(notes.items(), key=lambda x: x[1].get("last_updated", 0))
                for tag, _ in sorted_notes[:len(notes) - self.TOPIC_NOTE_MAX_NOTES]:
                    del notes[tag]

            self._save_notes(notes, self._notes_path)
            return new_count
        except Exception as exc:
            logger.error("后台巩固 话题笔记生成失败: %s", exc)
            return 0

    @staticmethod
    def _load_notes(notes_path: str) -> dict:
        if not os.path.exists(notes_path):
            return {}
        try:
            with open(notes_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _save_notes(notes: dict, notes_path: str):
        try:
            atomic_write(notes_path, notes)
        except OSError as exc:
            logger.debug("dmn OSError: %s", exc)

    def get_topic_notes(self, tags: list[str]) -> list[dict]:
        """根据标签列表检索匹配的话题笔记。"""
        if not tags:
            return []
        notes = self._load_notes(self._notes_path)
        matched = []
        for t in tags:
            if t in notes:
                matched.append(notes[t])
        matched.sort(key=lambda x: x.get("memory_count", 0), reverse=True)
        return matched[:3]

    # ── Prompt 注入 ──────────────────────────────────────

    def get_state_update(self) -> dict:
        """返回结构化状态更新，供认知引擎消费。

        返回::
          {"topics": [str], "conflicts": [(tag, old, new)], "mood_warning": bool}
        """
        try:
            state = self._read_state()
            topics = state.get("today_topics", []) or []
            conflicts_raw = state.get("pending_conflicts", []) or []
            conflicts = []
            seen = set()
            for c in conflicts_raw:
                tag = (c.get("shared_tags") or [None])[0]
                if not tag or tag in seen:
                    continue
                seen.add(tag)
                conflicts.append((tag, c.get("new_summary", "") or "", c.get("old_summary", "") or ""))
            return {
                "topics": topics[:8],
                "conflicts": conflicts[:3],
                "mood_warning": state.get("mood_warning", False),
            }
        except Exception as exc:
            logger.debug("后台巩固 get_state_update 异常: %s", exc)
            return {"topics": [], "conflicts": [], "mood_warning": False}

    def apply_to_cognitive_state(self, cognitive_state) -> None:
        """将 后台巩固 结构化状态写入 CognitiveState。"""
        update = self.get_state_update()
        cognitive_state.today_topics = update.get("topics", [])
        for tag, old, new in update.get("conflicts", []):
            cognitive_state.add_conflict(tag, old, new)

    # ── 可观测状态 ──────────────────────────────────────────

    def get_status(self) -> dict:
        """返回 后台巩固 当前状态（供可观测端点用）。"""
        try:
            state = self._read_state()
            idle_hours = 0.0
            try:
                if self._chat_history and len(self._chat_history.records) >= 1:
                    last_ts = self._chat_history.records[-1].get("timestamp", "")
                    if last_ts:
                        gap = datetime.now() - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                        idle_hours = round(gap.total_seconds() / 3600, 1)
            except Exception as exc:
                logger.debug("dmn exception: %s", exc)
            lsc = state.get("last_shallow_consolidation", 0) or 0
            ldc = state.get("last_deep_consolidation", 0) or 0
            return {
                "level3_triggered_today": state.get("level3_triggered_today", False),
                "last_idle_time": state.get("last_idle_time", ""),
                "last_review_date": state.get("last_review_date", ""),
                "last_preheat_time": state.get("last_preheat_time", ""),
                "preheat_queries": len(state.get("preheat_queries", [])),
                "pending_conflicts": len(state.get("pending_conflicts", [])),
                "today_topics": (state.get("today_topics", []) or [])[:5],
                "cache_size": len(self._preheat_cache),
                "current_idle_hours": idle_hours,
                "last_shallow_consolidation": lsc,
                "last_deep_consolidation": ldc,
                "hours_since_shallow": round((_time.time() - lsc) / 3600, 1) if lsc else None,
                "hours_since_deep": round((_time.time() - ldc) / 3600, 1) if ldc else None,
                "archived_topic_count": state.get("archived_topic_count", 0),
            }
        except Exception as exc:
            logger.debug("后台巩固 get_status 异常: %s", exc)
            return {"error": str(exc)}
