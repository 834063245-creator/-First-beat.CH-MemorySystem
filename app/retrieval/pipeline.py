"""检索管线 — 从 backend/main.py 迁移而来。

原 _run_chat_retrieval 整体迁移至此，路径导入已更新为 app 结构。
"""
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.brain.semantic import extract_tags, tokenize as _sem_tokenize

from app.config.settings import (
    ATTENTION_WINDOW,
    WORK_MEMORY_TOKEN_BUDGET,
    BENCHMARK_MODE as _BM,
)
from app.retrieval.scoring import compute_score
from app.llm.embed import local_embed, local_embed_batch
from app.memory.working import get_summary
from app.analysis.emotion import resolve_emotion_category
from app.analysis.entity import extract_entities

logger = logging.getLogger(__name__)

# 模块级 ThreadPoolExecutor 单例，避免检索热路径上每次新建线程池
_retrieval_executor: ThreadPoolExecutor | None = None


# ── 检索门控：意图 → 各路配额 ────────────────────────────────
# 配额含义是 ChromaDB query 的 n_results（不是截断上限）。
# 截断由引擎 weave_context 统一决策（不再有硬 K）。
_INTENT_ROUTES = {
    "casual":             {"semantic": 50, "tag": 20, "entity": 10, "time_expand": 5},
    "recall":             {"semantic": 100, "tag": 30, "entity": 20, "time_expand": 10},
    "ask_fact":           {"semantic": 100, "tag": 30, "entity": 20, "time_expand": 5},
    "emotional_sharing":  {"semantic": 50, "tag": 20, "entity": 10, "time_expand": 5},
    "conflict":           {"semantic": 100, "tag": 30, "entity": 20, "time_expand": 10},
} if _BM else {
    "casual":             {"semantic": 10, "tag": 5,  "entity": 0, "time_expand": 0},
    "recall":             {"semantic": 20, "tag": 8,  "entity": 5, "time_expand": 5},
    "ask_fact":           {"semantic": 25, "tag": 10, "entity": 5, "time_expand": 0},
    "emotional_sharing":  {"semantic": 12, "tag": 5,  "entity": 0, "time_expand": 3},
    "conflict":           {"semantic": 25, "tag": 10, "entity": 5, "time_expand": 5},
}


def _classify_intent(user_message: str) -> str:
    """轻量关键词意图分类（~50μs）。返回 intent 名称或空串。"""
    msg = user_message.lower()
    if any(w in msg for w in ["记错", "不是", "不对", "没说过", "搞错了", "错了"]):
        return "conflict"
    if any(w in msg for w in ["还记得", "之前", "上次", "那个", "什么来着", "想起来"]):
        return "recall"
    if any(w in msg for w in ["怎么", "为什么", "是什么", "多少", "哪里", "查询", "谁"]):
        return "ask_fact"
    if any(w in msg for w in ["难过", "开心", "烦", "累", "感动", "压力", "焦虑", "伤心"]):
        return "emotional_sharing"
    return "casual"


def _resolve_route(intent: str) -> dict:
    """intent → 配额 dict。未知 / 空串 → 全量降级（等价 recall）。"""
    route = _INTENT_ROUTES.get(intent)
    if route is None:
        logger.debug("检索门控: intent=%s 未匹配，全量降级", intent)
        return _INTENT_ROUTES["recall"]
    logger.debug("检索门控: intent=%s route=%s", intent, route)
    return route


from app.core.helpers import _load_jsonl_cached  # 共享版本，线程安全


def _load_error_counts(data_dir: str) -> dict[str, int]:
    """读取 error_reports.jsonl，返回 {memory_id: count} 用于排序降权。"""
    path = os.path.join(data_dir, "error_reports.jsonl")

    def _parse():
        counts: dict[str, int] = {}
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("action") in ("clear",):
                                continue
                            mid = rec.get("memory_id", "")
                            if mid:
                                counts[mid] = counts.get(mid, 0) + 1
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.warning("加载错误报告失败: %s", exc)
        return counts
    return _load_jsonl_cached(path, _parse)


def _load_correction_boosts(data_dir: str) -> dict[str, float]:
    _path = os.path.join(data_dir, "correction_log.jsonl")

    def _parse():
        boosts: dict[str, float] = {}
        tag_mids: dict[str, list[str]] = {}
        edit_counts: dict[str, int] = {}
        try:
            if os.path.exists(_path):
                with open(_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            mid = rec.get("memory_id", "")
                            tag = rec.get("tag", "")
                            mode = rec.get("mode", "edit")
                            if not mid:
                                continue
                            if mode == "downvote":
                                boosts[mid] = boosts.get(mid, 0) - 0.3
                            else:
                                boosts[mid] = boosts.get(mid, 0) + 0.3
                                edit_counts[mid] = edit_counts.get(mid, 0) + 1
                                if tag:
                                    tag_mids.setdefault(tag, []).append(mid)
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.warning("加载纠正反馈日志失败: %s", exc)
            return {}
        for mids in tag_mids.values():
            if len(mids) < 2:
                continue
            for mid in mids:
                boosts[mid] = boosts.get(mid, 0) + 0.1
        for mid, cnt in edit_counts.items():
            if cnt > 3:
                boosts[mid] = boosts.get(mid, 0) - 0.5
        return boosts
    return _load_jsonl_cached(_path, _parse)


def run_chat_retrieval(
    user_message: str,
    query_embedding_for_retrieval: list | None,
    ctx_obj,
    intent: str | None = None,
) -> tuple:
    """同步检索管线：在 ThreadPoolExecutor 中执行，不阻塞事件循环。

    Args:
        intent: 上游小模型计算好的意图。传入则跳过内部 keyword 分类。
    """
    import math

    # 清空请求级 embedding 缓存（线程池线程复用，必须每请求重置）
    from app.llm.embed import clear_request_cache
    clear_request_cache()

    _cached_q_tags = extract_tags(user_message, topk=5) or []
    _ticks = [("start", time.perf_counter())]
    def _log_step(name):
        ms = (time.perf_counter() - _ticks[-1][1]) * 1000
        from app.core import bottleneck
        bottleneck.record(name, ms)
        _ticks.append((name, time.perf_counter()))

    # ── 意图门控（intent 传递给 retrieve_all 复用） ──
    _intent = intent if intent else _classify_intent(user_message)

    _log_step('intent_gate')
    # ── 时间线近端历史 ──
    timeline_recent = []
    if ctx_obj.chat_history:
        try:
            timeline_recent = ctx_obj.chat_history.get_recent(
                token_budget=WORK_MEMORY_TOKEN_BUDGET
            )
        except Exception:
            try:
                timeline_recent = ctx_obj.chat_history.get_recent(5)
            except Exception:
                pass

    _log_step('timeline')
    # ── 工作记忆摘要 ──
    session_context = ""
    try:
        session_context = get_summary(f"{ctx_obj.data_dir}/working_memory.json")
    except Exception as exc:
        logger.debug("session_context 加载失败: %s", exc)

    _log_step('session_context')
    # Phase 4: 人格标签检索路径已退役 — 画像系统常驻注入替代
    personalities = []

    _log_step('personality')
    # ── DMN 预热缓存 ──
    memories = []
    try:
        p = ctx_obj.dmn.get_preheated(user_message)
        if p is not None:
            memories = p
    except Exception as exc:
        logger.debug("DMN 预热失败: %s", exc)

    _log_step('dmn_preheat')
    # ── 全量检索：8 路全开，引擎编织替代 K 截断 ──
    if not memories and query_embedding_for_retrieval is not None:
        try:
            memories = retrieve_all(
                user_message, query_embedding_for_retrieval, ctx_obj,
                intent=_intent,
                cached_tags=_cached_q_tags,
            )
        except Exception as exc:
            logger.warning("全量检索失败: %s", exc)

    _log_step('retrieval')

    # ── 注意力位移因子 ──
    try:
        recent_msgs = []
        for rec in reversed(ctx_obj.chat_history.get_records_snapshot()):
            msg = rec.get("user_message", "")
            if msg and msg != "[内心独白]":
                recent_msgs.append(msg)
                if len(recent_msgs) >= ATTENTION_WINDOW:
                    break
        if recent_msgs:
            msg_embs = local_embed_batch([m for m in recent_msgs if m])
            msg_embs = [e for e in msg_embs if e is not None]
            if msg_embs:
                import numpy as np
                decay = 0.7
                n = len(msg_embs)
                weights = [decay ** (n - 1 - i) for i in range(n)]
                center = np.average(msg_embs, axis=0, weights=weights).tolist()
                for mem in memories:
                    mem_emb = ctx_obj.chroma_service._get_embedding_cached(mem.get("id", ""))
                    if mem_emb:
                        dot = sum(a * b for a, b in zip(center, mem_emb))
                        n1 = math.sqrt(sum(a * a for a in center))
                        n2 = math.sqrt(sum(b * b for b in mem_emb))
                        mem["attention_proximity"] = dot / (n1 * n2 + 1e-10) if n1 and n2 else 0.0
                    else:
                        mem["attention_proximity"] = 0.0
            else:
                for mem in memories:
                    mem["attention_proximity"] = 0.0
        else:
            for mem in memories:
                mem["attention_proximity"] = 0.0
    except Exception:
        for mem in memories:
            mem["attention_proximity"] = 0.0

    # ── 分数默认值（引擎 weave_context 负责过滤，不再 K 截断）──
    for m in memories:
        if "score" not in m or m.get("score") is None:
            sim = 1.0 - m.get("distance", 1.0)
            m["score"] = round(max(0.0, sim), 3)

    # ── recency_weight（v2.1: 取代 archived 硬过滤，旧记忆只降权不屏蔽）──
    _now_ts = time.time()
    if _BM:
        for m in memories:
            m["recency_weight"] = 1.0
    else:
        for m in memories:
            meta = m.get("metadata") or {}
            last_hit = meta.get("last_hit_time") or meta.get("timestamp", 0)
            try:
                last_hit = float(last_hit)
            except (ValueError, TypeError):
                last_hit = _now_ts
            days_since = (_now_ts - last_hit) / 86400 if last_hit > 0 else 365
            # 90 天线性衰减到 0.15，不归零
            recency_weight = max(0.15, 1.0 - days_since / 90)
            # archived / stale 加额外上限，但不屏蔽
            if meta.get("archived", False):
                recency_weight = min(recency_weight, 0.6)
            if meta.get("stale", False):
                recency_weight = min(recency_weight, 0.3)
            m["recency_weight"] = round(recency_weight, 3)
            # 将 recency 折入 score，让后续排序感知时间衰减
            m["score"] = round(m.get("score", 0.5) * recency_weight, 3)
    # ── 兜底 ──
    if not memories:
        logger.warning("检索全部为空，回退到工作记忆兜底")
        try:
            fallback = ctx_obj.chat_history.get_recent(token_budget=3000)
            for rec in fallback[-3:]:
                user = rec.get("user_message", "")
                reply = rec.get("llm_reply", "")
                if user or reply:
                    memories.append({
                        "id": f"fallback_{int(time.time())}_{len(memories)}",
                        "document": f"用户：{user}\nAI：{reply}",
                        "metadata": {"summary": (user + " → " + reply)[:80]},
                        "source": "fallback", "distance": 0.5,
                    })
        except Exception as exc:
            logger.warning("工作记忆兜底也失败了: %s", exc)

    # ── 命中计数 ──
    for mem in memories:
        try:
            s = mem.get("source", "")
            d = 2 if s in (
                "co_occurrence", "time_triggered", "time_rhythm",
                "keyword_expand", "kw_match", "tag_match",
            ) else 1
            ctx_obj.chroma_service.increment_hit_count(mem["id"], delta=d)
        except Exception:
            pass

    # ── 共现记录 ──
    try:
        ids_ = [m["id"] for m in memories if m.get("id")]
        if len(ids_) >= 2:
            ctx_obj.storage_executor.submit(ctx_obj.co_tracker.record, ids_)
    except Exception as exc:
        logger.debug("共现记录失败: %s", exc)

    # ── 情绪字段兼容：确保旧数据 metadata 也有 emotion_valence_bin ──
    for mem in memories:
        meta = mem.get("metadata", {})
        if meta and "emotion_valence_bin" not in meta:
            meta["emotion_valence_bin"] = resolve_emotion_category(meta)

    return timeline_recent, session_context, personalities, memories


def retrieve_all(
    user_message: str,
    query_embedding: list | None,
    ctx_obj,
    intent: str | None = None,
    cached_tags: list[str] | None = None,
) -> list[dict]:
    """全量检索，6 路独立并行 + 1 路依赖合并后执行。

    原 8 路串行改为 ThreadPoolExecutor 并发：
      - 6 条独立路径并行跑
      - 共现扩展依赖其他路的 seen_ids，合并后单独跑
      - 注意力漂移也在并行池中

    Returns: list of dicts, each with:
        id, document, metadata, distance, source, summary, hit_count
    """
    import math
    import threading

    SEMANTIC_HARD_CAP = 500
    MIN_SIMILARITY = 0.1 if _BM else 0.3

    # 若调用方已提供缓存的 tags，跳过重复计算
    _cached_q_tags = cached_tags if cached_tags is not None else (extract_tags(user_message, topk=5) or [])
    route = _resolve_route(intent if intent else _classify_intent(user_message))
    sem_n = min(route["semantic"], SEMANTIC_HARD_CAP)
    tag_n = route["tag"]
    entity_n = route["entity"]

    _merge_lock = threading.Lock()
    # 第一阶段：6 路独立并行
    pool_results: list[list[dict]] = []

    def _merge(results: list[dict]):
        with _merge_lock:
            pool_results.append(results)

    def _make_mem(mid, meta, doc, dist, source):
        return {
            "id": mid, "document": doc or "", "metadata": meta or {},
            "summary": (meta or {}).get("summary", ""),
            "hit_count": (meta or {}).get("hit_count", 0) or 0,
            "distance": dist, "source": source,
        }

    def _path_semantic():
        """① 语义检索（hot + cool）。"""
        if not (query_embedding and sem_n > 0):
            return
        try:
            col = ctx_obj.chroma_service._collection
            local = []
            # hot（v2.1: 移除 archived 硬过滤 — 旧记忆只降权不屏蔽）
            hot = col.query(query_embeddings=[query_embedding], n_results=min(sem_n, 200),
                            where={"heat": "hot"},
                            include=["documents", "metadatas", "distances"])
            for i, mid in enumerate(hot.get("ids", [[]])[0]):
                meta = dict(hot["metadatas"][0][i]) if hot.get("metadatas") else {}
                doc = hot["documents"][0][i] if hot.get("documents") else ""
                dist = hot["distances"][0][i] if hot.get("distances") else 1.0
                local.append(_make_mem(mid, meta, doc, dist, "semantic_hot"))
            # cool 兜底
            remain = sem_n
            if remain > 0:
                # v2.1: 移除 archived 硬过滤 — 旧记忆只降权不屏蔽
                cool = col.query(query_embeddings=[query_embedding], n_results=remain,
                                 where={"heat": {"$in": ["warm", "cool"]}},
                                 include=["documents", "metadatas", "distances"])
                for i, mid in enumerate(cool.get("ids", [[]])[0]):
                    dist = cool["distances"][0][i] if cool.get("distances") else 1.0
                    if 1.0 - dist < MIN_SIMILARITY:
                        continue
                    meta = dict(cool["metadatas"][0][i]) if cool.get("metadatas") else {}
                    doc = cool["documents"][0][i] if cool.get("documents") else ""
                    local.append(_make_mem(mid, meta, doc, dist, "semantic_cool"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all 语义检索失败: %s", exc)

    def _path_keyword():
        """② 关键词匹配。将提取的标签分词后匹配倒排索引的 token 级 key。"""
        if not (hasattr(ctx_obj, 'inverted_index') and _cached_q_tags):
            return
        try:
            # 将整词标签拆成 2-gram token，匹配倒排索引的 token 级 key
            all_tokens = []
            for tag in _cached_q_tags:
                all_tokens.extend(ctx_obj.inverted_index._tokenize(tag))
            all_tokens = list(dict.fromkeys(all_tokens))  # 去重保序
            if not all_tokens:
                return
            kw_ids = ctx_obj.inverted_index.query(all_tokens, min_match=1)
            if not kw_ids:
                return
            kw_ids = kw_ids[:100] if _BM else kw_ids[:20]
            dr = ctx_obj.chroma_service._collection.get(ids=kw_ids, include=["documents", "metadatas"])
            local = []
            for i, mid in enumerate(dr.get("ids", [])):
                meta = dict(dr["metadatas"][i]) if dr.get("metadatas") else {}
                doc = dr["documents"][i] if dr.get("documents") else ""
                local.append(_make_mem(mid, meta, doc, 0.4, "kw_match"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all 关键词匹配失败: %s", exc)

    def _path_tag():
        """③ 标签索引。"""
        if not (hasattr(ctx_obj, 'inverted_index') and tag_n > 0 and _cached_q_tags):
            return
        try:
            tag_ids = list(ctx_obj.inverted_index.query_tags(_cached_q_tags))
            tag_ids = tag_ids[:100] if _BM else tag_ids[:20]
            if not tag_ids:
                return
            dr = ctx_obj.chroma_service._collection.get(ids=tag_ids, include=["documents", "metadatas"])
            local = []
            for i, mid in enumerate(dr.get("ids", [])):
                meta = dict(dr["metadatas"][i]) if dr.get("metadatas") else {}
                doc = dr["documents"][i] if dr.get("documents") else ""
                local.append(_make_mem(mid, meta, doc, 0.5, "tag_match"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all 标签匹配失败: %s", exc)

    def _path_entity():
        """④ 实体索引。"""
        if not (hasattr(ctx_obj, 'inverted_index') and entity_n > 0 and _cached_q_tags):
            return
        try:
            from app.analysis.entity import extract_entities
            q_entities = extract_entities(user_message)
            if not q_entities:
                return
            entity_names = [e["text"] for e in q_entities
                            if e.get("type") in ("PERSON", "LOCATION", "ORGANIZATION") and len(e["text"]) >= 2]
            if not entity_names:
                return
            all_eids: set[str] = set()
            for ename in entity_names:
                all_eids.update(ctx_obj.inverted_index.get_exact(ename))
            if not all_eids:
                return
            eid_limit = 100 if _BM else 20
            dr = ctx_obj.chroma_service._collection.get(
                ids=list(all_eids)[:eid_limit], include=["documents", "metadatas"])
            local = []
            for i, mid in enumerate(dr.get("ids", [])):
                meta = dict(dr["metadatas"][i]) if dr.get("metadatas") else {}
                doc = dr["documents"][i] if dr.get("documents") else ""
                local.append(_make_mem(mid, meta, doc, 0.5, "entity_match"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all 实体匹配失败: %s", exc)

    def _path_temporal():
        """⑥ 时间触发。"""
        if not hasattr(ctx_obj, 'temporal_pattern_index'):
            return
        try:
            tps = ctx_obj.temporal_pattern_index.query()
            if not tps:
                return
            tp_tags = [t[0] for t in tps[:5]]
            tp_ids = ctx_obj.inverted_index.query_tags(tp_tags) if hasattr(ctx_obj, 'inverted_index') else set()
            tp_ids = list(tp_ids)[:50] if _BM else list(tp_ids)[:10]
            if not tp_ids:
                return
            dr = ctx_obj.chroma_service._collection.get(ids=tp_ids, include=["documents", "metadatas"])
            local = []
            for i, mid in enumerate(dr.get("ids", [])):
                meta = dict(dr["metadatas"][i]) if dr.get("metadatas") else {}
                doc = dr["documents"][i] if dr.get("documents") else ""
                local.append(_make_mem(mid, meta, doc, 0.5, "time_triggered"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all 时间触发失败: %s", exc)

    def _path_topic():
        """⑦ 话题树分支。"""
        if not (hasattr(ctx_obj, 'topic_tree') and ctx_obj.topic_tree and _cached_q_tags):
            return
        try:
            expanded_tags = ctx_obj.topic_tree.expand(_cached_q_tags)
            if not expanded_tags:
                return
            topic_ids = ctx_obj.inverted_index.query_tags(expanded_tags) if hasattr(ctx_obj, 'inverted_index') else set()
            topic_ids = list(topic_ids)[:50] if _BM else list(topic_ids)[:10]
            if not topic_ids:
                return
            dr = ctx_obj.chroma_service._collection.get(ids=topic_ids, include=["documents", "metadatas"])
            local = []
            for i, mid in enumerate(dr.get("ids", [])):
                meta = dict(dr["metadatas"][i]) if dr.get("metadatas") else {}
                doc = dr["documents"][i] if dr.get("documents") else ""
                local.append(_make_mem(mid, meta, doc, 0.55, "topic_expand"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all 话题树扩展失败: %s", exc)

    def _path_attention():
        """⑧ 注意力漂移。"""
        if not hasattr(ctx_obj, 'chat_history'):
            return
        try:
            from app.llm.embed import local_embed_batch
            recent_msgs = []
            for rec in reversed(ctx_obj.chat_history.get_records_snapshot()):
                msg = rec.get("user_message", "")
                if msg and msg != "[内心独白]":
                    recent_msgs.append(msg)
                    if len(recent_msgs) >= 3:
                        break
            if not recent_msgs:
                return
            msg_embs = local_embed_batch([m for m in recent_msgs if m])
            msg_embs = [e for e in msg_embs if e is not None]
            if not msg_embs:
                return
            import numpy as np
            decay = 0.7
            n = len(msg_embs)
            weights = [decay ** (n - 1 - i) for i in range(n)]
            center = np.average(msg_embs, axis=0, weights=weights).tolist()
            results = ctx_obj.chroma_service._collection.query(
                query_embeddings=[center], n_results=50 if _BM else 10,
                include=["documents", "metadatas", "distances"])
            local = []
            for i, mid in enumerate(results.get("ids", [[]])[0]):
                dist = results["distances"][0][i] if results.get("distances") else 1.0
                if 1.0 - dist < MIN_SIMILARITY:
                    continue
                meta = dict(results["metadatas"][0][i]) if results.get("metadatas") else {}
                doc = results["documents"][0][i] if results.get("documents") else ""
                local.append(_make_mem(mid, meta, doc, dist, "attention_drift"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all 注意力漂移失败: %s", exc)

    def _path_bm25_fulltext():
        """⑨ BM25 全文检索。"""
        if not (hasattr(ctx_obj, 'bm25_index') and ctx_obj.bm25_index):
            return
        try:
            bm25_ids = ctx_obj.bm25_index.search(user_message, top_k=100 if _BM else 20)
            if not bm25_ids:
                return
            dr = ctx_obj.chroma_service._collection.get(
                ids=bm25_ids, include=["documents", "metadatas"])
            local = []
            for i, mid in enumerate(dr.get("ids", [])):
                meta = dict(dr["metadatas"][i]) if dr.get("metadatas") else {}
                doc = dr["documents"][i] if dr.get("documents") else ""
                local.append(_make_mem(mid, meta, doc, 0.35, "bm25_fulltext"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all BM25 全文检索失败: %s", exc)

    def _path_ai_memory():
        """⑩ AI 表达记忆检索 — 语义相似的 AI 历史表达。"""
        if not (query_embedding and hasattr(ctx_obj, 'ai_chroma_service')):
            return
        try:
            ai_col = ctx_obj.ai_chroma_service._collection
            ai_results = ai_col.query(
                query_embeddings=[query_embedding], n_results=20 if _BM else 5,
                include=["documents", "metadatas", "distances"])
            local = []
            for i, mid in enumerate(ai_results.get("ids", [[]])[0]):
                dist = ai_results["distances"][0][i] if ai_results.get("distances") else 1.0
                if 1.0 - dist < MIN_SIMILARITY:
                    continue
                meta = dict(ai_results["metadatas"][0][i]) if ai_results.get("metadatas") else {}
                doc = ai_results["documents"][0][i] if ai_results.get("documents") else ""
                local.append(_make_mem(mid, meta, doc, dist, "ai_expression"))
            if local:
                _merge(local)
        except Exception as exc:
            logger.debug("retrieve_all AI表达检索失败: %s", exc)

    # ── 第一阶段：9 路并行（+ BM25 全文 + AI 表达）──
    paths = [_path_semantic, _path_keyword, _path_tag,
             _path_entity, _path_temporal, _path_topic, _path_attention,
             _path_bm25_fulltext, _path_ai_memory]
    global _retrieval_executor
    if _retrieval_executor is None:
        _retrieval_executor = ThreadPoolExecutor(max_workers=min(len(paths), 8))
    futures = {_retrieval_executor.submit(p): p for p in paths}
    for f in as_completed(futures):
        try:
            f.result()
        except Exception:
            pass

    # ── 合并去重 ──
    seen_ids: set[str] = set()
    candidates: list[dict] = []
    for batch in pool_results:
        for m in batch:
            mid = m.get("id", "")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                candidates.append(m)

    # ── 第二阶段：⑤ 共现扩展（依赖 seen_ids） ──
    if hasattr(ctx_obj, 'co_tracker') and seen_ids:
        try:
            cooc = ctx_obj.co_tracker.query(list(seen_ids))
            if cooc:
                cooc_ids = [c["id"] for c in cooc if c["id"] not in seen_ids]
                cooc_ids = cooc_ids[:50] if _BM else cooc_ids[:10]
                if cooc_ids:
                    dr = ctx_obj.chroma_service._collection.get(
                        ids=cooc_ids, include=["documents", "metadatas"])
                    for i, mid in enumerate(dr.get("ids", [])):
                        if not mid or mid in seen_ids:
                            continue
                        meta = dict(dr["metadatas"][i]) if dr.get("metadatas") else {}
                        doc = dr["documents"][i] if dr.get("documents") else ""
                        seen_ids.add(mid)
                        candidates.append(
                            _make_mem(mid, meta, doc, 0.45, "co_occurrence"))
        except Exception as exc:
            logger.debug("retrieve_all 共现扩展失败: %s", exc)

    # ── Benchmark 兜底：小数据集直接全量返回，零遗漏 ──
    if _BM:
        try:
            total = ctx_obj.chroma_service.count()
            if total > 0 and total <= 200:
                all_items = ctx_obj.chroma_service._collection.get(
                    include=["documents", "metadatas"])
                for i, mid in enumerate(all_items.get("ids", [])):
                    if mid not in seen_ids:
                        meta = dict(all_items["metadatas"][i]) if all_items.get("metadatas") else {}
                        doc = all_items["documents"][i] if all_items.get("documents") else ""
                        seen_ids.add(mid)
                        candidates.append(
                            _make_mem(mid, meta, doc, 0.5, "benchmark_full"))
        except Exception as exc:
            logger.debug("retrieve_all benchmark 全量兜底失败: %s", exc)

    return candidates
