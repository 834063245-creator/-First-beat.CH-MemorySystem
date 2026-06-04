"""检索管线 — 从 backend/main.py 迁移而来。

原 _run_chat_retrieval 整体迁移至此，路径导入已更新为 app 结构。
"""
import json
import logging
import os
import re
import time

from app.brain.semantic import extract_tags, tokenize as _sem_tokenize

from app.config.settings import (
    RERANK_SEMANTIC_WEIGHT, RERANK_ATTENTION_WEIGHT, ATTENTION_WINDOW,
    RERANK_LN_MAX, DEFAULT_TOP_K, MAX_MEMORIES_IN_PROMPT,
    LITE_WORK_MEMORY_BUDGET,
)
from app.retrieval.scoring import compute_score
from app.llm.embed import local_embed, local_embed_batch
from app.memory.working import get_summary
from app.analysis.emotion import resolve_emotion_category
from app.analysis.entity import extract_entities

logger = logging.getLogger(__name__)


# ── 检索门控：意图 → 各路配额 ────────────────────────────────
# 配额含义是 ChromaDB query 的 n_results（不是截断上限）。
# 截断统一由 MAX_MEMORIES_IN_PROMPT 管理。
_INTENT_ROUTES = {
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

    _ACTUAL_TOP_K = max(
        DEFAULT_TOP_K,
        min(ctx_obj.chroma_service._read_collection.count() // 20, 100),
    )
    _cached_q_tags = extract_tags(user_message, topk=5) or []
    _ticks = [("start", time.perf_counter())]
    def _log_step(name):
        ms = (time.perf_counter() - _ticks[-1][1]) * 1000
        from app.core import bottleneck
        bottleneck.record(name, ms)
        _ticks.append((name, time.perf_counter()))

    # ── 意图门控 ──
    route = _resolve_route(intent if intent else _classify_intent(user_message))
    sem_n = route["semantic"]
    tag_n = route["tag"]
    entity_n = route["entity"]

    _log_step('intent_gate')
    # ── 时间线近端历史 ──
    timeline_recent = []
    if ctx_obj.chat_history:
        try:
            timeline_recent = ctx_obj.chat_history.get_recent(
                token_budget=LITE_WORK_MEMORY_BUDGET
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
    # ── 人格标签 ──
    personalities = []
    if ctx_obj.personality_store and query_embedding_for_retrieval is not None:
        try:
            personalities = ctx_obj.personality_store.rerank_tags(
                user_message, query_embedding_for_retrieval, top_k=3
            )
        except Exception:
            pass

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
    # ── 语义检索（ChromaDB） ──
    if not memories and query_embedding_for_retrieval is not None:
        try:
            cq = bool(
                re.search(
                    r"(?:多少|几个|几件|几台|哪些|how many|count|总数|所有)",
                    user_message,
                    re.I,
                )
            )
            nq = max(sem_n, 40) if cq else sem_n
            # 两段式：先捞 hot，不够再补 warm+cool
            hot_n = int(nq * 1.5)
            results = ctx_obj.chroma_service._read_collection.query(
                query_embeddings=[query_embedding_for_retrieval],
                n_results=hot_n,
                where={"$and": [{"heat": "hot"}, {"archived": {"$ne": True}}]},
                include=["documents", "metadatas", "distances"],
            )
            hot_count = len(results.get("ids", [[]])[0])
            if hot_count < nq:
                remain = nq  # 补到标准配额
                remain_results = ctx_obj.chroma_service._read_collection.query(
                    query_embeddings=[query_embedding_for_retrieval],
                    n_results=remain,
                    where={"$and": [{"heat": {"$in": ["warm", "cool"]}}, {"archived": {"$ne": True}}]},
                    include=["documents", "metadatas", "distances"],
                )
                for key in ("ids", "metadatas", "documents", "distances"):
                    if key in remain_results:
                        hot_val = results.get(key, [[]])[0]
                        warm_val = remain_results.get(key, [[]])[0]
                        results[key] = [hot_val + [v for v in warm_val if v not in hot_val]]
            for i, mid in enumerate(results.get("ids", [[]])[0]):
                meta = (
                    dict(results["metadatas"][0][i])
                    if results.get("metadatas")
                    else {}
                )
                if meta.get("stale", False) or meta.get("archived", False):
                    continue
                doc = (
                    results["documents"][0][i]
                    if results.get("documents")
                    else ""
                )
                memories.append({
                    "id": mid, "document": doc, "metadata": meta,
                    "summary": meta.get("summary", ""),
                    "hit_count": meta.get("hit_count", 0) or 0,
                    "source": "semantic", "summary_only": True,
                    "distance": results["distances"][0][i],
                })
        except Exception:
            pass

        # ── 关键词扩展 ──
        if len(memories) < 3:
            try:
                kws = _cached_q_tags
                if kws:
                    kw_emb = local_embed(" ".join(kws))
                    if kw_emb:
                        kr = ctx_obj.chroma_service._read_collection.query(
                            query_embeddings=[kw_emb],
                            n_results=_ACTUAL_TOP_K,
                            include=["documents", "metadatas", "distances"],
                        )
                        eids = {m["id"] for m in memories}
                        for i, mid in enumerate(kr.get("ids", [[]])[0]):
                            if mid in eids:
                                continue
                            meta = dict(kr["metadatas"][0][i]) if kr.get("metadatas") else {}
                            if meta.get("stale", False) or meta.get("archived", False):
                                continue
                            doc = kr["documents"][0][i] if kr.get("documents") else ""
                            memories.append({
                                "id": mid, "document": doc, "metadata": meta,
                                "summary": meta.get("summary", ""),
                                "hit_count": meta.get("hit_count", 0) or 0,
                                "source": "keyword_expand",
                            })
            except Exception:
                pass

        # ── 标签嵌入最近邻扩展（替代话题树：embedding cosine 相似度找近邻标签） ──
        try:
            tag_index = getattr(ctx_obj, '_tag_index', None)
            if tag_index is not None and tag_index.size() > 0:
                topic_expanded = tag_index.nearest(_cached_q_tags, top_k=5)
                if topic_expanded:
                    _cached_q_tags.extend(
                        [t for t in topic_expanded if t not in _cached_q_tags]
                    )
        except Exception:
            pass

        # ── 倒排索引标签匹配 + 多标签匹配 ──
        if tag_n > 0:
            try:
                q_tags = _cached_q_tags
                if q_tags:
                    eids = {m["id"] for m in memories}
                    candidate_ids = set()
                    # 标签索引优先（O(1)，启动时已构建）
                    candidate_ids = ctx_obj.inverted_index.query_tags(q_tags)
                    if not candidate_ids:
                        for tag in q_tags:
                            tag_ids = ctx_obj.inverted_index.get_exact(tag)
                            candidate_ids.update(tag_ids)
                    candidate_ids -= eids
                    if not candidate_ids:
                        for tag in q_tags:
                            tag_result = ctx_obj.chroma_service._read_collection.get(
                                where={"tags": {"$contains": tag}},
                                include=["metadatas"],
                            )
                            for i, mid in enumerate(tag_result.get("ids", [])):
                                if mid in eids:
                                    continue
                                if tag_result["metadatas"][i].get("stale", False):
                                    continue
                                candidate_ids.add(mid)
                    if candidate_ids:
                        dr = ctx_obj.chroma_service._read_collection.get(
                            ids=list(candidate_ids)[:50],
                            include=["documents", "metadatas"],
                        )
                        for i, mid in enumerate(dr.get("ids", [])):
                            if mid in eids:
                                continue
                            _m = dr["metadatas"][i] if dr.get("metadatas") else {}
                            _d = dr["documents"][i] if dr.get("documents") else ""
                            memories.append({
                                "id": mid, "document": _d, "metadata": _m,
                                "summary": _m.get("summary", ""),
                                "hit_count": _m.get("hit_count", 0) or 0,
                                "source": "tag_match", "distance": 0.5,
                            })
                            eids.add(mid)
                            if len([x for x in memories if x.get("source") == "tag_match"]) >= 20:
                                break
                    # ── 多标签匹配 ──
                    if q_tags:
                        mids = ctx_obj.inverted_index.query(q_tags, min_match=2)
                        if mids:
                            dr = ctx_obj.chroma_service._read_collection.get(
                                ids=mids, include=["documents", "metadatas"])
                            eids = {m["id"] for m in memories}
                            bc = 0
                            for i, mid in enumerate(dr.get("ids", [])):
                                if mid in eids:
                                    continue
                                meta = dr["metadatas"][i] if dr.get("metadatas") else {}
                                doc = dr["documents"][i] if dr.get("documents") else ""
                                if meta.get("stale", False) or meta.get("archived", False):
                                    continue
                                memories.append({
                                    "id": mid, "document": doc, "metadata": meta,
                                    "summary": meta.get("summary", ""),
                                    "hit_count": meta.get("hit_count", 0) or 0,
                                    "source": "kw_match", "distance": 0.4,
                                })
                                eids.add(mid)
                                bc += 1
                                if bc >= 10:
                                    break
            except Exception:
                pass

        _log_step('tag_retrieval')
    # ── 实体检索 + 实体共现扩展 ──
        if entity_n > 0:
            try:
                q_entities = extract_entities(user_message)
                if q_entities:
                    entity_names = [
                        e["text"] for e in q_entities
                        if e.get("type") in ("PERSON", "LOCATION", "ORGANIZATION")
                    ]
                    seen = {m["id"] for m in memories}
                    all_entity_ids = set()
                    for ename in entity_names:
                        ids = ctx_obj.inverted_index.get_exact(ename)
                        all_entity_ids.update(ids)
                    all_entity_ids -= seen
                    if all_entity_ids:
                        dr = ctx_obj.chroma_service._read_collection.get(
                            ids=list(all_entity_ids),
                            include=["documents", "metadatas"],
                        )
                        for i, mid in enumerate(dr.get("ids", [])):
                            if mid in seen:
                                continue
                            meta = dr["metadatas"][i] if dr.get("metadatas") else {}
                            doc = dr["documents"][i] if dr.get("documents") else ""
                            memories.append({
                                "id": mid, "document": doc, "metadata": meta,
                                "summary": meta.get("summary", ""),
                                "hit_count": meta.get("hit_count", 0) or 0,
                                "source": "entity_match", "distance": 0.5,
                            })
                            seen.add(mid)
                # ── 实体共现扩展 ──
                if hasattr(ctx_obj, "entity_pair_tracker") and ctx_obj.entity_pair_tracker:
                    q_entities_ext = extract_entities(user_message)
                    if q_entities_ext:
                        enames = list(dict.fromkeys(
                            e["text"] for e in q_entities_ext
                            if e.get("type") in ("PERSON", "LOCATION", "ORGANIZATION")
                            and len(e["text"]) >= 2
                        ))
                        if enames:
                            pair_ids = ctx_obj.entity_pair_tracker.get_memory_ids(enames)
                            pair_ids = [mid for mid in pair_ids if mid not in seen]
                            if pair_ids:
                                dr = ctx_obj.chroma_service._read_collection.get(
                                    ids=pair_ids[:20], include=["documents", "metadatas"])
                                for i, mid in enumerate(dr.get("ids", [])):
                                    if mid in seen:
                                        continue
                                    meta = dr["metadatas"][i] if dr.get("metadatas") else {}
                                    doc = dr["documents"][i] if dr.get("documents") else ""
                                    memories.append({
                                        "id": mid, "document": doc, "metadata": meta,
                                        "summary": meta.get("summary", ""),
                                        "hit_count": meta.get("hit_count", 0) or 0,
                                        "source": "entity_co_occurrence", "distance": 0.45,
                                    })
                                    seen.add(mid)
            except Exception:
                pass

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

    # ── Reranker 精排 ──
    _errs = _load_error_counts(data_dir=ctx_obj.data_dir)
    _corrs = _load_correction_boosts(data_dir=ctx_obj.data_dir)

    try:
        from rank_bm25 import BM25Okapi
        qt = _sem_tokenize(user_message)
        docs = [m.get("document", "") or m.get("summary", "") for m in memories]
        corpus = [_sem_tokenize(d) for d in docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(qt)
        for i, m in enumerate(memories):
            m["_bm25"] = float(scores[i]) if i < len(scores) else 0.0
        if sum(1 for s in scores if s > 0) < 3 and len(memories) > 5:
            raise ValueError("BM25 过于稀疏，回退 embedding reranker")
        _hot = lambda m: 0.1 if m.get("metadata", {}).get("heat") == "hot" else 0.0
        memories.sort(
            key=lambda m: compute_score(
                similarity=1.0 - m.get("distance", 1.0),
                hit_count=m.get("hit_count", 0) or 0,
                attention_boost=m.get("attention_proximity", 0.0),
                source_bonus=(0.1 if m.get("source") in (
                    "text_match", "keyword_expand", "tag_match",
                    "bm25_fulltext", "kw_match", "entity_match",
                ) else 0.0) + _hot(m),
                error_penalty=_errs.get(m.get("id", ""), 0) * 0.05,
            )
            + _corrs.get(m.get("id", ""), 0.0) * 0.1
            + 0.001 * (m.get("_bm25", 0) or 0),
            reverse=True,
        )
        for m in memories:
            m["score"] = (
                compute_score(
                    similarity=1.0 - m.get("distance", 1.0),
                    hit_count=m.get("hit_count", 0) or 0,
                    attention_boost=m.get("attention_proximity", 0.0),
                    source_bonus=(0.1 if m.get("source") in (
                        "text_match", "keyword_expand", "tag_match",
                        "bm25_fulltext", "kw_match", "entity_match",
                    ) else 0.0) + _hot(m),
                    error_penalty=_errs.get(m.get("id", ""), 0) * 0.05,
                )
                + _corrs.get(m.get("id", ""), 0.0) * 0.1
                + 0.001 * (m.get("_bm25", 0) or 0)
            )
        memories = memories[:MAX_MEMORIES_IN_PROMPT]
    except Exception:
        try:
            from app.retrieval.reranker import rerank
            if len(memories) > 1:
                attn_boosts = {
                    m.get("id", ""): m.get("attention_proximity", 0.0)
                    for m in memories if m.get("id")
                }
                attn_all_zero = all(v == 0 for v in attn_boosts.values())
                attn_w = 0.0 if attn_all_zero else RERANK_ATTENTION_WEIGHT
                memories = rerank(
                    user_message, memories, top_k=MAX_MEMORIES_IN_PROMPT,
                    correction_boosts=_corrs, error_counts=_errs,
                    attention_boosts=attn_boosts, attention_weight=attn_w,
                )
                for m in memories:
                    m["score"] = m.get("_rr_score", 0.0)
            else:
                raise ValueError("候选<2条，跳过rerank")
        except Exception:
            pass

    for m in memories:
        if "score" not in m or m.get("score") is None:
            m["score"] = 0.0


    _log_step('entity_retrieval')
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

    if len(memories) > MAX_MEMORIES_IN_PROMPT:
        memories = memories[:MAX_MEMORIES_IN_PROMPT]

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
