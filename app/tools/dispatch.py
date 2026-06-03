"""记忆系统内部工具集 — 引擎使用，非 LLM 工具。

引擎内部使用的工具函数：
- query_memory: 统一记忆检索（语义+时间+组合）
- query_explore: 记忆探索（供引擎/审计使用）
- analyze_pattern: 记忆分析（供引擎/审计使用）
- count_memories: 统计记忆总数（供 DMN 使用）

这些函数由引擎内部调用（DMN/检索管线），
不暴露给 LLM。LLM 仅使用 main.py 中注册的纯功能工具。
"""

import json
import logging
import threading
import os
import re
from datetime import datetime, timedelta

from app.llm.embed import local_embed

logger = logging.getLogger(__name__)


# ===================================================================
# 自然语言日期解析（供 query_explore 使用）
# ===================================================================
def _parse_natural_date(text: str) -> dict | None:
    """解析自然语言日期描述，返回 {"from_date": "YYYY-MM-DD", "to_date": "YYYY-MM-DD"} 或 None。"""
    if not text:
        return None
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if text in ("今天", "今日"):
        return {"from_date": today.strftime("%Y-%m-%d"), "to_date": today.strftime("%Y-%m-%d")}
    if text in ("昨天", "昨日"):
        d = today - timedelta(days=1)
        return {"from_date": d.strftime("%Y-%m-%d"), "to_date": d.strftime("%Y-%m-%d")}
    if text in ("前天"):
        d = today - timedelta(days=2)
        return {"from_date": d.strftime("%Y-%m-%d"), "to_date": d.strftime("%Y-%m-%d")}

    # 支持中文数字映射
    _CN_NUM = {"零":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
    m = re.match(r'(\d+)天前', text)
    if m:
        d = today - timedelta(days=int(m.group(1)))
        return {"from_date": d.strftime("%Y-%m-%d"), "to_date": d.strftime("%Y-%m-%d")}
    m = re.match(r'([一二两三四五六七八九十\d]+)天前', text)
    if m:
        _n = _CN_NUM.get(m.group(1), 0)
        d = today - timedelta(days=_n)
        return {"from_date": d.strftime("%Y-%m-%d"), "to_date": d.strftime("%Y-%m-%d")}

    if text == "上周":
        d = today - timedelta(weeks=1)
        return {"from_date": (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d"),
                "to_date": (d + timedelta(days=6 - d.weekday())).strftime("%Y-%m-%d")}

    if text in ("上个月", "上月"):
        first = today.replace(day=1) - timedelta(days=1)
        return {"from_date": first.replace(day=1).strftime("%Y-%m-%d"),
                "to_date": first.strftime("%Y-%m-%d")}
    if text in ("这个月", "本月"):
        first = today.replace(day=1)
        return {"from_date": first.strftime("%Y-%m-%d"), "to_date": today.strftime("%Y-%m-%d")}

    m = re.match(r'(\d+)月(\d+)日', text)
    if m:
        d = today.replace(month=int(m.group(1)), day=int(m.group(2)))
        return {"from_date": d.strftime("%Y-%m-%d"), "to_date": d.strftime("%Y-%m-%d")}
    m = re.match(r'(\d+)月', text)
    if m:
        d = today.replace(month=int(m.group(1)), day=1)
        if d.month != today.month:
            import calendar
            last = calendar.monthrange(d.year, d.month)[1]
            return {"from_date": d.strftime("%Y-%m-%d"),
                    "to_date": d.replace(day=last).strftime("%Y-%m-%d")}
        return {"from_date": d.strftime("%Y-%m-%d"),
                "to_date": today.strftime("%Y-%m-%d")}

    return None


# ===================================================================
# query_memory — 统一记忆检索
# ===================================================================

QUERY_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_memory",
        "description": (
            "检索记忆库，返回完整原文和上下文。read_memory 已合并至此工具。"
            "只有当用户明确提到某件过去的事、且系统自动检索的记忆中"
            "明显没有相关内容时，才调用此工具。"
            "不要每条回复都调用——大多数情况下系统推送的记忆已经足够。"
            "支持语义搜索、时间范围搜索、或两者组合。"
            "返回完整记忆内容（含原文、时间戳、上下文）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，用用户原话或你猜测的相关描述。可选，但建议提供以获得更好结果。",
                },
                "from_date": {
                    "type": "string",
                    "description": "起始日期 YYYY-MM-DD，可选。与 to_date 一起使用时限定时间范围。",
                },
                "to_date": {
                    "type": "string",
                    "description": "结束日期 YYYY-MM-DD，可选。与 from_date 一起使用时限定时间范围。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回条数，默认5，最大15",
                },
                "filters": {
                    "type": "object",
                    "description": "结构化过滤条件（可选）。支持按时间维度、情绪强度等精确筛选，所有条件 AND 关系。",
                    "properties": {
                        "time_period": {
                            "type": "string",
                            "description": "时段，可选值：凌晨(0-5)、早晨(5-8)、上午(8-12)、下午(12-14)、傍晚(14-18)、晚上(18-21)、深夜(21-24)",
                        },
                        "day_of_week": {
                            "type": "integer",
                            "description": "星期几，0=周一，1=周二，2=周三，3=周四，4=周五，5=周六，6=周日",
                        },
                        "month": {
                            "type": "integer",
                            "description": "月份，1-12",
                        },
                        "emotional_intensity": {
                            "type": "integer",
                            "description": "情绪强度下限，0-3（0=普通，1=有一点情绪，2=有情绪，3=情绪激动）",
                        },
                        "year": {
                            "type": "integer",
                            "description": "年份，如 2026",
                        },
                    },
                },
            },
        },
    },
}


def query_memory(collection, query: str = "", from_date: str = "", to_date: str = "", top_k: int = 5,
                 co_tracker=None, chat_history_obj=None,
                 filters: dict | None = None) -> list[dict]:
    """统一记忆检索。语义搜索 + 可选时间/结构化过滤 + 原文匹配。"""
    top_k = min(top_k, 15)
    has_query = bool(query and query.strip())
    has_time = bool(from_date or to_date)

    # filters 兜底
    if not filters or not isinstance(filters, dict):
        filters = None
    elif not any(v is not None for v in filters.values()):
        filters = None
    has_filters = filters is not None

    if not has_query and not has_time and not has_filters:
        return [{"error": "请提供搜索关键词或时间范围"}]

    # ── 时间过滤条件 ──
    # ChromaDB query() 不支持 $and（内部崩溃），改用 get() 取 ID 集 + query() 取语义交集
    where_get = None  # 给 get() 用的格式
    where_query = None  # 给 query() 用的单操作符格式
    if from_date and to_date:
        try:
            ts_start = datetime.strptime(from_date, "%Y-%m-%d").timestamp()
            ts_end = datetime.strptime(to_date, "%Y-%m-%d").timestamp() + 86399
            where_get = {"$and": [{"timestamp": {"$gte": ts_start}}, {"timestamp": {"$lte": ts_end}}]}
        except ValueError:
            pass
    elif from_date:
        try:
            ts_start = datetime.strptime(from_date, "%Y-%m-%d").timestamp()
            where_get = {"timestamp": {"$gte": ts_start}}
            where_query = where_get
        except ValueError:
            pass
    elif to_date:
        try:
            ts_end = datetime.strptime(to_date, "%Y-%m-%d").timestamp() + 86399
            where_get = {"timestamp": {"$lte": ts_end}}
            where_query = where_get
        except ValueError:
            pass

    # ── 结构化过滤条件（filters）→ where 子句 ──
    if has_filters:
        try:
            filter_conditions = []
            if filters.get("time_period"):
                filter_conditions.append({"time_period": {"$eq": filters["time_period"]}})
            if filters.get("day_of_week") is not None:
                filter_conditions.append({"day_of_week": {"$eq": filters["day_of_week"]}})
            if filters.get("month"):
                filter_conditions.append({"month": {"$eq": filters["month"]}})
            if filters.get("year"):
                filter_conditions.append({"year": {"$eq": filters["year"]}})
            if filters.get("emotional_intensity") is not None:
                filter_conditions.append({"emotional_intensity": {"$gte": filters["emotional_intensity"]}})

            if filter_conditions:
                all_conds = []
                # 合并现有的时间范围条件
                if where_get:
                    if "$and" in where_get:
                        all_conds.extend(where_get["$and"])
                    else:
                        all_conds.append(where_get)
                all_conds.extend(filter_conditions)

                if len(all_conds) == 1:
                    where_get = all_conds[0]
                else:
                    where_get = {"$and": all_conds}
                # filters 使条件变复杂，query() 不支持，清空 where_query
                where_query = None
        except Exception as e:
            logger.warning("query_memory filters 构造失败: %s", e)

    # 时间/结构化范围预取 ID 集
    time_filtered_ids: set[str] | None = None
    if (has_time or has_filters) and where_get:
        try:
            time_filtered = collection.get(where=where_get, include=[])
            time_filtered_ids = set(time_filtered["ids"])
        except Exception as e:
            logger.warning("query_memory 时间/结构化范围预取失败: %s", e)

    # ── 语义搜索（含原文匹配增强） ──
    if has_query:
        try:
            query_emb = local_embed(query.strip())
            if query_emb is None:
                return []
            n_results = top_k * 3 if (from_date or to_date) else top_k
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                where=where_query,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("query_memory 语义搜索失败: %s", e)
            return []

        # 原文主动检索：去 ChromaDB 原文里搜字面匹配
        text_matched_ids = set()
        try:
            all_docs = collection.get(include=["documents", "metadatas"])
            query_lower = query.strip().lower()
            for i, doc in enumerate(all_docs.get("documents", [])):
                if doc and query_lower in doc.lower():
                    text_matched_ids.add(all_docs["ids"][i])
        except Exception:
            pass

        memories = []
        docs_for_bm25 = []
        # 补充 V2 没捞到的原文匹配结果
        if text_matched_ids:
            try:
                extra = collection.get(ids=list(text_matched_ids), include=["documents", "metadatas"])
                for i, mid in enumerate(extra.get("ids", [])):
                    if any(r.get("id") == mid for r in memories):
                        continue
                    meta = extra["metadatas"][i] if extra.get("metadatas") else {}
                    doc = extra["documents"][i] if extra.get("documents") else ""
                    mem_ctx = chat_history_obj.get_context_by_chroma_id(mid, before=10, after=10) if chat_history_obj else {"context_before": [], "context_after": []}
                    memories.append({
                        "id": mid,
                        "document": doc,
                        "metadata": dict(meta),
                        "summary": meta.get("summary", ""),
                        "hit_count": meta.get("hit_count", 0) or 0,
                        "date": meta.get("date", ""),
                        "source": "text_match",
                        "context_before": mem_ctx.get("context_before", []),
                        "context_after": mem_ctx.get("context_after", []),
                    })
                    docs_for_bm25.append(doc)
            except Exception:
                pass

        for i, mem_id in enumerate(results.get("ids", [[]])[0]):
            if time_filtered_ids is not None and mem_id not in time_filtered_ids:
                continue
            meta = results.get("metadatas", [[{}]])[0][i] if results.get("metadatas") else {}
            doc = results.get("documents", [[""]])[0][i] if results.get("documents") else ""
            dist = results.get("distances", [[0]])[0][i] if results.get("distances") else 0
            sim = 1.0 - dist if dist else 0
            ts = meta.get("timestamp", 0)
            time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            kw_match = mem_id in text_matched_ids

            ctx_data = chat_history_obj.get_context_by_chroma_id(mem_id, before=3, after=3) if chat_history_obj else {"context_before": [], "context_after": []}

            memories.append({
                "id": mem_id,
                "summary": meta.get("summary", "")[:80],
                "tags": meta.get("tags", ""),
                "timestamp": time_str,
                "similarity": round(sim, 3),
                "hit_count": meta.get("hit_count", 0),
                "_preview": doc[:120],
                "_kw_match": kw_match,
                "context_before": ctx_data.get("context_before", []),
                "context_after": ctx_data.get("context_after", []),
            })
            docs_for_bm25.append(doc)

        # 标签关键词补充检索：语义不足时用 jieba 关键词匹配 tags 字段
        if len(memories) < top_k:
            try:
                import jieba, jieba.analyse
                _kws = jieba.analyse.extract_tags(query, topK=5) if query else []
                if _kws:
                    _all_meta = collection.get(include=["metadatas"])
                    _matched_ids = []
                    for _i, _mid in enumerate(_all_meta.get("ids", [])):
                        _tags = (_all_meta["metadatas"][_i].get("tags", "") or "") if _all_meta.get("metadatas") else ""
                        if any(k in _tags for k in _kws):
                            if not (_all_meta["metadatas"][_i] if _all_meta.get("metadatas") else {}).get("stale", False):
                                _matched_ids.append(_mid)
                    if _matched_ids:
                        _existing = {m.get("id") for m in memories}
                        _extra = collection.get(ids=_matched_ids, include=["documents", "metadatas"])
                        _added = 0
                        for _j, _mid in enumerate(_extra.get("ids", [])):
                            if _mid in _existing:
                                continue
                            _m = _extra["metadatas"][_j] if _extra.get("metadatas") else {}
                            _d = _extra["documents"][_j] if _extra.get("documents") else ""
                            memories.append({
                                "id": _mid, "summary": _m.get("summary", "")[:80],
                                "tags": _m.get("tags", ""), "similarity": 0.5,
                                "hit_count": _m.get("hit_count", 0),
                                "source": "tag_match",
                            })
                            docs_for_bm25.append(_d)
                            _existing.add(_mid)
                            _added += 1
                            if _added >= 10:
                                break
                        if _added:
                            logger.info("query_memory 标签补充检索: 新增%d条", _added)
            except Exception:
                pass

        # BM25 关键词重排序（基于候选集原文）
        try:
            if not docs_for_bm25:
                raise ValueError("empty corpus")
            from rank_bm25 import BM25Okapi
            import jieba
            query_tokens = [t for t in jieba.cut(query.strip()) if t.strip()]
            if not query_tokens:
                query_tokens = [query.strip()[:10]]
            corpus_tokens = [list(jieba.cut(d)) for d in docs_for_bm25]
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
            for i, mem in enumerate(memories):
                mem["_bm25"] = scores[i] if i < len(scores) else 0
            from app.retrieval.scoring import compute_score
            # 从 error_reports.jsonl 统计 top20 结果的错误报告数
            error_counts: dict[str, int] = {}
            try:
                _err_path = os.path.join(os.environ.get("DATA_DIR", "./data"), "error_reports.jsonl")
                if os.path.exists(_err_path):
                    with open(_err_path, encoding="utf-8") as _ef:
                        for _line in _ef:
                            _line = _line.strip()
                            if _line:
                                try:
                                    _er = json.loads(_line)
                                    _mid = _er.get("memory_id", "")
                                    if _mid:
                                        error_counts[_mid] = error_counts.get(_mid, 0) + 1
                                except Exception:
                                    pass
            except Exception:
                pass
            memories.sort(key=lambda x:
                compute_score(
                    similarity=x.get("similarity", 0),
                    hit_count=x.get("hit_count", 0) or 0,
                    source_bonus=(0.1 if x.get("source") in ("text_match", "entity_match") else 0.0),
                    error_penalty=error_counts.get(x.get("id", ""), 0) * 0.1,
                    bm25_score=x.get("_bm25", 0) or 0,
                )
                + 0.001 * (x.get("_bm25", 0) or 0)
            , reverse=True)
        except Exception as e:
            logger.warning("query_memory BM25排序失败: %s", e)
            memories.sort(key=lambda x: (x.get("_kw_match", False), x.get("similarity", 0)), reverse=True)

        # ★ 先截断到 top_k，再扩展补充检索（防止补充条目被截掉）
        memories = memories[:top_k]

        # 关键词文本补充检索：截断后扫描全文，提取包含≥2个查询关键词的记忆（bonus）
        try:
            import jieba, jieba.analyse
            _kws = jieba.analyse.extract_tags(query, topK=5) if query else []
            if _kws:
                _all_docs = collection.get(include=["documents", "metadatas"])
                _existing_ids = {m["id"] for m in memories}
                _matched = []
                for _i, _mid in enumerate(_all_docs.get("ids", [])):
                    if _mid in _existing_ids:
                        continue
                    _m = _all_docs["metadatas"][_i] if _all_docs.get("metadatas") else {}
                    if _m.get("stale", False):
                        continue
                    _text = _m.get("summary", "") or ""
                    # 如果请求带了时间范围，跳过范围外的记忆
                    if from_date or to_date:
                        _ts = _m.get("timestamp", 0)
                        if _ts:
                            if from_date:
                                _from_ts = datetime.strptime(from_date, "%Y-%m-%d").timestamp()
                                if _ts < _from_ts:
                                    continue
                            if to_date:
                                _to_ts = datetime.strptime(to_date, "%Y-%m-%d").timestamp() + 86400
                                if _ts > _to_ts:
                                    continue
                    _cnt = sum(1 for k in _kws if k in _text)
                    if _cnt >= 2:
                        _matched.append((_mid, _m, _cnt))
                if _matched:
                    _matched.sort(key=lambda x: -x[2])
                    for _mid, _m, _ in _matched[:3]:
                        if _mid in _existing_ids:
                            continue
                        memories.append({
                            "id": _mid, "summary": _m.get("summary", "")[:80],
                            "tags": _m.get("tags", ""), "similarity": 0.6,
                            "hit_count": _m.get("hit_count", 0),
                            "source": "kw_match",
                        })
                        _existing_ids.add(_mid)
                    if _matched:
                        logger.info("query_memory 关键词文本补充: 候选%d条", len(_matched))
        except Exception:
            pass

        # 共现扩展：top3 结果的共现伙伴追加到结果中
        if co_tracker is not None and memories:
            try:
                top_ids = [m["id"] for m in memories[:3] if m.get("id")]
                related = co_tracker.query(top_ids)[:3]
                existing_ids = {m["id"] for m in memories}
                for item in related:
                    mem_id = item.get("id", "")
                    if mem_id and mem_id not in existing_ids:
                        try:
                            _extra = collection.get(ids=[mem_id], include=["documents", "metadatas"])
                            if _extra.get("ids"):
                                _meta = dict(_extra["metadatas"][0]) if _extra.get("metadatas") else {}
                                _doc = _extra["documents"][0] if _extra.get("documents") else ""
                                memories.append({
                                    "id": mem_id, "document": _doc, "metadata": _meta,
                                    "summary": _meta.get("summary", "")[:80],
                                    "tags": _meta.get("tags", ""),
                                    "hit_count": _meta.get("hit_count", 0),
                                    "similarity": 0, "source": "co_occurrence",
                                })
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("query_memory 共现扩展失败: %s", exc)

        logger.info("query_memory: query=%s 返回 %d 条", query, len(memories))
        return memories     # 前面已截断到 top_k，共现条目是额外 bonus

    # ── 纯时间/结构化检索 ──
    elif has_time or has_filters:
        try:
            results = collection.get(
                where=where_get,
                limit=top_k,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning("query_memory 时间检索失败: %s", e)
            return []

        memories = []
        for i, mem_id in enumerate(results.get("ids", [])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            doc = results["documents"][i] if results.get("documents") else ""
            ts = meta.get("timestamp", 0)
            ctx_data = chat_history_obj.get_context_by_chroma_id(mem_id, before=3, after=3) if chat_history_obj else {"context_before": [], "context_after": []}
            memories.append({
                "id": mem_id,
                "summary": meta.get("summary", "")[:80],
                "tags": meta.get("tags", ""),
                "timestamp": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "",
                "hit_count": meta.get("hit_count", 0),
                "_preview": doc[:120],
                "context_before": ctx_data.get("context_before", []),
                "context_after": ctx_data.get("context_after", []),
            })

        memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        logger.info("query_memory: 时间检索 from=%s to=%s 返回 %d 条", from_date, to_date, len(memories))
        return memories[:top_k]

    return []


COUNT_MEMORIES_TOOL = {
    "type": "function",
    "function": {
        "name": "count_memories",
        "description": "统计记忆库中的记忆总数。当用户问「有多少条记忆」「我们聊了多少轮」等统计类问题时调用。不会返回具体内容，只返回数字。",
    },
}


def count_memories(collection) -> dict:
    """统计记忆库中的记忆总数。"""
    try:
        total = collection.count()
        return {"total_memories": total}
    except Exception as e:
        logger.warning("count_memories 失败: %s", e)
        return {"error": "查询失败"}


# ===================================================================
# 查询工具（合并）— query_explore 替代 5 个独立工具
# ===================================================================
_query_explore_init_lock = threading.Lock()
_query_explore_clients: dict[str, object] = {}  # path → PersistentClient
_QUERY_EXPLORE_MAX_CLIENTS = 10  # LRU 上限，防止无限增长

def _get_chroma_collection(path: str, name: str = "memories"):
    """缓存 ChromaDB PersistentClient，避免每次查询新建。LRU 淘汰。"""
    import chromadb
    if path not in _query_explore_clients:
        with _query_explore_init_lock:
            if path not in _query_explore_clients:
                if len(_query_explore_clients) >= _QUERY_EXPLORE_MAX_CLIENTS:
                    oldest = next(iter(_query_explore_clients))
                    del _query_explore_clients[oldest]
                _query_explore_clients[path] = chromadb.PersistentClient(path=path)
    return _query_explore_clients[path].get_or_create_collection(name, embedding_function=None)

def query_explore(mode: str = "timeline", _collection=None, **kwargs) -> str:
    """统一探索接口。mode: timeline / emotion / topics / co_occurrence / rhythm"""
    from datetime import datetime, timedelta
    if _collection is None:
        from app.config.settings import CHROMA_PERSIST_DIR
        _collection = _get_chroma_collection(CHROMA_PERSIST_DIR)
    coll = _collection

    if mode == "timeline":
        from_date = kwargs.get("from_date", "")
        to_date = kwargs.get("to_date", "")
        when = kwargs.get("when", "")
        if when and not from_date and not to_date:
            parsed = _parse_natural_date(when)
            if parsed:
                from_date = parsed["from_date"]
                to_date = parsed["to_date"]
        group_by = kwargs.get("group_by", "day")
        try:
            ts_s = datetime.strptime(from_date, "%Y-%m-%d").timestamp() if from_date else 0
            ts_e = datetime.strptime(to_date, "%Y-%m-%d").timestamp() + 86400 if to_date else 0
        except ValueError:
            return "时间格式错误"
        w = {}
        if ts_s and ts_e: w = {"$and": [{"timestamp": {"$gte": ts_s}}, {"timestamp": {"$lte": ts_e}}]}
        elif ts_s: w = {"timestamp": {"$gte": ts_s}}
        elif ts_e: w = {"timestamp": {"$lte": ts_e}}
        else: return "请提供时间范围"
        try:
            r = coll.get(where=w, include=["metadatas"])
        except Exception as e:
            return f"查询失败: {e}"
        if not r["ids"]:
            return "该时间段内没有记忆"
        items = []
        for i, mid in enumerate(r["ids"]):
            m = r["metadatas"][i]; ts = m.get("timestamp", 0)
            d = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            items.append({"id": mid[:8], "time": d, "summary": (m.get("summary","") or "")[:80]})
        items.sort(key=lambda x: x["time"])
        if group_by == "week":
            from itertools import groupby
            def _wk(x):
                d = datetime.strptime(x["time"][:10], "%Y-%m-%d")
                return d.strftime("%Y-W%W")
            lines = [f"[{wk}] {len(list(g))}条" for wk, g in groupby(items, _wk)]
            return "\n".join(lines)
        return "\n".join(f"{it['time']} {it['summary']}" for it in items)

    elif mode == "emotion":
        min_i = kwargs.get("min_intensity", 2); v = kwargs.get("valence", ""); tk = kwargs.get("top_k", 10)
        w = {"emotional_intensity": {"$gte": min_i}}
        if v in ("positive","negative","neutral"):
            w = {"$and": [w, {"$or": [
                {"emotion_valence": {"$eq": v}},
                {"emotion_valence_bin": {"$eq": v}},
            ]}]}
        try:
            r = coll.get(where=w, include=["metadatas"])
        except Exception as e:
            return f"查询失败: {e}"
        if not r["ids"]: return f"未找到情绪强度≥{min_i}的记忆"
        items = []
        for i, mid in enumerate(r["ids"]):
            m = r["metadatas"][i]
            items.append({"id": mid[:8], "ei": m.get("emotional_intensity",0), "v": m.get("emotion_valence_bin", "") or m.get("emotion_valence",""), "s": (m.get("summary","") or "")[:60]})
        items.sort(key=lambda x: -x["ei"])
        return "\n".join(f"[强度{i['ei']}][{i['v']}] {i['s']}" for i in items[:tk])

    elif mode == "topics":
        from_date = kwargs.get("from_date", ""); to_date = kwargs.get("to_date", "")
        when = kwargs.get("when", "")
        if when and not from_date and not to_date:
            parsed = _parse_natural_date(when)
            if parsed:
                from_date = parsed["from_date"]
                to_date = parsed["to_date"]
        import jieba.analyse
        try:
            mid_ts = datetime.strptime(from_date, "%Y-%m-%d").timestamp() if from_date else 0
            end_ts = datetime.strptime(to_date, "%Y-%m-%d").timestamp() + 86400 if to_date else 0
        except ValueError:
            return "时间格式错误"
        try:
            r = coll.get(include=["metadatas","documents"])
        except Exception as e:
            return f"查询失败: {e}"
        bt, at = "", ""
        sp = (mid_ts + end_ts) / 2 if mid_ts and end_ts else (mid_ts or end_ts)
        for i, mid in enumerate(r.get("ids",[])):
            m = r["metadatas"][i] if r.get("metadatas") else {}; ts = m.get("timestamp",0); d = r["documents"][i] if r.get("documents") else ""
            if not d: continue
            if sp and ts < sp: bt += d + " "
            elif ts >= sp: at += d + " "
        if not bt or not at: return "数据不足"
        bk = set(jieba.analyse.extract_tags(bt, topK=15)); ak = set(jieba.analyse.extract_tags(at, topK=15))
        return (f"消失: {', '.join(list(bk-ak)[:8]) or '无'}\n新增: {', '.join(list(ak-bk)[:8]) or '无'}\n持续: {', '.join(list(bk&ak)[:8]) or '无'}")

    elif mode == "co_occurrence":
        mid = kwargs.get("memory_id", ""); tk = kwargs.get("top_k", 5)
        from app.memory.cooccur import CoOccurrenceTracker
        co = CoOccurrenceTracker(); related = co.get_co_with(mid)
        if not related: return f"未找到共现关系"
        pids = [r["id"] for r in related[:tk]]
        r2 = coll.get(ids=pids, include=["metadatas"])
        lines = []
        for i, pid in enumerate(r2.get("ids",[])):
            m = r2["metadatas"][i] if r2.get("metadatas") else {}
            c = next((r["count"] for r in related if r["id"] == pid), 0)
            lines.append(f"[共现{c}次] {(m.get('summary','') or '')[:60]} ({pid[:8]})")
        return "\n".join(lines)

    elif mode == "rhythm":
        from calendar import monthrange
        from datetime import timedelta
        now = datetime.now()
        wl = []
        lr_s = datetime(now.year-1, now.month, max(1, now.day-3))
        lr_e = datetime(now.year-1, now.month, min(28, now.day+3))
        wl.append((lr_s, lr_e, "去年同期"))
        py, pm = (now.year-1, 12) if now.month == 1 else (now.year, now.month-1)
        md = monthrange(py, pm)[1]; pd = min(now.day, md)
        lm_s = datetime(py, pm, max(1, pd-3)); lm_e = datetime(py, pm, min(md, pd+3))
        wl.append((lm_s, lm_e, "上月同日"))
        lw = now - timedelta(days=7)
        wl.append((datetime(lw.year, lw.month, max(1, lw.day-1)),
                   datetime(lw.year, lw.month, min(monthrange(lw.year, lw.month)[1], lw.day+1)),
                   "上周同日"))
        iso_week = now.isocalendar()[1]
        jan4 = datetime(now.year-1, 1, 4)
        jan4_week = jan4.isocalendar()[1]
        monday = jan4 + timedelta(days=(iso_week - jan4_week) * 7 - jan4.weekday())
        wl.append((monday - timedelta(days=3), monday + timedelta(days=10), "去年同周"))
        out = []
        for ws, we, wn in wl:
            try:
                rr = coll.get(where={"$and": [{"timestamp": {"$gte": ws.timestamp()}}, {"timestamp": {"$lte": we.timestamp()+86399}}]}, include=["metadatas"])
            except: continue
            if rr["ids"]:
                its = []
                for i, mid in enumerate(rr["ids"]):
                    m = rr["metadatas"][i]; ts = m.get("timestamp",0)
                    its.append(f"{datetime.fromtimestamp(ts).strftime('%m-%d') if ts else ''} {(m.get('summary','') or '')[:50]}")
                out.append(f"【{wn}】\n" + "\n".join(its[:5]))
        return "\n\n".join(out) if out else "未找到匹配"

    elif mode == "entity":
        name = kwargs.get("name", "")
        etype = kwargs.get("etype", "")
        if not name:
            return "请提供要查询的实体名称"
        if _collection is not None:
            coll = _collection
        else:
            from app.config.settings import CHROMA_PERSIST_DIR
            coll = _get_chroma_collection(CHROMA_PERSIST_DIR)
        try:
            r = coll.get(include=["metadatas"])
        except Exception as e:
            return f"查询失败: {e}"
        results = []
        for i, mid in enumerate(r.get("ids", [])):
            m = r["metadatas"][i] if r.get("metadatas") else {}
            raw = m.get("entities", "")
            if not raw:
                continue
            try:
                ents = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            for ent in ents:
                if etype and ent.get("type", "") != etype:
                    continue
                if name in ent.get("text", ""):
                    sm = (m.get("summary", "") or "")[:80]
                    ts = m.get("timestamp", 0)
                    dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
                    results.append(f"[{dt}] {sm} ({mid[:8]})")
                    break
        if not results:
            return f"未找到包含「{name}」的记忆"
        return "\n".join(results[:20])

    elif mode == "forgiving":
        """宽松匹配：常规检索不足时放宽条件。"""
        query = kwargs.get("query", "")
        emb = kwargs.get("emb")
        ctx = kwargs.get("ctx")
        if not query or emb is None or ctx is None:
            return "forgiving 模式需要 query / emb / ctx 参数"
        from app.retrieval.pipeline import run_chat_retrieval
        results = run_chat_retrieval(query, emb, ctx)
        if len(results) < 3:
            try:
                import jieba.analyse
                kws = jieba.analyse.extract_tags(query, topK=3)
                seen_ids = {r["id"] for r in results if r.get("id")}
                for kw in kws:
                    if len(kw) < 2:
                        continue
                    kw_results = coll.get(
                        where={"tags": {"$contains": kw}},
                        include=["documents", "metadatas"],
                        limit=5,
                    )
                    for i, doc in enumerate(kw_results.get("documents", []) or []):
                        mid = kw_results["ids"][i]
                        if mid in seen_ids:
                            continue
                        seen_ids.add(mid)
                        meta = kw_results["metadatas"][i] if kw_results.get("metadatas") else {}
                        results.append({
                            "id": mid, "document": doc, "metadata": meta,
                            "source": "kw_match", "distance": 0.4,
                        })
            except Exception as exc:
                logger.debug("forgiving kw放宽失败: %s", exc)
        if not results:
            return "宽松检索后仍未找到匹配记忆"
        lines = []
        for r in results[:10]:
            meta = r.get("metadata") or {}
            ts = meta.get("timestamp", 0)
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            sm = meta.get("summary", "") or r.get("document", "")[:60]
            lines.append(f"[{dt}] {sm} ({r['source']})")
        return "\n".join(lines)

    return f"未知模式: {mode}，支持: timeline/emotion/topics/co_occurrence/rhythm/entity/forgiving"


# ===================================================================
# ⑥ analyze_pattern — LLM 对检索到的数据做二次分析
# ===================================================================
def analyze_pattern(memory_ids: list[str] = None, analysis_type: str = "summary") -> str:
    """将一批记忆的原文取出，供 LLM 自己分析规律。结果不进记忆库。"""
    if not memory_ids:
        return "请提供要分析的记忆 ID 列表"
    from app.config.settings import CHROMA_PERSIST_DIR
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    coll = client.get_or_create_collection("memories", embedding_function=None)
    try:
        results = coll.get(ids=memory_ids, include=["documents", "metadatas"])
    except Exception as e:
        return f"获取记忆失败: {e}"
    if not results["ids"]:
        return "未找到指定记忆"
    parts = [f"分析类型: {analysis_type}", f"共 {len(results['ids'])} 条记忆", ""]
    for i, mid in enumerate(results["ids"]):
        doc = results["documents"][i] if results.get("documents") else ""
        meta = results["metadatas"][i] if results.get("metadatas") else {}
        ts = meta.get("timestamp", 0)
        dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
        parts.append(f"--- [{dt_str}] ---\n{doc[:500]}")
    return "\n\n".join(parts)
