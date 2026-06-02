"""工具分发 — 跨模块可复用的查询/操作函数。"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from app.llm.embed import local_embed

logger = logging.getLogger(__name__)


# ── 语义检索 ──────────────────────────────────────────────

def query_memory(collection: Any, query: str, *, top_k: int = 5) -> list[dict]:
    """对 ChromaDB collection 执行语义检索。

    Args:
        collection: ChromaDB 的 Collection 对象（只读）。
        query: 查询文本。
        top_k: 返回结果数。

    Returns:
        [{"id": str, "summary": str, "_preview": str, "hit_count": int, "similarity": float}, ...]
        失败返回 [{"error": str}]。
    """
    try:
        embedding = local_embed(query)
        if embedding is None:
            return [{"error": "embedding 失败", "query": query[:100]}]

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        items: list[dict] = []
        ids_list = results["ids"][0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for i in range(len(ids_list)):
            meta = metas_list[i] if i < len(metas_list) and metas_list[i] else {}
            doc = docs_list[i] if i < len(docs_list) else ""
            similarity = round(1.0 - float(dists_list[i] if i < len(dists_list) else 0.5), 4)

            items.append({
                "id": ids_list[i],
                "_preview": meta.get("summary", doc[:200]),
                "summary": meta.get("summary", doc[:200]),
                "hit_count": int(meta.get("hit_count", 0) or 0),
                "similarity": similarity,
            })

        return items

    except Exception as e:
        logger.warning("query_memory 失败: %s", e)
        return [{"error": str(e)}]


# ── 记忆探索 ──────────────────────────────────────────────

def query_explore(
    mode: str,
    *,
    _collection: Any = None,
    min_intensity: int = 1,
    from_date: str = "",
    to_date: str = "",
    top_k: int = 5,
    memory_id: str = "",
) -> str:
    """多模式记忆探索查询。

    mode 支持:
      - "emotion":     情绪筛选（min_intensity 以上）
      - "timeline":    时间线查询（from_date ~ to_date）
      - "co_occurrence": 共现标签查询（按 memory_id）
      - "rhythm":      时间节律分析
      - "topics":      话题聚类统计
      - 其他:          返回 "未知模式"
    """
    from datetime import datetime as dt

    if mode == "emotion":
        if _collection is None:
            return "未提供 collection"
        try:
            all_data = _collection.get(include=["metadatas", "documents"])
            lines = []
            for i, mid in enumerate(all_data.get("ids", [])):
                meta = (all_data.get("metadatas") or [{}])[i]
                if int(meta.get("emotional_intensity", 0) or 0) >= min_intensity:
                    lines.append(f"- [{meta.get('timestamp', '?')}] {meta.get('summary', all_data['documents'][i][:80])}")
            return "\n".join(lines) if lines else "无符合条件的情绪记忆"
        except Exception as e:
            return f"查询失败: {e}"

    elif mode == "timeline":
        if _collection is None:
            return "未提供 collection"
        try:
            all_data = _collection.get(include=["metadatas", "documents"])
            items = []
            for i, mid in enumerate(all_data.get("ids", [])):
                meta = (all_data.get("metadatas") or [{}])[i]
                ts_raw = meta.get("timestamp", "")
                if from_date and ts_raw < from_date:
                    continue
                if to_date and ts_raw > to_date:
                    continue
                items.append((ts_raw, meta.get("summary", all_data["documents"][i][:80])))
            items.sort(key=lambda x: x[0])
            return "\n".join(f"- [{ts}] {summary}" for ts, summary in items) if items else "该时间范围内无记忆"
        except Exception as e:
            return f"查询失败: {e}"

    elif mode == "co_occurrence":
        if not memory_id:
            return "请提供 memory_id"
        if _collection is None:
            return "未提供 collection"
        try:
            # 查找共现文件中的关联记录（不查记忆自身标签）
            import json, os
            from app.config.settings import CO_OCCURRENCE_FILE
            co_related: list[str] = []
            if os.path.exists(CO_OCCURRENCE_FILE):
                with open(CO_OCCURRENCE_FILE, "r", encoding="utf-8") as f:
                    pairs = json.load(f)
                for pair in pairs:
                    src = pair.get("source_id") or pair.get("src_id", "")
                    tgt = pair.get("target_id") or pair.get("tgt_id", "")
                    if (src == memory_id and tgt) or (tgt == memory_id and src):
                        label = pair.get("label", "") or pair.get("tag", "") or tgt
                        if label and label not in co_related:
                            co_related.append(label)

            if not co_related:
                return "未找到共现关联"
            return f"共现标签: {', '.join(co_related[:15])}"
        except Exception as e:
            return f"查询失败: {e}"

    elif mode == "rhythm":
        if _collection is None:
            return "未提供 collection"
        try:
            all_data = _collection.get(include=["metadatas"])
            hour_counter = Counter()
            for meta in (all_data.get("metadatas") or []):
                ts_raw = (meta or {}).get("timestamp", "")
                if ts_raw:
                    try:
                        hour = dt.fromisoformat(str(ts_raw)[:19]).hour
                        period = (
                            "深夜" if 0 <= hour < 6 else
                            "早晨" if 6 <= hour < 9 else
                            "上午" if 9 <= hour < 12 else
                            "中午" if 12 <= hour < 14 else
                            "下午" if 14 <= hour < 18 else
                            "傍晚" if 18 <= hour < 21 else
                            "晚上"
                        )
                        hour_counter[period] += 1
                    except (ValueError, OSError):
                        pass
            if not hour_counter:
                return "无法分析时间节律"
            lines = [f"- {period}: {count} 条" for period, count in hour_counter.most_common()]
            return "\n".join(lines)
        except Exception as e:
            return f"查询失败: {e}"

    elif mode == "topics":
        if _collection is None:
            return "未提供 collection"
        try:
            all_data = _collection.get(include=["metadatas"])
            tag_counter = Counter()
            for meta in (all_data.get("metadatas") or []):
                tags_raw = (meta or {}).get("tags", "")
                for t in tags_raw.replace("，", ",").split(","):
                    t = t.strip()
                    if t:
                        tag_counter[t] += 1
            if not tag_counter:
                return "无话题标签"
            lines = [f"- {tag}: {cnt} 条" for tag, cnt in tag_counter.most_common(10)]
            return "\n".join(lines)
        except Exception as e:
            return f"查询失败: {e}"

    else:
        return f"未知模式: {mode}"


# ── 模式分析 ──────────────────────────────────────────────

def analyze_pattern(memory_ids: list[str], *, _collection: Any = None) -> str:
    """分析指定记忆的模式特征。

    Args:
        memory_ids: 要分析的内存 ID 列表。
        _collection: 可选，ChromaDB collection。

    Returns:
        分析结果的文本描述。
    """
    if not memory_ids:
        return "请提供至少一个 memory_id"

    if _collection is None:
        return "未提供 collection，无法获取记忆"

    try:
        result = _collection.get(ids=memory_ids, include=["metadatas", "documents"])
        if not result or not result.get("metadatas"):
            return f"未找到指定记忆: {memory_ids}"

        # 分析情绪分布
        emotions = []
        tags_set = set()
        for meta in (result.get("metadatas") or []):
            if meta:
                v = meta.get("emotion_valence", "neutral")
                emotions.append(v)
                for t in (meta.get("tags", "") or "").replace("，", ",").split(","):
                    t = t.strip()
                    if t:
                        tags_set.add(t)

        emotion_dist = Counter(emotions)
        lines = [
            f"分析 {len(memory_ids)} 条记忆:",
            f"- 情绪分布: {dict(emotion_dist)}",
            f"- 共同标签: {', '.join(sorted(tags_set)[:10]) if tags_set else '无'}",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"分析失败: {e}"
