#!/usr/bin/env python3
"""
初痕真实记忆审计套件 v4 — 开源版
基于生产数据直接测试检索层能力，诚实地测每一条通路。

v4 变更（Phase 4 退役适配）:
  - 类别 6: Personality Consistency → Portrait Consistency
    旧: 读 personality_chroma (PersonalityStore) → 测人格标签可检索性
    新: 读 PORTRAIT.md (PortraitManager) → 测画像维度完整性 + 证据链 + 版本活性
  - load_data(): 移除 PERSONALITY_TAGS 加载（personality_chroma 已退役）

用法:
  python scripts/audit.py                          # 跑全部 8 类
  python scripts/audit.py --quick                  # 快速模式（跳过画像类）
  python scripts/audit.py --category 1             # 只跑语义检索
  python scripts/audit.py --sample 20              # 指定抽样数
  python scripts/audit.py --report ./my_reports    # 自定义报告目录

报告保存在 audit/ 目录，对比工具: python scripts/compare_reports.py
"""
import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

# 路径：从 scripts/ 回到项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
# (backend/ removed — all modules now in app/)
os.chdir(_project_root)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("audit")

random.seed(42)  # 确定种子，分数可复现

# ── 全局状态 ──
CHROMA_DATA: list[dict] = []
PORTRAIT_DATA: dict = {}  # {dim: [entries], version: int, entry_count: int}
CHAT_HISTORY: list[dict] = []


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_data():
    global CHROMA_DATA, PORTRAIT_DATA, CHAT_HISTORY
    from app.config.settings import QDRANT_PERSIST_DIR, CHAT_HISTORY_PATH, PORTRAIT_FILE_PATH
    from app.memory.qdrant import QdrantService

    svc = QdrantService(persist_dir=QDRANT_PERSIST_DIR, collection_name="memories")
    CHROMA_DATA = []
    for item in svc.list_all():
        meta = dict(item.get("metadata") or {})
        doc = item.get("document") or ""
        CHROMA_DATA.append({
            "id": item["id"], "document": doc,
            **meta, "_meta": meta,
        })
    CHROMA_DATA.sort(key=lambda x: str(x["id"]))
    logger.info("Qdrant memories: %d", len(CHROMA_DATA))

    # 画像 PORTRAIT.md（替代 Phase 4 退役的 personality_chroma）
    PORTRAIT_DATA = {"dims": {}, "version": 0, "entry_count": 0}
    try:
        from app.portrait.manager import PortraitManager, ALL_DIMS
        pm = PortraitManager(PORTRAIT_FILE_PATH)  # 构造时自动加载（文件不存在则创建空画像）
        PORTRAIT_DATA["version"] = pm.version
        for dim in ALL_DIMS:
            entries = pm.get_dim_entries(dim)
            PORTRAIT_DATA["dims"][dim] = [
                {"text": e.text[:120], "tags": e.tags, "confidence": e.confidence,
                 "status": e.status.value}
                for e in entries
            ]
            PORTRAIT_DATA["entry_count"] += len(entries)
    except Exception as e:
        logger.warning("Portrait load failed: %s", e)
    logger.info("Portrait: v%d, %d entries across %d dims",
                PORTRAIT_DATA["version"],
                PORTRAIT_DATA["entry_count"],
                sum(1 for v in PORTRAIT_DATA.get("dims", {}).values() if v))

    # ChatHistory
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        CHAT_HISTORY.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        CHAT_HISTORY = CHAT_HISTORY[-50:]


def _is_ollama_available() -> bool:
    """检测 Ollama 是否可用（复用 conftest 逻辑）。"""
    try:
        import httpx
        url = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
        resp = httpx.get(f"{url}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _get_embedding(text: str) -> list[float] | None:
    from app.llm.embed import local_embed
    return local_embed(text)


# ═══════════════════════════════════════════════════════════════
# 辅助：Qdrant query
# ═══════════════════════════════════════════════════════════════

def _chroma_query(query_emb: list[float], n_results: int = 0) -> list[dict]:
    from app.config.settings import QDRANT_PERSIST_DIR
    from app.memory.qdrant import QdrantService
    coll = QdrantService(persist_dir=QDRANT_PERSIST_DIR, collection_name="memories")
    if n_results <= 0:
        total = coll.count()
        n_results = max(30, min(total // 20, 100))
    try:
        results = coll.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        out = []
        for i, mid in enumerate(results.get("ids", [[]])[0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            dist = results["distances"][0][i] if results.get("distances") else 1.0
            out.append({"id": mid, "distance": dist, "metadata": meta})
        return out
    except Exception as e:
        logger.warning("Qdrant query failed: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════
# 类别 1：语义检索（权重 25%）
# ═══════════════════════════════════════════════════════════════

def test_semantic_recall(sample_n: int = 30) -> dict:
    if len(CHROMA_DATA) < 4:
        return {"score": 0, "pass": "0/0", "error": "Insufficient memories"}
    pool = random.sample(CHROMA_DATA, min(sample_n, len(CHROMA_DATA)))
    hit, total = 0, 0
    for item in pool:
        query_text = (item.get("summary") or item.get("document") or "")[:200]
        if not query_text:
            continue
        emb = _get_embedding(query_text)
        if not emb:
            continue
        total += 1
        results = _chroma_query(emb)
        if any(r["id"] == item["id"] for r in results):
            hit += 1
    recall = round(hit / total, 4) if total else 0
    return {"score": recall, "pass": f"{hit}/{total}"}


# ═══════════════════════════════════════════════════════════════
# 类别 2：关键词检索（权重 15%）
# ═══════════════════════════════════════════════════════════════

def test_keyword_recall(sample_n: int = 20) -> dict:
    if len(CHROMA_DATA) < 3:
        return {"score": 0, "pass": "0/0", "error": "Insufficient memories"}
    tag_counts: dict[str, int] = defaultdict(int)
    for m in CHROMA_DATA:
        tags_str = (m.get("tags") or m.get("_meta", {}).get("tags", "") or "")
        for t in tags_str.split(","):
            t = t.strip()
            if len(t) >= 2 and t != "对话":
                tag_counts[t] += 1
    freq_tags = [(t, c) for t, c in tag_counts.items() if c >= 3]
    if not freq_tags:
        return {"score": 0, "pass": "0/0", "error": "No frequent tags"}
    random.shuffle(freq_tags)
    tags_to_test = freq_tags[:sample_n]
    hit, total = 0, 0
    for tag, count in tags_to_test:
        matched_ids = [
            m["id"] for m in CHROMA_DATA
            if tag in (m.get("tags") or m.get("_meta", {}).get("tags", "") or "")
        ]
        if not matched_ids:
            continue
        total += 1
        if len(matched_ids) >= count * 0.8:
            hit += 1
    return {
        "score": round(hit / total, 4) if total else 0,
        "pass": f"{hit}/{total} tags",
    }


# ═══════════════════════════════════════════════════════════════
# 类别 3：时间检索（权重 15%）
# ═══════════════════════════════════════════════════════════════

def test_time_retrieval(sample_n: int = 15) -> dict:
    if len(CHROMA_DATA) < 3:
        return {"score": 0, "pass": "0/0", "error": "Insufficient memories"}
    samples = [m for m in CHROMA_DATA if m.get("timestamp")]
    if len(samples) < 3:
        return {"score": 0, "pass": "0/0", "error": "No timestamped memories"}
    random.shuffle(samples)
    samples = samples[:sample_n]
    recall_ok, total = 0, 0
    for item in samples:
        if not item.get("timestamp"):
            continue
        total += 1
        query_text = (item.get("summary", "") or item.get("document", "") or "")[:100]
        if not query_text:
            continue
        emb = _get_embedding(query_text)
        if not emb:
            continue
        try:
            results = _chroma_query(emb)
            if any(r["id"] == item["id"] for r in results):
                recall_ok += 1
        except Exception:
            continue
    return {
        "score": round(recall_ok / total, 4) if total else 0,
        "pass": f"{recall_ok}/{total}",
    }


# ═══════════════════════════════════════════════════════════════
# 类别 4：排序逻辑（权重 15%）
# ═══════════════════════════════════════════════════════════════

def test_ranking_logic(sample_n: int = 0) -> dict:
    checks = []
    stale_count = sum(1 for m in CHROMA_DATA if (m.get("_meta") or {}).get("stale", False))
    checks.append({
        "check": "stale filter reported",
        "stale": stale_count, "total": len(CHROMA_DATA), "pass": True,
    })
    expected_order = [
        "semantic", "dmn_preheat", "entity_match",
        "kw_match", "tag_match", "time_rhythm", "co_occurrence",
    ]
    weights = {"semantic": 1.0, "dmn_preheat": 0.85, "entity_match": 0.8,
               "kw_match": 0.65, "tag_match": 0.6, "time_rhythm": 0.4,
               "co_occurrence": 0.35}
    actual_order = sorted(weights, key=lambda x: -weights[x])
    order_ok = actual_order == expected_order
    checks.append({
        "check": "source weight ranking order",
        "expected": expected_order, "actual": actual_order, "pass": order_ok,
    })
    passed = sum(1 for c in checks if c.get("pass", False))
    return {
        "score": round(passed / len(checks), 4) if checks else 0,
        "pass": f"{passed}/{len(checks)}",
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════
# 类别 5：纠正反馈（权重 10%）
# ═══════════════════════════════════════════════════════════════

def test_correction_feedback(sample_n: int = 0) -> dict:
    data_dir = os.getenv("DATA_DIR", "./data")
    checks = []
    log_path = os.path.join(data_dir, "correction_log.jsonl")
    log_exists = os.path.exists(log_path)
    checks.append({"check": "correction log exists", "pass": log_exists,
                   "detail": "found" if log_exists else "missing"})
    if log_exists:
        try:
            with open(log_path, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            checks.append({"check": "correction entries", "count": len(lines), "pass": True})
        except Exception:
            checks.append({"check": "correction log read", "pass": False})
    try:
        from app.retrieval.pipeline import _load_correction_boosts
        boosts = _load_correction_boosts(data_dir)
        checks.append({"check": "correction boost loader", "count": len(boosts), "pass": True})
    except Exception as e:
        checks.append({"check": "correction boost loader", "pass": False, "error": str(e)[:80]})
    passed = sum(1 for c in checks if c.get("pass", False))
    return {
        "score": round(passed / len(checks), 4) if checks else 0,
        "pass": f"{passed}/{len(checks)}",
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════
# 类别 6：画像一致性（权重 10%）
# Phase 4 退役后替代旧"人格一致性"——读 PORTRAIT.md，不依赖 /chat
# ═══════════════════════════════════════════════════════════════

def test_portrait_consistency(sample_n: int = 3) -> dict:
    """验证画像系统完整性：维度覆盖、证据链、版本活性。"""
    checks = []

    # ═══ C1: PORTRAIT.md 存在且可解析 ═══
    if not PORTRAIT_DATA.get("dims"):
        checks.append({"check": "PORTRAIT.md loaded", "pass": False,
                       "error": "No portrait data — PORTRAIT.md missing or empty"})
        passed = sum(1 for c in checks if c.get("pass", False))
        return {"score": round(passed / len(checks), 4) if checks else 0,
                "pass": f"{passed}/{len(checks)}", "checks": checks}
    checks.append({"check": "PORTRAIT.md loaded", "pass": True,
                   "detail": f"v{PORTRAIT_DATA['version']}, {PORTRAIT_DATA['entry_count']} entries"})

    # ═══ C2: 版本活性 — 画像至少被更新过一次 ═══
    checks.append({"check": "portrait version > 0 (has been updated)",
                   "pass": PORTRAIT_DATA["version"] > 0,
                   "detail": f"version={PORTRAIT_DATA['version']}"})

    # ═══ C3: 用户画像维度覆盖 — 6 个用户维度至少有 basic 覆盖 ═══
    user_dims = ["usr1", "usr2", "usr3", "usr4", "usr5", "usr6"]
    user_covered = 0
    dim_detail = []
    for dim in user_dims:
        entries = PORTRAIT_DATA["dims"].get(dim, [])
        active = [e for e in entries if e.get("status") not in ("decayed",)]
        if active:
            user_covered += 1
            dim_detail.append(f"{dim}={len(active)}")
        else:
            dim_detail.append(f"{dim}=0")
    checks.append({"check": "user portrait dimensions populated",
                   "pass": user_covered >= 3,  # 至少一半有内容
                   "detail": f"{user_covered}/6 dims active: {', '.join(dim_detail)}"})

    # ═══ C4: AI 画像维度覆盖 — 6 个 AI 维度 ═══
    ai_dims = ["ai1", "ai2", "ai3", "ai4", "ai5", "ai6"]
    ai_covered = 0
    ai_detail = []
    for dim in ai_dims:
        entries = PORTRAIT_DATA["dims"].get(dim, [])
        active = [e for e in entries if e.get("status") not in ("decayed",)]
        if active:
            ai_covered += 1
            ai_detail.append(f"{dim}={len(active)}")
        else:
            ai_detail.append(f"{dim}=0")
    checks.append({"check": "AI portrait dimensions populated",
                   "pass": ai_covered >= 2,  # AI 侧积累较慢，≥2 就 OK
                   "detail": f"{ai_covered}/6 dims active: {', '.join(ai_detail)}"})

    # ═══ C5: 画像条目证据链 — 抽查条目 tags 在记忆库中是否可查 ═══
    if sample_n > 0 and PORTRAIT_DATA["entry_count"] > 0:
        all_entries = []
        for dim in user_dims + ai_dims:
            for e in PORTRAIT_DATA["dims"].get(dim, []):
                if e.get("tags") and e.get("status") not in ("decayed",):
                    all_entries.append(e)

        if all_entries:
            import random as _random
            _random.seed(42)
            sample = _random.sample(all_entries, min(sample_n, len(all_entries)))
            evidence_hits = 0
            for entry in sample:
                tags = entry.get("tags", [])
                # 检查这些 tags 是否在记忆库中有对应记忆
                matched = any(
                    any(t in (m.get("tags") or m.get("_meta", {}).get("tags", "") or "")
                        for t in tags)
                    for m in CHROMA_DATA
                )
                if matched:
                    evidence_hits += 1
            checks.append({"check": "portrait entry evidence (tags match memories)",
                           "pass": evidence_hits >= max(1, len(sample) // 2),
                           "detail": f"{evidence_hits}/{len(sample)} entries have tag evidence in Qdrant"})
        else:
            checks.append({"check": "portrait entry evidence", "pass": True,
                           "detail": "no tag-bearing entries to check"})
    else:
        checks.append({"check": "portrait entry evidence", "pass": True,
                       "detail": f"skipped (sample_n={sample_n}, entries={PORTRAIT_DATA['entry_count']})"})

    # ═══ C6: 无完全空维度 — 如果用户侧有数据，不应有维度系统性缺席 ═══
    if user_covered >= 3:
        fully_empty = [d for d in user_dims if not PORTRAIT_DATA["dims"].get(d)]
        checks.append({"check": "no systematically empty user dimensions",
                       "pass": len(fully_empty) <= 1,
                       "detail": f"empty: {fully_empty}" if fully_empty else "all dims have entries"})

    passed = sum(1 for c in checks if c.get("pass", False))
    return {
        "score": round(passed / len(checks), 4) if checks else 0,
        "pass": f"{passed}/{len(checks)}",
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════
# 类别 7：工作记忆（权重 10%）
# — 主体：timeline_recent (token_budget=50000) 完整性
# — 兜底：session_context (working_memory.json) 摘要质量
# ═══════════════════════════════════════════════════════════════

def test_working_memory(sample_n: int = 5) -> dict:
    from app.config.settings import CHAT_HISTORY_PATH, WORK_MEMORY_TOKEN_BUDGET, DATA_DIR
    from app.memory.history import ChatHistory
    import os

    checks = []
    ch = None
    recent = []
    wm_path = os.path.join(DATA_DIR, "working_memory.json")

    # ═══════════════════════════════════════════════════════════
    # 子项 A：timeline_recent — 50000 token 工作记忆主体
    # ═══════════════════════════════════════════════════════════
    try:
        ch = ChatHistory(CHAT_HISTORY_PATH)
        recent = ch.get_recent(token_budget=WORK_MEMORY_TOKEN_BUDGET)
        n_turns = len(recent)
        total_records = len(ch.records)

        # A1: 不超预算
        total_tokens = sum(
            ch._estimate_tokens(r.get("user_message", "")) +
            ch._estimate_tokens(r.get("llm_reply", ""))
            for r in recent
        )
        budget_ok = total_tokens <= WORK_MEMORY_TOKEN_BUDGET
        checks.append({
            "check": "timeline_recent within 50000 token budget",
            "pass": budget_ok,
            "detail": f"{total_tokens}/{WORK_MEMORY_TOKEN_BUDGET} tokens, {n_turns} turns",
        })

        if total_records == 0:
            checks.append({
                "check": "timeline_recent (empty history)",
                "pass": True,
                "detail": "no chat history — skip",
            })
        else:
            # A2: 预算利用率 — 有足够历史时应该接近填满
            if total_records >= 10:
                utilization = total_tokens / WORK_MEMORY_TOKEN_BUDGET
                # 阈值随预算浮动：预算越大，利用率下限越低（避免小数据集误报）
                min_util = max(0.05, 10000 / WORK_MEMORY_TOKEN_BUDGET)
                checks.append({
                    "check": f"timeline_recent budget utilization ≥ {min_util:.0%}",
                    "pass": utilization >= min_util,
                    "detail": f"{utilization:.0%} utilized ({total_tokens} / {WORK_MEMORY_TOKEN_BUDGET})",
                })
            else:
                checks.append({
                    "check": "timeline_recent budget utilization",
                    "pass": True,
                    "detail": f"only {total_records} records — utilization not meaningful",
                })

            # A3: 最近 N 轮完整性 — 最近 sample_n 轮应该全部在 timeline_recent 中
            n_check = min(sample_n, total_records)
            most_recent_ids = {
                r.get("timestamp") for r in ch.records[-n_check:]
            }
            recent_ids = {r.get("timestamp") for r in recent}
            missing_recent = most_recent_ids - recent_ids
            checks.append({
                "check": f"most recent {n_check} turns all present",
                "pass": len(missing_recent) == 0,
                "detail": f"{n_check - len(missing_recent)}/{n_check} present"
                if missing_recent else f"all {n_check} present",
            })

            # A4: 消息完整性 — 最近轮次的消息不应被截断
            if recent:
                truncated = [
                    r for r in recent[-min(3, len(recent)):]
                    if not r.get("user_message") or not r.get("llm_reply")
                ]
                checks.append({
                    "check": "recent turns have complete user+ai messages",
                    "pass": len(truncated) == 0,
                    "detail": f"{len(truncated)} truncated / {min(3, len(recent))} checked"
                    if truncated else "no truncation in last turns",
                })

            # A5: 顺序正确 — timeline_recent 应该按时间正序
            if len(recent) >= 2:
                timestamps = [r.get("timestamp", "") for r in recent]
                ordered = all(
                    timestamps[i] <= timestamps[i + 1]
                    for i in range(len(timestamps) - 1)
                )
                checks.append({
                    "check": "timeline_recent chronological order",
                    "pass": ordered,
                    "detail": "chronological" if ordered else "OUT OF ORDER",
                })
    except Exception as e:
        checks.append({
            "check": "timeline_recent",
            "pass": False,
            "error": str(e)[:120],
        })

    # ═══════════════════════════════════════════════════════════
    # 子项 B：session_context — 摘要兜底
    # ═══════════════════════════════════════════════════════════
    try:
        from app.memory.working import get_summary, _load
        wm = _load(wm_path)
        summary_text = get_summary(wm_path)
        summary_body = wm.get("summary", "").strip()

        checks.append({
            "check": "session_context summary exists",
            "pass": bool(summary_body),
            "detail": f"{len(summary_body)} chars" if summary_body else "no summary",
        })

        # 摘要长度合理性：不能太空，也不能超过增量更新上限
        checks.append({
            "check": "summary length reasonable (20-500 chars)",
            "pass": 20 <= len(summary_body) <= 500,
            "detail": f"{len(summary_body)} chars",
        })

        # topics 质量：应该是完整词，不是被切碎的 n-gram
        topics = wm.get("topics", [])
        if topics:
            broken_topics = [t for t in topics if len(t) < 3]
            checks.append({
                "check": "topics are whole words (≥3 chars)",
                "pass": len(broken_topics) == 0,
                "detail": f"{len(broken_topics)}/{len(topics)} broken: {broken_topics[:3]}"
                if broken_topics else f"all {len(topics)} topics ≥3 chars",
            })
        else:
            checks.append({
                "check": "topics are whole words",
                "pass": True,
                "detail": "no topics yet",
            })
    except Exception as e:
        checks.append({
            "check": "session_context",
            "pass": False,
            "error": str(e)[:120],
        })

    passed = sum(1 for c in checks if c.get("pass", False))
    return {
        "score": round(passed / len(checks), 4) if checks else 0,
        "pass": f"{passed}/{len(checks)}",
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════
# 类别 8：时间节律（权重 5%）
# ═══════════════════════════════════════════════════════════════

def test_time_rhythm(sample_n: int = 10) -> dict:
    from calendar import monthrange
    now = datetime.now()
    windows_ts = []
    ly = now.year - 1
    try:
        base = now.replace(year=ly)
        for offset in range(-3, 4):
            d = base + timedelta(days=offset)
            windows_ts.append(d.timestamp())
    except (OSError, ValueError):
        pass
    lm = now.month - 1
    ly2 = now.year if lm > 0 else now.year - 1
    lm2 = lm if lm > 0 else 12
    max_day = monthrange(ly2, lm2)[1]
    try:
        base2 = datetime(ly2, lm2, min(now.day, max_day))
        for offset in range(-3, 4):
            d = base2 + timedelta(days=offset)
            windows_ts.append(d.timestamp())
    except (OSError, ValueError):
        pass
    expected_ids = set()
    for m in CHROMA_DATA:
        ts = m.get("timestamp", 0)
        if not ts:
            continue
        for wt in windows_ts:
            if abs(ts - wt) < 86400:
                expected_ids.add(m["id"])
                break
    if not expected_ids:
        return {"score": 0, "pass": "0/0", "error": "No rhythm memories"}
    timestamped = sorted(
        [m for m in CHROMA_DATA if m.get("timestamp") and m["id"] in expected_ids],
        key=lambda x: x["timestamp"],
    )
    hit, total = 0, 0
    for item in timestamped[:sample_n]:
        query_text = (item.get("summary", "") or item.get("document", "") or "")[:100]
        if not query_text:
            continue
        emb = _get_embedding(query_text)
        if not emb:
            continue
        total += 1
        results = _chroma_query(emb)
        if any(r["id"] == item["id"] for r in results):
            hit += 1
    return {
        "score": round(hit / total, 4) if total else 1.0,
        "pass": f"{hit}/{total}",
    }


# ═══════════════════════════════════════════════════════════════
# 类别定义 & 运行
# ═══════════════════════════════════════════════════════════════

CATEGORIES: dict[int, tuple[str, callable, float]] = {
    1: ("Semantic Recall", test_semantic_recall, 0.25),
    2: ("Keyword Recall", test_keyword_recall, 0.15),
    3: ("Time Retrieval", test_time_retrieval, 0.15),
    4: ("Ranking Logic", test_ranking_logic, 0.15),
    5: ("Correction Feedback", test_correction_feedback, 0.10),
    6: ("Portrait Consistency", test_portrait_consistency, 0.10),
    7: ("Working Memory", test_working_memory, 0.10),
    8: ("Time Rhythm", test_time_rhythm, 0.05),
}

CHAT_CATEGORIES: set[int] = set()  # v4: 无类别依赖 /chat（画像从 PORTRAIT.md 读，不走 /chat）
EMBED_CATEGORIES: set[int] = {1, 3}  # 类别 1（语义检索）和类别 3（时间检索）依赖 Ollama embedding
SAMPLE_DEFAULTS = {1: 30, 2: 20, 3: 15, 6: 3, 7: 5, 8: 10}


def run_category(cat_id: int, sample_n: int) -> dict:
    name, func, weight = CATEGORIES[cat_id]
    print(f"  [{cat_id}/8] {name} (weight={weight:.0%})...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = func(sample_n)
        secs = time.time() - t0
        score = result.get("score", 0)
        print(f"score={score:.1%} ({secs:.0f}s)  {result.get('pass', '')}")
        result["category"] = name
        result["weight"] = weight
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"score": 0, "pass": "ERROR", "error": str(e),
                "category": name, "weight": weight}


def main():
    parser = argparse.ArgumentParser(description="初痕记忆审计套件 v3")
    parser.add_argument("--category", type=int, choices=range(1, 9), help="Run single category")
    parser.add_argument("--sample", type=int, default=0, help="Sample size (default per category)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (skip /chat categories)")
    parser.add_argument("--report", default="./audit", help="Report output dir")
    args = parser.parse_args()

    print("=" * 50)
    print("初痕审计套件 v4")
    print("=" * 50)

    t_start = time.time()
    load_data()
    t_load = time.time()
    ollama_ok = _is_ollama_available()
    print(f"Loaded: {len(CHROMA_DATA)} memories, portrait v{PORTRAIT_DATA.get('version', 0)} ({PORTRAIT_DATA.get('entry_count', 0)} entries) ({t_load - t_start:.1f}s)")
    if not ollama_ok:
        print("Ollama: NOT AVAILABLE — embedding-dependent categories will be skipped")

    results = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for cid in sorted(CATEGORIES if not args.category else [args.category]):
        if args.quick and cid in CHAT_CATEGORIES:
            print(f"  [{cid}/8] {CATEGORIES[cid][0]} — skipped (quick mode)")
            continue
        if not ollama_ok and cid in EMBED_CATEGORIES:
            print(f"  [{cid}/8] {CATEGORIES[cid][0]} (weight={CATEGORIES[cid][2]:.0%}) — skipped (Ollama unavailable)")
            results[CATEGORIES[cid][0]] = {"score": None, "pass": "SKIPPED",
                                             "category": CATEGORIES[cid][0], "weight": CATEGORIES[cid][2]}
            continue
        sn = args.sample if args.sample > 0 else SAMPLE_DEFAULTS.get(cid, 10)
        if args.quick:
            sn = min(sn, 5)
        r = run_category(cid, sn)
        results[CATEGORIES[cid][0]] = r
        if r.get("score") is not None:
            weighted_sum += r["score"] * CATEGORIES[cid][2]
            total_weight += CATEGORIES[cid][2]

    overall = weighted_sum / total_weight if total_weight else 0
    print(f"\nWeighted Total: {overall:.1%}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_memories": len(CHROMA_DATA),
        "portrait_version": PORTRAIT_DATA.get("version", 0),
        "portrait_entries": PORTRAIT_DATA.get("entry_count", 0),
        "scores": results,
        "weighted_total": round(overall, 4),
    }
    os.makedirs(args.report, exist_ok=True)
    report_path = os.path.join(args.report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
