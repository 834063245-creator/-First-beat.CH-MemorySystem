#!/usr/bin/env python
"""Phase 0 数据迁移脚本：ChromaDB + SQLite → Qdrant。

功能:
  - 分批读取 ChromaDB (500条/批)，避免全量加载
  - 格式转换: ChromaDB metadata → Qdrant payload (原生类型)
  - 从旧 SQLite 聚合 entity_co_counts 到 payload
  - 迁移 CoOccurrence / HyperEdge 数据到独立 Qdrant collection
  - 进度条 + 吞吐量统计 (tqdm)
  - 迁移后校验: count() 对比 + 抽样 100 条逐字段对比
  - 断点续传: data/migration_checkpoint.json
  - 干跑模式: --dry-run 只校验不写入

用法:
  python scripts/migrate_to_qdrant.py                        # 全量迁移
  python scripts/migrate_to_qdrant.py --dry-run              # 干跑：只校验
  python scripts/migrate_to_qdrant.py --source user           # 仅迁移 user memories
  python scripts/migrate_to_qdrant.py --source ai             # 仅迁移 ai_memories
  python scripts/migrate_to_qdrant.py --source cooccur        # 仅迁移 co_occurrence
  python scripts/migrate_to_qdrant.py --source hyperedge      # 仅迁移 hyper_edges
  python scripts/migrate_to_qdrant.py --batch-size 200        # 自定义批次大小
  python scripts/migrate_to_qdrant.py --resume                # 从断点续传
  python scripts/migrate_to_qdrant.py --no-verify             # 跳过校验
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate")

# ============================================================
# 配置
# ============================================================
DEFAULT_BATCH_SIZE = 500
CHECKPOINT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "migration_checkpoint.json",
)
VERIFY_SAMPLE_SIZE = 100

# 尝试导入依赖
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    logger.warning("chromadb 未安装，无法读取源数据")

try:
    from qdrant_client import QdrantClient, models
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    logger.warning("qdrant-client 未安装，运行: pip install qdrant-client")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm 未安装，进度条不可用。运行: pip install tqdm")

# 环境变量
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
))

# Qdrant collection naming
USER_PREFIX = os.getenv("MIGRATE_USER_PREFIX", "")  # e.g., "admin_" for multi-user


# ============================================================
# Helpers
# ============================================================
def load_checkpoint():
    """加载断点文件"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(cp: dict):
    """保存断点文件（atomic write via temp + rename）"""
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CHECKPOINT_FILE)


def progress(iterable, desc: str, total: int = None, unit: str = "条"):
    """条件 tqdm 包装"""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, unit=unit)
    return iterable


def format_throughput(count: int, elapsed: float) -> str:
    if elapsed > 0:
        rate = count / elapsed
        if rate > 1000:
            return f"{rate/1000:.1f}k/s"
        return f"{rate:.0f}/s"
    return "N/A"


# ============================================================
# 格式转换
# ============================================================
def _compute_year_month_quarter(ts: float) -> dict:
    """从 Unix 时间戳预计算时间特征"""
    import datetime
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    month = dt.month
    quarter = (month - 1) // 3 + 1
    season_map = {1: "winter", 2: "spring", 3: "spring",
                  4: "spring", 5: "summer", 6: "summer",
                  7: "summer", 8: "autumn", 9: "autumn",
                  10: "autumn", 11: "winter", 12: "winter"}
    return {
        "year": dt.year,
        "month": month,
        "day": dt.day,
        "week": dt.isocalendar()[1],
        "day_of_week": dt.weekday(),  # 0=Monday
        "quarter": quarter,
        "season": season_map[month],
        "year_month": f"{dt.year}-{month:02d}",
    }


def convert_metadata_to_payload(meta: dict) -> dict:
    """ChromaDB metadata → Qdrant payload.

    关键类型转换:
      - entities: JSON string → native list[dict]
      - 预计算时间特征 (year/month/day/week/day_of_week/quarter/season/year_month)
      - 其他字段保持原类型
    """
    payload = {}

    # ── 直接复制的字段 ──
    direct_fields = [
        "user_message", "ai_message", "document", "summary",
        "tags", "timestamp", "hit_count", "last_hit_time",
        "heat", "embed_model", "stale", "archived",
        "superseded_by", "supersede_reason", "superseded_at",
        "storage_complete", "source", "date_tag",
        "emotion_valence", "emotion_arousal",
        "emotion_valence_bin", "emotional_intensity",
    ]
    for f in direct_fields:
        if f in meta:
            payload[f] = meta[f]

    # ── entities: JSON string → native list[dict] ──
    if "entities" in meta:
        raw = meta["entities"]
        if isinstance(raw, str):
            try:
                payload["entities"] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                payload["entities"] = []
        elif isinstance(raw, list):
            payload["entities"] = raw
        else:
            payload["entities"] = []

    # ── 预计算时间特征 ──
    ts = meta.get("timestamp", 0)
    if isinstance(ts, (int, float)) and ts > 0:
        time_features = _compute_year_month_quarter(float(ts))
        payload.update(time_features)

    # ── 默认值补全 ──
    if "stale" not in payload:
        payload["stale"] = False
    if "archived" not in payload:
        payload["archived"] = False
    if "storage_complete" not in payload:
        payload["storage_complete"] = True
    if "source" not in payload:
        payload["source"] = "user"
    if "hit_count" not in payload:
        payload["hit_count"] = 0
    if "heat" not in payload or not payload["heat"]:
        payload["heat"] = "cool"

    return payload


# ============================================================
# Entity co_counts 聚合 (从旧 SQLite entity_pairs.db)
# ============================================================
def load_entity_co_counts_from_sqlite(entity_pairs_db: str,
                                      memory_ids: list[str]) -> dict[str, dict[str, int]]:
    """从旧 SQLite entity_pairs 表为指定 memory_ids 聚合 entity_co_counts。

    返回: {memory_id: {entity_name: count, ...}, ...}
    """
    result: dict[str, dict[str, int]] = {}
    if not os.path.exists(entity_pairs_db):
        return result

    try:
        conn = sqlite3.connect(entity_pairs_db)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join(["?"] * len(memory_ids))

        # 尝试读取 entity_pair 表
        cursor = conn.execute(
            f"SELECT memory_id, entity_name, count FROM entity_pair "
            f"WHERE memory_id IN ({placeholders})",
            memory_ids,
        )
        for row in cursor.fetchall():
            mid = row["memory_id"]
            ename = row["entity_name"]
            cnt = row["count"]
            result.setdefault(mid, {})[ename] = cnt

        conn.close()
    except Exception as e:
        logger.warning("读取 entity_pairs.db 失败: %s", e)

    return result


# ============================================================
# Source readers
# ============================================================
def read_chromadb_source(chroma_path: str, collection_name: str,
                         batch_size: int = DEFAULT_BATCH_SIZE):
    """生成器: 分批 yield ChromaDB documents。

    Yields: (batch_index, list[{id, embedding, metadata, document}])
    """
    if not os.path.exists(chroma_path):
        logger.error("ChromaDB 路径不存在: %s", chroma_path)
        return

    client = chromadb.PersistentClient(
        path=os.path.dirname(chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        logger.error("获取 ChromaDB collection '%s' 失败: %s", collection_name, e)
        return

    total = collection.count()
    logger.info("ChromaDB collection '%s': %d 条记录", collection_name, total)

    offset = 0
    batch_idx = 0
    while offset < total:
        limit = min(batch_size, total - offset)
        results = collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        embs = results.get("embeddings", [])

        batch = []
        for i, mid in enumerate(ids):
            batch.append({
                "id": mid,
                "embedding": embs[i] if embs and i < len(embs) else None,
                "metadata": metas[i] if metas and i < len(metas) else {},
                "document": docs[i] if docs and i < len(docs) else "",
            })

        yield batch_idx, batch
        batch_idx += 1
        offset += limit


def read_sqlite_cooccur(cooccur_db: str, batch_size: int = 2000):
    """生成器: 分批 yield SQLite cooccurrence 行。

    Yields: (batch_index, list[{id_a, id_b, count, last_time}])
    """
    if not os.path.exists(cooccur_db):
        logger.warning("co_occurrence.db 不存在: %s", cooccur_db)
        return

    conn = sqlite3.connect(cooccur_db)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM cooccurrence").fetchone()[0]
    logger.info("SQLite cooccurrence: %d 条记录", total)

    offset = 0
    batch_idx = 0
    while offset < total:
        rows = conn.execute(
            "SELECT id_a, id_b, count, last_time FROM cooccurrence "
            "ORDER BY count DESC LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()
        if not rows:
            break
        yield batch_idx, [dict(r) for r in rows]
        batch_idx += 1
        offset += batch_size

    conn.close()


def read_sqlite_hyperedges(hyperedge_db: str, batch_size: int = 1000):
    """生成器: 分批 yield SQLite hyper_edges 行。

    Yields: (batch_index, list[{edge_id, entities, memory_ids, created_at, edge_size}])
    """
    if not os.path.exists(hyperedge_db):
        logger.warning("hyper_edges.db 不存在: %s", hyperedge_db)
        return

    conn = sqlite3.connect(hyperedge_db)
    conn.row_factory = sqlite3.Row

    # 尝试读取 hyper_edge 表
    total = 0
    try:
        total = conn.execute("SELECT COUNT(*) FROM hyper_edge").fetchone()[0]
    except Exception:
        logger.warning("hyper_edge 表不存在，跳过")
        conn.close()
        return

    logger.info("SQLite hyper_edges: %d 条记录", total)

    offset = 0
    batch_idx = 0
    while offset < total:
        rows = conn.execute(
            "SELECT id, entities, memory_ids, created_at FROM hyper_edge "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()
        if not rows:
            break

        batch = []
        for r in rows:
            entities_raw = r["entities"]
            memory_ids_raw = r["memory_ids"]
            batch.append({
                "edge_id": r["id"],
                "entities": json.loads(entities_raw) if isinstance(entities_raw, str) else (entities_raw or []),
                "memory_ids": json.loads(memory_ids_raw) if isinstance(memory_ids_raw, str) else (memory_ids_raw or []),
                "created_at": r["created_at"] or "",
                "edge_size": len(json.loads(entities_raw) if isinstance(entities_raw, str) else (entities_raw or [])),
            })
        yield batch_idx, batch
        batch_idx += 1
        offset += batch_size

    conn.close()


# ============================================================
# Qdrant writers
# ============================================================
def ensure_qdrant_collections(client: QdrantClient, prefix: str = "",
                              recreate: bool = False):
    """确保 Qdrant collections 存在，不存在则创建。

    Collections:
      - {prefix}memories: 主记忆库
      - {prefix}ai_memories: AI 自我记忆
      - {prefix}co_occurrence: 共现对
      - {prefix}hyper_edges: 超边
    """
    vector_config = models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
    )

    collections_spec = {
        f"{prefix}memories": "主记忆库",
        f"{prefix}ai_memories": "AI 自我记忆",
        f"{prefix}co_occurrence": "共现对",
        f"{prefix}hyper_edges": "超边",
    }

    existing = {c.name for c in client.get_collections().collections}

    for coll_name, desc in collections_spec.items():
        if coll_name in existing:
            if recreate:
                client.delete_collection(coll_name)
                logger.info("删除现有 collection: %s", coll_name)
            else:
                logger.info("Collection 已存在: %s (%s)", coll_name, desc)
                continue

        client.create_collection(
            collection_name=coll_name,
            vectors_config=vector_config,
        )
        logger.info("创建 collection: %s (%s)", coll_name, desc)


def migrate_memories(client: QdrantClient, chroma_path: str, collection_name: str,
                     qdrant_coll: str, entity_pairs_db: str,
                     batch_size: int, dry_run: bool, checkpoint: dict) -> int:
    """迁移 ChromaDB memories → Qdrant。"""
    key = f"chroma_{qdrant_coll}"
    start_batch = checkpoint.get(key, 0)

    total_migrated = 0
    t_start = time.time()

    # 预加载 entity_co_counts (仅非 dry-run)
    all_ids = []
    entity_co_map: dict[str, dict[str, int]] = {}
    if not dry_run and entity_pairs_db and os.path.exists(entity_pairs_db):
        logger.info("预加载 entity_co_counts 映射...")
        # 先从 ChromaDB 收集所有 IDs
        for _, batch in read_chromadb_source(chroma_path, collection_name, batch_size):
            all_ids.extend([b["id"] for b in batch])
        entity_co_map = load_entity_co_counts_from_sqlite(entity_pairs_db, all_ids)
        logger.info("entity_co_counts 映射: %d 条记忆有实体共现数据", len(entity_co_map))

    for batch_idx, batch in read_chromadb_source(chroma_path, collection_name, batch_size):
        if batch_idx < start_batch:
            logger.info("跳过 batch %d (已迁移)", batch_idx)
            continue

        points = []
        for item in batch:
            payload = convert_metadata_to_payload(item["metadata"])

            # 注入 entity_co_counts
            mid = item["id"]
            if mid in entity_co_map:
                payload["entity_co_counts"] = entity_co_map[mid]

            embedding = item["embedding"]
            if embedding is None:
                logger.warning("记忆 %s 缺少 embedding，跳过", mid)
                continue

            points.append(models.PointStruct(
                id=mid,
                vector=embedding,
                payload=payload,
            ))

        if not dry_run and points:
            client.upsert(collection_name=qdrant_coll, points=points)

        total_migrated += len(points)
        checkpoint[key] = batch_idx + 1

        # 每 2 批保存断点
        if batch_idx % 2 == 0:
            save_checkpoint(checkpoint)

        elapsed = time.time() - t_start
        rate = format_throughput(total_migrated, elapsed)
        logger.info("Batch %d: %d 条 | 累计 %d | %s",
                     batch_idx, len(points), total_migrated, rate)

    save_checkpoint(checkpoint)
    return total_migrated


def migrate_cooccurrence(client: QdrantClient, cooccur_db: str,
                         qdrant_coll: str, batch_size: int,
                         dry_run: bool, checkpoint: dict) -> int:
    """迁移 SQLite cooccurrence → Qdrant co_occurrence collection。"""
    key = f"cooccur_{qdrant_coll}"
    start_batch = checkpoint.get(key, 0)

    total_migrated = 0
    t_start = time.time()

    for batch_idx, batch in read_sqlite_cooccur(cooccur_db, batch_size):
        if batch_idx < start_batch:
            logger.info("跳过 batch %d (已迁移)", batch_idx)
            continue

        points = []
        for row in batch:
            point_id = f"{row['id_a']}||{row['id_b']}"
            points.append(models.PointStruct(
                id=point_id,
                vector=[0.0] * 1024,  # 占位向量 (cooccurrence 核心查询走 scroll)
                payload={
                    "id_a": row["id_a"],
                    "id_b": row["id_b"],
                    "count": row["count"],
                    "last_time": row.get("last_time", 0),
                },
            ))

        if not dry_run and points:
            client.upsert(collection_name=qdrant_coll, points=points)

        total_migrated += len(points)
        checkpoint[key] = batch_idx + 1

        if batch_idx % 5 == 0:
            save_checkpoint(checkpoint)

        elapsed = time.time() - t_start
        rate = format_throughput(total_migrated, elapsed)
        logger.info("CoOccur Batch %d: %d 条 | 累计 %d | %s",
                     batch_idx, len(points), total_migrated, rate)

    save_checkpoint(checkpoint)
    return total_migrated


def migrate_hyperedges(client: QdrantClient, hyperedge_db: str,
                       qdrant_coll: str, batch_size: int,
                       dry_run: bool, checkpoint: dict) -> int:
    """迁移 SQLite hyper_edges → Qdrant hyper_edges collection。"""
    key = f"hyper_{qdrant_coll}"
    start_batch = checkpoint.get(key, 0)

    total_migrated = 0
    t_start = time.time()

    for batch_idx, batch in read_sqlite_hyperedges(hyperedge_db, batch_size):
        if batch_idx < start_batch:
            logger.info("跳过 batch %d (已迁移)", batch_idx)
            continue

        points = []
        for row in batch:
            entities = row["entities"]
            memory_ids = row["memory_ids"]

            if not entities or len(entities) < 2:
                continue

            point_id = row.get("edge_id") or str(uuid.uuid4())
            points.append(models.PointStruct(
                id=point_id,
                vector=[0.0] * 1024,  # 占位向量
                payload={
                    "entities": entities,            # native list[str]
                    "memory_ids": memory_ids,        # native list[str]
                    "created_at": row.get("created_at", ""),
                    "edge_size": len(entities),
                },
            ))

        if not dry_run and points:
            client.upsert(collection_name=qdrant_coll, points=points)

        total_migrated += len(points)
        checkpoint[key] = batch_idx + 1

        if batch_idx % 5 == 0:
            save_checkpoint(checkpoint)

        elapsed = time.time() - t_start
        rate = format_throughput(total_migrated, elapsed)
        logger.info("HyperEdge Batch %d: %d 条 | 累计 %d | %s",
                     batch_idx, len(points), total_migrated, rate)

    save_checkpoint(checkpoint)
    return total_migrated


# ============================================================
# Verification
# ============================================================
def verify_migration(client: QdrantClient, chroma_path: str,
                     collection_name: str, qdrant_coll: str,
                     verify_sample: int = VERIFY_SAMPLE_SIZE) -> bool:
    """对比 ChromaDB 和 Qdrant 的数据一致性。

    验证项:
      1. count() 对比
      2. 抽样 N 条逐字段对比 (id / document / embedding)
    """
    logger.info("=" * 50)
    logger.info("开始校验: %s → %s", collection_name, qdrant_coll)

    all_ok = True

    # ── 1. count() 对比 ──
    try:
        chroma_client = chromadb.PersistentClient(
            path=os.path.dirname(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        chroma_coll = chroma_client.get_collection(collection_name)
        chroma_count = chroma_coll.count()
    except Exception as e:
        logger.error("读取 ChromaDB count 失败: %s", e)
        return False

    try:
        qdrant_count = client.count(collection_name=qdrant_coll).count
    except Exception as e:
        logger.error("读取 Qdrant count 失败: %s", e)
        return False

    if chroma_count == qdrant_count:
        logger.info("✅ count 一致: %d", chroma_count)
    else:
        logger.warning("⚠️  count 不一致: ChromaDB=%d, Qdrant=%d (差异: %d)",
                       chroma_count, qdrant_count, abs(chroma_count - qdrant_count))
        all_ok = False

    # ── 2. 抽样校验 ──
    sample_size = min(verify_sample, chroma_count)
    if sample_size == 0:
        return all_ok

    # 从 ChromaDB 随机抽样
    import random
    sample_offset = random.randint(0, max(0, chroma_count - sample_size))
    chroma_sample = chroma_coll.get(
        limit=sample_size,
        offset=sample_offset,
        include=["documents", "embeddings"],
    )

    sample_ids = chroma_sample.get("ids", [])
    qdrant_points = client.retrieve(
        collection_name=qdrant_coll,
        ids=sample_ids,
        with_payload=True,
        with_vectors=True,
    )
    qdrant_map = {p.id: p for p in qdrant_points}

    doc_mismatches = 0
    emb_mismatches = 0
    missing = 0

    for i, mid in enumerate(sample_ids):
        qpt = qdrant_map.get(mid)
        if qpt is None:
            missing += 1
            logger.warning("抽样缺失: %s", mid)
            continue

        # 对比 document
        chroma_doc = chroma_sample["documents"][i] if i < len(chroma_sample["documents"]) else ""
        qdrant_doc = (qpt.payload or {}).get("document", "")
        if chroma_doc != qdrant_doc:
            doc_mismatches += 1

        # 对比 embedding (余弦相似度)
        chroma_emb = chroma_sample["embeddings"][i] if i < len(chroma_sample.get("embeddings", [])) else None
        qdrant_emb = qpt.vector
        if chroma_emb is not None and qdrant_emb is not None:
            try:
                dot = sum(a * b for a, b in zip(chroma_emb, qdrant_emb))
                norm_a = (sum(a * a for a in chroma_emb) ** 0.5)
                norm_b = (sum(b * b for b in qdrant_emb) ** 0.5)
                cos_sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 1.0
                if cos_sim < 0.999:  # float32 precision tolerance
                    emb_mismatches += 1
            except Exception:
                emb_mismatches += 1

    if missing == 0 and doc_mismatches == 0 and emb_mismatches == 0:
        logger.info("✅ 抽样 %d 条全部一致 (id/document/embedding)", sample_size)
    else:
        logger.warning("⚠️  抽样 %d 条问题: 缺失=%d, document不一致=%d, embedding不一致=%d",
                       sample_size, missing, doc_mismatches, emb_mismatches)
        all_ok = False

    return all_ok


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 数据迁移: ChromaDB + SQLite → Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑模式：只校验不写入 Qdrant")
    parser.add_argument("--source", choices=["user", "ai", "cooccur", "hyperedge", "all"],
                        default="all", help="迁移目标 (default: all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"每批条数 (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--resume", action="store_true",
                        help="从断点续传")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过迁移后校验")
    parser.add_argument("--user-prefix", type=str, default=USER_PREFIX,
                        help=f"Qdrant collection 前缀 (default: '{USER_PREFIX}')")

    args = parser.parse_args()

    if not HAS_CHROMA:
        logger.error("chromadb 未安装。运行: pip install chromadb")
        return 1
    if not HAS_QDRANT:
        logger.error("qdrant-client 未安装。运行: pip install qdrant-client")
        return 1

    # ── 连接 Qdrant ──
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        client.health()
        logger.info("✅ Qdrant 连接成功: %s", QDRANT_URL)
    except Exception as e:
        logger.error("❌ Qdrant 连接失败: %s", e)
        return 1

    # ── 断点管理 ──
    if args.resume:
        checkpoint = load_checkpoint()
        logger.info("从断点续传: %s", json.dumps(checkpoint, indent=2))
    else:
        checkpoint = {}
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info("已清除旧断点文件")

    prefix = args.user_prefix

    # ── 确保 collections 存在 ──
    if not args.dry_run:
        ensure_qdrant_collections(client, prefix=prefix)

    # ── 路径 ──
    chroma_user_path = os.path.join(DATA_DIR, "chroma")
    chroma_ai_path = os.path.join(DATA_DIR, "ai_chroma")
    cooccur_db = os.path.join(DATA_DIR, "co_occurrence.db")
    entity_pairs_db = os.path.join(DATA_DIR, "entity_pairs.db")
    hyperedge_db = os.path.join(DATA_DIR, "hyper_edges.db")

    stats = {}
    t_total = time.time()

    # ── 迁移 user memories ──
    if args.source in ("user", "all"):
        if os.path.exists(chroma_user_path):
            t0 = time.time()
            logger.info("=" * 60)
            logger.info("迁移: user memories → %smemories", prefix)
            logger.info("=" * 60)
            count = migrate_memories(
                client=client,
                chroma_path=chroma_user_path,
                collection_name="memories",
                qdrant_coll=f"{prefix}memories",
                entity_pairs_db=entity_pairs_db,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                checkpoint=checkpoint,
            )
            stats["user_memories"] = count
            logger.info("完成: %d 条, 耗时 %.1fs", count, time.time() - t0)
        else:
            logger.warning("跳过 user memories: 路径不存在 %s", chroma_user_path)

    # ── 迁移 ai memories ──
    if args.source in ("ai", "all"):
        if os.path.exists(chroma_ai_path):
            t0 = time.time()
            logger.info("=" * 60)
            logger.info("迁移: ai_memories → %sai_memories", prefix)
            logger.info("=" * 60)
            count = migrate_memories(
                client=client,
                chroma_path=chroma_ai_path,
                collection_name="ai_memories",
                qdrant_coll=f"{prefix}ai_memories",
                entity_pairs_db="",  # AI memories don't have entity pairs
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                checkpoint=checkpoint,
            )
            stats["ai_memories"] = count
            logger.info("完成: %d 条, 耗时 %.1fs", count, time.time() - t0)
        else:
            logger.warning("跳过 ai_memories: 路径不存在 %s", chroma_ai_path)

    # ── 迁移 cooccurrence ──
    if args.source in ("cooccur", "all"):
        if os.path.exists(cooccur_db):
            t0 = time.time()
            logger.info("=" * 60)
            logger.info("迁移: cooccurrence → %sco_occurrence", prefix)
            logger.info("=" * 60)
            count = migrate_cooccurrence(
                client=client,
                cooccur_db=cooccur_db,
                qdrant_coll=f"{prefix}co_occurrence",
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                checkpoint=checkpoint,
            )
            stats["cooccurrence"] = count
            logger.info("完成: %d 条, 耗时 %.1fs", count, time.time() - t0)
        else:
            logger.warning("跳过 cooccurrence: 文件不存在 %s", cooccur_db)

    # ── 迁移 hyperedges ──
    if args.source in ("hyperedge", "all"):
        if os.path.exists(hyperedge_db):
            t0 = time.time()
            logger.info("=" * 60)
            logger.info("迁移: hyper_edges → %shyper_edges", prefix)
            logger.info("=" * 60)
            count = migrate_hyperedges(
                client=client,
                hyperedge_db=hyperedge_db,
                qdrant_coll=f"{prefix}hyper_edges",
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                checkpoint=checkpoint,
            )
            stats["hyperedges"] = count
            logger.info("完成: %d 条, 耗时 %.1fs", count, time.time() - t0)
        else:
            logger.warning("跳过 hyper_edges: 文件不存在 %s", hyperedge_db)

    # ── 汇总 ──
    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("迁移汇总 (总耗时 %.1fs):", total_elapsed)
    for key, count in stats.items():
        logger.info("  %s: %d 条", key, count)
    logger.info("  %s", "干跑模式，未实际写入" if args.dry_run else "已写入 Qdrant")

    # ── 校验 ──
    if not args.no_verify and not args.dry_run:
        logger.info("=" * 60)
        logger.info("开始数据一致性校验...")
        verify_ok = True

        if args.source in ("user", "all") and os.path.exists(chroma_user_path):
            verify_ok &= verify_migration(
                client, chroma_user_path, "memories", f"{prefix}memories"
            )
        if args.source in ("ai", "all") and os.path.exists(chroma_ai_path):
            verify_ok &= verify_migration(
                client, chroma_ai_path, "ai_memories", f"{prefix}ai_memories"
            )

        if verify_ok:
            logger.info("✅ 全部校验通过")
        else:
            logger.warning("⚠️  校验发现问题，请检查上述差异")
            return 1

    # ── 清理断点 ──
    if not args.resume:
        # 迁移成功后清理断点
        verify_fully_ok = all(
            v == (chroma_coll.count() if not args.dry_run else v)
            for v in stats.values()
        )
        if verify_fully_ok and not args.dry_run:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                logger.info("迁移成功，已清理断点文件")

    return 0


if __name__ == "__main__":
    sys.exit(main())
