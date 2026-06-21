#!/usr/bin/env python3
# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
"""
Qdrant → AuraSDK 迁移脚本

将现有 Qdrant 记忆批量写入 AuraSDK 事实召回引擎。
- 幂等：重复运行只追加新记忆，不会重复（AuraSDK 内部去重）
- 增量：跳过已存在的 AuraSDK 数据目录中的记录
- Level 映射：所有记忆默认 → Level.Domain

用法:
  python scripts/migrate_to_aura.py                     # 全量迁移
  python scripts/migrate_to_aura.py --dry-run            # 只统计不写入
  python scripts/migrate_to_aura.py --limit 500          # 限制数量
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_aura")


def main():
    parser = argparse.ArgumentParser(description="Qdrant → AuraSDK 迁移")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    parser.add_argument("--limit", type=int, default=0, help="限制迁移条数 (0=全量)")
    parser.add_argument("--qdrant-path", type=str, default="",
                        help="Qdrant 数据路径 (默认 data/qdrant)")
    parser.add_argument("--aura-path", type=str, default="",
                        help="AuraSDK 数据路径 (默认 data/aura)")
    parser.add_argument("--collection", type=str, default="memories",
                        help="Qdrant collection 名")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="每批写入后打印进度")
    args = parser.parse_args()

    # 路径
    data_dir = _PROJ_ROOT / "data"
    qdrant_path = args.qdrant_path or os.getenv("QDRANT_URL", "") or str(data_dir / "qdrant")
    aura_path = args.aura_path or str(data_dir / "aura")

    # AuraSDK init
    from app.memory.aurasdk import Aura, Level
    os.makedirs(aura_path, exist_ok=True)
    aura = Aura(aura_path)

    # Qdrant connect
    client = QdrantClient(path=qdrant_path)
    coll_name = args.collection

    # 统计现有数据
    try:
        info = client.get_collection(coll_name)
        total = info.points_count
        logger.info("Qdrant collection '%s': %d records", coll_name, total)
    except Exception:
        logger.warning("Qdrant collection '%s' 不存在或为空", coll_name)
        total = 0

    if total == 0:
        logger.info("没有需要迁移的数据。")
        aura.close()
        return

    # AuraSDK 现有统计
    aura_stats = aura.stats()
    logger.info("AuraSDK 现有: %d records", aura_stats.get("total_records", 0))

    # ── 滚动读取 Qdrant ──
    migrated = 0
    skipped = 0
    errors = 0
    batch_count = 0
    t_start = time.time()

    offset = None
    page = 0

    while True:
        pts, next_offset = client.scroll(
            collection_name=coll_name,
            with_vectors=False,
            with_payload=True,
            limit=500,
            offset=offset,
        )
        if not pts:
            break
        page += 1
        logger.info("Page %d: scanning %d points...", page, len(pts))

        for pt in pts:
            payload = pt.payload or {}
            doc = payload.get("document", "") or payload.get("text", "") or ""
            if not doc or not doc.strip():
                skipped += 1
                continue

            tags = payload.get("tags", []) or []
            # 确保 tags 是字符串列表
            tags = [str(t) for t in tags if t][:8]

            if args.dry_run:
                migrated += 1
                batch_count += 1
                if batch_count % args.batch_size == 0:
                    logger.info("  [dry-run] scanned %d...", migrated)
                continue

            try:
                aura.store(doc, level=Level.Domain, tags=tags)
                migrated += 1
                batch_count += 1
            except Exception as e:
                logger.debug("  store failed: %s", e)
                errors += 1
                if errors <= 3:
                    logger.warning("  第 %d 条写入失败: %s", errors, str(e)[:100])

            if batch_count % args.batch_size == 0:
                elapsed = time.time() - t_start
                rate = migrated / elapsed if elapsed > 0 else 0
                logger.info(
                    "  ... %d migrated, %d skipped, %d errors (%.0f/s)",
                    migrated, skipped, errors, rate,
                )

            if args.limit > 0 and migrated >= args.limit:
                break

        if args.limit > 0 and migrated >= args.limit:
            logger.info("达到 limit=%d，停止。", args.limit)
            break

        if next_offset is None or len(pts) < 500:
            break
        offset = next_offset

    elapsed = time.time() - t_start
    aura_stats = aura.stats()
    logger.info("=" * 50)
    logger.info("迁移完成 (%.1fs)", elapsed)
    logger.info("  Qdrant 扫描: %d 条", total)
    logger.info("  AuraSDK 写入: %d 条", migrated)
    logger.info("  跳过 (空内容): %d 条", skipped)
    logger.info("  错误: %d 条", errors)
    logger.info("  AuraSDK 总数: %d 条", aura_stats.get("total_records", 0))
    if migrated > 0 and not args.dry_run:
        logger.info("  速率: %.1f 条/s", migrated / elapsed)

    aura.close()
    client.close()


if __name__ == "__main__":
    main()
