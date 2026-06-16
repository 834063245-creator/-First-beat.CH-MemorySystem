---
name: phase3-sqlite-migration-complete
description: Phase 3 complete — SQLite (cooccur/entity_pair/hyperedge) migrated to Qdrant collections
metadata:
  type: project
---

Phase 3 已完成：三个 SQLite 模块（cooccur.py 292行 / entity_pair.py 236行 / hyperedge.py 461行）已迁移到 Qdrant。

**新增代码**（在 `app/memory/qdrant.py`）：
- `CoOccurrenceStore` (~280行) — 独立 Qdrant collection `co_occurrence`，完整 API 兼容旧 CoOccurrenceTracker
- `HyperEdgeStore` (~280行) — 独立 Qdrant collection `hyper_edges`，8 个公开方法
- `QdrantService.update_entity_co_counts()` / `get_entity_co_counts()` — payload 中预计算实体共现

**已删除**：cooccur.py, entity_pair.py, hyperedge.py
**保留**：db.py 兼容桩（close_all no-op），Phase 5 删除

**测试**：1063 passed, 8 failed (全部预存 flaky/环境依赖)

**Why:** Phase 3 是 SPEC_MIGRATION.md 规定的第 3 步——将 SQLite 元数据迁移到 Qdrant，消除 ChromaDB+SQLite 双写冗余，为百万级规模做准备。

**How to apply:** 新代码中的共现/超边/实体对逻辑全部通过 `app/memory/qdrant.py` 中的 `CoOccurrenceStore` / `HyperEdgeStore` / `QdrantService` 访问。不再使用 SQLite。context.py 中 `ctx.co_tracker` / `ctx.ai_co_tracker` / `ctx.hyperedge_index` 属性保持不变，透明接入新 stores。[[phase2-kill-list-all]] [[spec-migration]]
