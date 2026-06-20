#!/usr/bin/env python3
# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
"""
M@q 记忆场原型 — 验证 M = VᵀV 残差注入方案的可行性。

核心逻辑:
  1. 从 Qdrant 拉取全量记忆向量 → V [N × 3584]
  2. 构建 Gram 矩阵 M = VᵀV [3584 × 3584] (~51MB float32)
  3. 查询 q → R = M@q [3584维] → 残差注入方向
  4. 验证: R 是否指向正确的语义领域

验证场景:
  A. 基础方向验证 — M@q 产出的 R 是否与查询领域一致
  B. 噪音鲁棒性 — 200 条同领域噪音中 R 是否仍然稳定
  C. 领域分离度 — 两个不同领域的 R 向量是否正交/可区分
  D. 与 Top-K 对比 — M@q 加权质心 vs 离散 Top-K 的差异

用法:
  python scripts/memory_field_prototype.py                    # 全部验证
  python scripts/memory_field_prototype.py --quick             # 仅基础验证
  python scripts/memory_field_prototype.py --scene noise       # 仅噪音鲁棒性
  python scripts/memory_field_prototype.py --rebuild-matrix    # 强制重建 M 矩阵
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Windows GBK 编码修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 确保项目根在 sys.path
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m@q")

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

QWEN_EMBED_DIM = 3584
DATA_DIR = _PROJ_ROOT / "data"
QDRANT_PATH = os.getenv("QDRANT_URL", "") or str(DATA_DIR / "qdrant")
COLLECTION_NAME = "memories"
M_MATRIX_PATH = DATA_DIR / "m_matrix_f32.npy"
V_IDS_PATH = DATA_DIR / "m_matrix_ids.json"

# M 矩阵缓存: float32 — 3584×3584×4 = ~51.4 MB
DTYPE = np.float32


# ═══════════════════════════════════════════════════════════════
# 0. 种子数据生成 (测试用)
# ═══════════════════════════════════════════════════════════════

# 两个不同领域的种子记忆
SEED_DOMAIN_A = "编程"  # 技术/编程
SEED_DOMAIN_B = "生活"  # 情绪/日常

SEED_MEMORIES = {
    "编程": [
        "Python 的列表推导式真的很优雅，一行代码就能替代 for 循环",
        "Rust 的 borrow checker 太难了，学了三个月还是经常编译不过",
        "今天用 Docker 部署了一个 FastAPI 服务，感觉容器化确实方便",
        "TypeScript 的类型系统比 JavaScript 好太多，重构时很安心",
        "写了一个异步爬虫，asyncio + aiohttp 并发 1000 个请求只用 3 秒",
        "数据库索引没建对，一条查询跑了 30 秒，加了复合索引后降到 0.1 秒",
        "VSCode 的 Copilot 插件有时候能猜中我想写的代码，挺神奇的",
        "学了设计模式之后看老代码发现到处都是策略模式和观察者模式",
        "Linux 命令行真的强大，grep + awk + sed 组合拳处理日志无敌",
        "Git rebase 把我的提交历史搞得一团糟，还是 merge 更安全",
        "微服务架构是双刃剑，拆太细运维成本爆炸，拆太少又没意义",
        "写单元测试很烦但确实有用，上周一个边界条件 bug 就是测试发现的",
        "redis 做缓存层真的很香，但要注意缓存穿透和雪崩的问题",
        "学完了 CSAPP 这本书，对计算机底层有了全新的理解",
        "k8s 的配置太复杂了，一个小小的 YAML 错误排查了一下午",
        "用 pytest 的参数化测试功能，一个测试函数覆盖了 20 个用例",
        "重构了一个 2000 行的函数，拆成 15 个小函数后清晰多了",
        "Python 的 GIL 限制了多线程性能，但多进程 + Queue 可以绕过",
        "学 Rust 最大的收获不是语言本身，是对内存管理的理解加深了",
        "用 Nginx 做反向代理，配 SSL 证书踩了好多坑",
        "前后端分离项目里 API 文档用 OpenAPI 自动生成很爽",
        "ELK 日志系统搭建起来费劲，但线上排查问题时是真方便",
        "代码 review 是提高团队代码质量最有效的方式之一",
        "用了三年的 Python 想换 Go，但公司的技术栈全是 Python",
        "MongoDB 的聚合管道比 SQL 的 GROUP BY 灵活太多了",
    ],
    "生活": [
        "今天天气很好，和朋友去公园散步聊了很多",
        "最近工作压力好大，晚上总是失眠到两三点",
        "养了一盆多肉植物，每天看着它慢慢长大很有成就感",
        "周末做了一顿红烧肉，按照网上的教程一步步来居然成功了",
        "和父母视频通话时他们总问我什么时候回家",
        "下雨天待在家里看书喝茶是最舒服的事情",
        "最近在减肥，已经坚持跑步两周了，瘦了 2 公斤",
        "买了一个新的机械键盘，打字的感觉比薄膜键盘好太多了",
        "昨天晚上做了一个很奇怪的梦，梦见自己在飞",
        "楼下新开了一家咖啡店，拿铁的味道还不错",
        "朋友推荐了一部电影《肖申克的救赎》，看完很震撼",
        "最近在学做饭，发现自己对烹饪还挺有兴趣的",
        "周末去爬山，山顶的风景真的很美",
        "家里的猫最近总是半夜跑酷，睡眠质量严重下降",
        "好久没联系的老同学突然发消息，聊了很多以前的趣事",
        "换了一个新的枕头，脖子不再酸痛了",
        "今天在地铁上看到一个老人给孕妇让座，挺感动的",
        "最近在学吉他，手指按弦按得生疼",
        "清理了一天的房间，扔掉了好多一直舍不得丢的东西",
        "感冒了在家躺了两天，突然觉得健康真的很重要",
        "和朋友一起玩桌游到深夜，笑到肚子疼",
        "一个人去吃火锅，感觉也挺自在的",
        "最近开始记账，发现每个月在奶茶上花了快三百块",
        "终于把拖了很久的体检做了，结果一切正常",
        "刷到了一首老歌，一下子想起大学时的事情",
    ],
}

# 噪音记忆 — 介于编程和生活之间的内容
SEED_NOISE = [
    "写了一个记账的 Python 脚本，分析每个月的开销",
    "用 Excel 做了一个健身记录表，公式自动计算卡路里",
    "给家里的智能家居写了个自动化脚本",
    "在网上看到一个用代码生成菜谱的项目",
    "用 Python 分析了自己的微信聊天记录，发现最常用的词是'好的'",
    "做一个健康管理 App 的想法，用机器学习预测睡眠质量",
    "用爬虫抓了豆瓣电影 Top 250 的数据分析评分规律",
    "写了个脚本自动备份手机照片到 NAS",
    "把跑步数据导出成 CSV 用 matplotlib 画了趋势图",
    "写了一个提醒喝水的桌面小程序",
    "用 Python 分析了自己的信用卡账单",
    "做了个网页来记录每天的心情和日记",
    "计划做一个家庭菜谱管理的 Flutter App",
    "用技术手段优化了自己的早起习惯",
    "写了个 Telegram bot 每天推送天气预报",
]


def seed_qdrant(
    qdrant_path: str = QDRANT_PATH,
    collection: str = COLLECTION_NAME,
    embed_fn=None,
) -> int:
    """向 Qdrant 写入种子数据。"""
    if embed_fn is None:
        from app.llm.embed import local_embed
        embed_fn = local_embed

    client = QdrantClient(path=qdrant_path)

    # 确保 collection 存在
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=qdrant_models.VectorParams(
                size=QWEN_EMBED_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info("创建 collection: %s", collection)

    count = 0
    all_texts = []

    # 编程领域 (25条)
    for text in SEED_MEMORIES["编程"]:
        mid = str(uuid.uuid4())
        emb = embed_fn(text)
        if emb is None:
            continue
        client.upsert(
            collection_name=collection,
            points=[qdrant_models.PointStruct(
                id=mid,
                vector=emb,
                payload={"document": text, "tags": ["编程", "技术"], "timestamp": time.time()},
            )],
        )
        count += 1
        all_texts.append(text)

    # 生活领域 (25条)
    for text in SEED_MEMORIES["生活"]:
        mid = str(uuid.uuid4())
        emb = embed_fn(text)
        if emb is None:
            continue
        client.upsert(
            collection_name=collection,
            points=[qdrant_models.PointStruct(
                id=mid,
                vector=emb,
                payload={"document": text, "tags": ["生活", "日常"], "timestamp": time.time()},
            )],
        )
        count += 1
        all_texts.append(text)

    # 噪音 (15条) — 中间地带
    for text in SEED_NOISE:
        mid = str(uuid.uuid4())
        emb = embed_fn(text)
        if emb is None:
            continue
        client.upsert(
            collection_name=collection,
            points=[qdrant_models.PointStruct(
                id=mid,
                vector=emb,
                payload={"document": text, "tags": ["编程", "生活"], "timestamp": time.time()},
            )],
        )
        count += 1
        all_texts.append(text)

    logger.info("种子数据写入完成: %d 条 (编程25 + 生活25 + 噪音15)", count)
    return count


# ═══════════════════════════════════════════════════════════════
# 1. 数据加载: Qdrant → V 矩阵
# ═══════════════════════════════════════════════════════════════

def load_vectors_from_qdrant(
    qdrant_path: str = QDRANT_PATH,
    collection: str = COLLECTION_NAME,
) -> Tuple[np.ndarray, list[str], list[dict]]:
    """从 Qdrant scroll 拉取全部向量 + payload。

    Returns:
        V:  [N × 3584] float32, 已归一化
        ids: [N] 记忆ID列表
        payloads: [N] payload dict 列表
    """
    client = QdrantClient(path=qdrant_path)
    logger.info("连接 Qdrant: %s / %s", qdrant_path, collection)

    # 先看有多少条
    try:
        info = client.count(collection_name=collection)
        total = info.count
    except Exception:
        # 本地模式可能不支持 count，fallback
        pts, _ = client.scroll(collection_name=collection, limit=1)
        total = "?" if pts else 0

    logger.info("总记忆数: %s，开始 scroll...", total)

    vectors_list = []
    ids_list = []
    payloads_list = []
    offset = None
    batch_count = 0
    t0 = time.time()

    while True:
        pts, next_offset = client.scroll(
            collection_name=collection,
            with_vectors=True,
            with_payload=True,
            limit=1000,
            offset=offset,
        )
        if not pts:
            break

        for pt in pts:
            if pt.vector is None:
                continue
            vec = np.array(pt.vector, dtype=DTYPE)
            # 确保归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors_list.append(vec)
            ids_list.append(str(pt.id))
            payloads_list.append(pt.payload or {})

        batch_count += 1
        if batch_count % 5 == 0:
            logger.info("  已加载 %d 条 (%d batches)...", len(vectors_list), batch_count)

        if next_offset is None or len(pts) < 1000:
            break
        offset = next_offset

    elapsed = time.time() - t0
    V = np.stack(vectors_list, axis=0).astype(DTYPE) if vectors_list else np.empty((0, QWEN_EMBED_DIM), dtype=DTYPE)
    logger.info("加载完成: N=%d, shape=%s, 耗时 %.1fs", len(vectors_list), V.shape, elapsed)
    return V, ids_list, payloads_list


# ═══════════════════════════════════════════════════════════════
# 2. M 矩阵构建与缓存
# ═══════════════════════════════════════════════════════════════

def build_m_matrix(V: np.ndarray) -> np.ndarray:
    """构建 Gram 矩阵 M = VᵀV。

    V: [N × d], M: [d × d] = Vᵀ @ V

    时间复杂度: O(N × d²) — N=10000, d=3584 → ~460M FLOPs, ~1-2s on CPU
    空间: d × d = 3584² × 4 bytes ≈ 51.4 MB (float32)
    """
    N, d = V.shape
    logger.info("构建 M = VᵀV: V[%d × %d] → M[%d × %d] (~%.1f MB)",
                N, d, d, d, d * d * 4 / 1024 / 1024)
    t0 = time.time()
    M = V.T @ V  # [d×N] @ [N×d] = [d×d]
    elapsed = time.time() - t0
    logger.info("M 矩阵构建完成: %.2fs, 非零元素: %d", elapsed, np.count_nonzero(M))
    return M.astype(DTYPE)


def save_m_matrix(M: np.ndarray, ids: list[str]):
    """缓存 M 矩阵和 ID 列表到磁盘。"""
    np.save(M_MATRIX_PATH, M)
    with open(V_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"count": len(ids), "ids": ids}, f, ensure_ascii=False)
    logger.info("M 矩阵已缓存: %s (%.1f MB)", M_MATRIX_PATH, os.path.getsize(M_MATRIX_PATH) / 1024 / 1024)


def load_m_matrix() -> Tuple[Optional[np.ndarray], Optional[list[str]]]:
    """从磁盘加载缓存的 M 矩阵。"""
    if not M_MATRIX_PATH.exists() or not V_IDS_PATH.exists():
        return None, None
    M = np.load(M_MATRIX_PATH)
    with open(V_IDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("M 矩阵从缓存加载: shape=%s, N=%d", M.shape, data["count"])
    return M, data.get("ids", [])


# ═══════════════════════════════════════════════════════════════
# 3. 查询: q → R = M@q
# ═══════════════════════════════════════════════════════════════

def embed_query(text: str) -> np.ndarray:
    """用 qwen_embed 嵌入查询文本，返回归一化向量 [d]."""
    from app.llm.embed import local_embed
    vec = local_embed(text)
    if vec is None:
        raise RuntimeError(f"嵌入失败: {text[:50]}...")
    v = np.array(vec, dtype=DTYPE)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def compute_residual(M: np.ndarray, q: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """计算残差方向 R = M @ q × α。

    M: [d × d], q: [d], R: [d] — 所有记忆对查询的加权响应。
    """
    R = M @ q  # [d×d] @ [d] = [d]
    R = R * alpha
    return R.astype(DTYPE)


def top_k_memories_by_residual(
    R: np.ndarray,
    V: np.ndarray,
    ids: list[str],
    payloads: list[dict],
    k: int = 10,
) -> list[dict]:
    """用残差方向 R 对全量记忆做相似度排序。

    计算 score_i = dot(V[i], R)，取 Top-K。
    等价于 V @ R = V @ (M @ q) = V @ Vᵀ @ q = (V @ Vᵀ) @ q
    而 V @ q 是标准余弦检索 — R 版本的排序会不同，体现"记忆场"效应。
    """
    scores = V @ R  # [N] — 每条记忆在 R 方向上的投影
    top_indices = np.argsort(-scores)[:k]
    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "id": ids[idx],
            "score": float(scores[idx]),
            "text": (payloads[idx].get("document", "") or
                     payloads[idx].get("text", "") or
                     payloads[idx].get("summary", ""))[:200],
            "tags": payloads[idx].get("tags", []),
            "timestamp": payloads[idx].get("timestamp", 0),
        })
    return results


def top_k_memories_by_cosine(
    q: np.ndarray,
    V: np.ndarray,
    ids: list[str],
    payloads: list[dict],
    k: int = 10,
) -> list[dict]:
    """标准余弦检索 Top-K（对照基线）。"""
    scores = V @ q  # [N]
    top_indices = np.argsort(-scores)[:k]
    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "id": ids[idx],
            "score": float(scores[idx]),
            "text": (payloads[idx].get("document", "") or
                     payloads[idx].get("text", "") or
                     payloads[idx].get("summary", ""))[:200],
            "tags": payloads[idx].get("tags", []),
            "timestamp": payloads[idx].get("timestamp", 0),
        })
    return results


# ═══════════════════════════════════════════════════════════════
# 4. 验证场景
# ═══════════════════════════════════════════════════════════════

def _print_results(title: str, results: list[dict]):
    """格式化打印检索结果。"""
    print(f"\n  [{title}]")
    for r in results:
        text = r["text"].replace("\n", " ")[:120]
        tags = ", ".join(r.get("tags", [])[:5])
        print(f"    #{r['rank']} [score={r['score']:.4f}] {text}")
        if tags:
            print(f"        tags: {tags}")


def scene_a_basic_direction(M: np.ndarray, V: np.ndarray, ids: list[str], payloads: list[dict]):
    """场景 A: 基础方向验证 — M@q 产出的 R 是否指向正确记忆。"""
    print("\n" + "=" * 70)
    print("场景 A: 基础方向验证 — R = M@q 是否指向正确语义领域")
    print("=" * 70)

    queries = [
        "Python 编程学习中遇到的困难",
        "今天心情很好，和朋友聊了很多",
        "Rust borrow checker 怎么理解",
    ]

    for query_text in queries:
        print(f"\n  查询: \"{query_text}\"")
        q = embed_query(query_text)

        # 标准余弦检索（对照）
        cos_results = top_k_memories_by_cosine(q, V, ids, payloads, k=5)
        _print_results("余弦 Top-5（对照基线）", cos_results)

        # M@q 残差检索
        R = compute_residual(M, q)
        residual_results = top_k_memories_by_residual(R, V, ids, payloads, k=5)
        _print_results("M@q 残差 Top-5", residual_results)

        # 对比: 两个排序的重叠度
        cos_ids = set(r["id"] for r in cos_results[:5])
        res_ids = set(r["id"] for r in residual_results[:5])
        overlap = len(cos_ids & res_ids)
        print(f"    → 余弦 Top-5 vs M@q Top-5 重叠: {overlap}/5")

        # R 与 q 的余弦相似度 — R 是记忆场加权方向，不一定等于 q
        cos_Rq = float(np.dot(R, q) / (np.linalg.norm(R) * np.linalg.norm(q)))
        print(f"    → cos(R, q) = {cos_Rq:.4f} (R 与原始查询的偏离度)")


def scene_b_noise_robustness(M: np.ndarray, V: np.ndarray, ids: list[str], payloads: list[dict]):
    """场景 B: 噪音鲁棒性 — M@q 在噪音中是否保持方向稳定。

    模拟 AuraSDK 压力测试中的场景: 200 条同领域噪音。
    M@q 不依赖离散选择 → 噪音对它是信号，应保持领域方向不变。
    """
    print("\n" + "=" * 70)
    print("场景 B: 噪音鲁棒性 — 同领域噪音中 R 方向是否稳定")
    print("=" * 70)

    # 先找到有足够多样本的领域
    # 统计 tags 分布，找一个有足够记忆的领域
    from collections import Counter
    tag_counter = Counter()
    for p in payloads:
        for t in p.get("tags", []):
            tag_counter[t] += 1

    # 找最多的两个不同领域
    top_tags = tag_counter.most_common(10)
    print(f"  标签分布 Top-10: {top_tags}")

    if len(top_tags) < 2:
        print("  [WARN] 标签太少，跳过噪音鲁棒性测试")
        return

    # 选两个不同领域
    tag_a = top_tags[0][0] if len(top_tags) > 0 else "编程"
    tag_b = top_tags[1][0] if len(top_tags) > 1 else "情绪"

    # 收集中等相似度的记忆
    mid_memories = []
    for i, p in enumerate(payloads):
        tags = p.get("tags", [])
        # 排除同时有两个标签的
        if tag_a in tags and tag_b in tags:
            continue
        mid_memories.append(i)

    if len(mid_memories) < 10:
        print(f"  [WARN] 记忆不足 ({len(mid_memories)}条)，跳过")
        return

    print(f"\n  领域 A: \"{tag_a}\", 领域 B: \"{tag_b}\"")
    print(f"  记忆池: {len(mid_memories)} 条 (排除双标签)")

    # 测试: 查询领域 A 的内容
    query_a = f"关于{tag_a}的讨论"
    q_a = embed_query(query_a)
    R_a = compute_residual(M, q_a)

    # 计算 R_a 与全量记忆的投影分布
    projections_a = V @ R_a  # [N]
    # 按记忆是否属于领域 A 分组
    idx_a = [i for i in mid_memories if tag_a in payloads[i].get("tags", [])]
    idx_b = [i for i in mid_memories if tag_b in payloads[i].get("tags", [])]

    if idx_a and idx_b:
        proj_a = projections_a[idx_a].mean()
        proj_b = projections_a[idx_b].mean()
        separation = proj_a - proj_b
        print(f"\n  查询: \"{query_a}\"")
        print(f"    领域 A ({tag_a}) 平均投影: {proj_a:.4f}")
        print(f"    领域 B ({tag_b}) 平均投影: {proj_b:.4f}")
        print(f"    分离度 (A-B): {separation:.4f} {'[OK]' if separation > 0 else '[FAIL]'}")

    # 查询领域 B
    query_b = f"关于{tag_b}的感受"
    q_b = embed_query(query_b)
    R_b = compute_residual(M, q_b)
    projections_b = V @ R_b

    if idx_a and idx_b:
        proj_a2 = projections_b[idx_a].mean()
        proj_b2 = projections_b[idx_b].mean()
        separation2 = proj_b2 - proj_a2
        print(f"\n  查询: \"{query_b}\"")
        print(f"    领域 A ({tag_a}) 平均投影: {proj_a2:.4f}")
        print(f"    领域 B ({tag_b}) 平均投影: {proj_b2:.4f}")
        print(f"    分离度 (B-A): {separation2:.4f} {'[OK]' if separation2 > 0 else '[FAIL]'}")

    # R_a 与 R_b 的余弦相似度 — 应为低或中等（不同领域）
    cos_Ra_Rb = float(np.dot(R_a, R_b) / (np.linalg.norm(R_a) * np.linalg.norm(R_b)))
    print(f"\n    cos(R_a, R_b) = {cos_Ra_Rb:.4f} (跨领域 R 相似度，越低越好)")


def scene_c_mfield_vs_topk(M: np.ndarray, V: np.ndarray, ids: list[str], payloads: list[dict]):
    """场景 C: M@q 记忆场 vs 离散 Top-K 的差异分析。

    核心问题: M@q 的"全量加权" vs Top-K 的"离散截断"到底差在哪？
    """
    print("\n" + "=" * 70)
    print("场景 C: M@q 连续场 vs Top-K 离散截断")
    print("=" * 70)

    queries = [
        "我最近在学习新的编程语言",
        "今天工作压力很大",
    ]

    for query_text in queries:
        print(f"\n  查询: \"{query_text}\"")
        q = embed_query(query_text)

        R = compute_residual(M, q)

        # Top-K=5 离散检索的质心
        cos_top = top_k_memories_by_cosine(q, V, ids, payloads, k=5)
        top_vectors = np.stack([V[ids.index(r["id"])] for r in cos_top])
        centroid_top5 = top_vectors.mean(axis=0)
        centroid_top5 = centroid_top5 / np.linalg.norm(centroid_top5)

        # M@q 残差方向（归一化）
        R_norm = R / np.linalg.norm(R)

        # 差异: R 与 Top-5 质心的余弦相似度
        diff = float(np.dot(R_norm, centroid_top5))
        print(f"    cos(R_mfield, centroid_top5) = {diff:.4f}")
        print(f"    → R 包含 Top-5 质心的信息 ({diff*100:.1f}%)，但额外编码了其余 {V.shape[0]-5} 条记忆的加权贡献")

        # 展示 Top-3 分别来自余弦和 R
        print(f"\n    余弦 Top-3:")
        for r in cos_top[:3]:
            print(f"      [{r['score']:.4f}] {r['text'][:100]}")

        res_top = top_k_memories_by_residual(R, V, ids, payloads, k=3)
        print(f"\n    M@q Top-3:")
        for r in res_top[:3]:
            print(f"      [{r['score']:.4f}] {r['text'][:100]}")


def scene_d_alpha_sweep(M: np.ndarray, V: np.ndarray, ids: list[str], payloads: list[dict]):
    """场景 D: α 值扫参 — 全局强度系数对 R 方向的影响。

    R = M@q × α，α 不影响方向，只影响幅度。验证这一点。
    """
    print("\n" + "=" * 70)
    print("场景 D: α 扫参 — 验证 α 只影响幅度不影响方向")
    print("=" * 70)

    q = embed_query("最近的学习进展")
    R_base = compute_residual(M, q, alpha=1.0)
    R_base_norm = R_base / np.linalg.norm(R_base)

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"\n  查询: \"最近的学习进展\"")
    for alpha in alphas:
        R = compute_residual(M, q, alpha=alpha)
        R_norm = R / (np.linalg.norm(R) + 1e-10)
        cos_with_base = float(np.dot(R_norm, R_base_norm))
        magnitude = float(np.linalg.norm(R))
        print(f"    α={alpha:5.1f}  |R|={magnitude:8.2f}  cos(R, R_base)={cos_with_base:.6f}  "
              f"{'[OK] 方向不变' if abs(cos_with_base - 1.0) < 1e-5 else '[FAIL] 方向变化!'}")


def scene_e_cold_start(M: np.ndarray, V: np.ndarray, ids: list[str], payloads: list[dict]):
    """场景 E: 冷启动 — 无记忆时的行为。

    没有记忆时应优雅退化，M 为零矩阵或空。
    """
    print("\n" + "=" * 70)
    print("场景 E: 冷启动 — 零记忆退化")
    print("=" * 70)

    if V.shape[0] == 0:
        print("  [OK] 记忆库为空，M 矩阵为零，R=0 — 不注入残差，等价于纯 prompt 模式")
    else:
        print(f"  当前有 {V.shape[0]} 条记忆，跳过冷启动测试")


# ═══════════════════════════════════════════════════════════════
# 5. 性能基准
# ═══════════════════════════════════════════════════════════════

def benchmark(M: np.ndarray, V: np.ndarray):
    """性能基准: 单次 M@q 的延迟。"""
    print("\n" + "=" * 70)
    print("性能基准")
    print("=" * 70)

    q = np.random.randn(QWEN_EMBED_DIM).astype(DTYPE)
    q = q / np.linalg.norm(q)

    # warmup
    for _ in range(3):
        _ = M @ q

    N_runs = 1000
    t0 = time.time()
    for _ in range(N_runs):
        _ = M @ q
    elapsed = (time.time() - t0) / N_runs * 1000  # ms

    print(f"  M 矩阵: {M.shape}, 大小: {M.nbytes / 1024 / 1024:.1f} MB")
    print(f"  V 矩阵: {V.shape}, 大小: {V.nbytes / 1024 / 1024:.1f} MB")
    print(f"  M@q 延迟: {elapsed:.3f} ms (avg over {N_runs} runs)")
    print(f"  M@q + qwen_embed: ~{elapsed + 1.0:.1f} ms (含嵌入)")

    if elapsed < 1.0:
        print("  [OK] 延迟远低于实时对话要求 (100ms)")
    elif elapsed < 10:
        print("  [OK] 延迟可接受 (<10ms)")
    else:
        print("  [WARN] 延迟偏高")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M@q 记忆场原型")
    parser.add_argument("--quick", action="store_true", help="仅基础验证")
    parser.add_argument("--scene", choices=["basic", "noise", "mfield", "alpha", "cold", "bench", "all"],
                        default="all", help="运行指定场景")
    parser.add_argument("--rebuild-matrix", action="store_true", help="强制重建 M 矩阵")
    parser.add_argument("--seed", action="store_true", help="先写入种子数据 (65条) 再验证")
    parser.add_argument("--qdrant-path", default=QDRANT_PATH, help="Qdrant 路径")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Collection 名")
    args = parser.parse_args()

    scene = args.scene
    if args.quick:
        scene = "basic"

    # —— 种子数据 ——
    if args.seed:
        logger.info("写入种子数据...")
        seed_qdrant(args.qdrant_path, args.collection)

    # —— 加载或构建 M 矩阵 ——
    M, cached_ids = load_m_matrix()
    V = None
    ids = None
    payloads = None

    if M is not None and not args.rebuild_matrix:
        logger.info("使用缓存的 M 矩阵，但仍需加载 V 用于验证")
        V, ids, payloads = load_vectors_from_qdrant(args.qdrant_path, args.collection)
        if V.shape[0] == 0:
            logger.warning("Qdrant 中无记忆，M 矩阵可能过期。考虑 --rebuild-matrix")
    else:
        logger.info("构建新的 M 矩阵...")
        V, ids, payloads = load_vectors_from_qdrant(args.qdrant_path, args.collection)
        if V.shape[0] == 0:
            logger.warning("Qdrant 中无记忆，创建零 M 矩阵 (冷启动)")
            M = np.zeros((QWEN_EMBED_DIM, QWEN_EMBED_DIM), dtype=DTYPE)
        else:
            M = build_m_matrix(V)
        save_m_matrix(M, ids)

    N = V.shape[0]
    print(f"\n═══════════════════════════════════════════════════════════")
    print(f"  M@q 记忆场原型")
    print(f"  记忆数: {N}")
    print(f"  M 矩阵: {M.shape} ({M.nbytes / 1024 / 1024:.1f} MB)")
    print(f"═══════════════════════════════════════════════════════════")

    if N == 0:
        print("\n  [WARN] 记忆库为空，仅运行冷启动测试")
        scene_e_cold_start(M, V, ids, payloads)
        return

    # —— 运行验证场景 ——
    scene_map = {
        "basic": [scene_a_basic_direction],
        "noise": [scene_b_noise_robustness],
        "mfield": [scene_c_mfield_vs_topk],
        "alpha": [scene_d_alpha_sweep],
        "cold": [scene_e_cold_start],
        "bench": [benchmark],
        "all": [
            scene_a_basic_direction,
            scene_b_noise_robustness,
            scene_c_mfield_vs_topk,
            scene_d_alpha_sweep,
            scene_e_cold_start,
            benchmark,
        ],
    }

    for fn in scene_map[scene]:
        if fn == benchmark:
            fn(M, V)
        else:
            fn(M, V, ids, payloads)

    print("\n" + "=" * 70)
    print("全部验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
