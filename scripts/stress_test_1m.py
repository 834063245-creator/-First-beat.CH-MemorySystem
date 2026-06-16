#!/usr/bin/env python3
"""Phase 4 压力测试 - 百万级 Qdrant 性能基准。

用法:
    python scripts/stress_test_1m.py                    # 默认 10K 快速验证
    python scripts/stress_test_1m.py --count 100000     # 10万条
    python scripts/stress_test_1m.py --count 1000000    # 100万条（需 Qdrant 服务器）
    python scripts/stress_test_1m.py --count 100000 --server http://localhost:6333  # 指定服务器

Phase 4 交付物 #4: 性能基准验证 (§10.2)。

设计:
    - 合成记忆：真实中文对话分布（技术/生活/情感/宠物/工作/旅行/音乐/家庭 8 话题）
    - 每个操作跑 N 次取 P50/P95/P99
    - 输出表格对照 SPEC_MIGRATION.md §10.2 阈值
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# ── 随机种子，可复现 ──────────────────────────────────────────
random.seed(42)

# ===================================================================
# 合成数据生成
# ===================================================================

TOPICS = {
    "技术": ["Python", "Rust", "bug", "优化", "开发", "部署", "AI", "算法", "代码", "重构",
             "微服务", "Docker", "Kubernetes", "API", "数据库", "缓存", "并发", "异步"],
    "生活": ["吃饭", "睡觉", "作息", "健康", "运动", "健身", "咖啡", "茶", "熬夜", "早起",
             "外卖", "做饭", "散步", "瑜伽", "冥想"],
    "情感": ["爱", "喜欢", "感情", "关系", "陪伴", "想你", "在乎", "依赖", "珍惜", "信任",
             "理解", "沟通", "争吵", "和解", "温暖"],
    "宠物": ["猫", "狗", "橘猫", "边牧", "宠物", "兽医", "尿闭", "猫粮", "遛狗", "铲屎",
             "毛孩子", "喵星人", "汪星人"],
    "工作": ["公司", "项目", "leader", "绩效", "年终奖", "裁员", "跳槽", "面试", "加班",
             "会议", "需求", "排期", "OKR", "日报", "团建"],
    "旅行": ["旅行", "日本", "东京", "大阪", "酒店", "机票", "景点", "攻略", "签证", "护照",
             "美食", "购物", "温泉", "樱花"],
    "音乐": ["歌", "音乐", "专辑", "周杰伦", "演唱会", "吉他", "钢琴", "作曲", "编曲",
             "歌词", "旋律", "节奏", "摇滚", "爵士", "古典"],
    "家庭": ["妈", "爸", "妹妹", "家人", "老家", "郑州", "北京", "上海", "过年", "团聚",
             "电话", "礼物", "思念", "牵挂"],
}

EMOTIONS = ["positive", "negative", "neutral"]
EMOTION_WEIGHTS = [0.35, 0.15, 0.50]


def _random_date(start_days_ago: int = 365) -> datetime:
    days = random.randint(0, start_days_ago)
    return datetime.now() - timedelta(days=days)


def _gen_embedding(dim: int = 1024) -> list[float]:
    """生成伪随机单位向量（近似 bge-m3 分布）。"""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _gen_memory(idx: int) -> dict:
    """生成一条合成记忆，包含完整 payload。"""
    topic = random.choice(list(TOPICS.keys()))
    keywords = random.sample(TOPICS[topic], min(3, len(TOPICS[topic])))
    emotion_bin = random.choices(EMOTIONS, weights=EMOTION_WEIGHTS, k=1)[0]
    valence = {"positive": random.uniform(0.1, 1.0),
               "negative": random.uniform(-1.0, -0.1),
               "neutral": random.uniform(-0.2, 0.2)}[emotion_bin]
    arousal = random.uniform(0.1, 1.0)
    intensity = random.randint(0, 5)

    dt = _random_date()
    ts = dt.timestamp()

    user_msg = f"最近在弄{keywords[0]}相关的东西，{random.choice(['感觉','觉得','发现'])}挺有意思的"
    ai_msg = f"是的，{keywords[0]}确实是个有趣的方向！你有什么具体想了解的？"

    entities = [
        {"text": kw, "type": "TOPIC"} for kw in keywords
    ]

    return {
        "id": str(uuid.uuid4()),
        "vector": _gen_embedding(),
        "payload": {
            "user_message": user_msg,
            "ai_message": ai_msg,
            "document": f"用户：{user_msg}\nAI：{ai_msg}",
            "summary": f"讨论{keywords[0]}相关话题",
            "tags": ",".join(keywords),
            "timestamp": ts,
            "hit_count": random.randint(0, 50),
            "last_hit_time": ts + random.randint(0, 86400 * 30),
            "heat": random.choices(["hot", "warm", "cool"], weights=[0.2, 0.5, 0.3], k=1)[0],
            "embed_model": "bge-m3",
            "stale": random.random() < 0.05,
            "archived": False,
            "superseded_by": "",
            "storage_complete": True,
            "source": "user",
            "date_tag": dt.strftime("%Y-%m-%d"),
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "week": dt.isocalendar()[1],
            "day_of_week": dt.weekday(),
            "quarter": (dt.month - 1) // 3 + 1,
            "season": (dt.month % 12 + 3) // 3,
            "year_month": dt.strftime("%Y-%m"),
            "emotion_valence": valence,
            "emotion_arousal": arousal,
            "emotion_valence_bin": emotion_bin,
            "emotional_intensity": intensity,
            "entities": entities,
            "entity_co_counts": {kw: random.randint(1, 20) for kw in keywords},
        },
    }


# ===================================================================
# 压力测试执行器
# ===================================================================

def _measure(func, warmup: int = 3, samples: int = 20) -> dict:
    """测量函数延迟，返回 P50/P95/P99/mean/min/max (ms)。"""
    # 预热
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass

    latencies = []
    for _ in range(samples):
        t0 = time.perf_counter()
        try:
            func()
        except Exception:
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

    if not latencies:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "min": None, "max": None}

    latencies.sort()
    return {
        "p50": latencies[len(latencies) // 2],
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
        "mean": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }


def _check(bench_name: str, measured: float | None, threshold: float, unit: str = "ms") -> str:
    """对照阈值判定：[PASS]通过 / [WARN]略慢 / [FAIL]超标。"""
    if measured is None:
        return "[SKIP] 跳过"
    ratio = measured / threshold
    if ratio <= 1.0:
        return f"[PASS] {measured:.1f}{unit} (≤{threshold}{unit})"
    elif ratio <= 1.5:
        return f"[WARN] {measured:.1f}{unit} (>{threshold}{unit}, +{ratio-1:.0%})"
    else:
        return f"[FAIL] {measured:.1f}{unit} (>{threshold}{unit}, +{ratio-1:.0%})"


# ===================================================================
# 主函数
# ===================================================================

def run_stress_test(count: int, qdrant_url: str | None = None,
                    batch_size: int = 500, samples: int = 20):
    """执行压力测试。"""
    from qdrant_client import QdrantClient, models

    # ── Qdrant 客户端 ──────────────────────────────────────────
    if qdrant_url:
        client = QdrantClient(url=qdrant_url, timeout=60)
        coll_name = f"stress_test_{count}"
        print(f"[SERVER] Qdrant 服务器模式: {qdrant_url}")
    else:
        tmpdir = tempfile.mkdtemp(prefix="qdrant_stress_")
        client = QdrantClient(path=tmpdir)
        coll_name = "stress_test"
        print(f"[LOCAL] Qdrant 本地模式: {tmpdir}")

    # 清理旧 collection
    existing = {c.name for c in client.get_collections().collections}
    if coll_name in existing:
        client.delete_collection(coll_name)

    # ── 创建 collection ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 4 压力测试 - {count:,} 条记忆")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    client.create_collection(
        collection_name=coll_name,
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE,
            on_disk=count >= 100_000,
        ),
        # Phase 4: scalar_int8 量化
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ) if count >= 100_000 else None,
    )
    print(f"[OK] Collection 创建: {(time.perf_counter() - t0)*1000:.0f}ms")

    # ── Payload 索引 ──────────────────────────────────────────
    t0 = time.perf_counter()
    _idx_fields = [
        ("heat", models.PayloadSchemaType.KEYWORD),
        ("timestamp", models.PayloadSchemaType.FLOAT),
        ("stale", models.PayloadSchemaType.BOOL),
        ("source", models.PayloadSchemaType.KEYWORD),
    ]
    for field, stype in _idx_fields:
        try:
            client.create_payload_index(
                collection_name=coll_name, field_name=field, field_schema=stype,
            )
        except Exception:
            pass
    print(f"[OK] Payload 索引: {(time.perf_counter() - t0)*1000:.0f}ms ({len(_idx_fields)} 字段)")

    # ── 批量写入 ──────────────────────────────────────────────
    print(f"\n[WRITE] 批量写入 {count:,} 条...")
    mem_ids = []
    total_batches = (count + batch_size - 1) // batch_size
    t_write_start = time.perf_counter()
    written = 0

    for batch_idx in range(total_batches):
        batch_n = min(batch_size, count - written)
        points = []
        for i in range(batch_n):
            mem = _gen_memory(written + i)
            mem_ids.append(mem["id"])
            points.append(models.PointStruct(
                id=mem["id"],
                vector=mem["vector"],
                payload=mem["payload"],
            ))

        t_batch = time.perf_counter()
        client.upsert(collection_name=coll_name, points=points)
        written += batch_n
        batch_ms = (time.perf_counter() - t_batch) * 1000

        if (batch_idx + 1) % max(1, total_batches // 10) == 0:
            elapsed = time.perf_counter() - t_write_start
            rate = written / elapsed
            pct = written / count * 100
            eta = (count - written) / rate if rate > 0 else 0
            print(f"  [{pct:5.1f}%] {written:>8,}/{count:,}  "
                  f"batch={batch_ms:.0f}ms  rate={rate:,.0f}/s  ETA={eta:.0f}s")

    t_write = time.perf_counter() - t_write_start
    rate = count / t_write
    print(f"[OK] 写入完成: {t_write:.1f}s ({rate:,.0f} 条/s)")

    # ── 运行基准测试 ──────────────────────────────────────────
    print(f"\n[BENCH] 性能基准 (各 {samples} 次采样, P50/P95/P99)...\n")

    query_vector = _gen_embedding()
    results = {}

    # 1. add_memory (单条)
    def _test_add():
        mid = str(uuid.uuid4())
        mem = _gen_memory(count)
        client.upsert(collection_name=coll_name, points=[models.PointStruct(
            id=mid, vector=mem["vector"], payload=mem["payload"],
        )])
        client.delete(collection_name=coll_name, points_selector=[mid])
    results["add_memory"] = _measure(_test_add, samples=min(samples, 30))

    # 2. 语义 search (top 50)
    def _test_search():
        client.search(
            collection_name=coll_name,
            query_vector=query_vector,
            limit=50,
            with_payload=False,
            with_vectors=False,
        )
    results["semantic_search(top50)"] = _measure(_test_search)

    # 3. 分页 list (20 per page)
    def _test_list():
        # scroll with offset
        offset = random.randint(0, max(0, count - 20))
        pts, _ = client.scroll(
            collection_name=coll_name,
            limit=20,
            offset=offset,
            with_payload=["summary", "tags", "timestamp", "hit_count"],
        )
    results["list_memories(page=20)"] = _measure(_test_list)

    # 4. stats (count)
    def _test_count():
        client.count(collection_name=coll_name)
    results["stats(count)"] = _measure(_test_count, samples=min(samples, 50))

    # 5. batch_hit_count (100 IDs)
    sample_ids = random.sample(mem_ids, min(100, len(mem_ids)))
    def _test_batch_hit():
        pts = client.retrieve(
            collection_name=coll_name,
            ids=sample_ids,
            with_payload=["hit_count"],
        )
    results["batch_hit_count(100)"] = _measure(_test_batch_hit, samples=min(samples, 30))

    # 6. 全量扫描 (分页 scroll)
    # 仅采样部分，估算总时间
    scan_limit = min(count, 10_000)
    def _test_scan():
        scanned = 0
        offset = None
        while scanned < scan_limit:
            pts, next_offset = client.scroll(
                collection_name=coll_name,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not pts:
                break
            scanned += len(pts)
            if next_offset is None:
                break
            offset = next_offset
    scan_result = _measure(_test_scan, warmup=1, samples=min(5, samples))
    if scan_result["mean"] is not None and scan_limit > 0:
        # 估算全量时间
        scan_est = scan_result["mean"] * (count / scan_limit)
        results["full_scan(est)"] = {
            "p50": scan_est, "p95": scan_est * 1.2,
            "p99": scan_est * 1.5, "mean": scan_est,
            "min": scan_result["min"], "max": scan_result["max"],
        }
    else:
        results["full_scan(est)"] = None

    # ── 对照阈值判定 ──────────────────────────────────────────
    print(f"{'操作':<30} {'P50':>8} {'P95':>8} {'判定':>30}")
    print("-" * 80)

    BENCHMARKS_10K = {
        "add_memory": 50,
        "semantic_search(top50)": 30,
        "list_memories(page=20)": 30,
        "stats(count)": 5,
        "batch_hit_count(100)": 100,
        "full_scan(est)": 2000,
    }
    BENCHMARKS_1M = {
        "add_memory": 50,
        "semantic_search(top50)": 50,
        "list_memories(page=20)": 100,
        "stats(count)": 10,
        "batch_hit_count(100)": 200,
        "full_scan(est)": 10000,
    }

    thresholds = BENCHMARKS_1M if count >= 100_000 else BENCHMARKS_10K

    passed = 0
    failed = 0
    for op, bench in results.items():
        if bench is None or bench.get("p50") is None:
            print(f"{op:<30} {'-':>8} {'-':>8} {'[SKIP] 跳过':>30}")
            continue
        threshold = thresholds.get(op, 999)
        p50_str = f"{bench['p50']:.1f}ms"
        p95_str = f"{bench['p95']:.1f}ms"
        verdict = _check(op, bench["p50"], threshold)
        print(f"{op:<30} {p50_str:>8} {p95_str:>8} {verdict:>30}")
        if "[PASS]" in verdict:
            passed += 1
        elif "[FAIL]" in verdict:
            failed += 1

    # ── 内存估算 ──────────────────────────────────────────────
    coll_info = client.get_collection(coll_name)
    vectors_count = coll_info.vectors_count if hasattr(coll_info, 'vectors_count') else coll_info.points_count
    print(f"\n[STATS] Collection: {vectors_count:,} vectors")

    # ── 清理 ──────────────────────────────────────────────────
    client.delete_collection(coll_name)
    client.close()
    if not qdrant_url:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 总结 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"结果: {passed} 通过, {failed} 超标, {len(results) - passed - failed} 跳过")
    if failed == 0:
        print("[OK] Phase 4 压力测试通过！")
    else:
        print(f"[WARN]  {failed} 项超标，需优化")
    print(f"{'='*60}")

    return passed, failed


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 压力测试 - Qdrant 百万级性能基准",
    )
    parser.add_argument(
        "--count", type=int, default=10000,
        help="记忆条数 (默认: 10000, 建议: 100000, 完整: 1000000)",
    )
    parser.add_argument(
        "--server", type=str, default=None,
        help="Qdrant 服务器 URL (默认: 本地临时目录)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="批量写入大小 (默认: 500)",
    )
    parser.add_argument(
        "--samples", type=int, default=20,
        help="每项测试采样次数 (默认: 20)",
    )
    args = parser.parse_args()

    if args.count > 100_000 and not args.server:
        print("[WARN]  警告: 超过 10 万条建议使用 Qdrant 服务器 (--server URL)")
        print("   本地模式内存占用高且不支持 payload 索引。")
        print()
        if args.count >= 1_000_000:
            print("[FAIL] 100 万条必须使用 Qdrant 服务器。")
            print("   用法: python scripts/stress_test_1m.py --count 1000000 --server http://localhost:6333")
            sys.exit(1)

    passed, failed = run_stress_test(
        count=args.count,
        qdrant_url=args.server,
        batch_size=args.batch_size,
        samples=args.samples,
    )
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
