#!/usr/bin/env python

# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 072fbc77

"""Phase 0 基础设施连通性验证脚本。

验证项:
  1. Qdrant 服务可达 — 创建临时 collection → 插入 point → 查询 → 删除 collection
  2. vLLM Embed 实例 — POST /v1/embeddings → 验证返回 1024 维向量
  3. vLLM Chat 实例 — POST /v1/chat/completions → 验证返回非空文本
  4. (可选) Ollama 回退可用 — 验证 bge-m3 embedding + qwen2.5 chat

用法:
  python scripts/verify_infra.py                          # 全部验证
  python scripts/verify_infra.py --qdrant-only            # 仅 Qdrant
  python scripts/verify_infra.py --vllm-only              # 仅 vLLM
  python scripts/verify_infra.py --ollama                 # 仅 Ollama (回退验证)
  python scripts/verify_infra.py --quick                  # 跳过耗时测试
"""

import argparse
import json
import os
import sys
import time
import uuid

# 项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None

try:
    import httpx
except ImportError:
    httpx = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ============================================================
# 配置（环境变量优先）
# ============================================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

VLLM_EMBED_URL = os.getenv("VLLM_EMBED_URL", "http://localhost:8001")
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "BAAI/bge-m3")

VLLM_CHAT_URL = os.getenv("VLLM_CHAT_URL", "http://localhost:8002")
VLLM_CHAT_MODEL = os.getenv("VLLM_CHAT_MODEL", "Qwen/Qwen2.5-3B-Instruct")

OLLAMA_URL = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
OLLAMA_CHAT_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b")

TEST_COLLECTION = "_verify_infra_test_" + str(uuid.uuid4())[:8]
VERIFY_DIM = 1024  # bge-m3 expected dimension

# ============================================================
# Rich / plain output
# ============================================================
if HAS_RICH:
    console = Console()

    def print_ok(msg): console.print(f"  [green]✅[/green] {msg}")
    def print_fail(msg): console.print(f"  [red]❌[/red] {msg}")
    def print_warn(msg): console.print(f"  [yellow]⚠️[/yellow] {msg}")
    def print_info(msg): console.print(f"  [dim]ℹ️[/dim] {msg}")
    def print_header(title): console.print(Panel.fit(title, style="bold cyan"))
else:
    def print_ok(msg): print(f"  ✅ {msg}")
    def print_fail(msg): print(f"  ❌ {msg}")
    def print_warn(msg): print(f"  ⚠️  {msg}")
    def print_info(msg): print(f"  ℹ️  {msg}")
    def print_header(title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")


# ============================================================
# 1. Qdrant connectivity
# ============================================================
def verify_qdrant():
    print_header("1. Qdrant 向量数据库")

    if QdrantClient is None:
        print_fail("qdrant-client 未安装。运行: pip install qdrant-client")
        return False

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
        # 健康检查
        health = client.health()
        print_ok(f"Qdrant 服务可达 ({QDRANT_URL}) — status: {health}")

        # 创建临时 collection
        client.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config=models.VectorParams(
                size=VERIFY_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print_ok(f"创建临时 collection: {TEST_COLLECTION}")

        # 写入测试 point
        test_id = str(uuid.uuid4())
        test_vector = [0.0] * VERIFY_DIM
        test_vector[0] = 1.0
        client.upsert(
            collection_name=TEST_COLLECTION,
            points=[models.PointStruct(
                id=test_id,
                vector=test_vector,
                payload={"test": True, "message": "Phase 0 verification"},
            )],
        )
        print_ok(f"写入测试 point: {test_id}")

        # 查询验证
        results = client.search(
            collection_name=TEST_COLLECTION,
            query_vector=test_vector,
            limit=1,
            with_payload=True,
        )
        if results and results[0].id == test_id:
            print_ok(f"语义查询成功 — 命中 {test_id}")
        else:
            print_fail(f"语义查询未命中预期 point")
            return False

        # count
        count = client.count(collection_name=TEST_COLLECTION).count
        if count == 1:
            print_ok(f"count() = {count}")
        else:
            print_fail(f"count() = {count}, 预期 1")
            return False

        # 删除临时 collection
        client.delete_collection(collection_name=TEST_COLLECTION)
        print_ok(f"删除临时 collection (清理)")

        # 列出 collections (验证删除)
        collections = client.get_collections()
        coll_names = [c.name for c in collections.collections]
        if TEST_COLLECTION not in coll_names:
            print_ok("临时 collection 确认已删除")
        else:
            print_warn(f"临时 collection 可能未完全删除")

        print_info(f"当前 collections: {coll_names if coll_names else '(空)'}")
        return True

    except Exception as e:
        print_fail(f"Qdrant 连接失败: {e}")
        return False


# ============================================================
# 2. vLLM Embed 实例
# ============================================================
def verify_vllm_embed():
    print_header("2. vLLM Embedding 实例 (bge-m3)")

    if httpx is None:
        print_fail("httpx 未安装。运行: pip install httpx")
        return False

    try:
        # 健康检查
        with httpx.Client(timeout=VLLM_EMBED_TIMEOUT) as client:
            try:
                health_resp = client.get(f"{VLLM_EMBED_URL}/health")
                print_ok(f"vLLM Embed 服务可达 ({VLLM_EMBED_URL}) — HTTP {health_resp.status_code}")
            except Exception:
                print_warn(f"vLLM Embed /health 端点不可用，尝试直接调用 API")

            test_texts = [
                "你好，这是一条测试消息",
                "Python is a great programming language",
            ]

            # 单条 embed
            resp = client.post(
                f"{VLLM_EMBED_URL}/v1/embeddings",
                json={
                    "model": VLLM_EMBED_MODEL,
                    "input": test_texts[0],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            single_emb = data["data"][0]["embedding"]
            single_dim = len(single_emb)

            if single_dim == VERIFY_DIM:
                print_ok(f"单条 embed 维度正确: {single_dim}")
            else:
                print_fail(f"单条 embed 维度错误: {single_dim}, 预期 {VERIFY_DIM}")
                return False

            # 批量 embed
            resp = client.post(
                f"{VLLM_EMBED_URL}/v1/embeddings",
                json={
                    "model": VLLM_EMBED_MODEL,
                    "input": test_texts,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if len(data["data"]) == 2:
                batch_dim = len(data["data"][0]["embedding"])
                if batch_dim == VERIFY_DIM:
                    print_ok(f"批量 embed ({len(test_texts)}条) 维度正确: {batch_dim}")
                else:
                    print_fail(f"批量 embed 维度错误: {batch_dim}")
                    return False
            else:
                print_fail(f"批量 embed 返回数量错误: {len(data['data'])}, 预期 2")
                return False

            # 验证向量非全零
            non_zero = sum(1 for v in single_emb if abs(v) > 1e-6)
            print_info(f"单条向量非零元素: {non_zero}/{single_dim}")
            if non_zero == 0:
                print_fail("向量全为零，模型可能未正确加载")
                return False

            print_ok(f"vLLM Embed 全部验证通过 (模型: {VLLM_EMBED_MODEL})")
            return True

    except Exception as e:
        print_fail(f"vLLM Embed 验证失败: {e}")
        return False

# Timeout for vLLM embed
VLLM_EMBED_TIMEOUT = int(os.getenv("VLLM_EMBED_TIMEOUT", "30"))


# ============================================================
# 3. vLLM Chat 实例
# ============================================================
def verify_vllm_chat():
    print_header("3. vLLM Chat 实例 (qwen2.5:3b)")

    if httpx is None:
        print_fail("httpx 未安装。运行: pip install httpx")
        return False

    try:
        with httpx.Client(timeout=VLLM_CHAT_TIMEOUT) as client:
            try:
                health_resp = client.get(f"{VLLM_CHAT_URL}/health")
                print_ok(f"vLLM Chat 服务可达 ({VLLM_CHAT_URL}) — HTTP {health_resp.status_code}")
            except Exception:
                print_warn(f"vLLM Chat /health 端点不可用，尝试直接调用 API")

            # 简单生成测试
            resp = client.post(
                f"{VLLM_CHAT_URL}/v1/chat/completions",
                json={
                    "model": VLLM_CHAT_MODEL,
                    "messages": [
                        {"role": "user", "content": "请用一句话回复：你好，世界。"}
                    ],
                    "max_tokens": 32,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            if content and len(content.strip()) > 0:
                print_ok(f"Chat 生成成功 — 返回 {len(content)} 字符: '{content[:60]}...'")
            else:
                print_fail("Chat 返回空内容")
                return False

            # 实体抽取测试
            resp = client.post(
                f"{VLLM_CHAT_URL}/v1/chat/completions",
                json={
                    "model": VLLM_CHAT_MODEL,
                    "messages": [
                        {"role": "user", "content": (
                            "Extract entities from this text. Return ONLY a JSON array "
                            "of objects with 'text' and 'type' fields.\n"
                            "Text: 小明用Python写了一款VSCode插件。\n"
                            "Example: [{\"text\": \"Python\", \"type\": \"TECHNOLOGY\"}, "
                            "{\"text\": \"小明\", \"type\": \"PERSON\"}]"
                        )},
                    ],
                    "max_tokens": 128,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            entity_content = resp.json()["choices"][0]["message"]["content"]
            print_ok(f"实体抽取测试完成 — 返回: '{entity_content[:100]}'")

            print_ok(f"vLLM Chat 全部验证通过 (模型: {VLLM_CHAT_MODEL})")
            return True

    except Exception as e:
        print_fail(f"vLLM Chat 验证失败: {e}")
        return False

# Timeout for vLLM chat
VLLM_CHAT_TIMEOUT = int(os.getenv("VLLM_CHAT_TIMEOUT", "60"))


# ============================================================
# 4. Ollama 回退验证
# ============================================================
def verify_ollama(quick: bool = False):
    print_header("4. Ollama 回退验证")

    if httpx is None:
        print_fail("httpx 未安装")
        return False

    all_ok = True

    try:
        with httpx.Client(timeout=30) as client:
            # bge-m3 embedding
            resp = client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": "测试消息"},
            )
            resp.raise_for_status()
            emb = resp.json().get("embedding", [])
            if len(emb) == VERIFY_DIM:
                print_ok(f"Ollama bge-m3 embedding 维度正确: {len(emb)}")
            else:
                print_fail(f"Ollama bge-m3 维度错误: {len(emb)}")
                all_ok = False

            if not quick:
                # qwen2.5 chat
                resp = client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_CHAT_MODEL,
                        "prompt": "回复'OK'",
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                response_text = resp.json().get("response", "")
                if response_text.strip():
                    print_ok(f"Ollama {OLLAMA_CHAT_MODEL} 生成成功")
                else:
                    print_fail(f"Ollama {OLLAMA_CHAT_MODEL} 返回空")
                    all_ok = False

        return all_ok

    except Exception as e:
        print_fail(f"Ollama 验证失败: {e}")
        return False


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 基础设施连通性验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/verify_infra.py                   # 全部验证
  python scripts/verify_infra.py --qdrant-only     # 仅 Qdrant
  python scripts/verify_infra.py --vllm-only       # 仅 vLLM
  python scripts/verify_infra.py --ollama          # 仅 Ollama
  python scripts/verify_infra.py --quick           # 跳过耗时测试
        """,
    )
    parser.add_argument("--qdrant-only", action="store_true",
                        help="仅验证 Qdrant")
    parser.add_argument("--vllm-only", action="store_true",
                        help="仅验证 vLLM (embed + chat)")
    parser.add_argument("--ollama", action="store_true",
                        help="验证 Ollama 回退")
    parser.add_argument("--quick", action="store_true",
                        help="跳过耗时测试 (Ollama chat)")

    args = parser.parse_args()

    results: dict[str, bool] = {}
    t0 = time.time()

    if args.qdrant_only:
        results["Qdrant"] = verify_qdrant()
    elif args.vllm_only:
        results["vLLM Embed"] = verify_vllm_embed()
        results["vLLM Chat"] = verify_vllm_chat()
    elif args.ollama:
        results["Ollama"] = verify_ollama(quick=args.quick)
    else:
        # 全量验证
        results["Qdrant"] = verify_qdrant()
        results["vLLM Embed"] = verify_vllm_embed()
        results["vLLM Chat"] = verify_vllm_chat()
        results["Ollama"] = verify_ollama(quick=args.quick)

    elapsed = time.time() - t0

    # ── 汇总 ──
    print_header("验证汇总")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        if ok:
            print_ok(f"{name}: 通过")
        else:
            print_fail(f"{name}: 失败")

    print_info(f"耗时: {elapsed:.1f}s | {passed}/{total} 项通过")

    if passed == total:
        print_ok("🎉 全部验证通过！基础设施就绪，可以进入 Phase 1。")
        return 0
    else:
        print_fail("⚠️  部分验证失败。请检查以上服务是否已启动。")
        print_info("  docker compose up -d qdrant vllm-embed vllm-chat")
        return 1


if __name__ == "__main__":
    sys.exit(main())
