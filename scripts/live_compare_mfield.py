#!/usr/bin/env python3
# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
"""
M@q 记忆场 live 生成对比。

直接在本地 qwen2.5 模型上跑三种路径，对比生成质量：
  1. no_steering    — 纯 prompt，零残差注入
  2. direct_no_mfield — steering_direct 10 模块，去掉 memory_field
  3. direct_mfield  — steering_direct 11 模块，含 M@q 记忆场

用法:
  set LOCAL_LLM_MODE=true
  python scripts/live_compare_mfield.py                     # 3 场景 × 3 路径
  python scripts/live_compare_mfield.py --quick              # 1 场景
  python scripts/live_compare_mfield.py --seed-500           # 先扩到 500 条种子再测
  python scripts/live_compare_mfield.py --save results.json  # 保存结果
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

# 强制设置 LOCAL_LLM_MODE（必须在 import settings 之前）
os.environ["LOCAL_LLM_MODE"] = "true"
os.environ["STEERING_ENABLED"] = "true"
os.environ["STEERING_DIRECT"] = "true"


SCENARIOS = {
    "emotional_sharing": {
        "user_message": "最近工作压力好大，感觉自己做什么都不太行，有时候真想放弃了",
        "portrait_tags": ["编程", "学习"],
        "emotion": {"valence": -0.6, "arousal": 0.7},
    },
    "tech_question": {
        "user_message": "Python 的异步编程好难理解，asyncio 和 await 之间的关系到底是什么？",
        "portrait_tags": ["编程", "Python"],
        "emotion": {"valence": -0.1, "arousal": 0.3},
    },
    "casual_chat": {
        "user_message": "今天天气真不错，去公园散了会儿步，看到很多花开得很漂亮",
        "portrait_tags": ["生活", "日常"],
        "emotion": {"valence": 0.5, "arousal": 0.2},
    },
    "learning_progress": {
        "user_message": "学了三个月 Rust，borrow checker 还是经常搞不定，是不是我不太适合学系统编程",
        "portrait_tags": ["编程", "Rust", "学习"],
        "emotion": {"valence": -0.4, "arousal": 0.5},
    },
}


def seed_500_memories():
    """写入 500 条多领域合成记忆，测试 M@q 区分度。"""
    from scripts.memory_field_prototype import seed_qdrant, SEED_MEMORIES, SEED_NOISE

    # 先清空已有种子（可选），直接追加新数据
    import uuid
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client import models as qdrant_models
    from app.llm.embed import local_embed

    client = QdrantClient(path="data/qdrant")
    count_before = client.count(collection_name="memories").count
    print(f"Current count: {count_before}")

    # 扩展领域
    domains = {
        "编程": [
            "Python", "Rust", "Go", "TypeScript", "Java", "C++", "算法", "数据结构",
            "设计模式", "重构", "测试", "调试", "部署", "Docker", "Kubernetes",
            "数据库", "SQL", "NoSQL", "Redis", "消息队列", "微服务", "API设计",
            "性能优化", "并发编程", "函数式编程", "面向对象", "版本控制", "CI/CD",
            "代码审查", "技术文档", "开源项目", "技术博客", "编程竞赛", "面试准备",
            "前端框架", "后端框架", "命令行工具", "正则表达式", "编译器", "操作系统",
        ],
        "生活": [
            "做饭", "运动", "旅行", "读书", "看电影", "听音乐", "养宠物", "种花",
            "健身", "跑步", "游泳", "瑜伽", "冥想", "摄影", "画画", "写作",
            "朋友聚会", "家庭聚餐", "逛街", "网购", "整理房间", "装修", "搬家",
            "理财", "记账", "投资", "保险", "体检", "看医生", "心理咨询",
            "学乐器", "学外语", "考证", "考研", "留学", "工作面试", "跳槽",
            "加班", "休假", "通勤", "租房", "买房", "买车", "养孩子",
            "追剧", "打游戏", "桌游", "露营", "钓鱼", "烘焙", "喝咖啡", "喝茶",
        ],
        "情绪": [
            "焦虑", "抑郁", "开心", "难过", "愤怒", "恐惧", "惊喜", "平静",
            "孤独", "思念", "感激", "愧疚", "自豪", "嫉妒", "失望", "期待",
            "压力", "放松", "疲惫", "兴奋", "无聊", "满足", "迷茫", "坚定",
        ],
        "技术": [
            "机器学习", "深度学习", "自然语言处理", "计算机视觉", "强化学习",
            "神经网络", "Transformer", "GPT", "BERT", "embedding", "tokenizer",
            "GPU", "CUDA", "分布式训练", "模型部署", "推理优化", "量化",
            "数据清洗", "特征工程", "A/B测试", "推荐系统", "搜索排序",
            "知识图谱", "图神经网络", "时间序列", "异常检测", "聚类",
            "Linux", "Shell", "Vim", "VSCode", "Git", "GitHub",
            "网络协议", "HTTP", "TCP", "WebSocket", "gRPC", "GraphQL",
        ],
    }

    templates = [
        "最近在学{topic}，感觉{opinion}",
        "今天用{topic}做了一个项目，{result}",
        "{topic}真的是{opinion}，花了好多时间才搞明白",
        "关于{topic}，我有一个问题想问",
        "分享一个{topic}的小技巧：{result}",
        "最近对{topic}产生了兴趣，想深入学习一下",
        "在工作中遇到了{topic}相关的问题，{result}",
        "看了一篇关于{topic}的文章，{opinion}",
        "和朋友讨论了{topic}，他觉得{opinion}",
        "终于把{topic}学完了，{result}",
    ]

    opinions = {
        "编程": ["很有意思", "有点难", "非常实用", "需要多练习", "比想象中简单"],
        "生活": ["很放松", "很有意义", "挺累的", "很开心", "值得一试"],
        "情绪": ["很难控制", "需要调节", "很正常", "会过去的", "要学会接纳"],
        "技术": ["太强大了", "更新好快", "学不过来", "非常有趣", "很烧脑"],
    }

    results = {
        "编程": ["确实方便了很多", "效率提高了不少", "发现了很多坑", "很有意思", "学到了很多"],
        "生活": ["感觉很不错", "值得记录一下", "以后还会继续", "体验很好", "推荐给大家"],
        "情绪": ["慢慢好转了", "还在调整中", "感觉好多了", "需要时间", "有所领悟"],
        "技术": ["效果很好", "还有很多要学", "解决了实际问题", "确实强大", "踩了不少坑"],
    }

    total = 0
    batch_points = []
    BATCH_SIZE = 100

    for domain, topics in domains.items():
        for topic in topics:
            # 每个 topic 生成 1-3 条变体
            n_variants = min(3, max(1, 500 // len(topics) // len(domains) + 1))
            for vi in range(n_variants):
                tmpl = templates[(hash(topic) + vi) % len(templates)]
                op_pool = opinions.get(domain, ["挺有意思的"])
                res_pool = results.get(domain, ["效果不错"])
                opinion = op_pool[hash(topic + str(vi)) % len(op_pool)]
                result = res_pool[(hash(topic) + vi * 7) % len(res_pool)]
                text = tmpl.format(topic=topic, opinion=opinion, result=result)

                emb = local_embed(text)
                if emb is None:
                    continue

                batch_points.append(qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "document": text,
                        "tags": [domain, topic],
                        "timestamp": time.time() - (500 - total) * 3600,  # 分散时间
                    },
                ))
                total += 1

                if len(batch_points) >= BATCH_SIZE:
                    client.upsert(collection_name="memories", points=batch_points)
                    print(f"  Seeded {total} memories...")
                    batch_points = []

    if batch_points:
        client.upsert(collection_name="memories", points=batch_points)

    count_after = client.count(collection_name="memories").count
    print(f"Seeded {total} new memories, total: {count_after}")
    return count_after


def build_spec(scenario: dict):
    """构建 UtteranceSpec。"""
    from app.core.state import UtteranceSpec, UserMessageAnalysis

    return UtteranceSpec(
        user=UserMessageAnalysis(
            raw_text=scenario["user_message"],
            topics=scenario.get("portrait_tags", []),
            emotion=scenario.get("emotion", {}),
        ),
    )


def run_live_compare(scenarios: dict, model_loaded: bool, save_path: str = None):
    """运行 live 三路对比。"""
    if not model_loaded:
        print("\n[SKIP] Model not loaded. Make sure LOCAL_LLM_MODE=true and GGUF exists.")
        print(f"  QWEN_GGUF_PATH={os.getenv('QWEN_GGUF_PATH', 'NOT SET')}")
        return None

    from app.llm.steering import get_steering_injector
    from app.llm.steering_direct import MODULE_DIRECT_CONFIG, _EXTRACTOR_REGISTRY
    from app.config import settings

    injector = get_steering_injector()
    if not injector.is_loaded:
        print("\n[SKIP] Model not loaded after injector creation.")
        return None

    all_results = {}

    for name, scenario in scenarios.items():
        spec = build_spec(scenario)

        print(f"\n{'='*70}")
        print(f"  Scenario: {name}")
        print(f"  User: {scenario['user_message'][:80]}...")
        print(f"{'='*70}")

        results = {}

        # Path 1: no steering at all
        old_enabled = settings.STEERING_ENABLED
        old_direct = settings.STEERING_DIRECT
        try:
            settings.STEERING_ENABLED = False
            settings.STEERING_DIRECT = False
            t0 = time.time()
            r = injector.generate(scenario["user_message"], spec, max_tokens=200, temperature=0.7)
            results["no_steering"] = {"content": r["content"], "time": time.time() - t0}
            print(f"\n  [no_steering] ({results['no_steering']['time']:.1f}s)")
            print(f"  {'─'*60}")
            for line in results["no_steering"]["content"].strip().split("\n")[:15]:
                print(f"  {line}")

            # Path 2: direct with M@q (default config)
            settings.STEERING_ENABLED = True
            settings.STEERING_DIRECT = True
            t0 = time.time()
            r = injector.generate(scenario["user_message"], spec, max_tokens=200, temperature=0.7)
            results["direct_mfield"] = {"content": r["content"], "time": time.time() - t0}
            print(f"\n  [direct_mfield] ({results['direct_mfield']['time']:.1f}s)")
            print(f"  {'─'*60}")
            for line in results["direct_mfield"]["content"].strip().split("\n")[:15]:
                print(f"  {line}")

            # Path 3: direct WITHOUT M@q (temporarily remove memory_field)
            saved_configs = list(MODULE_DIRECT_CONFIG)
            MODULE_DIRECT_CONFIG.clear()
            for cfg in saved_configs:
                if cfg.extractor != "memory_field":
                    MODULE_DIRECT_CONFIG.append(cfg)

            settings.STEERING_ENABLED = True
            settings.STEERING_DIRECT = True
            t0 = time.time()
            r = injector.generate(scenario["user_message"], spec, max_tokens=200, temperature=0.7)
            results["direct_no_mfield"] = {"content": r["content"], "time": time.time() - t0}

            # Restore config
            MODULE_DIRECT_CONFIG.clear()
            MODULE_DIRECT_CONFIG.extend(saved_configs)

            print(f"\n  [direct_no_mfield] ({results['direct_no_mfield']['time']:.1f}s)")
            print(f"  {'─'*60}")
            for line in results["direct_no_mfield"]["content"].strip().split("\n")[:15]:
                print(f"  {line}")

        finally:
            settings.STEERING_ENABLED = old_enabled
            settings.STEERING_DIRECT = old_direct

        # Quick quality analysis
        print(f"\n  ── Quick Analysis ──")
        for label in ["no_steering", "direct_no_mfield", "direct_mfield"]:
            content = results[label]["content"]
            words = len(content)
            t = results[label]["time"]
            # 简单的共情指标
            empathy_words = ["理解", "知道", "感觉", "正常", "没关系", "加油", "相信",
                           "不容易", "辛苦", "慢慢", "一起", "坚持", "可以的", "没问题"]
            empathy_count = sum(1 for w in empathy_words if w in content)
            # 技术回复指标
            tech_words = ["代码", "函数", "编程", "Python", "Rust", "异步", "await", "asyncio",
                        "可以试试", "建议", "文档", "官方", "教程", "示例"]
            tech_count = sum(1 for w in tech_words if w in content)
            print(f"  {label:20s}: {words:3d} words, {t:.1f}s, "
                  f"empathy={empathy_count}, tech={tech_count}")

        all_results[name] = results

    if save_path:
        # 保存可读结果
        save_data = {}
        for name, results in all_results.items():
            save_data[name] = {
                k: {"content": v["content"], "time": v["time"]}
                for k, v in results.items()
            }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {save_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="M@q 记忆场 live 生成对比")
    parser.add_argument("--quick", action="store_true", help="仅 1 个场景")
    parser.add_argument("--seed-500", action="store_true", help="先扩到 500 条种子数据")
    parser.add_argument("--save", type=str, default=None, help="保存结果 JSON")
    parser.add_argument("--rebuild-m", action="store_true", help="重建 M 矩阵")
    args = parser.parse_args()

    print("=" * 70)
    print("  M@q 记忆场 Live 生成对比")
    print("=" * 70)

    # Step 1: seed 500 if requested
    if args.seed_500:
        print("\n[Step 1] Seeding 500 memories...")
        count = seed_500_memories()
        print(f"  Done: {count} memories total")

    # Step 2: rebuild M matrix
    if args.rebuild_m or args.seed_500:
        print("\n[Step 2] Rebuilding M matrix...")
        from app.llm.steering_direct import _rebuild_m_matrix, _load_m_matrix
        ok = _rebuild_m_matrix()
        M = _load_m_matrix()
        if M is not None:
            import numpy as np
            print(f"  M shape: {M.shape}, nonzero: {np.count_nonzero(M)}/{M.size}")
        print(f"  {'[OK]' if ok else '[FAIL]'}")

    # Step 3: load model and compare
    print("\n[Step 3] Loading model & running comparison...")
    print("  (this will take 1-2 minutes for model loading + generation)")

    from app.llm.steering import get_steering_injector
    t0 = time.time()
    injector = get_steering_injector()
    model_loaded = injector.is_loaded
    print(f"  Model {'loaded' if model_loaded else 'FAILED'} in {time.time() - t0:.1f}s")

    # Select scenarios
    if args.quick:
        scenarios = {"emotional_sharing": SCENARIOS["emotional_sharing"]}
    else:
        scenarios = SCENARIOS

    results = run_live_compare(scenarios, model_loaded, args.save)

    print("\n" + "=" * 70)
    print("  Live compare complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
