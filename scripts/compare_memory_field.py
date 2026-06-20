#!/usr/bin/env python3
# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
"""
M@q 记忆场 vs 原 memory_1~5 端到端对比。

对比维度:
  1. 轨迹差异 — M@q 残差向量 vs memory_1~5 残差向量的层分布
  2. 幅度对比 — 各层 L2 norm 对比
  3. 语义方向 — M@q 方向 vs memory 质心方向
  4. 如果有本地 LLM → 实际生成对比

用法:
  python scripts/compare_memory_field.py              # 轨迹对比
  python scripts/compare_memory_field.py --live       # 需要 LOCAL_LLM_MODE=true
"""

import argparse
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np


def build_trajectory_mfield(spec) -> np.ndarray:
    """使用 M@q memory_field 模块构建 trajectory。"""
    from app.llm.steering_direct import build_steering_trajectory
    return build_steering_trajectory(spec, global_strength=1.0)


def build_trajectory_no_memory(spec) -> np.ndarray:
    """去掉 memory_field 模块，只保留其他 10 个模块。"""
    from app.llm.steering_direct import (
        build_steering_trajectory,
        MODULE_DIRECT_CONFIG,
    )
    # 临时移除 memory_field 模块
    saved = list(MODULE_DIRECT_CONFIG)
    MODULE_DIRECT_CONFIG.clear()
    for cfg in saved:
        if cfg.extractor != "memory_field":
            MODULE_DIRECT_CONFIG.append(cfg)
    try:
        traj = build_steering_trajectory(spec, global_strength=1.0)
    finally:
        MODULE_DIRECT_CONFIG.clear()
        MODULE_DIRECT_CONFIG.extend(saved)
    return traj


def build_trajectory_no_steering(spec) -> np.ndarray:
    """完全无 steering 的零 trajectory（对照基线）。"""
    return np.zeros((28, 3584), dtype=np.float32)


def analyze_trajectory(traj: np.ndarray, label: str):
    """分析 trajectory 的层分布。"""
    n_layer, n_embd = traj.shape
    layer_norms = np.linalg.norm(traj, axis=1)  # [n_layer]
    total_norm = np.linalg.norm(traj)
    active_layers = int(np.sum(layer_norms > 1e-6))

    print(f"\n  [{label}]")
    print(f"    shape: {traj.shape}")
    print(f"    total L2 norm: {total_norm:.4f}")
    print(f"    active layers: {active_layers}/{n_layer}")
    print(f"    top-5 layers by norm:")
    top5 = np.argsort(-layer_norms)[:5]
    for l in top5:
        print(f"      L{l+1:2d}: {layer_norms[l]:.4f}")


def compare_scenarios():
    """对比三种配置的 trajectory。"""
    from app.core.state import UtteranceSpec, UserMessageAnalysis

    scenarios = [
        ("编程学习", "Python 的列表推导式学了三天还是不太会用，有没有好的学习方法？"),
        ("情绪低落", "最近工作压力好大，感觉自己什么都做不好，很沮丧"),
        ("日常聊天", "今天天气很好，想去公园散步"),
    ]

    print("=" * 70)
    print("  M@q 记忆场 vs 无记忆 vs 无steering — Trajectory 对比")
    print("=" * 70)

    for scene_name, msg in scenarios:
        print(f"\n{'─'*70}")
        print(f"  场景: {scene_name}")
        print(f"  消息: \"{msg}\"")
        print(f"{'─'*70}")

        spec = UtteranceSpec(user=UserMessageAnalysis(raw_text=msg))

        # 三种 trajectory
        traj_mfield = build_trajectory_mfield(spec)
        traj_no_mem = build_trajectory_no_memory(spec)
        traj_zero = build_trajectory_no_steering(spec)

        analyze_trajectory(traj_mfield, "M@q 记忆场 (11 modules)")
        analyze_trajectory(traj_no_mem, "无记忆场 (10 modules)")
        analyze_trajectory(traj_zero, "无steering (0 modules)")

        # 关键指标: memory_field 模块的贡献
        memory_contribution = traj_mfield - traj_no_mem
        mem_norm = np.linalg.norm(memory_contribution)
        total_norm = np.linalg.norm(traj_mfield)
        ratio = mem_norm / total_norm * 100 if total_norm > 0 else 0

        print(f"\n  >> memory_field 模块贡献: L2={mem_norm:.4f} ({ratio:.1f}% of total)")

        # memory_field 贡献的层分布
        mem_layer_norms = np.linalg.norm(memory_contribution, axis=1)
        active_mem_layers = np.sum(mem_layer_norms > 1e-6)
        print(f"  >> memory_field 活跃层: {active_mem_layers}/28 (L5-L12, gradient_down)")

        # 关键: M@q 方向 vs 其他模块的质心
        mem_vec = memory_contribution.ravel()
        other_vec = traj_no_mem.ravel()
        mem_unit = mem_vec / (np.linalg.norm(mem_vec) + 1e-10)
        other_unit = other_vec / (np.linalg.norm(other_vec) + 1e-10)
        cos_mem_other = float(np.dot(mem_unit, other_unit))

        print(f"  >> cos(memory_field, other_modules) = {cos_mem_other:.4f}")
        print(f"     {'[OK] 方向正交/互补' if abs(cos_mem_other) < 0.5 else '[INFO] 方向部分重叠'}")

    # ── 多场景一致性 ──
    print(f"\n{'='*70}")
    print("  跨场景 M@q 向量稳定性")
    print(f"{'='*70}")

    messages = [
        "Python编程学习中遇到的困难",
        "今天心情很好和朋友聊了很多",
        "Rust borrow checker怎么理解",
        "最近想学做饭但不知道从哪开始",
    ]

    R_vectors = []
    for msg in messages:
        spec = UtteranceSpec(user=UserMessageAnalysis(raw_text=msg))
        traj_full = build_trajectory_mfield(spec)
        traj_no_m = build_trajectory_no_memory(spec)
        R = (traj_full - traj_no_m).ravel()  # memory_field 贡献
        R_vectors.append(R)

    print(f"\n  跨场景 memory_field 向量余弦相似度矩阵:")
    print(f"  {'':20s}", end="")
    for i, msg in enumerate(messages):
        print(f"  [{i}]", end="")
    print()

    for i, (msg_i, Ri) in enumerate(zip(messages, R_vectors)):
        print(f"  [{i}] {msg_i[:18]:18s}", end="")
        for j, Rj in enumerate(R_vectors):
            cos_ij = float(np.dot(Ri, Rj) / (np.linalg.norm(Ri) * np.linalg.norm(Rj) + 1e-10))
            print(f"  {cos_ij:+.3f}", end="")
        print()

    # 计算平均跨场景相似度（对角线除外）
    n = len(R_vectors)
    cross_sims = []
    for i in range(n):
        for j in range(i+1, n):
            cos_ij = float(np.dot(R_vectors[i], R_vectors[j]) /
                          (np.linalg.norm(R_vectors[i]) * np.linalg.norm(R_vectors[j]) + 1e-10))
            cross_sims.append(cos_ij)
    avg_cross = np.mean(cross_sims) if cross_sims else 0
    print(f"\n  平均跨场景 cos: {avg_cross:.4f}")
    if avg_cross < 0.5:
        print(f"  [OK] 不同场景产生明显不同的记忆场方向 — M@q 有区分度")
    elif avg_cross < 0.9:
        print(f"  [INFO] 跨场景有一定区分度但不高 — N=65 的限制")
    else:
        print(f"  [WARN] 跨场景方向几乎相同 — N 太小导致区分度不足")


def main():
    parser = argparse.ArgumentParser(description="M@q 记忆场对比")
    parser.add_argument("--live", action="store_true", help="实际 LLM 生成对比 (需 LOCAL_LLM_MODE=true)")
    args = parser.parse_args()

    compare_scenarios()

    if args.live:
        from app.config.settings import LOCAL_LLM_MODE
        if not LOCAL_LLM_MODE:
            print("\n[SKIP] LOCAL_LLM_MODE=false, 跳过 live 生成对比")
            print("  设置 LOCAL_LLM_MODE=true 并确保 llama.cpp server 运行后再试")
            return

        # TODO: live generation comparison
        print("\n[INFO] live 生成对比待实现")

    print("\n" + "=" * 70)
    print("  对比完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
