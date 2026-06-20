# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: c4a7b2d1

"""
Trajectory 标定实验 — 逐模块扫参找最优 shape + alpha。

用法:
  # 扫单个模块的所有 shape（使用模拟 UtteranceSpec，不加载模型）
  python scripts/calibrate_trajectory.py --module gate_tone --dry-run

  # 扫单个模块，实际加载模型生成回复
  python scripts/calibrate_trajectory.py --module gate_tone --live

  # 扫所有高优先级模块
  python scripts/calibrate_trajectory.py --priority high --dry-run

  # 扫 alpha（固定 shape）
  python scripts/calibrate_trajectory.py --module portrait_emotion --scan-alpha --live

设计:
  dry-run 模式只检查 trajectory 产出是否合理（shape/alpha 变化在数值上是否可区分），
  不需要加载 GGUF 模型。适合快速迭代。
  live 模式加载模型实际生成，用于最终确认。
"""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# 测试场景
# ═══════════════════════════════════════════════════════════════════

SCENARIOS = {
    "tech_question": {
        "user_message": "微服务拆得太碎了，现在调用链很深，排查一个问题要跳四五个服务，怎么办",
        "emotion": "困扰",
        "emotion_intensity": 0.4,
        "trust": 0.7,
        "mode": "collaborator",
        "tone": "direct",
        "formality": 0.5,
        "drift": "",
        "impulse_target": "",
        "memories": [
            {"summary": "用户之前提到在做微服务架构重构"},
            {"summary": "用户关注系统可观测性"},
        ],
    },
    "emotional_sharing": {
        "user_message": "今天真的累了，Rust搞了一周感觉还在原地，是不是我不适合编程",
        "emotion": "沮丧",
        "emotion_intensity": 0.7,
        "trust": 0.6,
        "mode": "casual",
        "tone": "caring",
        "formality": 0.2,
        "drift": "偏移: drift_放弃(-40%) 用户想放弃",
        "impulse_target": "编程信心重建",
        "memories": [
            {"summary": "用户06/15说Rust太难了想放弃"},
            {"summary": "用户05/28说Python写起来确实舒服"},
            {"summary": "用户05/12架构重构搞了一周终于差不多了"},
        ],
    },
    "casual_chat": {
        "user_message": "周末有什么好玩的地方推荐吗",
        "emotion": "期待",
        "emotion_intensity": 0.3,
        "trust": 0.5,
        "mode": "casual",
        "tone": "warm",
        "formality": 0.2,
        "drift": "",
        "impulse_target": "",
        "memories": [
            {"summary": "用户之前说喜欢户外活动"},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════
# 构建 mock UtteranceSpec
# ═══════════════════════════════════════════════════════════════════

def build_spec(scenario: dict) -> MagicMock:
    spec = MagicMock()
    spec.portrait_stable = "用户是程序员，偏好Rust和Python"
    spec.portrait_dynamic = f"用户当前情绪：{scenario['emotion']}"
    spec.personality_notes_ai = [{"content": "擅长技术问题分析"}]
    spec.personality_notes = [
        {"content": "用户关注：架构设计、技术栈选型"},
    ]

    user = MagicMock()
    user.emotion = scenario["emotion"]
    user.emotion_intensity = scenario["emotion_intensity"]
    spec.user = user

    spec.memories = scenario["memories"]

    rel = MagicMock()
    rel.trust = scenario["trust"]
    rel.closeness = 0.4
    rel.familiarity = 0.3
    rel.interaction_mode = scenario["mode"]
    spec.relationship = rel

    gate = MagicMock()
    gate.tone = scenario["tone"]
    gate.formality = scenario["formality"]
    spec.gate = gate

    spec.drift_text = scenario["drift"]
    spec.self_mirror_text = ""

    if scenario["impulse_target"]:
        imp = MagicMock()
        imp.target_concept = scenario["impulse_target"]
        spec.impulses = [imp]
    else:
        spec.impulses = []

    spec.mirror_prediction = {"next_intents": ["casual"]}
    return spec


# ═══════════════════════════════════════════════════════════════════
# Dry-run: 检查 trajectory 数值特征
# ═══════════════════════════════════════════════════════════════════

SHAPES = ["uniform", "gradient_up", "gradient_down", "early", "late",
          "peak:12", "peak:20:5"]
ALPHAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]


def analyze_trajectory(trajectory: np.ndarray, label: str = ""):
    """分析 trajectory 数值特征。"""
    layer_norms = np.linalg.norm(trajectory, axis=1)
    active_layers = int(np.sum(layer_norms > 1e-6))
    max_norm = float(np.max(layer_norms))
    mean_norm = float(np.mean(layer_norms[layer_norms > 1e-6])) if active_layers > 0 else 0.0
    top5 = np.argsort(layer_norms)[-5:][::-1]

    info = {
        "label": label,
        "active_layers": active_layers,
        "max_norm": round(max_norm, 6),
        "mean_norm": round(mean_norm, 6),
        "top5_layers": [(int(i) + 1, round(float(layer_norms[i]), 6)) for i in top5],
    }
    return info


def dry_run_module(module_name: str):
    """对单个模块，用 3 个场景 × 7 shape 扫一遍，对比 trajectory 差异。"""
    from app.llm.steering_direct import (
        MODULE_DIRECT_CONFIG, _EXTRACTOR_REGISTRY, ConceptVectorBuilder,
        TrajectoryShaper,
    )

    cvb = ConceptVectorBuilder()

    # 找到模块配置
    cfg = None
    for c in MODULE_DIRECT_CONFIG:
        if c.name == module_name:
            cfg = c
            break
    if cfg is None:
        print(f"模块 '{module_name}' 不在 MODULE_DIRECT_CONFIG 中")
        print(f"可用模块: {[c.name for c in MODULE_DIRECT_CONFIG]}")
        return

    extractor_fn = _EXTRACTOR_REGISTRY.get(cfg.extractor)
    if extractor_fn is None:
        print(f"提取器 '{cfg.extractor}' 未注册")
        return

    print(f"\n{'='*60}")
    print(f"模块: {cfg.name}  提取器: {cfg.extractor}")
    print(f"当前配置: shape={cfg.shape}, alpha={cfg.alpha}, layers=[{cfg.layer_start}-{cfg.layer_end}]")
    print(f"{'='*60}")

    results = {}

    for scenario_name, scenario in SCENARIOS.items():
        spec = build_spec(scenario)
        try:
            base_vec = extractor_fn(spec, cvb)
        except Exception as e:
            print(f"  {scenario_name}: 提取失败 - {e}")
            continue

        if base_vec is None:
            print(f"  {scenario_name}: 无产出（该场景下模块无激活）")
            continue

        base_norm = float(np.linalg.norm(base_vec))
        print(f"\n  [{scenario_name}] base_vec norm={base_norm:.4f}, "
              f"message=\"{scenario['user_message'][:40]}...\"")

        shape_results = {}
        for shape in SHAPES:
            traj = TrajectoryShaper.expand(base_vec, shape, cfg.alpha, 28)
            info = analyze_trajectory(traj, f"{scenario_name}/{shape}")
            shape_results[shape] = info
            print(f"    {shape:16s}: active={info['active_layers']:2d}/28, "
                  f"max={info['max_norm']:.4f}, "
                  f"top3={[l for l, _ in info['top5_layers'][:3]]}")

        results[scenario_name] = {
            "base_norm": base_norm,
            "shapes": shape_results,
        }

    # 总结：跨场景的 shape 稳定性
    print(f"\n  ── 跨场景一致性 ──")
    for shape in SHAPES:
        active_counts = []
        for sn in results:
            if shape in results[sn]["shapes"]:
                active_counts.append(results[sn]["shapes"][shape]["active_layers"])
        if active_counts:
            print(f"    {shape:16s}: active_layers min={min(active_counts)} max={max(active_counts)}")

    return results


def dry_run_alpha_scan(module_name: str, shape: str = "uniform"):
    """固定 shape，扫 alpha。"""
    from app.llm.steering_direct import (
        MODULE_DIRECT_CONFIG, _EXTRACTOR_REGISTRY, ConceptVectorBuilder,
        TrajectoryShaper,
    )

    cvb = ConceptVectorBuilder()

    cfg = None
    for c in MODULE_DIRECT_CONFIG:
        if c.name == module_name:
            cfg = c
            break
    if cfg is None:
        print(f"模块 '{module_name}' 未找到")
        return

    extractor_fn = _EXTRACTOR_REGISTRY.get(cfg.extractor)

    print(f"\n{'='*60}")
    print(f"Alpha Scan: {module_name}  shape={shape}")
    print(f"{'='*60}")

    for scenario_name, scenario in SCENARIOS.items():
        spec = build_spec(scenario)
        try:
            base_vec = extractor_fn(spec, cvb)
        except Exception as e:
            print(f"  {scenario_name}: 提取失败 - {e}")
            continue
        if base_vec is None:
            continue

        print(f"\n  [{scenario_name}]")
        for alpha in ALPHAS:
            traj = TrajectoryShaper.expand(base_vec, shape, alpha, 28)
            info = analyze_trajectory(traj, f"a={alpha:.2f}")
            print(f"    a={alpha:.2f}: max={info['max_norm']:.4f}, "
                  f"mean={info['mean_norm']:.4f}, active={info['active_layers']}")


# ═══════════════════════════════════════════════════════════════════
# Live 模式：加载模型实际生成（需要 GGUF）
# ═══════════════════════════════════════════════════════════════════

def live_test(module_name: str, shape_override: str = None, alpha_override: float = None):
    """加载模型，用不同 shape/alpha 实际生成回复对比。"""
    from app.llm.steering import get_steering_injector

    injector = get_steering_injector()
    if not injector.is_loaded:
        print("模型加载失败，无法运行 live 模式")
        return

    # 临时覆盖模块配置
    from app.llm import steering_direct as sd
    original_configs = list(sd.MODULE_DIRECT_CONFIG)

    scenarios_to_test = ["emotional_sharing", "tech_question"]

    for scenario_name in scenarios_to_test:
        scenario = SCENARIOS[scenario_name]
        print(f"\n{'='*60}")
        print(f"场景: {scenario_name}")
        print(f"用户: {scenario['user_message']}")
        print(f"{'='*60}")

        spec = build_spec(scenario)

        # 对照: 无 steering
        from app.config import settings
        old_enabled = settings.STEERING_ENABLED
        old_direct = settings.STEERING_DIRECT

        try:
            settings.STEERING_ENABLED = False
            settings.STEERING_DIRECT = False
            result_no = injector.generate(scenario["user_message"], spec, max_tokens=256)
            print(f"\n  [无steering]: {result_no['content'][:200]}")

            # 文本路径
            settings.STEERING_ENABLED = True
            settings.STEERING_DIRECT = False
            result_text = injector.generate(scenario["user_message"], spec, max_tokens=256)
            print(f"\n  [文本路径]: {result_text['content'][:200]}")

            # 直接向量路径
            settings.STEERING_DIRECT = True
            result_direct = injector.generate(scenario["user_message"], spec, max_tokens=256)
            print(f"\n  [直接向量]: {result_direct['content'][:200]}")
        finally:
            settings.STEERING_ENABLED = old_enabled
            settings.STEERING_DIRECT = old_direct
            sd.MODULE_DIRECT_CONFIG = original_configs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

HIGH_PRIORITY_MODULES = [
    "gate_tone", "portrait_emotion", "relationship_state",
    "portrait_identity", "impulse_signal", "drift_context",
]


def main():
    parser = argparse.ArgumentParser(description="Trajectory 标定实验")
    parser.add_argument("--module", type=str, help="要标定的模块名")
    parser.add_argument("--priority", choices=["high", "all"], help="标定优先级组")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="dry-run 模式：只检查 trajectory 数值（默认）")
    parser.add_argument("--live", action="store_true", help="live 模式：加载模型实际生成")
    parser.add_argument("--scan-alpha", action="store_true", help="扫 alpha（固定 shape）")
    parser.add_argument("--shape", type=str, default="uniform", help="alpha scan 时固定的 shape")
    parser.add_argument("--list", action="store_true", help="列出所有可用模块")
    args = parser.parse_args()

    if args.list:
        from app.llm.steering_direct import MODULE_DIRECT_CONFIG
        print("可用模块:")
        for c in MODULE_DIRECT_CONFIG:
            print(f"  {c.name:25s} extractor={c.extractor:15s} "
                  f"shape={c.shape:15s} alpha={c.alpha:.2f} layers=[{c.layer_start}-{c.layer_end}]")
        return 0

    modules = []
    if args.module:
        modules = [args.module]
    elif args.priority == "high":
        modules = HIGH_PRIORITY_MODULES
    elif args.priority == "all":
        from app.llm.steering_direct import MODULE_DIRECT_CONFIG
        modules = [c.name for c in MODULE_DIRECT_CONFIG]

    if not modules:
        parser.print_help()
        return 1

    if args.live:
        live_test(modules[0])
        return 0

    for mod in modules:
        if args.scan_alpha:
            dry_run_alpha_scan(mod, args.shape)
        else:
            dry_run_module(mod)

    print(f"\n完成。共 {len(modules)} 个模块。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
