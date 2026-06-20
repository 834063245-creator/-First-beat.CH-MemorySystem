# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: c4a7b2d1

"""
Trajectory 标定实验 — 逐模块扫参找最优 shape + alpha。

用法:
  # 列出所有模块
  python scripts/calibrate_trajectory.py --list

  # dry-run: 扫单个模块的所有 shape（数值分析，不需模型）
  python scripts/calibrate_trajectory.py --module gate_tone

  # dry-run: 扫 alpha（固定 shape）
  python scripts/calibrate_trajectory.py --module portrait_emotion --scan-alpha

  # live: 单模块扫 shape（加载模型实际生成，场景默认 emotional_sharing）
  python scripts/calibrate_trajectory.py --module gate_tone --live

  # live: 指定场景 + 扫 alpha
  python scripts/calibrate_trajectory.py --module gate_tone --live --scan-alpha --shape late --scenario tech_question

  # live: 全量对比（无steering / 文本 / direct 三路）
  python scripts/calibrate_trajectory.py --compare --scenario emotional_sharing

  # live: 高优先级模块逐个扫 shape
  python scripts/calibrate_trajectory.py --priority high --live

  # 保存结果到 JSON
  python scripts/calibrate_trajectory.py --module gate_tone --live --save results.json

设计:
  dry-run: 只检查 trajectory 数值特征（shape/alpha 变化是否可区分），不需模型。
  live: 加载 GGUF 模型实际生成回复，用于人工对比质量。
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
# Mock UtteranceSpec 构建
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
# Dry-run: 数值分析
# ═══════════════════════════════════════════════════════════════════

SHAPES = ["uniform", "gradient_up", "gradient_down", "early", "late",
          "peak:12:4", "peak:20:5"]
ALPHAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]


def analyze_trajectory_detail(trajectory: np.ndarray) -> dict:
    """详细分析 trajectory 的层分布特征。"""
    layer_norms = np.linalg.norm(trajectory, axis=1)
    active_layers = int(np.sum(layer_norms > 1e-6))
    max_norm = float(np.max(layer_norms))
    mean_norm = float(np.mean(layer_norms[layer_norms > 1e-6])) if active_layers > 0 else 0.0
    var_norm = float(np.var(layer_norms))
    total_norm = float(np.sum(layer_norms))

    # 强层（>50% max）
    threshold_50 = max_norm * 0.5
    strong = np.where(layer_norms > threshold_50)[0]
    strong_range = f"{int(strong[0])+1}-{int(strong[-1])+1}" if len(strong) > 0 else "none"

    # 前 5 层
    top5 = np.argsort(layer_norms)[-5:][::-1]
    top5_info = [(int(i) + 1, round(float(layer_norms[i]), 6)) for i in top5]

    # 重心（加权平均层号）
    if total_norm > 0:
        center_of_mass = float(np.sum(layer_norms * np.arange(1, len(layer_norms) + 1)) / total_norm)
    else:
        center_of_mass = 0.0

    return {
        "active_layers": active_layers,
        "max_norm": round(max_norm, 6),
        "mean_norm": round(mean_norm, 6),
        "var_norm": round(var_norm, 6),
        "total_norm": round(total_norm, 6),
        "center_of_mass": round(center_of_mass, 1),
        "strong_range": strong_range,
        "strong_count": len(strong),
        "top5_layers": top5_info,
    }


def dry_run_module(module_name: str):
    """单模块 × 3 场景 × 7 shapes 扫一遍。"""
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
        print(f"Module '{module_name}' not found. Available: {[c.name for c in MODULE_DIRECT_CONFIG]}")
        return

    extractor_fn = _EXTRACTOR_REGISTRY.get(cfg.extractor)
    if extractor_fn is None:
        print(f"Extractor '{cfg.extractor}' not registered")
        return

    print(f"\n{'='*70}")
    print(f"Module: {cfg.name}  |  extractor: {cfg.extractor}")
    print(f"Current: shape={cfg.shape}, alpha={cfg.alpha}, layers=[{cfg.layer_start}-{cfg.layer_end}]")
    print(f"{'='*70}")

    for scenario_name, scenario in SCENARIOS.items():
        spec = build_spec(scenario)
        try:
            base_vec = extractor_fn(spec, cvb)
        except Exception as e:
            print(f"  [{scenario_name}] extract failed: {e}")
            continue

        if base_vec is None:
            print(f"  [{scenario_name}] no output (module inactive)")
            continue

        base_norm = float(np.linalg.norm(base_vec))
        msg_short = scenario["user_message"][:45]
        print(f"\n  [{scenario_name}] base_norm={base_norm:.4f}  msg=\"{msg_short}...\"")

        # 表头
        print(f"    {'shape':16s} {'active':>6s} {'max':>8s} {'var':>8s} "
              f"{'CoM':>6s} {'strong':>10s} {'top3':>20s}")
        print(f"    {'-'*16} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*20}")

        for shape in SHAPES:
            traj = TrajectoryShaper.expand(base_vec, shape, cfg.alpha, 28)
            info = analyze_trajectory_detail(traj)
            top3_str = ",".join(f"L{l}" for l, _ in info["top5_layers"][:3])
            print(f"    {shape:16s} {info['active_layers']:>3d}/28 "
                  f"{info['max_norm']:>8.4f} {info['var_norm']:>8.6f} "
                  f"{info['center_of_mass']:>5.1f} {info['strong_range']:>10s} "
                  f"{top3_str:>20s}")

    # 跨场景一致性
    print(f"\n  -- Cross-scenario consistency --")
    print(f"    {'shape':16s} {'CoM(min)':>9s} {'CoM(max)':>9s} {'CoM(range)':>11s}")
    print(f"    {'-'*16} {'-'*9} {'-'*9} {'-'*11}")
    for shape in SHAPES:
        coms = []
        for sn, scenario in SCENARIOS.items():
            spec = build_spec(scenario)
            try:
                base_vec = extractor_fn(spec, cvb)
            except Exception:
                continue
            if base_vec is None:
                continue
            traj = TrajectoryShaper.expand(base_vec, shape, cfg.alpha, 28)
            info = analyze_trajectory_detail(traj)
            coms.append(info["center_of_mass"])
        if coms:
            print(f"    {shape:16s} {min(coms):>9.1f} {max(coms):>9.1f} {max(coms)-min(coms):>11.1f}")


def dry_run_alpha_scan(module_name: str, shape: str = "uniform"):
    """固定 shape，扫 alpha 范围。"""
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
        print(f"Module '{module_name}' not found")
        return

    extractor_fn = _EXTRACTOR_REGISTRY.get(cfg.extractor)

    print(f"\n{'='*70}")
    print(f"Alpha Scan: {module_name}  shape={shape}")
    print(f"{'='*70}")

    for scenario_name, scenario in SCENARIOS.items():
        spec = build_spec(scenario)
        try:
            base_vec = extractor_fn(spec, cvb)
        except Exception as e:
            print(f"  [{scenario_name}] extract failed: {e}")
            continue
        if base_vec is None:
            continue

        print(f"\n  [{scenario_name}]")
        print(f"    {'alpha':>6s} {'max_norm':>10s} {'mean_norm':>10s} "
              f"{'active':>6s} {'strong_range':>12s}")
        print(f"    {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*12}")

        for alpha in ALPHAS:
            traj = TrajectoryShaper.expand(base_vec, shape, alpha, 28)
            info = analyze_trajectory_detail(traj)
            print(f"    {alpha:>5.2f}  {info['max_norm']:>10.4f} {info['mean_norm']:>10.4f} "
                  f"{info['active_layers']:>3d}/28 {info['strong_range']:>12s}")


# ═══════════════════════════════════════════════════════════════════
# Live 模式：加载模型实际生成
# ═══════════════════════════════════════════════════════════════════

LIVE_SHAPES = ["uniform", "gradient_up", "gradient_down", "early", "late", "peak:12:4"]
LIVE_ALPHAS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]


def _override_module_config(module_name: str, **overrides):
    """临时覆盖模块配置，返回 (old_cfg, module_index)。"""
    from app.llm import steering_direct as sd

    for i, cfg in enumerate(sd.MODULE_DIRECT_CONFIG):
        if cfg.name == module_name:
            old = {
                "shape": cfg.shape,
                "alpha": cfg.alpha,
                "layer_start": cfg.layer_start,
                "layer_end": cfg.layer_end,
            }
            for key, val in overrides.items():
                setattr(cfg, key, val)
            return old, i
    return None, -1


def _restore_module_config(module_name: str, old: dict):
    """恢复模块配置。"""
    from app.llm import steering_direct as sd

    for cfg in sd.MODULE_DIRECT_CONFIG:
        if cfg.name == module_name:
            for key, val in old.items():
                setattr(cfg, key, val)
            return


def live_shape_sweep(module_name: str, scenario_name: str = "emotional_sharing",
                     shapes: list[str] | None = None):
    """单模块扫 shape — 每个 shape 实际生成回复。

    Prints all responses side-by-side for human comparison.
    """
    from app.llm.steering import get_steering_injector
    from app.config import settings

    injector = get_steering_injector()
    if not injector.is_loaded:
        print("Model not loaded. Check QWEN_GGUF_PATH.")
        return

    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIOS.keys())}")
        return

    shapes = shapes or LIVE_SHAPES
    scenario = SCENARIOS[scenario_name]
    spec = build_spec(scenario)

    print(f"\n{'='*70}")
    print(f"LIVE SHAPE SWEEP: {module_name}")
    print(f"Scenario: {scenario_name}")
    print(f"User: {scenario['user_message']}")
    print(f"Emotion: {scenario['emotion']}  Tone: {scenario['tone']}  Trust: {scenario['trust']}")
    print(f"{'='*70}")

    # 1) Baseline: no steering
    old_enabled = settings.STEERING_ENABLED
    old_direct = settings.STEERING_DIRECT
    try:
        settings.STEERING_ENABLED = False
        settings.STEERING_DIRECT = False
        t0 = time.time()
        result_no = injector.generate(scenario["user_message"], spec, max_tokens=256)
        dt_no = time.time() - t0
    finally:
        settings.STEERING_ENABLED = old_enabled
        settings.STEERING_DIRECT = old_direct

    print(f"\n  [BASELINE - no steering] ({dt_no:.1f}s)")
    print(f"  {'─'*60}")
    for line in result_no["content"].strip().split("\n"):
        print(f"  {line}")

    # 2) Sweep shapes
    results = {"baseline": {"content": result_no["content"], "time": dt_no}, "shapes": {}}

    for shape in shapes:
        old_cfg, _ = _override_module_config(module_name, shape=shape)

        try:
            settings.STEERING_ENABLED = True
            settings.STEERING_DIRECT = True
            t0 = time.time()
            result = injector.generate(scenario["user_message"], spec, max_tokens=256)
            dt = time.time() - t0
        finally:
            settings.STEERING_ENABLED = old_enabled
            settings.STEERING_DIRECT = old_direct
            if old_cfg:
                _restore_module_config(module_name, old_cfg)

        content = result["content"].strip()
        results["shapes"][shape] = {"content": content, "time": dt}

        print(f"\n  [{shape}] ({dt:.1f}s)")
        print(f"  {'─'*60}")
        for line in content.split("\n"):
            print(f"  {line}")

    return results


def live_alpha_scan(module_name: str, shape: str, scenario_name: str = "emotional_sharing"):
    """单模块扫 alpha — 固定 shape，扫 alpha 范围。"""
    from app.llm.steering import get_steering_injector
    from app.config import settings

    injector = get_steering_injector()
    if not injector.is_loaded:
        print("Model not loaded.")
        return

    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario '{scenario_name}'")
        return

    scenario = SCENARIOS[scenario_name]
    spec = build_spec(scenario)

    print(f"\n{'='*70}")
    print(f"LIVE ALPHA SCAN: {module_name}  shape={shape}")
    print(f"Scenario: {scenario_name}")
    print(f"User: {scenario['user_message']}")
    print(f"{'='*70}")

    results = {}

    for alpha in LIVE_ALPHAS:
        old_cfg, _ = _override_module_config(module_name, shape=shape, alpha=alpha)

        try:
            settings.STEERING_ENABLED = True
            settings.STEERING_DIRECT = True
            t0 = time.time()
            result = injector.generate(scenario["user_message"], spec, max_tokens=256)
            dt = time.time() - t0
        finally:
            settings.STEERING_ENABLED = settings.STEERING_ENABLED
            settings.STEERING_DIRECT = settings.STEERING_DIRECT
            if old_cfg:
                _restore_module_config(module_name, old_cfg)

        content = result["content"].strip()
        results[f"a={alpha:.2f}"] = {"content": content, "time": dt}

        print(f"\n  [α={alpha:.2f}] ({dt:.1f}s)")
        print(f"  {'─'*60}")
        for line in content.split("\n"):
            print(f"  {line}")

    return results


def live_compare(scenario_name: str = "emotional_sharing"):
    """全量对比: no steering / text / direct 三种路径。"""
    from app.llm.steering import get_steering_injector
    from app.config import settings

    injector = get_steering_injector()
    if not injector.is_loaded:
        print("Model not loaded.")
        return

    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario '{scenario_name}'")
        return

    scenario = SCENARIOS[scenario_name]
    spec = build_spec(scenario)

    print(f"\n{'='*70}")
    print(f"LIVE COMPARE: 3 paths")
    print(f"Scenario: {scenario_name}")
    print(f"User: {scenario['user_message']}")
    print(f"{'='*70}")

    results = {}

    # Path 1: no steering
    old_enabled = settings.STEERING_ENABLED
    old_direct = settings.STEERING_DIRECT
    try:
        settings.STEERING_ENABLED = False
        settings.STEERING_DIRECT = False
        t0 = time.time()
        r = injector.generate(scenario["user_message"], spec, max_tokens=256)
        results["no_steering"] = {"content": r["content"], "time": time.time() - t0}
        settings.STEERING_ENABLED = True
        settings.STEERING_DIRECT = False
        t0 = time.time()
        r = injector.generate(scenario["user_message"], spec, max_tokens=256)
        results["text_path"] = {"content": r["content"], "time": time.time() - t0}
        settings.STEERING_DIRECT = True
        t0 = time.time()
        r = injector.generate(scenario["user_message"], spec, max_tokens=256)
        results["direct"] = {"content": r["content"], "time": time.time() - t0}
    finally:
        settings.STEERING_ENABLED = old_enabled
        settings.STEERING_DIRECT = old_direct

    for label, data in results.items():
        print(f"\n  [{label}] ({data['time']:.1f}s)")
        print(f"  {'─'*60}")
        for line in data["content"].strip().split("\n"):
            print(f"  {line}")

    return results


def live_all_high_priority():
    """逐个扫高优先级模块的 shape。"""
    from app.config import settings

    HIGH = ["gate_tone", "portrait_emotion", "relationship_state"]
    all_results = {}

    for mod in HIGH:
        print(f"\n{'#'*70}")
        print(f"# MODULE: {mod}")
        print(f"{'#'*70}")
        results = live_shape_sweep(mod, "emotional_sharing")
        if results:
            all_results[mod] = results

    return all_results


# ═══════════════════════════════════════════════════════════════════
# Cross-module analysis (dry-run)
# ═══════════════════════════════════════════════════════════════════

def cross_module_analysis():
    """分析所有模块在同一场景下的层贡献分布，检测冲突。"""
    from app.llm.steering_direct import (
        MODULE_DIRECT_CONFIG, _EXTRACTOR_REGISTRY, ConceptVectorBuilder,
        TrajectoryShaper,
    )

    cvb = ConceptVectorBuilder()
    scenario = SCENARIOS["emotional_sharing"]
    spec = build_spec(scenario)

    print(f"\n{'='*70}")
    print("CROSS-MODULE LAYER CONTRIBUTION ANALYSIS")
    print(f"Scenario: emotional_sharing")
    print(f"{'='*70}")

    # 收集每个模块的 trajectory
    module_trajs = {}
    for cfg in MODULE_DIRECT_CONFIG:
        extractor_fn = _EXTRACTOR_REGISTRY.get(cfg.extractor)
        if extractor_fn is None:
            continue
        try:
            base_vec = extractor_fn(spec, cvb)
        except Exception:
            continue
        if base_vec is None:
            continue
        traj = TrajectoryShaper.expand(base_vec, cfg.shape, cfg.alpha, 28)
        # 只在配置的层范围内保留
        masked = np.zeros_like(traj)
        l0 = max(0, cfg.layer_start - 1)
        l1 = min(28, cfg.layer_end)
        masked[l0:l1] = traj[l0:l1]
        module_trajs[cfg.name] = masked

    # 每层的总 norm（各模块贡献的 L2 norm 之和）
    print(f"\n  {'Layer':>5s} {'total_norm':>10s} {'#modules':>8s}  top contributors")
    print(f"  {'-'*5} {'-'*10} {'-'*8}  {'-'*40}")

    for layer_idx in range(28):
        layer_norms = {}
        for name, traj in module_trajs.items():
            n = float(np.linalg.norm(traj[layer_idx]))
            if n > 1e-6:
                layer_norms[name] = n
        total = sum(layer_norms.values())
        top3 = sorted(layer_norms.items(), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{n}({v:.4f})" for n, v in top3)
        print(f"  L{layer_idx+1:>4d} {total:>10.4f} {len(layer_norms):>8d}  {top_str}")

    # 层间冲突检测：同层不同模块向量的余弦相似度
    print(f"\n  -- Layer co-inhabitation conflicts (cosine similarity > 0.7 = aligned, < 0 = opposing) --")
    for layer_idx in range(28):
        layer_vecs = {}
        for name, traj in module_trajs.items():
            v = traj[layer_idx]
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                layer_vecs[name] = v / n

        if len(layer_vecs) < 2:
            continue

        names = list(layer_vecs.keys())
        conflicts = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = float(np.dot(layer_vecs[names[i]], layer_vecs[names[j]]))
                if sim < 0.2:  # near-orthogonal or opposing
                    conflicts.append((names[i], names[j], sim))

        if conflicts:
            for a, b, sim in conflicts:
                print(f"  L{layer_idx+1:>4d}: {a:25s} vs {b:25s}  cos={sim:+.3f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

HIGH_PRIORITY_MODULES = [
    "gate_tone", "portrait_emotion", "relationship_state",
    "portrait_identity", "impulse_signal", "drift_context",
]


def main():
    parser = argparse.ArgumentParser(description="Trajectory Calibration Experiments")
    parser.add_argument("--module", type=str, help="Module name to calibrate")
    parser.add_argument("--priority", choices=["high", "all"], help="Priority group")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry-run: numerical analysis only (default)")
    parser.add_argument("--live", action="store_true",
                        help="Live mode: load model and generate responses")
    parser.add_argument("--compare", action="store_true",
                        help="Compare 3 paths (no-steering / text / direct)")
    parser.add_argument("--scan-alpha", action="store_true",
                        help="Scan alpha values (fix shape)")
    parser.add_argument("--shape", type=str, default="uniform",
                        help="Shape to fix when scanning alpha")
    parser.add_argument("--scenario", type=str, default="emotional_sharing",
                        choices=list(SCENARIOS.keys()),
                        help="Scenario for live testing")
    parser.add_argument("--cross-module", action="store_true",
                        help="Cross-module layer contribution analysis")
    parser.add_argument("--list", action="store_true",
                        help="List all available modules")
    parser.add_argument("--save", type=str, default="",
                        help="Save results to JSON file")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Skip dry-run (for --live without dry-run)")
    args = parser.parse_args()

    # --list
    if args.list:
        from app.llm.steering_direct import MODULE_DIRECT_CONFIG
        print("Available modules:")
        for c in MODULE_DIRECT_CONFIG:
            print(f"  {c.name:25s} extractor={c.extractor:15s} "
                  f"shape={c.shape:15s} alpha={c.alpha:.2f} layers=[{c.layer_start}-{c.layer_end}]")
        return 0

    # --cross-module
    if args.cross_module:
        cross_module_analysis()
        return 0

    # --compare (live)
    if args.compare:
        results = live_compare(args.scenario)
        if args.save and results:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {args.save}")
        return 0

    # Determine modules
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

    # Live mode
    if args.live:
        all_results = {}
        for mod in modules:
            if args.scan_alpha:
                results = live_alpha_scan(mod, args.shape, args.scenario)
            else:
                results = live_shape_sweep(mod, args.scenario)
            if results:
                all_results[mod] = results

        if args.save and all_results:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {args.save}")
        return 0

    # Dry-run mode
    for mod in modules:
        if args.scan_alpha:
            dry_run_alpha_scan(mod, args.shape)
        else:
            dry_run_module(mod)

    print(f"\nDone. {len(modules)} module(s) analyzed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
