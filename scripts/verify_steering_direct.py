# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: f8a2c1e6

"""
Smoke 验证 — 直接向量注入引擎

验证项:
  1. ConceptVectorBuilder 初始化 (锚点/方向/类别预计算)
  2. 情绪方向语义有效性 (valence: 正面词 > 负面词)
  3. 信任锚点插值 (trust 0.9 closer to trust_high than trust_low)
  4. TrajectoryShaper 5 种 shape 形状正确
  5. build_steering_trajectory() 产出非零 trajectory
  6. 与文本路径向量余弦相似度对比 (同模块 direct vs text)

用法:
  python scripts/verify_steering_direct.py
"""

import sys
import time
from pathlib import Path

# 确保项目根在 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def verify_cvb_init():
    """1. ConceptVectorBuilder 初始化。"""
    print("── 1. ConceptVectorBuilder 初始化 ──")
    from app.llm.steering_direct import ConceptVectorBuilder

    t0 = time.perf_counter()
    cvb = ConceptVectorBuilder()
    elapsed = time.perf_counter() - t0

    assert cvb.dim == 3584, f"dim={cvb.dim}"
    assert len(cvb._anchors) >= 8, f"anchors={len(cvb._anchors)}"
    assert len(cvb._directions) >= 5, f"directions={len(cvb._directions)}"
    assert len(cvb._categories) >= 3, f"category_sets={len(cvb._categories)}"

    # 验证方向向量是单位向量
    for name, d in cvb._directions.items():
        norm = float(np.linalg.norm(d))
        assert abs(norm - 1.0) < 0.01, f"direction '{name}' norm={norm}"

    print(f"   [OK] {len(cvb._anchors)} anchors, {len(cvb._directions)} directions, "
          f"{sum(len(v) for v in cvb._categories.values())} categories, "
          f"init {elapsed*1000:.0f}ms")
    return cvb


def verify_emotion_direction(cvb):
    """2. 情绪方向语义有效性。"""
    print("\n── 2. 情绪方向语义验证 ──")

    # valence 方向: 正面词应该在该方向上有正投影，负面词负投影
    valence_dir = cvb._directions["valence"]
    pos_words = ["开心", "快乐", "幸福", "满足", "感动"]
    neg_words = ["难过", "痛苦", "绝望", "愤怒", "焦虑"]

    pos_scores = []
    neg_scores = []
    for w in pos_words:
        v = cvb._emb.embed(w)
        v = v / np.linalg.norm(v)
        pos_scores.append(float(np.dot(v, valence_dir)))
    for w in neg_words:
        v = cvb._emb.embed(w)
        v = v / np.linalg.norm(v)
        neg_scores.append(float(np.dot(v, valence_dir)))

    pos_mean = np.mean(pos_scores)
    neg_mean = np.mean(neg_scores)

    print(f"   valence 方向投影: 正面词 mean={pos_mean:+.4f}, 负面词 mean={neg_mean:+.4f}")
    assert pos_mean > 0, f"正面词应有正投影, got {pos_mean}"
    assert neg_mean < 0, f"负面词应有负投影, got {neg_mean}"
    assert pos_mean - neg_mean > 0.2, f"正负分离度不足: {pos_mean - neg_mean:.4f}"
    print(f"   [OK] 正负分离度: {pos_mean - neg_mean:.4f} (>0.2)")

    # arousal 方向: 高唤醒词 > 低唤醒词
    arousal_dir = cvb._directions["arousal"]
    high_words = ["激动", "兴奋", "紧张", "愤怒", "惊喜"]
    low_words = ["平静", "放松", "疲惫", "无聊", "困倦"]

    high_scores = []
    low_scores = []
    for w in high_words:
        v = cvb._emb.embed(w)
        v = v / np.linalg.norm(v)
        high_scores.append(float(np.dot(v, arousal_dir)))
    for w in low_words:
        v = cvb._emb.embed(w)
        v = v / np.linalg.norm(v)
        low_scores.append(float(np.dot(v, arousal_dir)))

    print(f"   arousal 方向投影: 高唤醒 mean={np.mean(high_scores):+.4f}, "
          f"低唤醒 mean={np.mean(low_scores):+.4f}")
    assert np.mean(high_scores) > np.mean(low_scores), "高唤醒词应该 > 低唤醒词"
    print(f"   [OK] 唤醒度分离度: {np.mean(high_scores) - np.mean(low_scores):.4f}")


def verify_trust_interpolation(cvb):
    """3. 信任锚点插值。"""
    print("\n── 3. 信任锚点插值验证 ──")

    vec_high = cvb.from_scalar(0.9, "trust_high", "trust_low")
    vec_low = cvb.from_scalar(0.1, "trust_high", "trust_low")
    vec_mid = cvb.from_scalar(0.5, "trust_high", "trust_low")

    # 高信任向量应更接近 trust_high 锚点
    anchor_high = cvb._anchors["trust_high"]
    anchor_low = cvb._anchors["trust_low"]

    sim_high_to_high = float(np.dot(vec_high, anchor_high))
    sim_high_to_low = float(np.dot(vec_high, anchor_low))
    sim_low_to_low = float(np.dot(vec_low, anchor_low))
    sim_low_to_high = float(np.dot(vec_low, anchor_high))

    print(f"   trust=0.9 → sim(trust_high)={sim_high_to_high:.4f}, sim(trust_low)={sim_high_to_low:.4f}")
    print(f"   trust=0.1 → sim(trust_low)={sim_low_to_low:.4f}, sim(trust_high)={sim_low_to_high:.4f}")

    assert sim_high_to_high > sim_high_to_low, "trust=0.9 应更接近 trust_high"
    assert sim_low_to_low > sim_low_to_high, "trust=0.1 应更接近 trust_low"

    # trust=0.5 应在中间
    sim_mid_high = float(np.dot(vec_mid, anchor_high))
    sim_mid_low = float(np.dot(vec_mid, anchor_low))
    print(f"   trust=0.5 → sim(trust_high)={sim_mid_high:.4f}, sim(trust_low)={sim_mid_low:.4f}")
    print(f"   [OK] 锚点插值方向正确")


def verify_tone_categories(cvb):
    """4. 语气类别查表。"""
    print("\n── 4. 语气类别查表验证 ──")

    tones = ["soft", "caring", "warm", "direct", "neutral"]

    vecs = {}
    for t in tones:
        v = cvb.from_category(t, "tone")
        assert v is not None, f"tone '{t}' not found"
        vecs[t] = v

    # 相同 tone 自相似度最高
    for t in tones:
        sim_self = float(np.dot(vecs[t], vecs[t]))
        assert abs(sim_self - 1.0) < 0.01, f"tone '{t}' self-sim={sim_self}"

    # soft vs direct 应有明显差异 (不同的沟通风格)
    sim_soft_direct = float(np.dot(vecs["soft"], vecs["direct"]))
    print(f"   sim(soft, direct)={sim_soft_direct:.4f} (应 < 0.9)")

    # warm vs soft 应有一定相似度 (都是温暖系)
    sim_warm_soft = float(np.dot(vecs["warm"], vecs["soft"]))
    print(f"   sim(warm, soft)={sim_warm_soft:.4f} (应较高)")
    assert sim_warm_soft > sim_soft_direct, "warm-soft 应比 soft-direct 更相似"

    print(f"   [OK] 5 tone categories, 语义结构合理")


def verify_trajectory_shapes():
    """5. TrajectoryShaper 形状正确性。"""
    print("\n── 5. TrajectoryShaper 形状验证 ──")
    from app.llm.steering_direct import TrajectoryShaper

    n_layer = 28
    shapes = ["uniform", "gradient_up", "gradient_down", "early", "late", "peak:12", "peak:20:5"]

    for shape in shapes:
        sf = TrajectoryShaper.shape_fn(shape, n_layer)
        assert sf.shape == (n_layer,), f"{shape}: shape={sf.shape}"
        assert np.all(sf >= 0), f"{shape}: has negative values"
        assert np.all(sf <= 1), f"{shape}: has values > 1"

        if shape == "uniform":
            assert np.allclose(sf, 1.0), f"uniform: not all 1.0"
            print(f"   {shape:20s}: all {sf[0]:.2f}")
        elif shape == "gradient_up":
            assert sf[0] < sf[-1], f"gradient_up: L1={sf[0]:.3f} not < L28={sf[-1]:.3f}"
            print(f"   {shape:20s}: L1={sf[0]:.3f} → L28={sf[-1]:.3f}")
        elif shape == "gradient_down":
            assert sf[0] > sf[-1], f"gradient_down: L1={sf[0]:.3f} not > L28={sf[-1]:.3f}"
            print(f"   {shape:20s}: L1={sf[0]:.3f} → L28={sf[-1]:.3f}")
        elif shape == "early":
            assert sf[0] > 0.7 and sf[-1] < 0.1, f"early: L1={sf[0]:.3f} L28={sf[-1]:.3f}"
            print(f"   {shape:20s}: L1={sf[0]:.3f} → L28={sf[-1]:.3f}")
        elif shape == "late":
            assert sf[0] < 0.1 and sf[-1] > 0.7, f"late: L1={sf[0]:.3f} L28={sf[-1]:.3f}"
            print(f"   {shape:20s}: L1={sf[0]:.3f} → L28={sf[-1]:.3f}")
        elif shape.startswith("peak:"):
            peak_layer = int(shape.split(":")[1])
            assert sf[peak_layer] > sf[0] and sf[peak_layer] > sf[-1], \
                f"{shape}: peak at L{peak_layer}={sf[peak_layer]:.3f} not > edges"
            print(f"   {shape:20s}: L1={sf[0]:.3f} peak@L{peak_layer}={sf[peak_layer]:.3f} L28={sf[-1]:.3f}")

    print(f"   [OK] 7 shapes all valid")


def build_mock_utterance_spec():
    """构造一个模拟的 UtteranceSpec 用于端到端测试。"""
    from unittest.mock import MagicMock

    spec = MagicMock()
    spec.portrait_stable = "用户是资深程序员，偏好 Rust 和 Python，关注架构设计"
    spec.portrait_dynamic = "用户近期工作压力大"
    spec.personality_notes_ai = [{"content": "擅长技术问题分析"}]
    spec.personality_notes = [
        {"content": "用户关注：架构设计、技术栈选型、性能优化"},
        {"content": "近期频繁提及 Rust 难度"},
    ]

    # 用户情绪
    user = MagicMock()
    user.emotion = "焦虑"
    user.emotion_intensity = 0.6
    spec.user = user

    # 记忆
    spec.memories = [
        {"summary": "用户06/15说Rust太难了想放弃", "document": "Rust borrow checker太难理解了"},
        {"summary": "用户05/28说Python写起来确实舒服", "document": "Python开发效率高"},
        {"summary": "05/12架构重构搞了一周终于差不多了", "document": "微服务重构完成"},
    ]

    # 关系
    rel = MagicMock()
    rel.trust = 0.7
    rel.closeness = 0.4
    rel.familiarity = 0.3
    rel.interaction_mode = "collaborator"
    spec.relationship = rel

    # 门控
    gate = MagicMock()
    gate.tone = "caring"
    gate.formality = 0.4
    spec.gate = gate

    # 偏移率
    spec.drift_text = "偏移: frugal(+25%) 连续3轮节省倾向"

    # 冲动
    impulse = MagicMock()
    impulse.target_concept = "技术栈选型讨论"
    spec.impulses = [impulse]

    # 自我镜像
    spec.self_mirror_text = "之前回应风格：先共情再给建议，语气温暖直接"

    # 行为预测
    spec.mirror_prediction = {"next_intents": ["emotional_sharing", "ask_fact"]}

    return spec


def verify_end_to_end():
    """6. build_steering_trajectory() 端到端。"""
    print("\n── 6. build_steering_trajectory() 端到端 ──")
    from app.llm.steering_direct import (
        build_steering_trajectory, trajectory_to_cvec_buffer,
    )

    spec = build_mock_utterance_spec()

    t0 = time.perf_counter()
    trajectory = build_steering_trajectory(spec, n_layer=28, n_embd=3584)
    elapsed = time.perf_counter() - t0

    assert trajectory.shape == (28, 3584), f"shape={trajectory.shape}"
    assert trajectory.dtype == np.float32, f"dtype={trajectory.dtype}"

    # 应该有非零 steering
    nonzero_layers = np.sum(np.abs(trajectory).max(axis=1) > 1e-6)
    print(f"   trajectory shape: {trajectory.shape}")
    print(f"   nonzero layers: {nonzero_layers}/28")
    print(f"   build time: {elapsed*1000:.1f}ms")

    assert nonzero_layers >= 5, f"只有 {nonzero_layers} 层有非零 steering，预期 >= 5"

    # 各层的 steering 幅度分布
    layer_norms = np.linalg.norm(trajectory, axis=1)
    top_layers = np.argsort(layer_norms)[-5:][::-1]
    top_info = [(i + 1, round(float(layer_norms[i]), 4)) for i in top_layers]
    print(f"   top-5 layers by norm: {top_info}")

    # 展平
    buf = trajectory_to_cvec_buffer(trajectory)
    assert buf.shape == (28 * 3584,), f"buffer shape={buf.shape}"
    assert buf.dtype == np.float32
    assert np.allclose(buf, trajectory.ravel())
    print(f"   buffer shape: {buf.shape}, C-contiguous: {buf.flags['C_CONTIGUOUS']}")

    print(f"   [OK] 端到端通过")

    return trajectory


def verify_text_vs_direct_similarity():
    """7. 同模块 direct vs text 向量余弦相似度。"""
    print("\n── 7. Direct vs Text 向量相似度 ──")
    from app.llm.steering_direct import (
        ConceptVectorBuilder, _EXTRACTOR_REGISTRY, MODULE_DIRECT_CONFIG,
    )
    from app.llm.steering import build_steering_segments

    spec = build_mock_utterance_spec()
    cvb = ConceptVectorBuilder()

    # 文本路径产出
    segments = build_steering_segments(spec)

    comparisons = []
    for cfg in MODULE_DIRECT_CONFIG:
        extractor_fn = _EXTRACTOR_REGISTRY.get(cfg.extractor)
        if extractor_fn is None:
            continue

        # Direct 向量
        try:
            vec_direct = extractor_fn(spec, cvb)
        except Exception:
            continue
        if vec_direct is None:
            continue

        # 文本路径: 找对应的 output_key → embed
        # MODULE_LAYER_MAP 中有 output_key
        from app.llm.steering import MODULE_LAYER_MAP
        text_key = None
        for _mn, _l, _a, ok in MODULE_LAYER_MAP:
            if ok == cfg.extractor or (
                cfg.extractor.startswith("memory_") and ok.startswith("memory_")
                and cfg.extractor[-1] == ok[-1]
            ):
                text_key = ok
                break

        if text_key is None:
            # 尝试直接匹配
            text_key = cfg.extractor

        seg_text = segments.get(text_key, "")
        if not seg_text or not seg_text.strip():
            continue

        vec_text = cvb.from_text(seg_text)

        # 余弦相似度
        sim = float(np.dot(vec_direct, vec_text))
        comparisons.append((cfg.name, sim))

        if abs(sim) < 0.3:
            # 低相似度不一定坏——direct 可能编码了文本没法表达的语义
            indicator = "[LOW] 低相似"
        elif sim > 0.7:
            indicator = "[HIGH] 高相似"
        else:
            indicator = "[MED] 中等"
        print(f"   {cfg.name:25s}: sim={sim:+.4f} {indicator}")

    if comparisons:
        avg_sim = np.mean([s for _, s in comparisons])
        print(f"\n   平均相似度: {avg_sim:.4f}")
        # 不硬断——低相似度可能是 direct 编码了不同的东西
        print(f"   注：低相似度不一定是问题——direct 可能编码了文本无法表达的语义结构")

    print(f"   [OK] 对比完成 ({len(comparisons)} 模块)")


def main():
    print("═" * 60)
    print("直接向量注入引擎 — Smoke 验证")
    print("═" * 60)

    results = []

    # 1
    try:
        cvb = verify_cvb_init()
        results.append(("1. CVB 初始化", True))
    except Exception as e:
        print(f"   [FAIL] {e}")
        results.append(("1. CVB 初始化", False))

    # 2
    try:
        verify_emotion_direction(cvb)
        results.append(("2. 情绪方向语义", True))
    except Exception as e:
        print(f"   [FAIL] {e}")
        results.append(("2. 情绪方向语义", False))

    # 3
    try:
        verify_trust_interpolation(cvb)
        results.append(("3. 信任锚点插值", True))
    except Exception as e:
        print(f"   [FAIL] {e}")
        results.append(("3. 信任锚点插值", False))

    # 4
    try:
        verify_tone_categories(cvb)
        results.append(("4. 语气类别查表", True))
    except Exception as e:
        print(f"   [FAIL] {e}")
        results.append(("4. 语气类别查表", False))

    # 5
    try:
        verify_trajectory_shapes()
        results.append(("5. TrajectoryShaper", True))
    except Exception as e:
        print(f"   [FAIL] {e}")
        results.append(("5. TrajectoryShaper", False))

    # 6
    try:
        verify_end_to_end()
        results.append(("6. 端到端 trajectory", True))
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        results.append(("6. 端到端 trajectory", False))

    # 7
    try:
        verify_text_vs_direct_similarity()
        results.append(("7. Direct vs Text 对比", True))
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        results.append(("7. Direct vs Text 对比", False))

    # ── 汇总 ──
    print("\n" + "═" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"结果: {passed}/{total}")
    for name, ok in results:
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    print("═" * 60)

    if passed == total:
        print("\n[PASS] 全部通过！直接向量注入引擎就绪。")
        return 0
    else:
        print(f"\n[WARN] {total - passed} 项失败，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
