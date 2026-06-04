"""训练指标持久化 — 集中式 Metrics Manager.

JSON 文件: app/brain/training_metrics.json（不存在就创建）

用法:
    from app.brain.metrics import record_training, print_latest
    record_training("intent", {...})
    print_latest()
    或命令行: python -c "from app.brain.metrics import print_latest; print_latest()"
"""

import json
import os
import threading
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(HERE, "training_metrics.json")
_lock = threading.Lock()

SCHEMA_VERSION = "1.0"


def _default():
    return {
        "schema_version": SCHEMA_VERSION,
        "models": {},
        "shadow_tests": [],
        "benchmarks": [],
    }


def _load():
    """加载 JSON，不存在则返回默认结构。"""
    if not os.path.exists(METRICS_PATH):
        return _default()
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 补全缺失的顶层 key（兼容旧版 schema）
        for key, val in _default().items():
            data.setdefault(key, val)
        return data
    except (json.JSONDecodeError, IOError):
        return _default()


def _save(data):
    """线程安全写入 JSON。"""
    with _lock:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════
# 公开 API
# ══════════════════════════════════════════════════════════════════

def record_training(model_name, metrics_dict):
    """追加一次训练记录到 models[model_name]。

    metrics_dict 应包含: val_acc, val_loss, classification_report,
    epoch_curve, num_samples, seed, epochs, batch_size, lr 等。
    """
    data = _load()
    data.setdefault("models", {}).setdefault(model_name, [])
    record = {"timestamp": datetime.now().isoformat()}
    record.update(metrics_dict)
    data["models"][model_name].append(record)
    _save(data)


def record_shadow_test(results):
    """追加影子测试结果。

    results 应包含: intent_match_rate, emotion_match_rate,
    test_messages, results 等。
    """
    data = _load()
    data.setdefault("shadow_tests", []).append({
        "timestamp": datetime.now().isoformat(),
        **results,
    })
    _save(data)


def record_benchmark(results):
    """追加基准测试结果。

    results 应包含: avg_intent_ms, avg_emotion_ms, combined_ms, messages 等。
    """
    data = _load()
    data.setdefault("benchmarks", []).append({
        "timestamp": datetime.now().isoformat(),
        **results,
    })
    _save(data)


def print_latest(model_name=None):
    """打印最新指标 — 按模型名或全部。"""
    data = _load()

    # ── 模型指标 ──
    models = data.get("models", {})
    names = [model_name] if model_name else sorted(models.keys())
    for name in names:
        records = models.get(name, [])
        if not records:
            if model_name:
                print(f"{name}: 暂无训练记录")
            continue
        r = records[-1]
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        acc = r.get("val_acc", 0)
        loss = r.get("val_loss", 0)
        print(f"{name}模型 ({ts}) — val_acc: {acc:.4f}, val_loss: {loss:.4f}")

    # ── 影子测试 ──
    sts = data.get("shadow_tests", [])
    if sts:
        st = sts[-1]
        ts = st.get("timestamp", "")[:10]
        intent = st.get("intent_match_rate", 0)
        emotion = st.get("emotion_match_rate", 0)
        print(f"影子测试 ({ts}) — 意图一致率: {intent:.0%}, 情绪一致率: {emotion:.0%}")

    # ── 基准延迟 ──
    bms = data.get("benchmarks", [])
    if bms:
        bm = bms[-1]
        ts = bm.get("timestamp", "")[:10]
        avg_intent = bm.get("avg_intent_ms", 0)
        avg_emotion = bm.get("avg_emotion_ms", 0)
        combined = bm.get("combined_ms", 0)
        print(f"基准延迟 ({ts}) — 意图: {avg_intent:.0f}ms, "
              f"情绪: {avg_emotion:.0f}ms, 合计: {combined:.0f}ms")
