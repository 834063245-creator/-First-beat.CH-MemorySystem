"""语义脑 — 纯 semantic.py 兼容外壳。零 CNN 依赖。"""

from app.brain.semantic import (
    classify_intent,
    analyze_emotion,
    classify_urgency,
    detect_negation,
    extract_tags,
    tokenize,
    extract_entities,
)

__all__ = [
    "classify_intent",
    "analyze_emotion",
    "classify_urgency",
    "detect_negation",
    "extract_tags",
    "tokenize",
    "extract_entities",
]
