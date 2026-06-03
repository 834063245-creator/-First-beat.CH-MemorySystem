"""ChuchenBrain — 初痕智能引擎

用三个小模型替代 circuit.py 中的规则决策模块：
  - IntentClassifier   → 替代 keyword_match()（回路①意图）
  - EmotionAnalyzer    → 替代 _keyword_emotion()（回路①情绪）
  - GateDecisionMaker  → 替代 basal_ganglia_gate()（回路④门控）

当前阶段：影子模式。模型未就绪时自动回退到原版规则。
"""

from app.brain.models import (
    IntentClassifier,
    EmotionAnalyzer,
    GateDecisionMaker,
    ChuchenBrain,
)

__all__ = [
    "IntentClassifier",
    "EmotionAnalyzer",
    "GateDecisionMaker",
    "ChuchenBrain",
]
