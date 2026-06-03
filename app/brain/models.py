"""ChuchenBrain 三模型架构 — ChuchuCNN 本地推理 + 规则

模型加载优先级：
  IntentClassifier  → ChuchuCNN(字符CNN) > Ollama > 规则
  EmotionAnalyzer   → ChuchuCNN(字符CNN) > Ollama > 规则
  GateDecisionMaker → 纯规则（策略映射表）

ChuchuCNN 约 500KB，纯 CPU 推理 <5ms，不依赖 transformers/HuggingFace。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Ollama 客户端（简单 requests 封装，零额外依赖）
import requests as _requests

# ChuchuCNN 自研模型
from app.brain.chuchu_model import ChuchuCNN
from app.brain.keywords import INTENT_LABELS, INTENT_KEYWORDS, EMOTION_KEYWORDS, EMOTION_LABELS
from app.brain.chuchu_tok import ChuchuTok


def _ollama_chat(model: str, prompt: str, ollama_url: str = "http://localhost:11434",
                 timeout: int = 30) -> str:
    """调用 Ollama chat API，返回模型回复文本。"""
    resp = _requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_json_safely(raw: str) -> dict:
    """安全解析 LLM 返回的 JSON，处理 markdown 代码块等。"""
    raw = raw.strip()
    if "```" in raw:
        raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        raw = re.sub(r'\s*```$', '', raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r'\{[^{}]*\}', raw))
    if not matches:
        brace_count = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0 and start >= 0:
                    try:
                        return json.loads(raw[start:i+1])
                    except json.JSONDecodeError:
                        start = -1
                        continue
        raise ValueError(f"无法解析 JSON: {raw[:100]}")
    for m in matches:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法解析 JSON: {raw[:100]}")


# ── 数据类型 ──────────────────────────────────────────────

@dataclass
class IntentResult:
    """意图分类输出。"""
    intent: str
    confidence: float = 0.0
    source: str = "rule"


@dataclass
class EmotionResult:
    """情绪标注输出。"""
    primary: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    intensity: float = 0.0
    confidence: float = 0.0
    source: str = "rule"


class IntentClassifier:
    """意图分类器 — ChuchuCNN + Ollama + 规则三层兜底。

    优先级：ChuchuCNN(字符CNN, 500KB) > Ollama > 规则
    """

    _BRAIN_DIR = os.path.dirname(__file__)
    CHUCHU_PATH = os.path.join(_BRAIN_DIR, "model_intent", "chuchu_cnn.pt")
    CHUCHU_TOK = os.path.join(_BRAIN_DIR, "chuchu_tok.json")

    LABELS = INTENT_LABELS

    _INTENT_KEYWORDS = INTENT_KEYWORDS

    _INTENT_PROMPT = """对用户消息分类，只输出标签名。

分类规则：
- recall: 提起过去聊过的事、回忆、旧话题
- emotional_sharing: 表达情绪、感受、吐槽、抱怨、倾诉
- conflict: 否定、纠正、质疑AI说的话
- request: 要求AI做具体的事：写代码、查东西、改代码、帮忙做任务
- ask_fact: 问知识、概念、方法，但没有要求AI做具体行动
- meta: 问AI本身（你是谁、你能做什么）
- casual: 闲聊、问候、不属于以上

用户消息：{text}
标签："""

    def __init__(self, model_name: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self._ollama_ok = False

        # ChuchuCNN
        self._chuchu_ok = False
        self._chuchu_model = None
        self._chuchu_tok = None
        self._chuchu_id2label = None

    # ── 加载 ──

    def _load_chuchu(self) -> bool:
        if self._chuchu_ok and self._chuchu_model is not None:
            return True
        if not os.path.exists(self.CHUCHU_PATH):
            logger.info("ChuchuCNN 不存在: %s", self.CHUCHU_PATH)
            return False
        try:
            import torch
            ckpt = torch.load(self.CHUCHU_PATH, map_location="cpu", weights_only=False)
            self._chuchu_tok = ChuchuTok.load(self.CHUCHU_TOK)
            self._chuchu_model = ChuchuCNN(
                vocab_size=ckpt["vocab_size"],
                num_classes=ckpt["num_classes"],
            )
            self._chuchu_model.load_state_dict(ckpt["model_state_dict"])
            self._chuchu_model.eval()
            self._chuchu_id2label = {str(k): v for k, v in ckpt["id2label"].items()}
            self._chuchu_ok = True
            logger.info("ChuchuCNN 加载成功 (%s)", self.CHUCHU_PATH)
            return True
        except Exception as exc:
            logger.warning("ChuchuCNN 加载失败: %s", exc)
            self._chuchu_model = None
            self._chuchu_tok = None
            return False

    def load(self) -> bool:
        """加载模型：ChuchuCNN > Ollama。任一成功即可。"""
        if self._chuchu_ok:
            return True
        if self._load_chuchu():
            return True
        # Ollama 兜底
        if self._ollama_ok:
            return True
        if self.model_name is None:
            return False
        try:
            resp = _requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if self.model_name in models or any(
                    m.startswith(self.model_name.split(":")[0]) for m in models
                ):
                    self._ollama_ok = True
                    return True
        except Exception:
            pass
        return False

    # ── 推理 ──

    def predict(self, text: str) -> IntentResult:
        """分类：ChuchuCNN > Ollama > 规则。"""
        if self._chuchu_ok:
            return self._chuchu_predict(text)
        if self._ollama_ok and self.model_name:
            return self._prompt_classify(text)
        return IntentResult(
            intent=self._rule_classify(text),
            confidence=0.6, source="rule",
        )

    def _chuchu_predict(self, text: str) -> IntentResult:
        import torch
        try:
            ids = self._chuchu_tok.encode(text)
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self._chuchu_model(x)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence, idx = probs.max(dim=-1)
            label = self._chuchu_id2label[str(idx.item())]
            return IntentResult(
                intent=label,
                confidence=round(confidence.item(), 4),
                source="model",
            )
        except Exception as exc:
            logger.warning("ChuchuCNN 推理失败 (%s)，回退规则", exc)
            return IntentResult(
                intent=self._rule_classify(text),
                confidence=0.6, source="rule",
            )


    def _prompt_classify(self, text: str) -> IntentResult:
        try:
            raw = _ollama_chat(
                self.model_name,
                self._INTENT_PROMPT.format(text=text),
                ollama_url=self.ollama_url, timeout=15,
            )
            label = raw.strip().lower()
            if label not in self.LABELS:
                label = "casual"
            return IntentResult(intent=label, confidence=0.85, source="model")
        except Exception as exc:
            logger.warning("Ollama 调用失败 (%s)，回退规则", exc)
            return IntentResult(
                intent=self._rule_classify(text),
                confidence=0.6, source="rule",
            )

    def _rule_classify(self, text: str) -> str:
        for intent in ["conflict", "emotional_sharing", "recall",
                        "request", "ask_fact", "meta"]:
            for kw in self._INTENT_KEYWORDS[intent]:
                if kw in text:
                    return intent
        return "casual"

    def close(self):
        self._chuchu_model = None
        self._chuchu_tok = None
        self._chuchu_ok = False


# ═══════════════════════════════════════════════════════
# 情绪分析器
# ═══════════════════════════════════════════════════════

class EmotionAnalyzer:
    """情绪分析器 — ChuchuCNN + Ollama + 规则三层兜底。

    优先级：ChuchuCNN(字符CNN, 500KB) > Ollama > 规则
    """

    _BRAIN_DIR = os.path.dirname(__file__)
    CHUCHU_PATH = os.path.join(_BRAIN_DIR, "model_emotion", "chuchu_cnn.pt")
    CHUCHU_TOK = os.path.join(_BRAIN_DIR, "chuchu_tok.json")

    LABELS = EMOTION_LABELS

    _EMOTION_KEYWORDS = EMOTION_KEYWORDS

    _RUSSELL_MAP = {
        "intimate":    (0.5, 0.4),
        "positive":    (0.7, 0.5),
        "negative":    (-0.5, 0.4),
        "frustrated":  (-0.6, 0.7),
        "neutral":     (0.0, 0.1),
    }

    _EMOTION_PROMPT = """分析情绪，输出JSON：{{"primary":"标签","valence":-1到1,"arousal":0到1}}

标签只有5种：
- intimate: 亲昵/温存/想你/抱抱
- positive: 开心/高兴/满足
- negative: 难过/累/疲惫/消沉
- frustrated: 愤怒/烦/被骂/不爽/讽刺/抱怨
- neutral: 纯事实、无情绪

用户消息：{text}"""

    def __init__(self, model_name: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self._ollama_ok = False

        # ChuchuCNN
        self._chuchu_ok = False
        self._chuchu_model = None
        self._chuchu_tok = None
        self._chuchu_id2label = None

    # ── 加载 ──

    def _load_chuchu(self) -> bool:
        if self._chuchu_ok and self._chuchu_model is not None:
            return True
        if not os.path.exists(self.CHUCHU_PATH):
            logger.info("Emotion ChuchuCNN 不存在: %s", self.CHUCHU_PATH)
            return False
        try:
            import torch
            ckpt = torch.load(self.CHUCHU_PATH, map_location="cpu", weights_only=False)
            self._chuchu_tok = ChuchuTok.load(self.CHUCHU_TOK)
            self._chuchu_model = ChuchuCNN(
                vocab_size=ckpt["vocab_size"],
                num_classes=ckpt["num_classes"],
            )
            self._chuchu_model.load_state_dict(ckpt["model_state_dict"])
            self._chuchu_model.eval()
            self._chuchu_id2label = {str(k): v for k, v in ckpt["id2label"].items()}
            self._chuchu_ok = True
            logger.info("Emotion ChuchuCNN 加载成功 (%s)", self.CHUCHU_PATH)
            return True
        except Exception as exc:
            logger.warning("Emotion ChuchuCNN 加载失败: %s", exc)
            self._chuchu_model = None
            self._chuchu_tok = None
            return False

    def load(self) -> bool:
        """加载模型：ChuchuCNN > Ollama。任一成功即可。"""
        if self._chuchu_ok:
            return True
        if self._load_chuchu():
            return True
        # Ollama
        if self._ollama_ok:
            return True
        if self.model_name is None:
            return False
        try:
            resp = _requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if self.model_name in models or any(
                    m.startswith(self.model_name.split(":")[0]) for m in models
                ):
                    self._ollama_ok = True
                    return True
        except Exception:
            pass
        return False

    # ── 推理 ──

    def analyze(self, text: str) -> EmotionResult:
        """分析：ChuchuCNN > Ollama > 规则。"""
        if self._chuchu_ok:
            return self._chuchu_analyze(text)
        if self._ollama_ok and self.model_name:
            return self._prompt_analyze(text)
        return self._rule_analyze(text)

    def _chuchu_analyze(self, text: str) -> EmotionResult:
        import torch
        try:
            ids = self._chuchu_tok.encode(text)
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self._chuchu_model(x)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence, idx = probs.max(dim=-1)
            label = self._chuchu_id2label[str(idx.item())]
            v, a = self._RUSSELL_MAP.get(label, (0.0, 0.1))
            return EmotionResult(
                primary=label,
                valence=v, arousal=a,
                intensity=round(a * confidence.item(), 4),
                confidence=round(confidence.item(), 4),
                source="model",
            )
        except Exception as exc:
            logger.warning("Emotion ChuchuCNN 推理失败 (%s)，回退规则", exc)
            return self._rule_analyze(text)

    def _prompt_analyze(self, text: str) -> EmotionResult:
        try:
            raw = _ollama_chat(
                self.model_name,
                self._EMOTION_PROMPT.format(text=text),
                ollama_url=self.ollama_url, timeout=15,
            )
            data = _parse_json_safely(raw)
            primary = data.get("primary", "neutral")
            if primary not in self._RUSSELL_MAP:
                primary = "neutral"
            return EmotionResult(
                primary=primary,
                valence=float(data.get("valence", 0)),
                arousal=float(data.get("arousal", 0)),
                intensity=max(0.3, float(data.get("arousal", 0))),
                confidence=0.8,
                source="model",
            )
        except Exception as exc:
            logger.warning("Emotion Ollama 调用失败 (%s)，回退规则", exc)
            return self._rule_analyze(text)

    def _rule_analyze(self, text: str) -> EmotionResult:
        for label in ["intimate", "frustrated", "negative", "positive"]:
            for kw in self._EMOTION_KEYWORDS[label]:
                if kw in text:
                    v, a = self._RUSSELL_MAP[label]
                    intensity = 0.5
                    if "!" in text or "！" in text:
                        intensity += 0.2
                    if "很" in text or "非常" in text or "特别" in text:
                        intensity += 0.15
                    intensity = min(intensity, 1.0)
                    return EmotionResult(
                        primary=label, valence=v, arousal=a,
                        intensity=intensity, confidence=0.7, source="rule",
                    )
        return EmotionResult(primary="neutral", valence=0.0, arousal=0.1,
                             intensity=0.0, confidence=0.8, source="rule")

    def close(self):
        self._chuchu_model = None
        self._chuchu_tok = None
        self._chuchu_ok = False

class ChuchenBrain:
    """初痕智能引擎 — 统一管三个模型的加载和调用。"""

    def __init__(self, model_name: str = "qwen2.5:3b",
                 ollama_url: str = "http://localhost:11434"):
        self.intent_classifier = IntentClassifier(model_name=model_name,
                                                   ollama_url=ollama_url)
        self.emotion_analyzer = EmotionAnalyzer(model_name=model_name,
                                                 ollama_url=ollama_url)
        from app.core.circuit import GateDecisionMaker
        self.gate_maker = GateDecisionMaker(model_name=model_name,
                                             ollama_url=ollama_url)

    def load_all(self) -> dict[str, bool]:
        return {
            "intent": self.intent_classifier.load(),
            "emotion": self.emotion_analyzer.load(),
            "gate": self.gate_maker.load(),
        }

    def classify_intent(self, text: str) -> IntentResult:
        return self.intent_classifier.predict(text)

    def analyze_emotion(self, text: str) -> EmotionResult:
        return self.emotion_analyzer.analyze(text)
