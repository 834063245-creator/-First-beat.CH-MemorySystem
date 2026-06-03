"""ChuchenBrain 三模型架构 — ChuchuCNN 本地推理 + MiniLM/Ollama/规则

模型加载优先级：
  IntentClassifier  → ChuchuCNN(字符CNN) > MiniLM > Ollama > 规则
  EmotionAnalyzer   → ChuchuCNN(字符CNN) > MiniLM > Ollama > 规则
  GateDecisionMaker → Ollama Prompt > 规则

ChuchuCNN 约 500KB，纯 CPU 推理 <5ms，不依赖 transformers/HuggingFace。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Ollama 客户端（简单 requests 封装，零额外依赖）
import requests as _requests

# ChuchuCNN 自研模型
from app.brain.chuchu_model import ChuchuCNN
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


@dataclass
class GateResult:
    """门控决策输出。"""
    tone: str = "warm"
    formality: float = 0.3
    response_mode: str = "auto"
    intimacy: float = 0.0
    suppression_reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "rule"


@dataclass
class ShadowResult:
    """影子对比结果 — 规则侧和模型侧并排数据。"""
    user_message: str
    rule_intent: IntentResult
    model_intent: IntentResult
    rule_emotion: EmotionResult
    model_emotion: EmotionResult
    rule_gate: GateResult
    model_gate: GateResult

    @property
    def intent_match(self) -> bool:
        return self.rule_intent.intent == self.model_intent.intent

    @property
    def emotion_match(self) -> bool:
        return self.rule_emotion.primary == self.model_emotion.primary


# ═══════════════════════════════════════════════════════
# 意图分类器
# ═══════════════════════════════════════════════════════

class IntentClassifier:
    """意图分类器 — ChuchuCNN + MiniLM + Ollama + 规则四层兜底。

    优先级：ChuchuCNN(字符CNN, 500KB) > MiniLM(90MB) > Ollama > 规则
    """

    _BRAIN_DIR = os.path.dirname(__file__)
    CHUCHU_PATH = os.path.join(_BRAIN_DIR, "model_intent", "chuchu_cnn.pt")
    CHUCHU_TOK = os.path.join(_BRAIN_DIR, "chuchu_tok.json")
    MINILM_PATH = os.path.join(_BRAIN_DIR, "model_intent")  # MiniLM 目录

    LABELS = [
        "recall", "emotional_sharing", "conflict",
        "ask_fact", "request", "meta", "casual",
    ]

    _INTENT_KEYWORDS = {
        "conflict": ["不对", "不是", "你错了", "别说了", "你搞错了", "乱说",
                     "你没听懂", "不是这样"],
        "emotional_sharing": ["想", "觉得", "感觉", "心情", "难过", "开心", "烦",
                              "累", "困", "疲惫", "焦虑", "担心", "感动", "温暖",
                              "梦到", "失眠", "心疼", "好烦", "好累", "好开心", "好难过"],
        "recall": ["记得", "之前", "上次", "以前", "曾经", "想起", "是不是说过",
                   "那时候", "那会儿", "还记得", "记不记得"],
        "request": ["帮我", "请你", "需要你", "帮我查", "帮我找", "帮我写",
                    "帮我改", "帮我看看", "能不能帮我"],
        "ask_fact": ["什么", "怎么", "为什么", "如何", "能不能", "请问", "啥", "是不是"],
        "meta": ["你是谁", "你能做什么", "你会什么", "你叫什么", "你有什么功能", "你是吗"],
    }

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

        # MiniLM（旧的）
        self._local_ok = False
        self._model = None
        self._tokenizer = None

        # ChuchuCNN（新的）
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

    def _load_minilm(self) -> bool:
        if self._local_ok and self._model is not None:
            return True
        if not os.path.exists(os.path.join(self.MINILM_PATH, "config.json")):
            return False
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._tokenizer = AutoTokenizer.from_pretrained(self.MINILM_PATH)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MINILM_PATH)
            self._model.eval()
            self._local_ok = True
            logger.info("MiniLM 加载成功 (%s)", self.MINILM_PATH)
            return True
        except Exception as exc:
            logger.warning("MiniLM 加载失败: %s", exc)
            self._model = None
            self._tokenizer = None
            return False

    def load(self) -> bool:
        """加载模型：ChuchuCNN > MiniLM > Ollama。任一成功即可。"""
        if self._chuchu_ok:
            return True
        if self._load_chuchu():
            return True
        if self._local_ok:
            return True
        if self._load_minilm():
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
        """分类：ChuchuCNN > MiniLM > Ollama > 规则。"""
        if self._chuchu_ok:
            return self._chuchu_predict(text)
        if self._local_ok:
            return self._minilm_predict(text)
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

    def _minilm_predict(self, text: str) -> IntentResult:
        import torch
        try:
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, idx = probs.max(dim=-1)
            id2label = self._model.config.id2label
            label = id2label[idx.item()]
            return IntentResult(
                intent=label,
                confidence=round(confidence.item(), 4),
                source="model",
            )
        except Exception as exc:
            logger.warning("MiniLM 推理失败 (%s)，回退规则", exc)
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
        self._model = None
        self._tokenizer = None
        self._local_ok = False


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

    LABELS = ["intimate", "positive", "negative", "frustrated", "neutral"]

    _EMOTION_KEYWORDS = {
        "intimate": ["想你", "爱", "心疼", "抱", "陪", "温暖", "梦到", "亲", "在乎", "爱你", "抱抱"],
        "positive": ["开心", "高兴", "好", "棒", "喜欢", "感动", "幸福", "感谢", "太棒", "太好了", "不错", "厉害"],
        "negative": ["难过", "烦", "累", "焦虑", "担心", "生气", "讨厌", "失望", "痛苦", "崩溃",
                     "孤独", "压力", "郁闷", "烦躁"],
        "frustrated": ["烦死了", "受不了", "无语", "气死", "崩溃", "不想说了", "够了", "算了吧"],
    }

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


# ═══════════════════════════════════════════════════════
# 门控决策器
# ═══════════════════════════════════════════════════════

class GateDecisionMaker:
    """门控决策器 — Ollama + 规则兜底。"""

    _GATE_PROMPT = """只输出一行JSON，不要解释。
{{"tone":"warm/caring/direct/soft/neutral之一","formality":0到1,"response_mode":"soothe/question_first/direct_answer/confirm/auto之一","intimacy":0到1}}

当前: 意图={intent}, 情绪={emotion}"""

    def __init__(self, model_name: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self._ollama_ok = False

    def load(self) -> bool:
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

    def decide(self, intent: str, emotion: str,
               context: dict | None = None) -> GateResult:
        if self._ollama_ok and self.model_name:
            return self._llm_decide(intent, emotion, context)
        return self._rule_decide(intent, emotion)

    def _llm_decide(self, intent: str, emotion: str,
                    context: dict | None = None) -> GateResult:
        try:
            prompt = self._GATE_PROMPT.format(intent=intent, emotion=emotion)
            raw = _ollama_chat(self.model_name, prompt,
                               ollama_url=self.ollama_url, timeout=15)
            data = _parse_json_safely(raw)
            tone = data.get("tone", "warm")
            if tone not in ("warm", "caring", "direct", "soft", "neutral"):
                tone = "warm"
            mode = data.get("response_mode", "auto")
            if mode not in ("soothe", "question_first", "direct_answer", "confirm", "auto"):
                mode = "auto"
            return GateResult(
                tone=tone,
                formality=float(data.get("formality", 0.3)),
                response_mode=mode,
                intimacy=float(data.get("intimacy", 0.3)),
                confidence=0.8,
                source="model",
            )
        except Exception as exc:
            logger.warning("Gate 调用失败 (%s)，回退规则", exc)
            return self._rule_decide(intent, emotion)

    def _rule_decide(self, intent: str, emotion: str) -> GateResult:
        tone = "warm"
        formality = 0.3
        response_mode = "auto"
        intimacy = 0.3

        if intent == "emotional_sharing":
            if emotion in ("negative", "intimate", "frustrated"):
                tone = "caring"
                formality = 0.1
                response_mode = "soothe"
                intimacy = 0.6
            else:
                tone = "warm"
                response_mode = "question_first"
        elif intent == "conflict":
            tone = "soft"
            formality = 0.5
            response_mode = "confirm"
            intimacy = 0.1
        elif intent == "recall":
            tone = "direct" if emotion == "neutral" else "warm"
            response_mode = "auto"
        elif intent in ("ask_fact", "meta"):
            tone = "direct"
            formality = 0.4
            response_mode = "direct_answer"
        elif intent == "request":
            tone = "direct"
            formality = 0.3
            response_mode = "direct_answer"

        if emotion == "intimate":
            intimacy = max(intimacy, 0.7)

        return GateResult(
            tone=tone, formality=formality, response_mode=response_mode,
            intimacy=intimacy, confidence=0.8, source="rule",
        )


# ═══════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════

class ChuchenBrain:
    """初痕智能引擎 — 统一管三个模型的加载和调用。"""

    def __init__(self, model_name: str = "qwen2.5:3b",
                 ollama_url: str = "http://localhost:11434"):
        self.intent_classifier = IntentClassifier(model_name=model_name,
                                                   ollama_url=ollama_url)
        self.emotion_analyzer = EmotionAnalyzer(model_name=model_name,
                                                 ollama_url=ollama_url)
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

    def decide_gate(self, intent: str, emotion: str,
                    context: dict | None = None) -> GateResult:
        return self.gate_maker.decide(intent, emotion, context)

    def shadow_analyze(self, text: str, context: dict | None = None) -> ShadowResult:
        rule_intent = IntentResult(
            intent=self.intent_classifier._rule_classify(text),
            confidence=0.7, source="rule",
        )
        rule_emotion = self.emotion_analyzer._rule_analyze(text)
        model_intent = self.intent_classifier.predict(text)
        model_emotion = self.emotion_analyzer.analyze(text)
        rule_gate = self.gate_maker._rule_decide(rule_intent.intent, rule_emotion.primary)
        model_gate = self.gate_maker.decide(model_intent.intent, model_emotion.primary, context)
        return ShadowResult(
            user_message=text,
            rule_intent=rule_intent,
            model_intent=model_intent,
            rule_emotion=rule_emotion,
            model_emotion=model_emotion,
            rule_gate=rule_gate,
            model_gate=model_gate,
        )
