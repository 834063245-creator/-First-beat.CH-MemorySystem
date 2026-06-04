"""语义层 — 零 CNN、零 jieba 的纯语义工具集。

替代：5 个 ChuchuCNN 模型 + 27 处 jieba 调用。
依赖：bge-m3 (local_embed) + qwen2.5:3b (实体抽取)。

7 个公开函数：
    extract_tags(text, topk=5)      — 语义关键词提取（KeyBERT）
    classify_intent(text)           — 意图分类（原型匹配）
    analyze_emotion(text)           — 情绪分析（原型匹配）
    detect_negation(text, word)     — 否定检测（规则）
    classify_urgency(text)          — 紧急度（启发式）
    tokenize(text)                  — BM25 分词
    extract_entities(text)          — 实体抽取（Ollama）
"""

import json as _json
import logging
import os
import re
import threading
from typing import List, Optional

import httpx
import numpy as np

from app.llm.embed import local_embed, local_embed_batch
from app.brain.keywords import NEGATION_WORDS

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════

_OLLAMA_URL = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
_OLLAMA_CHAT_MODEL = "qwen2.5:3b"

# 候选 n-gram 上限（控制 batch embedding 大小）
_MAX_NGRAM_CHARS = 200
# ═════════════════════════════════════════════════════════════

_INTENT_PROTOTYPES = {
    "recall": [
        "你还记得我之前说的吗",
        "上次我们聊到那个事情",
        "我之前提过一个项目",
        "我记得你之前说过",
        "你记不记得我们聊过",
        "之前说的那个方案怎么样了",
    ],
    "emotional_sharing": [
        "我今天心情不太好",
        "好开心啊终于搞定了",
        "我觉得好累啊",
        "最近压力好大",
        "真的好烦啊",
        "我今天太难了",
        "好难过，不知道怎么办",
        "感觉好焦虑",
    ],
    "conflict": [
        "你说的不对",
        "不是这样的",
        "你搞错了",
        "你理解错了",
        "你根本没听懂",
        "我不是这个意思",
    ],
    "ask_fact": [
        "今天天气怎么样",
        "你知道这个怎么用吗",
        "这是什么意思",
        "请问这个功能怎么用",
        "Python的GIL是什么",
        "这个命令怎么用",
    ],
    "request": [
        "帮我查一下这个",
        "帮我写一段代码",
        "能不能帮我看看这个问题",
        "帮我改一下这个",
        "帮我写一个脚本",
        "帮我查一下数据",
    ],
    "meta": [
        "你是谁开发的",
        "你能做什么事情",
        "你都有什么功能",
        "你是什么模型",
    ],
    "casual": [
        "你好",
        "今天天气不错",
        "吃了吗",
        "哈哈哈",
        "再见",
        "嗯嗯",
        "好的",
        "没什么事",
    ],
}

# ═════════════════════════════════════════════════════════════
# 情绪原型（每类 4-6 句）
# ═════════════════════════════════════════════════════════════

_EMOTION_PROTOTYPES = {
    "intimate": [
        "好想你", "抱抱", "有你真好", "想你", "亲亲", "爱",
    ],
    "positive": [
        "今天好开心", "太棒了", "心情真好", "太好了", "不错不错", "厉害",
    ],
    "negative": [
        "我好难过", "最近压力好大", "好累", "好烦", "焦虑", "心情不好",
    ],
    "frustrated": [
        "烦死了", "受不了", "无语", "气死了", "够了", "不想说了",
    ],
    "neutral": [
        "今天天气不错", "帮我查一下资料", "什么是Python", "你好",
    ],
}

# ═════════════════════════════════════════════════════════════
# 惰性原型 embedding 缓存
# ═════════════════════════════════════════════════════════════

_INTENT_PROTO: Optional[dict[str, np.ndarray]] = None
_EMOTION_PROTO: Optional[dict[str, np.ndarray]] = None
_PROTO_LOCK = threading.Lock()


def _warmup():
    """预计算所有原型 embedding（首次调用时自动触发）。"""
    _get_intent_protos()
    _get_emotion_protos()


def _compute_protos(examples: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """对每个类的原型句子取平均 embedding。"""
    result = {}
    for label, texts in examples.items():
        embs = local_embed_batch(texts)
        valid = [np.array(e, dtype=np.float32) for e in embs if e is not None]
        if valid:
            result[label] = np.mean(valid, axis=0)
    if result:
        logger.debug("原型 embedding 计算完成: %s", list(result.keys()))
    return result


def _get_intent_protos() -> dict[str, np.ndarray]:
    global _INTENT_PROTO
    if _INTENT_PROTO is not None:
        return _INTENT_PROTO
    with _PROTO_LOCK:
        if _INTENT_PROTO is not None:
            return _INTENT_PROTO
        _INTENT_PROTO = _compute_protos(_INTENT_PROTOTYPES)
        return _INTENT_PROTO


def _get_emotion_protos() -> dict[str, np.ndarray]:
    global _EMOTION_PROTO
    if _EMOTION_PROTO is not None:
        return _EMOTION_PROTO
    with _PROTO_LOCK:
        if _EMOTION_PROTO is not None:
            return _EMOTION_PROTO
        _EMOTION_PROTO = _compute_protos(_EMOTION_PROTOTYPES)
        return _EMOTION_PROTO


# ═════════════════════════════════════════════════════════════
# 否定检测白名单
# ═════════════════════════════════════════════════════════════

_NEGATION_WHITELIST = {
    "还不错", "不赖", "不简单", "不一般",
    "了不起", "不得了", "说不定", "差不多",
    "不由得", "不至于", "不怎么",
    "没什么", "没关系", "没事", "没问题",
    "受不了", "挡不住", "忍不住",
    "吃不下", "睡不着", "放不下", "停不下来",
    "不错",  # 常见正面表达
}

# ═════════════════════════════════════════════════════════════
# 公开 API
# ═════════════════════════════════════════════════════════════

def extract_tags(text: str, topk: int = 5) -> list[str]:
    """语义关键词提取（KeyBERT 思路，bge-m3 嵌入）。

    候选：汉字段内 2-6 字子串 + 完整英文 token。
    产出独立完整的词/短语，可被 InvertedIndex 精确匹配。
    Ollama 不可用时返回 []。
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    segments = re.findall(
        r'[\u4e00-\u9fff]+|[a-zA-Z_][a-zA-Z0-9_\.\-]+', text
    )
    candidates = []
    for seg in segments:
        if re.match(r'^[a-zA-Z]', seg):
            if len(seg) >= 2:
                candidates.append(seg)
        else:
            for n in range(2, min(7, len(seg) + 1)):
                for i in range(len(seg) - n + 1):
                    candidates.append(seg[i:i + n])
    if not candidates:
        return []
    candidates = list(dict.fromkeys(candidates))

    to_embed = [text] + candidates
    embs = local_embed_batch(to_embed)
    if embs[0] is None:
        return []
    text_emb = np.array(embs[0], dtype=np.float32)

    scores = []
    for i, cand in enumerate(candidates):
        e = embs[i + 1] if i + 1 < len(embs) else None
        if e is None:
            continue
        sim = float(np.dot(text_emb, np.array(e, dtype=np.float32)))
        if sim >= 0.3:
            scores.append((cand, sim))
    if not scores:
        return []

    scores.sort(key=lambda x: -x[1])
    result = []
    for cand, _ in scores:
        cand = cand.lstrip('\u7684')
        if len(cand) < 2:
            continue
        # 剔除末尾语气词（吗、呢、吧）
        if cand[-1] in '\u5417\u5462\u5427':
            continue
        if any(cand in r for r in result):
            continue
        result = [r for r in result if r not in cand]
        result.append(cand)
        if len(result) >= topk:
            break
    return result[:topk]


def classify_intent(text: str) -> str:
    """意图分类 — embedding 原型匹配。

    返回: recall | emotional_sharing | conflict | ask_fact | request | meta | casual
    """
    if not text or not text.strip():
        return "casual"

    protos = _get_intent_protos()
    if not protos:
        return "casual"

    vec = local_embed(text)
    if vec is None:
        return "casual"

    emb = np.array(vec, dtype=np.float32)
    best_label = "casual"
    best_sim = 0.0
    for label, proto in protos.items():
        sim = float(np.dot(emb, proto))
        if sim > best_sim:
            best_sim = sim
            best_label = label

    if best_sim < 0.45:
        return "casual"
    return best_label


def analyze_emotion(text: str) -> str:
    """情绪分析 — embedding 原型匹配。

    返回: intimate | positive | negative | frustrated | neutral
    """
    if not text or not text.strip():
        return "neutral"

    protos = _get_emotion_protos()
    if not protos:
        return "neutral"

    vec = local_embed(text)
    if vec is None:
        return "neutral"

    emb = np.array(vec, dtype=np.float32)
    best_label = "neutral"
    best_sim = 0.0
    for label, proto in protos.items():
        sim = float(np.dot(emb, proto))
        if sim > best_sim:
            best_sim = sim
            best_label = label

    if best_sim < 0.40:
        return "neutral"
    return best_label


def detect_negation(text: str, emotion_word: str) -> bool:
    """否定检测 — 规则驱动，比 CNN 更可解释。

    规则：
      1. 扫描 emotion_word 前后 3 字内是否有否定词
      2. 白名单跳过（还不错、不错、了不起……）
      3. 双重否定模式（不是不、没有不）→ False
    """
    if not text or not emotion_word:
        return False

    # 白名单
    for wl in _NEGATION_WHITELIST:
        if wl in text:
            return False

    idx = text.find(emotion_word)
    if idx < 0:
        return False

    # 前后窗口扫描
    start = max(0, idx - 3)
    end = min(len(text), idx + len(emotion_word) + 3)
    window = text[start:end]

    # 按否定词长度降序匹配（避免"没有"被"没"先吃掉）
    neg_words = sorted(NEGATION_WORDS, key=len, reverse=True)
    matched = set()
    remaining = window
    for neg in neg_words:
        if neg in remaining:
            # 只标记在窗口内找到的否定
            matched.add(neg)
            remaining = remaining.replace(neg, " " * len(neg), 1)

    if len(matched) == 0:
        return False

    # 双重否定：两种不同的否定词出现在同一窗口
    if len(matched) >= 2:
        return False

    return True


def classify_urgency(text: str) -> float:
    """紧急度评分 0~1。纯规则，零模型。"""
    urgency = 0.0
    if "!" in text or "！" in text:
        urgency += 0.3
    if len(text) > 100:
        urgency += 0.2
    if "急" in text or "马上" in text or "立刻" in text:
        urgency += 0.4
    return min(urgency, 1.0)


def tokenize(text: str) -> list[str]:
    """BM25 分词 — 字符 2-gram + 英文 token 保留。

    示例:
      "帮我写数据库迁移脚本"
      → ["帮我", "我写", "写数", "数据", "据库", "库迁", "迁移", "移脚", "脚本"]

      "consolidation.py 的那行代码"
      → ["co", "on", ..., "consolidation.py", "的那", "那行", "行代", "代码"]
    """
    if not text:
        return []

    tokens: list[str] = []

    # 英文 token（完整保留）
    for m in re.finditer(r'[a-zA-Z_][a-zA-Z0-9_\.\-]*', text):
        token = m.group()
        if len(token) >= 2:
            tokens.append(token)

    # 字符 2-gram
    for i in range(len(text) - 1):
        bigram = text[i:i + 2]
        if bigram.strip():
            tokens.append(bigram)

    return tokens


# ── 实体抽取 Ollama prompt ─────────────────────────────────

_ENTITY_PROMPT = (
    "从文本中识别人名、项目名、工具名等实体。"
    "只返回JSON数组，格式：[{\"text\":\"...\",\"type\":\"PERSON|PROJECT|TOOL|ORG\"}]"
    "如果没有任何实体，返回[]。"
    "不要输出实体以外的内容。"
    "文本："
)


def extract_entities(text: str) -> list[dict]:
    """实体抽取 — 调用本地 Ollama qwen2.5:3b。

    返回: [{"text": "...", "type": "PERSON|PROJECT|TOOL|ORG"}, ...]
    Ollama 不可用或超时 → 返回 []。
    """
    if not text or not text.strip():
        return []

    prompt = _ENTITY_PROMPT + text

    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
            resp = client.post(
                f"{_OLLAMA_URL}/api/chat",
                json={
                    "model": _OLLAMA_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 256},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()

            # 提取 JSON 数组
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                entities = _json.loads(json_match.group())
                if isinstance(entities, list):
                    return [
                        {"text": e["text"], "type": e["type"]}
                        for e in entities
                        if isinstance(e, dict) and "text" in e and "type" in e
                    ]
    except Exception as e:
        logger.debug("实体抽取失败（Ollama 不可用或超时）: %s", e)

    return []
