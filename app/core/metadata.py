"""入库元数据提取 — 存储层无损索引，非检索逻辑。"""

import logging
from datetime import datetime

import jieba.posseg as pseg

from app.analysis.emotion import analyze_emotion_2d
from app.config.settings import TIME_PERIOD_MAP, TOPIC_KEYWORDS

logger = logging.getLogger(__name__)


def map_hour_to_period(hour: int) -> str:
    for (lo, hi), name in TIME_PERIOD_MAP.items():
        if lo <= hour <= hi:
            return name
    return "其他"


def extract_topics(text: str) -> list[str]:
    """预定义话题关键词匹配，返回匹配的话题名列表。"""
    text_lower = text.lower()
    matched = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                matched.append(topic)
                break
    return matched


def extract_persons(text: str) -> list[str]:
    """jieba.posseg 识别人名 + 代词，去重。"""
    words = list(pseg.cut(text))
    persons = []
    for w, flag in words:
        if flag == "nr" and len(w) >= 2:
            persons.append(w)
        elif w in {"我", "你", "他", "她", "我们", "你们", "他们"}:
            persons.append(w)
    seen = set()
    unique = []
    for p in persons:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def build_memory_metadata(user_message: str, ai_message: str, timestamp: str) -> dict:
    """构建入库元数据（时间、话题、人名等索引字段）。"""
    full_text = f"用户：{user_message}\nAI：{ai_message}"
    meta = {}
    if timestamp and ' ' in timestamp:
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            meta["date"] = dt.strftime("%Y-%m-%d")
            meta["time_period"] = map_hour_to_period(dt.hour)
        except ValueError:
            pass
    topics = extract_topics(full_text)
    if topics:
        meta["topics"] = ",".join(topics)
    persons = extract_persons(full_text)
    if persons:
        meta["persons"] = ",".join(persons)
    meta["source_type"] = "chat"
    # ── Russell 二维情感坐标 + 向后兼容离散标签 ──
    valence, arousal, category = analyze_emotion_2d(user_message)
    meta["emotion_valence"] = valence
    meta["emotion_arousal"] = arousal
    meta["emotion_valence_bin"] = category
    return meta
