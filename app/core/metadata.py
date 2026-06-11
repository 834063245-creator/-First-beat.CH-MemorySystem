"""入库元数据提取 — 存储层无损索引，非检索逻辑。

⚠️ DEPRECATED (2026-06-11): 本模块未集成到任何生产路径。仅 test_metadata.py 引用。
TODO: 集成到 ChromaDB write() 路径或移入 legacy/。
"""

import logging
from datetime import datetime

from app.brain.semantic import extract_entities

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
    """语义实体抽取识别人名 + 代词，去重。Ollama 不可用时返回代词兜底。"""
    entities = extract_entities(text)
    persons = [e["text"] for e in entities if e.get("type") == "PERSON"]
    # 代词兜底
    pronouns_single = {"我", "你", "他", "她"}
    pronouns_multi = {"我们", "你们", "他们"}
    for ch in text:
        if ch in pronouns_single:
            persons.append(ch)
    for mp in pronouns_multi:
        if mp in text:
            persons.append(mp)
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
