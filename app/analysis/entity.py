"""实体抽取模块 — 基于 jieba 本地 POS 标注，零延迟零费用。

入库时从对话文本中抽取命名实体（人名、地名、组织名）和关键词。
不依赖任何外部 API，Ollama 不可用时不受影响。
"""

import json
import logging
import re

import jieba
import jieba.analyse
import jieba.posseg as pseg

logger = logging.getLogger(__name__)

ENTITY_TYPES = ["PERSON", "LOCATION", "ORGANIZATION", "AMOUNT", "KEYWORD"]


def extract_entities(text: str) -> list[dict]:
    """从文本中抽取命名实体和关键词。

    返回实体列表，格式为 [{"text": str, "type": str}, ...]。
    零 API 调用，纯 jieba 本地处理，永不失败。
    """
    if not text or not text.strip():
        return []

    entities = []

    # 1. 人名（nr）、地名（ns）、组织名（nt）
    words = pseg.cut(text)
    for w, flag in words:
        w = w.strip()
        if len(w) < 2:
            continue
        if flag == "nr":
            entities.append({"text": w, "type": "PERSON"})
        elif flag == "ns":
            entities.append({"text": w, "type": "LOCATION"})
        elif flag == "nt":
            entities.append({"text": w, "type": "ORGANIZATION"})

    # 2. 金额/数字+单位
    for m in re.finditer(r'(\d+[\.\d]*)\s*(万|亿|元|块|美元|欧元|日|天|小时|分钟|岁)', text):
        entities.append({"text": m.group(0).strip(), "type": "AMOUNT"})

    # 3. 关键词（TF-IDF top 5）
    try:
        tags = jieba.analyse.extract_tags(text, topK=5, withWeight=False)
        for tag in tags:
            if len(tag) >= 2 and not any(e["text"] == tag for e in entities):
                entities.append({"text": tag, "type": "KEYWORD"})
    except Exception:
        pass

    # 去重（同文本同类型只保留一条）
    seen = set()
    unique = []
    for e in entities:
        key = (e["text"], e["type"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique
