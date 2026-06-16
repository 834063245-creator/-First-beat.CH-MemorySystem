"""实体抽取模块 — 基于 jieba 本地 POS 标注，零延迟零费用。

入库时从对话文本中抽取命名实体（人名、地名、组织名）和关键词。
不依赖任何外部 API，Ollama 不可用时不受影响。
"""

import json
import logging
import re

from app.brain.semantic import extract_tags, extract_entities as _sem_extract_entities

logger = logging.getLogger(__name__)

def extract_entities(text: str) -> list[dict]:
    """从文本中抽取命名实体和关键词。

    实体：Ollama qwen2.5:3b（不可用时降级返回 []）。
    关键词：语义层 bge-m3 KeyBERT。
    金额/数字：正则提取。
    """
    if not text or not text.strip():
        return []

    entities = []

    # 1. 命名实体（Ollama）
    try:
        ollama_entities = _sem_extract_entities(text)
        entities.extend(ollama_entities)
    except Exception:
        logger.warning("Ollama 实体抽取失败，实体匹配路径将降级为空")

    # 2. 金额/数字+单位
    for m in re.finditer(r'(\d+[\.\d]*)\s*(万|亿|元|块|美元|欧元|日|天|小时|分钟|岁)', text):
        entities.append({"text": m.group(0).strip(), "type": "AMOUNT"})

    # 3. 关键词（bge-m3 KeyBERT top 5）
    try:
        tags = extract_tags(text, topk=5)
        for tag in tags:
            if not any(e["text"] == tag for e in entities):
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
