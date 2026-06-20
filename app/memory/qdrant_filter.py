# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 6758b6f2

"""旧 ChromaDB where 格式 → Qdrant Filter 翻译层 (Phase 2 遗留兼容)。

从 qdrant.py 中提取，消除模块内循环依赖。纯函数，无 app 内部依赖。
"""
from qdrant_client import models


def _build_condition(key: str, value) -> models.FieldCondition | models.Filter:
    """构建单个字段条件。"""
    if not isinstance(value, dict):
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    for op, val in value.items():
        if op == "$gte":
            return models.FieldCondition(key=key, range=models.Range(gte=val))
        elif op == "$lte":
            return models.FieldCondition(key=key, range=models.Range(lte=val))
        elif op == "$gt":
            return models.FieldCondition(key=key, range=models.Range(gt=val))
        elif op == "$lt":
            return models.FieldCondition(key=key, range=models.Range(lt=val))
        elif op == "$eq":
            return models.FieldCondition(key=key, match=models.MatchValue(value=val))
        elif op == "$ne":
            return models.Filter(must_not=[
                models.FieldCondition(key=key, match=models.MatchValue(value=val))
            ])
        elif op == "$in":
            return models.FieldCondition(key=key, match=models.MatchAny(any=val))
        elif op == "$contains":
            return models.FieldCondition(key=key, match=models.MatchText(text=str(val)))

    raise ValueError(f"Unsupported operator in: {value}")


def _translate_filter(chroma_where: dict) -> models.Filter:
    """旧 ChromaDB where dict 格式 → Qdrant Filter。

    支持的运算符: $gte, $lte, $gt, $lt, $eq, $ne, $in, $contains, $and, $or
    """
    if not chroma_where:
        return models.Filter()

    conditions = []
    for key, value in chroma_where.items():
        if key == "$and":
            sub_conditions = []
            for sub_clause in value:
                if not isinstance(sub_clause, dict):
                    continue
                for sk, sv in sub_clause.items():
                    sub_conditions.append(_build_condition(sk, sv))
            if sub_conditions:
                conditions.append(models.Filter(must=sub_conditions))
        elif key == "$or":
            sub_conditions = []
            for sub_clause in value:
                if not isinstance(sub_clause, dict):
                    continue
                for sk, sv in sub_clause.items():
                    sub_conditions.append(_build_condition(sk, sv))
            if sub_conditions:
                conditions.append(models.Filter(should=sub_conditions))
        else:
            conditions.append(_build_condition(key, value))

    if not conditions:
        return models.Filter()
    if len(conditions) == 1 and isinstance(conditions[0], models.Filter):
        return conditions[0]
    return models.Filter(must=[
        c if isinstance(c, models.Filter) else models.Filter(must=[c])
        for c in conditions
    ])
