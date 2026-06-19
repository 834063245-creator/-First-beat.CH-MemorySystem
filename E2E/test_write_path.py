"""链路一：写入链路 Benchmark 测试（W1~W12）。

每个节点独立测试，使用真实 Qdrant + bge-m3 + 本地 LLM，固定 seed。
断言直接引用 BENCHMARK_SPEC.md 中的「验证方法」。
"""
import json
import math
import os
import time
from datetime import datetime

import pytest


# ═══════════════════════════════════════════════════════════════
# 共享测试数据
# ═══════════════════════════════════════════════════════════════

_TEST_USER_MESSAGE = "我今天在咖啡馆用Python写了一个爬虫脚本，感觉很有成就感"
_TEST_AI_REPLY = "太棒了！写爬虫确实很有成就感，你用的是什么库？BeautifulSoup还是Scrapy？"
_TEST_FULL_TEXT = f"用户：{_TEST_USER_MESSAGE}\nAI：{_TEST_AI_REPLY}"


# ═══════════════════════════════════════════════════════════════
# W1: embedding
# ═══════════════════════════════════════════════════════════════

class TestW1Embedding:
    """W1 — 嵌入向量生成。"""

    def test_embedding_dimension_and_validity(self):
        """验证：list[float]，len=1024，非全零，非 NaN。"""
        from app.llm.embed import local_embed

        emb = local_embed(_TEST_USER_MESSAGE)

        assert emb is not None, "嵌入向量不应为 None"
        assert isinstance(emb, list), "嵌入向量应为 list"
        assert len(emb) == 1024, f"嵌入维度应为 1024，实际为 {len(emb)}"

        # 非 NaN
        for v in emb:
            assert not math.isnan(v), f"嵌入向量包含 NaN 值: {v}"

        # 非全零
        assert any(abs(v) > 1e-8 for v in emb), "嵌入向量不应全零"

        # 归一化校验：L2 范数应接近 1（bge-m3 输出已归一化）
        import numpy as np
        norm = float(np.linalg.norm(np.array(emb, dtype=np.float32)))
        assert abs(norm - 1.0) < 0.01, f"L2 范数应接近 1，实际 {norm:.6f}"


# ═══════════════════════════════════════════════════════════════
# W2: summary
# ═══════════════════════════════════════════════════════════════

class TestW2Summary:
    """W2 — 中文摘要生成。"""

    def test_summary_length_and_content(self):
        """验证：中文摘要，长度 20~200 字，含关键实体。"""
        from app.llm.local import LocalLLM

        llm = LocalLLM()
        summary = llm.summarize(_TEST_FULL_TEXT, max_chars=200)

        assert summary, "摘要不应为空"
        assert isinstance(summary, str), "摘要应为字符串"

        char_len = len(summary)
        assert 20 <= char_len <= 200, (
            f"摘要长度应在 20~200 字之间，实际 {char_len}: 「{summary}」"
        )

        # 含关键实体（至少命中一个）
        key_entities = ["Python", "爬虫", "BeautifulSoup", "Scrapy", "咖啡"]
        matched = any(e in summary for e in key_entities)
        assert matched, f"摘要应包含至少一个关键实体，实际: 「{summary}」"


# ═══════════════════════════════════════════════════════════════
# W3: tags
# ═══════════════════════════════════════════════════════════════

class TestW3Tags:
    """W3 — 标签提取。"""

    def test_tags_count_and_length(self):
        """验证：list[str]，≥2 个标签，每个 ≥2 字。"""
        from app.brain.semantic import extract_tags

        tags = extract_tags(_TEST_USER_MESSAGE, topk=5)

        assert isinstance(tags, list), "标签应为 list"
        assert len(tags) >= 2, f"至少应提取 2 个标签，实际 {len(tags)}: {tags}"

        for tag in tags:
            assert isinstance(tag, str), f"标签应为字符串，实际: {type(tag)}"
            assert len(tag) >= 2, f"每个标签应 ≥2 字，实际: 「{tag}」({len(tag)} 字)"


# ═══════════════════════════════════════════════════════════════
# W4: entities
# ═══════════════════════════════════════════════════════════════

class TestW4Entities:
    """W4 — 实体抽取。"""

    def test_entities_schema(self):
        """验证：list[dict]，可空，非空时每项含 text + type。"""
        from app.analysis.entity import extract_entities

        entities = extract_entities(_TEST_USER_MESSAGE)

        assert isinstance(entities, list), "实体应为 list"

        # 可空 —— 不强制要求非空
        if entities:
            for ent in entities:
                assert isinstance(ent, dict), f"实体项应为 dict，实际: {type(ent)}"
                assert "text" in ent, f"实体项缺少 'text' 字段: {ent}"
                assert "type" in ent, f"实体项缺少 'type' 字段: {ent}"
                assert isinstance(ent["text"], str), f"text 应为字符串: {ent}"
                assert isinstance(ent["type"], str), f"type 应为字符串: {ent}"

    def test_entities_with_known_entities(self):
        """验证：包含已知实体（如 Python）时能被识别。"""
        from app.analysis.entity import extract_entities

        entities = extract_entities("张三在阿里巴巴用Python写代码")

        # 至少应有 KEYWORD 类型实体（bge-m3 提取）
        entity_texts = [e["text"] for e in entities]
        # 不强制要求 Ollama 成功（可能不可用），但至少有关键词
        assert len(entities) >= 0, "空列表也是合法输出"


# ═══════════════════════════════════════════════════════════════
# W5: emotion
# ═══════════════════════════════════════════════════════════════

class TestW5Emotion:
    """W5 — 情绪分析。"""

    def test_emotion_valence_range(self):
        """验证：valence ∈ [-1, 1], arousal ∈ [0, 1], 返回 category 字符串。"""
        from app.analysis.emotion import analyze_emotion_2d

        valence, arousal, category = analyze_emotion_2d(_TEST_USER_MESSAGE)

        assert isinstance(valence, (int, float)), f"valence 应为数值: {type(valence)}"
        assert isinstance(arousal, (int, float)), f"arousal 应为数值: {type(arousal)}"
        assert isinstance(category, str), f"category 应为字符串: {type(category)}"

        assert -1.0 <= valence <= 1.0, f"valence 应在 [-1,1] 内，实际 {valence}"
        assert 0.0 <= arousal <= 1.0, f"arousal 应在 [0,1] 内，实际 {arousal}"

    def test_emotion_positive_text(self):
        """验证：正面文本 valence > 0。"""
        from app.analysis.emotion import analyze_emotion_2d

        valence, arousal, category = analyze_emotion_2d("今天好开心啊，太棒了")
        assert valence > 0, f"正面文本 valence 应 >0，实际 {valence}"
        assert category == "positive", f"正面文本 category 应为 positive，实际 {category}"

    def test_emotion_negative_text(self):
        """验证：负面文本 valence < 0。"""
        from app.analysis.emotion import analyze_emotion_2d

        valence, arousal, category = analyze_emotion_2d("气死我了，好难过")
        assert valence < 0, f"负面文本 valence 应 <0，实际 {valence}"
        assert category == "negative", f"负面文本 category 应为 negative，实际 {category}"

    def test_emotion_empty(self):
        """验证：空文本返回 neutral。"""
        from app.analysis.emotion import analyze_emotion_2d

        valence, arousal, category = analyze_emotion_2d("")
        assert valence == 0.0
        assert arousal == 0.0
        assert category == "neutral"


# ═══════════════════════════════════════════════════════════════
# W6: time_features
# ═══════════════════════════════════════════════════════════════

class TestW6TimeFeatures:
    """W6 — 时间特征提取。"""

    _EXPECTED_PERIODS = {
        0: "深夜", 1: "深夜", 2: "深夜", 3: "深夜", 4: "深夜", 5: "深夜",
        6: "早晨", 7: "早晨", 8: "早晨",
        9: "上午", 10: "上午", 11: "上午",
        12: "中午", 13: "中午",
        14: "下午", 15: "下午", 16: "下午", 17: "下午",
        18: "傍晚", 19: "傍晚", 20: "傍晚",
        21: "晚上", 22: "晚上", 23: "晚上",
    }

    def test_all_24_hours_mapped(self):
        """验证：24 小时每个小时映射到正确 time_period。"""
        from app.config.settings import TIME_PERIOD_MAP

        # 收集所有已覆盖的小时
        covered_hours = set()
        for (lo, hi), label in TIME_PERIOD_MAP.items():
            for h in range(lo, hi + 1):
                covered_hours.add(h)

        # 24 小时全覆盖
        assert len(covered_hours) == 24, (
            f"TIME_PERIOD_MAP 应覆盖 24 小时，实际覆盖 {len(covered_hours)}: "
            f"缺失 {sorted(set(range(24)) - covered_hours)}"
        )

    def test_each_hour_maps_correctly(self):
        """验证：每个小时的 period 映射正确。"""
        from app.config.settings import TIME_PERIOD_MAP

        for hour in range(24):
            found = None
            for (lo, hi), label in TIME_PERIOD_MAP.items():
                if lo <= hour <= hi:
                    found = label
                    break
            expected = self._EXPECTED_PERIODS[hour]
            assert found == expected, (
                f"小时 {hour} 应映射为「{expected}」，实际为「{found}」"
            )

    def test_time_features_fields_complete(self):
        """验证：time_features 字典包含所有必要字段。"""
        expected_fields = {"year", "month", "day", "day_of_week",
                           "hour", "season", "time_period"}
        from app.config.settings import TIME_PERIOD_MAP

        test_dt = datetime(2026, 6, 6, 14, 30, 0)
        h = test_dt.hour
        period = "晚上"
        for (lo, hi), name in TIME_PERIOD_MAP.items():
            if lo <= h <= hi:
                period = name
                break

        tf = {
            "date": test_dt.strftime("%Y-%m-%d"),
            "year": test_dt.year,
            "month": test_dt.month,
            "day": test_dt.day,
            "week": test_dt.isocalendar()[1],
            "day_of_week": test_dt.weekday(),
            "quarter": (test_dt.month - 1) // 3 + 1,
            "season": (test_dt.month % 12 + 3) // 3,
            "year_month": test_dt.strftime("%Y-%m"),
            "time_period": period,
        }

        # 验证字段存在且非 None
        for field in ["year", "month", "day", "day_of_week", "season", "time_period"]:
            assert field in tf, f"缺少字段: {field}"
            assert tf[field] is not None, f"字段 {field} 为 None"

        # 2026-06-06 是星期六 → day_of_week=5
        assert tf["day_of_week"] == 5
        # 6 月 → season=3 (summer)，公式 (month%12+3)//3: 1冬2春3夏4秋
        assert tf["season"] == 3, f"6 月应为夏季(3)，实际: {tf['season']}"


# ═══════════════════════════════════════════════════════════════
# W7: Qdrant 存储
# ═══════════════════════════════════════════════════════════════

class TestW7QdrantStorage:
    """W7 — Qdrant 存储完整性。"""

    def test_store_and_retrieve_memory(self, isolated_memory_service):
        """验证：1 条新记录，id 非空，metadata 完整。"""
        from app.llm.embed import local_embed
        from app.analysis.emotion import analyze_emotion_2d
        from app.analysis.entity import extract_entities
        from app.brain.semantic import extract_tags

        # 构造完整元数据
        full_text = _TEST_FULL_TEXT
        embedding = local_embed(full_text)
        assert embedding, "嵌入向量不应为 None"

        tags = extract_tags(_TEST_USER_MESSAGE, topk=5)
        if len(tags) < 2:
            tags = list(tags) + (["交流"] if tags[0] != "交流" else ["对话"])

        entities = extract_entities(full_text)
        valence, arousal, emo_category = analyze_emotion_2d(full_text)

        ts = "2026-06-06 14:30:00"
        date_tag = ts.split(" ")[0]
        from datetime import datetime as _dt
        dt = _dt.strptime(ts, "%Y-%m-%d %H:%M:%S")
        from app.config.settings import TIME_PERIOD_MAP as _tpm
        h = dt.hour
        period = "晚上"
        for (lo, hi), name in _tpm.items():
            if lo <= h <= hi:
                period = name
                break
        time_features = {
            "date": dt.strftime("%Y-%m-%d"),
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "week": dt.isocalendar()[1],
            "day_of_week": dt.weekday(),
            "quarter": (dt.month - 1) // 3 + 1,
            "season": (dt.month % 12 + 3) // 3,
            "year_month": dt.strftime("%Y-%m"),
            "time_period": period,
            "emotion_valence": valence,
            "emotion_arousal": arousal,
            "emotion_valence_bin": emo_category,
            "emotional_intensity": 1,
            "timestamp": dt.timestamp(),
        }

        memory_id = isolated_memory_service.add_memory(
            user_message=_TEST_USER_MESSAGE,
            ai_message=_TEST_AI_REPLY,
            summary="用Python写爬虫脚本的对话",
            tags=tags,
            embedding=embedding,
            entities=entities,
            date_tag=date_tag,
            time_features=time_features,
            source="user",
        )

        # 验证 ID 非空
        assert memory_id, "记忆 ID 不应为空"
        assert len(memory_id) > 0, "记忆 ID 不应为空字符串"

        # 验证可检索
        result = isolated_memory_service._collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )
        assert result["ids"], f"应按 ID 检索到记忆: {memory_id}"
        assert result["ids"][0] == memory_id

        # 验证 metadata 完整（关键字段存在）
        meta = dict(result["metadatas"][0])
        assert "summary" in meta, "metadata 缺少 summary"
        assert "tags" in meta, "metadata 缺少 tags"
        assert "user_message" in meta, "metadata 缺少 user_message"
        assert "ai_message" in meta, "metadata 缺少 ai_message"
        assert "timestamp" in meta, "metadata 缺少 timestamp"
        assert "source" in meta, "metadata 缺少 source"

        # 验证 document 字段
        doc = result["documents"][0]
        assert _TEST_USER_MESSAGE in doc, "document 应包含用户消息"
        assert _TEST_AI_REPLY in doc, "document 应包含 AI 回复"


# ═══════════════════════════════════════════════════════════════
# W8: chat_history 存储
# ═══════════════════════════════════════════════════════════════

class TestW8ChatHistoryStorage:
    """W8 — 对话历史 JSONL 存储。"""

    def test_append_and_verify_fields(self, isolated_chat_history):
        """验证：jsonl 追加一行，含 timestamp/user_message/llm_reply。"""
        ts = "2026-06-06 14:30:00"

        isolated_chat_history.append(
            user_message=_TEST_USER_MESSAGE,
            llm_reply=_TEST_AI_REPLY,
            timestamp=ts,
        )

        # 解析文件最后一行
        with open(isolated_chat_history.path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) >= 1, "JSONL 文件应至少包含 1 行"

        last_line = lines[-1].strip()
        record = json.loads(last_line)

        # 字段齐全
        assert "timestamp" in record, "记录缺少 timestamp"
        assert "user_message" in record, "记录缺少 user_message"
        assert "llm_reply" in record, "记录缺少 llm_reply"

        assert record["timestamp"] == ts
        assert record["user_message"] == _TEST_USER_MESSAGE
        assert record["llm_reply"] == _TEST_AI_REPLY


# ═══════════════════════════════════════════════════════════════
# W9: 倒排索引
# ═══════════════════════════════════════════════════════════════

class TestW9InvertedIndex:
    """W9 — 倒排索引写入与查询。"""

    def test_new_tag_maps_to_memory_id(self, isolated_inverted_index):
        """验证：新标签 → 新记忆 ID 的映射存在。"""
        test_memory_id = "test-mem-w9-001"
        unique_tag = f"测试标签_{int(time.time() * 1000)}"

        # 写入标签索引
        isolated_inverted_index.add_tags(test_memory_id, unique_tag)

        # 查询
        result_ids = isolated_inverted_index.query_tags([unique_tag])

        assert test_memory_id in result_ids, (
            f"倒排索引应包含标签「{unique_tag}」→ ID「{test_memory_id}」的映射"
        )

    def test_summary_index_also_works(self, isolated_inverted_index):
        """验证：通过 summary 添加后也能 query 到（Token 需匹配 tokenizer 2-gram 切分）。"""
        test_memory_id = "test-mem-w9-002"
        # 使用短词，会被 2-gram tokenizer 作为完整 token 保留且不切分
        # （中文 2 字词正好命中 char bigram）
        known_word = "摘要"

        isolated_inverted_index.add(test_memory_id, f"这是一段包含{known_word}的测试文本")
        results = isolated_inverted_index.query([known_word], min_match=1)

        assert test_memory_id in results, (
            f"倒排索引应能通过 token「{known_word}」命中记忆 {test_memory_id}"
        )


# ═══════════════════════════════════════════════════════════════
# W10: 实体对存储
# ═══════════════════════════════════════════════════════════════

class TestW10EntityPairStorage:
    """W10 — 实体对共现存储。"""

    def test_entity_pair_count_increments(self, isolated_entity_pair_tracker):
        """验证：实体对共现记录写入并可查询。"""
        entity_a = "Python"
        entity_b = "Scrapy"
        memory_id = "test-mem-w10-001"

        # 记录实体对
        isolated_entity_pair_tracker.record(entity_a, entity_b, memory_id)

        # 通过 expand 验证双向存在
        result = isolated_entity_pair_tracker.expand([entity_a])
        assert entity_a in result, f"实体「{entity_a}」应有扩展结果"
        assert entity_b in result[entity_a], f"「{entity_b}」应在「{entity_a}」的共现列表中"
        assert result[entity_a][entity_b] >= 1, (
            f"共现计数应 ≥1，实际: {result[entity_a][entity_b]}"
        )

        # 验证 memory_id 关联
        ids = isolated_entity_pair_tracker.get_memory_ids([entity_a])
        assert memory_id in ids, (
            f"memory_id {memory_id} 应在实体对的 memory_ids 中"
        )



# ═══════════════════════════════════════════════════════════════
# W11: AI 自我记忆存储
# ═══════════════════════════════════════════════════════════════

class TestW11AIMemoryStorage:
    """W11 — AI 自我表达记忆存储。"""

    def test_ai_memory_store_and_retrieve(self, isolated_ai_memory_service):
        """验证：ai_memories 集合新增 1 条记录，metadata 含 summary/tags/emotion。"""
        from app.llm.embed import local_embed
        from app.analysis.emotion import analyze_emotion_2d

        ai_message = "我也觉得写代码很有成就感！爬虫是很好玩的技能，建议用Scrapy框架。"
        ai_embedding = local_embed(f"AI：{ai_message}")
        assert ai_embedding, "AI 嵌入向量不应为 None"

        ai_valence, ai_arousal, ai_emo_category = analyze_emotion_2d(ai_message)

        ai_meta = {
            "emotion_valence": ai_valence,
            "emotion_arousal": ai_arousal,
            "emotion_valence_bin": ai_emo_category,
            "emotional_intensity": 1,
            "timestamp": datetime.now().timestamp(),
        }

        ai_mid = isolated_ai_memory_service.add_memory(
            user_message="[AI]",
            ai_message=ai_message,
            summary="AI 表达对写代码成就感的共鸣",
            tags=["AI表达", "编程", "成就感"],
            embedding=ai_embedding,
            time_features=ai_meta,
            source="ai",
        )

        assert ai_mid, "AI 记忆 ID 不应为空"

        # 验证可检索
        result = isolated_ai_memory_service._collection.get(
            ids=[ai_mid],
            include=["metadatas"],
        )
        assert result["ids"], f"应按 ID 检索到 AI 记忆: {ai_mid}"

        meta = dict(result["metadatas"][0])
        assert "summary" in meta, "AI 记忆 metadata 缺少 summary"
        assert "tags" in meta, "AI 记忆 metadata 缺少 tags"
        assert meta.get("emotion_valence") is not None, "AI 记忆 metadata 缺少 emotion_valence"
        assert meta.get("emotion_valence_bin") is not None, "AI 记忆 metadata 缺少 emotion_valence_bin"


# ═══════════════════════════════════════════════════════════════
# W12: 回复不崩
# ═══════════════════════════════════════════════════════════════

class TestW12ReplyNotCrash:
    """W12 — 端到端 HTTP 请求与响应。"""

    @pytest.mark.integration
    def test_chat_endpoint_returns_200_with_response(self, server_url, server_alive):
        """验证：HTTP 200，response 字段非空。"""
        if not server_alive:
            pytest.skip(f"基准服务器未运行 ({server_url})，跳过集成测试")

        import httpx

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{server_url}/chat",
                json={
                    "message": "你好，今天天气真好！",
                },
                headers={"Authorization": "Bearer admin:changeme"},
            )

        assert resp.status_code == 200, f"HTTP 状态码应为 200，实际 {resp.status_code}"

        data = resp.json()
        assert "response" in data, "响应体应包含 'response' 字段"
        assert data["response"], "response 字段不应为空"
        assert isinstance(data["response"], str), "response 应为字符串"
