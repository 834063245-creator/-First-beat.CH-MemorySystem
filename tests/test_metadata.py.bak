"""测试 app/core/metadata.py — 入库元数据提取。

覆盖：map_hour_to_period / extract_topics / extract_persons / build_memory_metadata。
"""
import pytest
from app.core.metadata import (
    map_hour_to_period,
    extract_topics,
    extract_persons,
    build_memory_metadata,
)


class TestMapHourToPeriod:
    def test_deep_night(self):
        assert map_hour_to_period(2) == "深夜"

    def test_morning(self):
        assert map_hour_to_period(7) == "早晨"

    def test_forenoon(self):
        assert map_hour_to_period(10) == "上午"

    def test_noon(self):
        assert map_hour_to_period(13) == "中午"

    def test_afternoon(self):
        assert map_hour_to_period(15) == "下午"

    def test_evening(self):
        assert map_hour_to_period(19) == "傍晚"

    def test_night(self):
        assert map_hour_to_period(22) == "晚上"

    def test_boundary_midnight(self):
        assert map_hour_to_period(0) in ("深夜", "其他")
        assert map_hour_to_period(5) in ("深夜", "其他")

    @pytest.mark.parametrize("hour,expected", [
        (6, "早晨"), (8, "早晨"),
        (9, "上午"), (11, "上午"),
        (12, "中午"), (13, "中午"),
        (14, "下午"), (17, "下午"),
        (18, "傍晚"), (20, "傍晚"),
        (21, "晚上"), (23, "晚上"),
    ])
    def test_all_period_bounds(self, hour, expected):
        assert map_hour_to_period(hour) == expected


class TestExtractTopics:
    def test_matches_tech_topic(self):
        topics = extract_topics("我最近在写Python代码和调试bug")
        assert "技术" in topics

    def test_matches_multiple_topics(self):
        # "工作" 主题的关键词是"公司/项目/leader/重构"等，需要包含具体关键词
        topics = extract_topics("我的猫生病了带它去宠物医院，最近公司项目也很累")
        assert "宠物" in topics
        assert "工作" in topics

    def test_no_match_returns_empty(self):
        topics = extract_topics("今天天气不错")
        assert topics == []

    def test_case_insensitive(self):
        topics = extract_topics("我喜欢PYTHON编程")
        assert "技术" in topics


class TestExtractPersons:
    def test_extracts_pronouns(self):
        # 代词兜底
        persons = extract_persons("我觉得你很厉害")
        assert any(p in persons for p in ["我", "你"])

    def test_deduplicates(self):
        # "我" 出现多次也只出现一次
        persons = extract_persons("我觉得我觉得我很开心")
        assert persons.count("我") == 1


class TestBuildMemoryMetadata:
    def test_default_fields(self):
        meta = build_memory_metadata("你好", "你好呀", "2025-06-01 14:30:00")
        assert meta["source_type"] == "chat"
        assert "emotion_valence" in meta
        assert "emotion_arousal" in meta
        assert "emotion_valence_bin" in meta

    def test_date_and_time_period(self):
        meta = build_memory_metadata("测试", "回复", "2025-06-01 10:30:00")
        assert meta["date"] == "2025-06-01"
        assert meta["time_period"] == "上午"

    def test_topics_extraction(self):
        meta = build_memory_metadata("我的猫生病了", "需要去宠物医院", "2025-06-01 10:30:00")
        assert "宠物" in meta.get("topics", "")

    def test_persons_extraction(self):
        meta = build_memory_metadata("我和妈妈", "好的", "2025-06-01 10:30:00")
        # 至少包含"我"（代词兜底）
        persons = meta.get("persons", "")
        assert len(persons) > 0

    def test_invalid_timestamp_no_date(self):
        meta = build_memory_metadata("测试", "回复", "invalid")
        assert "date" not in meta
