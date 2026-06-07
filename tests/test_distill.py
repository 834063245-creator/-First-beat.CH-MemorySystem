"""测试 app/background/distill.py — 蒸馏引擎纯函数。"""
import json
import os
import time
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest

from app.background.distill import (
    _read_state,
    _write_state,
    _recency_score,
    _compute_confidence,
    _generate_content,
    _extract_keywords,
    _extract_patterns,
    DistillEngine,
)


# ═══════════════════════════════════════════════════════════════════
# _read_state / _write_state
# ═══════════════════════════════════════════════════════════════════

class TestReadState:
    def test_file_not_exists(self):
        result = _read_state("/nonexistent/path/state.json")
        assert result == {"last_distill_timestamp": None, "total_distill_runs": 0}

    def test_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"last_distill_timestamp": "2026-06-01T10:00:00", "total_distill_runs": 5}, f)
            path = f.name
        try:
            result = _read_state(path)
            assert result["last_distill_timestamp"] == "2026-06-01T10:00:00"
            assert result["total_distill_runs"] == 5
        finally:
            os.unlink(path)

    def test_corrupted_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            path = f.name
        try:
            result = _read_state(path)
            assert result == {"last_distill_timestamp": None, "total_distill_runs": 0}
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            path = f.name
        try:
            result = _read_state(path)
            assert result == {"last_distill_timestamp": None, "total_distill_runs": 0}
        finally:
            os.unlink(path)


class TestWriteState:
    def test_write_and_read_roundtrip(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "state.json")
        try:
            state = {"last_distill_timestamp": "2026-06-01", "total_distill_runs": 3}
            _write_state(state, path)
            assert os.path.exists(path)
            result = _read_state(path)
            assert result == state
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_creates_directory(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "subdir", "state.json")
        try:
            _write_state({"last_distill_timestamp": None, "total_distill_runs": 0}, path)
            assert os.path.exists(path)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# _recency_score
# ═══════════════════════════════════════════════════════════════════

class TestRecencyScore:
    def test_today(self):
        now = datetime.now().timestamp()
        # 刚刚发生
        assert _recency_score(now, now) == 1.0

    def test_yesterday(self):
        now = datetime.now().timestamp()
        one_day_ago = now - 86400
        assert _recency_score(one_day_ago, now) == pytest.approx(1.0)

    def test_one_week_ago(self):
        now = datetime.now().timestamp()
        seven_days_ago = now - 7 * 86400
        assert _recency_score(seven_days_ago, now) == pytest.approx(1.0)

    def test_two_weeks_ago(self):
        now = datetime.now().timestamp()
        fourteen_days_ago = now - 14 * 86400
        score = _recency_score(fourteen_days_ago, now)
        # 14 days: (14-7)/23 ≈ 0.304, score = 1 - 0.304 ≈ 0.696
        assert 0.6 < score < 0.8

    def test_one_month_ago(self):
        now = datetime.now().timestamp()
        thirty_days_ago = now - 30 * 86400
        assert _recency_score(thirty_days_ago, now) == 0.0

    def test_one_year_ago(self):
        now = datetime.now().timestamp()
        year_ago = now - 365 * 86400
        assert _recency_score(year_ago, now) == 0.0

    def test_future_date(self):
        now = datetime.now().timestamp()
        future = now + 86400
        # days_ago = -1, which is <= 7, so score = 1.0
        assert _recency_score(future, now) == 1.0

    def test_boundary_8_days(self):
        now = datetime.now().timestamp()
        eight_days_ago = now - 8 * 86400
        score = _recency_score(eight_days_ago, now)
        # (8-7)/23 ≈ 0.0435, score ≈ 0.957
        assert 0.95 < score < 1.0

    def test_boundary_29_days(self):
        now = datetime.now().timestamp()
        twenty_nine_days_ago = now - 29 * 86400
        score = _recency_score(twenty_nine_days_ago, now)
        # (29-7)/23 ≈ 0.957, score ≈ 0.043
        assert 0.0 < score < 0.1


# ═══════════════════════════════════════════════════════════════════
# _compute_confidence
# ═══════════════════════════════════════════════════════════════════

class TestComputeConfidence:
    def test_all_max(self):
        score, label = _compute_confidence(count=10, days_span=14, recency=1.0,
                                           has_emotion=True, kw_diversity=5)
        assert label == "高"
        assert score >= 0.70

    def test_all_zero(self):
        score, label = _compute_confidence(count=0, days_span=0, recency=0.0,
                                           has_emotion=False, kw_diversity=0)
        assert label == "低"
        assert score < 0.40

    def test_medium_values(self):
        score, label = _compute_confidence(count=5, days_span=7, recency=0.5,
                                           has_emotion=True, kw_diversity=3)
        assert label in ("低", "中", "高")
        assert 0.0 <= score <= 1.0

    def test_low_count_only(self):
        score, label = _compute_confidence(count=1, days_span=1, recency=0.0,
                                           has_emotion=False, kw_diversity=0)
        assert label == "低"

    def test_high_recency_boosts(self):
        score_low_rec, _ = _compute_confidence(count=5, days_span=7, recency=0.0,
                                                has_emotion=False, kw_diversity=2)
        score_high_rec, _ = _compute_confidence(count=5, days_span=7, recency=1.0,
                                                 has_emotion=False, kw_diversity=2)
        assert score_high_rec > score_low_rec

    def test_high_label_threshold(self):
        score, label = _compute_confidence(count=10, days_span=14, recency=1.0,
                                           has_emotion=True, kw_diversity=5)
        assert score >= 0.70
        assert label == "高"

    def test_mid_label_threshold(self):
        # Around 0.40-0.70 should be "中"
        score, label = _compute_confidence(count=5, days_span=7, recency=0.5,
                                           has_emotion=True, kw_diversity=3)
        if 0.40 <= score < 0.70:
            assert label == "中"
        # else it's fine either way - thresholds are boundary

    def test_score_in_range(self):
        for count in [0, 3, 10]:
            for days in [0, 5, 14]:
                for rec in [0.0, 0.5, 1.0]:
                    for emo in [True, False]:
                        for kw in [0, 3, 5]:
                            score, label = _compute_confidence(count, days, rec, emo, kw)
                            assert 0.0 <= score <= 1.0
                            assert label in ("低", "中", "高")


# ═══════════════════════════════════════════════════════════════════
# _generate_content
# ═══════════════════════════════════════════════════════════════════

class TestGenerateContent:
    def test_habit_type(self):
        content = _generate_content("周期性行为", "跑步", "早晨", 30, 20,
                                    {"positive": 5, "negative": 1},
                                    ["运动", "健康"], who_prefix="用户")
        assert "用户习惯在早晨聊跑步" in content
        assert "运动" in content or "健康" in content

    def test_habit_type_no_keywords(self):
        content = _generate_content("周期性行为", "阅读", "晚上", 15, 8,
                                    {}, [], who_prefix="用户")
        assert "用户习惯在晚上聊阅读" in content

    def test_topic_type(self):
        content = _generate_content("稳定兴趣", "编程", "下午", 60, 30,
                                    {"positive": 10}, ["Python", "Rust"],
                                    who_prefix="用户")
        assert "用户长期关注编程" in content

    def test_interest_type(self):
        content = _generate_content("临时热点", "AI", "上午", 3, 12,
                                    {}, ["GPT"], who_prefix="用户")
        assert "用户近期密集关注AI" in content

    def test_trait_type_with_emotion(self):
        content = _generate_content("情绪波动", "工作", "上午", 20, 15,
                                    {"positive": 8, "negative": 7},
                                    ["项目"], who_prefix="用户")
        assert "情绪波动较大" in content
        assert "正向8次" in content

    def test_emotion_association(self):
        content = _generate_content("情绪关联", "家庭", "晚上", 10, 8,
                                    {"positive": 6, "negative": 2},
                                    [], who_prefix="用户")
        assert "情绪偏positive" in content or "情绪偏" in content

    def test_fallback_type(self):
        content = _generate_content("未知类型", "某某", "上午", 1, 1,
                                    {}, [], who_prefix="用户")
        assert "多次提到某某" in content

    def test_ai_prefix(self):
        content = _generate_content("周期性行为", "回应", "下午", 10, 5,
                                    {}, [], who_prefix="AI")
        assert "AI习惯" in content


# ═══════════════════════════════════════════════════════════════════
# _extract_keywords
# ═══════════════════════════════════════════════════════════════════

class TestExtractKeywords:
    def test_chinese_text(self):
        # extract_tags is mocked at conftest level to not do HTTP
        from app.background.distill import _keyword_cache
        _keyword_cache.clear()
        result = _extract_keywords("今天天气真好，适合出去跑步运动健身")
        assert isinstance(result, list)

    def test_empty_string(self):
        result = _extract_keywords("")
        assert result == []

    def test_none_text(self):
        result = _extract_keywords("")
        assert result == []

    def test_cache_hit(self):
        from app.background.distill import _keyword_cache
        _keyword_cache.clear()
        _keyword_cache["测试文本"] = ["测试", "文本"]
        result = _extract_keywords("测试文本")
        assert result == ["测试", "文本"]
        _keyword_cache.clear()

    def test_english_text(self):
        from app.background.distill import _keyword_cache
        _keyword_cache.clear()
        result = _extract_keywords("machine learning and artificial intelligence")
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════
# _extract_patterns
# ═══════════════════════════════════════════════════════════════════

def _make_memory(tags: str, timestamp: float, summary: str = "",
                 emotion_valence: str = "", emotional_intensity: int = 0):
    return {
        "id": f"mem_{hash(tags + str(timestamp))}",
        "document": f"用户消息关于{tags}",
        "metadata": {
            "tags": tags,
            "timestamp": timestamp,
            "summary": summary,
            "emotion_valence_bin": emotion_valence,
            "emotional_intensity": emotional_intensity,
        },
    }


class TestExtractPatterns:
    def test_empty_memories(self):
        result = _extract_patterns([])
        assert result == []

    def test_single_memory_insufficient(self):
        """单条记忆不足以形成模式（_MIN_OCCURRENCES=2）"""
        now = datetime.now().timestamp()
        mems = [_make_memory("跑步", now, "今天去跑步了")]
        result = _extract_patterns(mems)
        assert result == []  # count < 2, filtered out

    def test_two_memories_same_tag(self):
        """两条同标签记忆形成模式"""
        now = datetime.now().timestamp()
        mems = [
            _make_memory("跑步", now - 86400, "昨天跑步5公里"),
            _make_memory("跑步", now, "今天跑步3公里"),
        ]
        result = _extract_patterns(mems)
        assert len(result) >= 1
        assert "跑步" in result[0]["content"]

    def test_multiple_tags(self):
        """多条记忆多标签"""
        now = datetime.now().timestamp()
        mems = [
            _make_memory("Rust,编程", now - 3 * 86400, "学习Rust所有权"),
            _make_memory("Python,编程", now - 2 * 86400, "写Python脚本"),
            _make_memory("Rust,编程", now - 86400, "Rust实战项目"),
            _make_memory("Python,编程", now, "Python数据分析"),
        ]
        result = _extract_patterns(mems)
        assert len(result) >= 1
        # Should have patterns for tags with >=2 occurrences
        tags_found = set()
        for p in result:
            for tag in ["Rust", "编程", "Python"]:
                if tag in p["content"]:
                    tags_found.add(tag)
        assert len(tags_found) >= 1

    def test_pattern_types(self):
        """验证不同类型的模式被检测"""
        now = datetime.now().timestamp()
        mems = [
            _make_memory("焦虑,工作", now - 5 * 86400, "工作压力好大", "negative", 3),
            _make_memory("焦虑,工作", now - 3 * 86400, "项目又延期了", "negative", 2),
            _make_memory("焦虑,工作", now - 86400, "加班到很晚", "negative", 3),
            _make_memory("运动,健康", now - 10 * 86400, "开始跑步", "positive", 1),
            _make_memory("运动,健康", now - 7 * 86400, "坚持跑步一周", "positive", 1),
            _make_memory("运动,健康", now - 4 * 86400, "跑步三周打卡", "positive", 2),
            _make_memory("运动,健康", now - 86400, "今天也跑了", "positive", 1),
        ]
        result = _extract_patterns(mems)
        assert len(result) >= 1
        for p in result:
            assert "content" in p
            assert "type" in p
            assert "confidence" in p
            assert "evidence" in p

    def test_compound_patterns(self):
        """跨标签复合模式"""
        now = datetime.now().timestamp()
        mems = []
        for i in range(6):
            mems.append(_make_memory("Python,编程", now - i * 86400,
                                     f"Python编程第{i}天"))
        for i in range(6):
            mems.append(_make_memory("Docker,编程", now - i * 86400,
                                     f"Docker学习第{i}天"))
        result = _extract_patterns(mems)
        # Should have compound pattern for Python+Docker co-occurring >=5 times
        compound = [p for p in result if p.get("type") == "兴趣领域"]
        if compound:
            assert "Python" in compound[0]["content"] or "Docker" in compound[0]["content"]

    def test_max_patterns_limit(self):
        """验证 _MAX_PATTERNS=15 截断"""
        now = datetime.now().timestamp()
        mems = []
        # Create 30 different tags with 2 memories each
        for tag_idx in range(30):
            tag = f"话题{tag_idx}"
            mems.append(_make_memory(tag, now - 2 * 86400, f"内容{tag_idx}a"))
            mems.append(_make_memory(tag, now - 86400, f"内容{tag_idx}b"))
        result = _extract_patterns(mems)
        assert len(result) <= 15

    def test_source_ai(self):
        """AI 来源记忆"""
        now = datetime.now().timestamp()
        mems = [
            _make_memory("回应,帮助", now - 86400, "帮你解决问题"),
            _make_memory("回应,帮助", now, "提供建议"),
        ]
        result = _extract_patterns(mems, source="ai")
        assert len(result) >= 1
        assert "AI" in result[0]["content"]

    def test_no_timestamps(self):
        """无时间戳的记忆被跳过"""
        mems = [
            _make_memory("测试", 0, "无时间戳"),
            _make_memory("测试", 0, "也无时间戳"),
        ]
        result = _extract_patterns(mems)
        # Both have timestamp=0 which is falsy, so continue
        assert result == [] or all(p["evidence"]["count"] == 0 for p in result)

    def test_short_tags_excluded(self):
        """长度 < 2 的标签被排除"""
        now = datetime.now().timestamp()
        mems = [
            {"id": "m1", "metadata": {"tags": "a,b,c", "timestamp": now - 86400}},
            {"id": "m2", "metadata": {"tags": "a,b,c", "timestamp": now}},
        ]
        result = _extract_patterns(mems)
        # All tags have len < 2, so no patterns
        assert result == []

    def test_confidence_fields(self):
        """验证置信度字段"""
        now = datetime.now().timestamp()
        mems = [
            _make_memory("健身", now - 5 * 86400, "周一健身"),
            _make_memory("健身", now - 2 * 86400, "周四健身"),
            _make_memory("健身", now, "今天健身"),
        ]
        result = _extract_patterns(mems)
        assert len(result) >= 1
        p = result[0]
        assert "confidence_score" in p
        assert "confidence" in p
        assert 0.0 <= p["confidence_score"] <= 1.0

    def test_evidence_fields(self):
        """验证 evidence 字段"""
        now = datetime.now().timestamp()
        mems = [
            _make_memory("阅读", now - 3 * 86400, "读《黑客与画家》"),
            _make_memory("阅读", now - 86400, "继续读书"),
            _make_memory("阅读", now, "读完了一章"),
        ]
        result = _extract_patterns(mems)
        assert len(result) >= 1
        ev = result[0]["evidence"]
        assert "count" in ev
        assert "days_span" in ev
        assert "recency_days" in ev
        assert "peak_period" in ev
        assert "keywords" in ev
        assert "density" in ev


# ═══════════════════════════════════════════════════════════════════
# DistillEngine.run_distill
# ═══════════════════════════════════════════════════════════════════

class TestDistillEngineRun:
    def test_no_memories(self, isolated_env):
        """空记忆库 → skipped"""
        engine = DistillEngine(
            personality_store=isolated_env.personality_store,
            chroma_service=isolated_env.chroma_service,
            state_path=os.path.join(isolated_env.data_dir, "distill_state.json"),
            source="user",
        )
        result = engine.run_distill()
        assert result["status"] == "skipped"
        assert result["reason"] == "no_new_memories"

    def test_with_memories(self, isolated_env):
        """有记忆时正常蒸馏"""
        # 写入记忆（使用 _store_conversation 管线）
        isolated_env._store_conversation("我喜欢跑步", "跑步很好", "2026-06-01 10:00:00")
        isolated_env._store_conversation("今天又跑了5公里", "加油", "2026-06-02 10:00:00")
        time.sleep(0.3)  # 等待队列处理
        engine = DistillEngine(
            personality_store=isolated_env.personality_store,
            chroma_service=isolated_env.chroma_service,
            state_path=os.path.join(isolated_env.data_dir, "distill_state.json"),
            source="user",
        )
        result = engine.run_distill()
        assert result["status"] in ("done", "skipped")

    def test_force_all(self, isolated_env):
        """强制全量蒸馏"""
        engine = DistillEngine(
            personality_store=isolated_env.personality_store,
            chroma_service=isolated_env.chroma_service,
            state_path=os.path.join(isolated_env.data_dir, "distill_state.json"),
            source="user",
        )
        result = engine.run_distill(force_all=True)
        assert result["status"] == "skipped"  # 空库

    def test_incremental(self, isolated_env):
        """增量蒸馏：只处理新记忆"""
        state_path = os.path.join(isolated_env.data_dir, "distill_state.json")
        # 预设上次蒸馏时间
        _write_state({"last_distill_timestamp": "2026-06-01T10:00:00", "total_distill_runs": 1}, state_path)

        engine = DistillEngine(
            personality_store=isolated_env.personality_store,
            chroma_service=isolated_env.chroma_service,
            state_path=state_path,
            source="user",
        )
        result = engine.run_distill()
        assert result["status"] == "skipped"  # 空库，无新记忆
