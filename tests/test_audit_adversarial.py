"""边界对抗审计 — 同义句鲁棒性 + 多话题混合查询。"""

import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from config import DATA_DIR
from app.memory.chroma import ChromaService


_CS = None


def _get_store():
    global _CS
    if _CS is None:
        _CS = ChromaService(persist_dir=os.path.join(DATA_DIR, "chroma"))
    return _CS


def _embed(text: str) -> list[float]:
    from local_embed import local_embed
    return local_embed(text)


# ── 同义句对 ──
SYNONYM_PAIRS = [
    ("那个项目预算多少", "A项目花了好多钱吧"),
    ("我家猫生病了", "橘猫最近身体怎么样"),
    ("代码重构搞完了吗", "项目重写进度如何"),
    ("今天几号", "现在什么日期"),
    ("你记得我说过什么吗", "我之前提过哪个事"),
    ("我最近在学 Rust", "我在写 Rust 代码"),
    ("上次那个 BUG 修好了没", "之前报的什么什么错误解决了没有"),
    ("今天心情不太好", "我今天不是很开心"),
    ("能帮我查个资料吗", "我查一下用户之前讨论过的东西"),
    ("那个项目后来怎么样了", "项目后来的结果如何"),
]


class TestSynonymRobustness:
    """4.5 同义句鲁棒性。"""

    @pytest.mark.parametrize("q1,q2", SYNONYM_PAIRS)
    def test_synonym_recall(self, q1, q2):
        emb1 = _embed(q1)
        emb2 = _embed(q2)
        store = _get_store()
        r1 = store._read_collection.query(
            query_embeddings=[emb1], n_results=10,
            include=["metadatas", "distances"]
        )
        r2 = store._read_collection.query(
            query_embeddings=[emb2], n_results=10,
            include=["metadatas", "distances"]
        )
        ids1 = set(r1["ids"][0]) if r1.get("ids") else set()
        ids2 = set(r2["ids"][0]) if r2.get("ids") else set()
        if not ids1 or not ids2:
            pytest.skip("无检索结果")
        overlap = len(ids1 & ids2) / max(len(ids1), len(ids2))
        # 同义句语义检索应达到 40%+ 重合率
        assert overlap >= 0.4, f"同义句 '{q1[:15]}' / '{q2[:15]}' 重合率 {overlap:.0%}"

    def test_overall_pass_rate(self):
        """10 组中 ≥8 组通过即视为整体通过。"""
        passed = 0
        for q1, q2 in SYNONYM_PAIRS:
            try:
                self.test_synonym_recall(q1, q2)
                passed += 1
            except (AssertionError, Exception):
                pass
        assert passed >= 8, f"同义句通过率 {passed}/10 (要求 ≥8)"


class TestMultiTopicQuery:
    """4.6 多话题混合查询 —— ⚠️ 出数据不评分。"""

    MULTI_TOPIC_CASES = [
        ("A项目预算和猫的病", ["项目", "预算", "猫"]),
        ("今天天气怎么样代码部署到哪了", ["天气", "部署", "代码"]),
        ("上次那个bug修好没对了晚上吃啥", ["bug", "修复", "吃饭"]),
    ]

    def test_show_coverage(self):
        """跑一轮，输出每组的话题覆盖率，不断言。"""
        store = _get_store()
        results = []
        for query, expected_tags in self.MULTI_TOPIC_CASES:
            emb = _embed(query)
            r = store._read_collection.query(
                query_embeddings=[emb], n_results=15,
                include=["metadatas"]
            )
            retrieved_tags = set()
            for meta in (r["metadatas"][0] if r.get("metadatas") else []):
                tag_str = (meta.get("tags") or "") if isinstance(meta, dict) else ""
                retrieved_tags.update(t.lower() for t in tag_str.split(",") if len(t) > 1)
            covered = [t for t in expected_tags if t in retrieved_tags]
            results.append({
                "query": query[:30],
                "topics_expected": len(expected_tags),
                "topics_covered": len(covered),
                "topic_list": covered,
            })
        print(f"\n多话题覆盖率简报:")
        for r in results:
            print(f"  [{r['topics_covered']}/{r['topics_expected']}] {r['query']}: {r['topic_list']}")
        # 不 assert，只输出
