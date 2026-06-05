"""巩固安全审计 — 验证 DMN 不修改记忆原文。"""

import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from app.config.settings import DATA_DIR
from app.memory.chroma import ChromaService


def _memory_digest(persist_dir: str, n: int = 100) -> dict[str, str]:
    """从 ChromaDB 随机采样 N 条记忆，返回 {id: document_hash}。"""
    import random as _r
    _r.seed(42)
    cs = ChromaService(persist_dir=persist_dir)
    all_data = cs._collection.get(include=["documents"])
    all_ids = all_data.get("ids", [])
    all_docs = all_data.get("documents", [])
    if not all_ids:
        return {}
    sample = _r.sample(list(range(len(all_ids))), min(n, len(all_ids)))
    return {
        all_ids[i]: hashlib.md5((all_docs[i] or "").encode()).hexdigest()
        for i in sample
    }


class TestConsolidationSafety:
    """4.4 巩固不修改原文。"""

    @pytest.fixture
    def chroma_persist_dir(self):
        return os.path.join(DATA_DIR, "chroma")

    @pytest.fixture
    def temp_state_dir(self):
        import tempfile
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def test_shallow_does_not_modify_documents(self, chroma_persist_dir, temp_state_dir):
        """触发一次浅巩固，确认 document 字段未被修改。"""
        before = _memory_digest(chroma_persist_dir, 50)
        if not before:
            pytest.skip("没有足够记忆样本")

        cs = ChromaService(persist_dir=chroma_persist_dir)
        import types
        mock_store = types.SimpleNamespace()
        mock_store.list_all = lambda: []

        from app.background.consolidation import ConsolidationEngine
        dmn = ConsolidationEngine(
            chroma_service=cs,
            personality_store=mock_store,
            behavior_store=mock_store,
            chat_history=None,
            co_tracker=None,
            state_path=os.path.join(temp_state_dir, "dmn_state.json"),
            notes_path=os.path.join(temp_state_dir, "topic_notes.json"),
        )
        dmn.consolidate_shallow()

        after = _memory_digest(chroma_persist_dir, 50)
        shared = set(before.keys()) & set(after.keys())
        mismatches = [mid for mid in shared if before[mid] != after[mid]]
        assert len(mismatches) == 0, (
            f"巩固修改了 {len(mismatches)} 条记忆的 document"
        )
