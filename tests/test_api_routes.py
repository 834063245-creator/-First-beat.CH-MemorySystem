"""测试薄路由端点 — chat_history / consolidation / distill / personalities。

覆盖：所有依赖 AppContext 的 GET/POST/DELETE 端点。
使用 dependency_overrides 注入统一假 AppContext。
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def _make_fake_ctx():
    ctx = MagicMock()
    ctx.data_dir = "/tmp/test"

    # chat_history
    ctx.chat_history.get_recent.return_value = [
        {"user": "你好", "ai": "你好呀", "timestamp": "2025-01-01 10:00:00"},
    ]
    ctx.chat_history.delete_by_timestamp.return_value = True

    # dmn (consolidation)
    ctx.dmn.get_status.return_value = {"shallow_last": 1700000000, "deep_last": 1700000100}

    # distill_engine
    ctx.distill_engine.get_state.return_value = {"patterns_count": 5, "last_run": 1700000000}
    ctx.distill_engine.run_distill.return_value = ["pattern_1", "pattern_2"]

    # personality_store
    ctx.personality_store.list_tags.return_value = {
        "items": [{"id": "p1", "content": "标签1", "type": "行为模式", "confidence": "高"}],
        "total": 1, "page": 1, "page_size": 20,
    }
    ctx.personality_store.get_tag.return_value = {
        "id": "p1", "content": "标签1", "type": "行为模式", "confidence": "高",
        "outdated": False, "hit_count": 5, "last_hit_time": 1700000000, "created_at": 1700000000,
    }
    ctx.personality_store.delete_tag = MagicMock()

    return ctx


@pytest.fixture
def client():
    from app.api.app import app
    from app.api.deps import get_user_context
    fake = _make_fake_ctx()
    app.dependency_overrides[get_user_context] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════
# Chat History
# ═══════════════════════════════════════════════════════

class TestChatHistory:
    def test_get_recent(self, client):
        resp = client.get("/api/chat/history")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["user"] == "你好"

    def test_delete_by_timestamp_ok(self, client):
        resp = client.delete("/api/chat/history/2025-01-01%2010:00:00")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_delete_not_found(self, client):
        from app.api.app import app
        from app.api.deps import get_user_context
        fake = _make_fake_ctx()
        fake.chat_history.delete_by_timestamp.return_value = False
        app.dependency_overrides[get_user_context] = lambda: fake
        resp = client.delete("/api/chat/history/nonexistent")
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════
# Consolidation
# ═══════════════════════════════════════════════════════

class TestConsolidation:
    def test_status(self, client):
        resp = client.get("/api/consolidation/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "shallow_last" in data

    def test_status_error(self, client):
        from app.api.app import app
        from app.api.deps import get_user_context
        fake = _make_fake_ctx()
        fake.dmn.get_status.side_effect = RuntimeError("db down")
        app.dependency_overrides[get_user_context] = lambda: fake
        resp = client.get("/api/consolidation/status")
        assert resp.status_code == 200
        assert "error" in resp.json()


# ═══════════════════════════════════════════════════════
# Distill
# ═══════════════════════════════════════════════════════

class TestDistill:
    def test_status(self, client):
        resp = client.get("/api/distill/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["patterns_count"] == 5

    def test_trigger(self, client):
        resp = client.post("/api/distill")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["patterns"] == 2

    def test_trigger_error(self, client):
        from app.api.app import app
        from app.api.deps import get_user_context
        fake = _make_fake_ctx()
        fake.distill_engine.run_distill.side_effect = RuntimeError("engine fail")
        app.dependency_overrides[get_user_context] = lambda: fake
        resp = client.post("/api/distill")
        assert resp.status_code == 200
        assert resp.json()["ok"] is False


# ═══════════════════════════════════════════════════════
# Personalities
# ═══════════════════════════════════════════════════════

class TestPersonalitiesAPI:
    def test_list(self, client):
        resp = client.get("/api/personalities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    def test_detail(self, client):
        resp = client.get("/api/personalities/p1")
        assert resp.status_code == 200
        assert resp.json()["content"] == "标签1"

    def test_detail_not_found(self, client):
        from app.api.app import app
        from app.api.deps import get_user_context
        fake = _make_fake_ctx()
        fake.personality_store.get_tag.return_value = None
        app.dependency_overrides[get_user_context] = lambda: fake
        resp = client.get("/api/personalities/nonexistent")
        assert resp.status_code == 404

    def test_delete(self, client):
        resp = client.delete("/api/personalities/p1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
