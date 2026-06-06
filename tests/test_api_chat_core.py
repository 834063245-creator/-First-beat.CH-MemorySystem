"""测试 app/api/chat.py 核心端点（mock 全链路）。

覆盖：/benchmark/inject /admin/reset /chat 非流式 /chat/stream 空消息/流式。
"""
import asyncio
import concurrent.futures
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from fastapi.testclient import TestClient


def _make_chat_ctx():
    """构建 chat 端点需要的完整假 AppContext。"""
    ctx = MagicMock()
    ctx.data_dir = "/tmp/test"
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _run_in_executor(executor, fn, *args, **kwargs):
        """模拟 run_in_executor：同步调用 fn（测试环境不用真线程）。"""
        return fn(*args, **kwargs)

    ctx.retrieval_executor = pool
    ctx.storage_executor = pool
    # 把 run_in_executor 换成同步版避免线程问题
    ctx._real_run = _run_in_executor

    # chroma_service
    ctx.chroma_service.clear_all = MagicMock()
    ctx.chroma_service._collection = MagicMock()
    ctx.chroma_service._collection.name = "test_collection"

    # ai_chroma_service
    ctx.ai_chroma_service.clear_all = MagicMock()

    # chat_history
    ctx.chat_history.append = MagicMock()
    ctx.chat_history.clear = MagicMock()
    ctx.chat_history.records = []

    # inverted_index / co_tracker
    ctx.inverted_index.clear = MagicMock()
    ctx.co_tracker.clear = MagicMock()
    ctx.ai_co_tracker.clear = MagicMock()

    # bm25_index
    ctx.bm25_index = None

    # llm_client
    ctx.llm_client = MagicMock()
    # personality_store / dmn / etc
    ctx.personality_store = MagicMock()
    ctx.impulse_scheduler = MagicMock()
    ctx.dmn = MagicMock()
    ctx.mirror_neuron = MagicMock()

    # _enqueue_store_task
    ctx._enqueue_store_task = MagicMock()
    # _store_conversation
    ctx._store_conversation = MagicMock()

    return ctx


@pytest.fixture
def client():
    from app.api.app import app
    from app.api.deps import get_user_context
    ctx = _make_chat_ctx()
    app.dependency_overrides[get_user_context] = lambda: ctx
    yield TestClient(app), ctx
    app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════
# Benchmark Inject
# ═══════════════════════════════════════════════════════

class TestBenchmarkInject:
    def test_injects_memory(self, client):
        cli, ctx = client
        resp = cli.post("/benchmark/inject", json={
            "user_message": "我喜欢咖啡",
            "ai_message": "咖啡确实很棒",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        ctx._store_conversation.assert_called_once()

    def test_with_timestamp(self, client):
        cli, ctx = client
        resp = cli.post("/benchmark/inject", json={
            "user_message": "test",
            "ai_message": "reply",
            "timestamp": "2025-06-01 12:00:00",
        })
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════
# Admin Reset
# ═══════════════════════════════════════════════════════

class TestAdminReset:
    def test_resets_all_stores(self, client):
        cli, ctx = client
        resp = cli.post("/admin/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        ctx.chroma_service.clear_all.assert_called_once()
        ctx.ai_chroma_service.clear_all.assert_called_once()
        ctx.chat_history.clear.assert_called_once()
        ctx.inverted_index.clear.assert_called_once()
        ctx.co_tracker.clear.assert_called_once()
        ctx.ai_co_tracker.clear.assert_called_once()


# ═══════════════════════════════════════════════════════
# Chat (non-stream) — fully mocked
# ═══════════════════════════════════════════════════════

class TestChatNonStream:
    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_returns_response(self, mock_circuit_cls, mock_retrieval, mock_embed, client):
        cli, ctx = client
        mock_embed.return_value = [0.1] * 1024
        mock_retrieval.return_value = ([], "", [], [])
        # fake utterance_spec
        fake_spec = MagicMock()
        fake_spec.user.intent = "casual"
        fake_spec.user.emotion = "neutral"
        fake_spec.memories = []
        fake_spec.impulses = []
        fake_spec.emotional_reversals = []
        mock_circuit_cls.return_value.process.return_value = fake_spec
        # fake LLM response
        ctx.llm_client.generate = AsyncMock(return_value={
            "content": "你好！很高兴见到你。",
            "tool_calls": [],
            "reasoning_content": "",
        })
        resp = cli.post("/chat", json={"message": "你好"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "你好" in data["response"]

    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_empty_message_returns_prompt(self, mock_circuit, mock_retrieval, mock_embed, client):
        cli, _ = client
        resp = cli.post("/chat", json={"message": ""})
        assert resp.status_code == 200
        assert "说点什么" in resp.json()["response"]

    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_llm_error_handled(self, mock_circuit_cls, mock_retrieval, mock_embed, client):
        cli, ctx = client
        mock_embed.return_value = [0.1] * 1024
        mock_retrieval.return_value = ([], "", [], [])
        fake_spec = MagicMock()
        fake_spec.user.intent = "casual"
        fake_spec.user.emotion = "neutral"
        fake_spec.memories = []
        fake_spec.impulses = []
        fake_spec.emotional_reversals = []
        mock_circuit_cls.return_value.process.return_value = fake_spec
        ctx.llm_client.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        resp = cli.post("/chat", json={"message": "测试"})
        assert resp.status_code == 200
        assert "暂时不可用" in resp.json()["response"]

    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_test_mode_skips_storage(self, mock_circuit_cls, mock_retrieval, mock_embed, client):
        cli, ctx = client
        mock_embed.return_value = [0.1] * 1024
        mock_retrieval.return_value = ([], "", [], [])
        fake_spec = MagicMock()
        fake_spec.user.intent = "casual"
        fake_spec.user.emotion = "neutral"
        fake_spec.memories = []
        fake_spec.impulses = []
        fake_spec.emotional_reversals = []
        mock_circuit_cls.return_value.process.return_value = fake_spec
        ctx.llm_client.generate = AsyncMock(return_value={
            "content": "ok", "tool_calls": [], "reasoning_content": "",
        })
        resp = cli.post("/chat", json={"message": "test", "test_mode": True})
        assert resp.status_code == 200
        # test_mode 下不应调用 append
        ctx.chat_history.append.assert_not_called()

    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_benchmark_inject_skips_history(self, mock_circuit_cls, mock_retrieval, mock_embed, client):
        cli, ctx = client
        mock_embed.return_value = [0.1] * 1024
        mock_retrieval.return_value = ([], "", [], [])
        fake_spec = MagicMock()
        fake_spec.user.intent = "casual"
        fake_spec.user.emotion = "neutral"
        fake_spec.memories = []
        fake_spec.impulses = []
        fake_spec.emotional_reversals = []
        mock_circuit_cls.return_value.process.return_value = fake_spec
        ctx.llm_client.generate = AsyncMock(return_value={
            "content": "ok", "tool_calls": [], "reasoning_content": "",
        })
        resp = cli.post("/chat", json={"message": "test", "benchmark_inject": True})
        assert resp.status_code == 200
        ctx.chat_history.append.assert_not_called()
        ctx._enqueue_store_task.assert_called_once()


# ═══════════════════════════════════════════════════════
# Chat Stream — minimal tests
# ═══════════════════════════════════════════════════════

class TestChatStream:
    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_stream_emits_content(self, mock_circuit_cls, mock_retrieval, mock_embed, client):
        cli, ctx = client
        mock_embed.return_value = [0.1] * 1024
        mock_retrieval.return_value = ([], "", [], [])
        fake_spec = MagicMock()
        fake_spec.user.intent = "casual"
        fake_spec.user.emotion = "neutral"
        fake_spec.memories = []
        fake_spec.impulses = []
        fake_spec.emotional_reversals = []
        mock_circuit_cls.return_value.process.return_value = fake_spec
        # 模拟流式输出
        async def fake_stream(*args, **kwargs):
            yield ("content", "你")
            yield ("content", "好")
        ctx.llm_client.generate_stream = fake_stream
        resp = cli.post("/chat/stream", json={"message": "你好"})
        assert resp.status_code == 200
        body = resp.text
        assert "你" in body or "[CONTENT]" in body

    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_stream_emits_trace(self, mock_circuit_cls, mock_retrieval, mock_embed, client):
        cli, ctx = client
        mock_embed.return_value = [0.1] * 1024
        mock_retrieval.return_value = ([], "", [], [])
        fake_spec = MagicMock()
        fake_spec.user.intent = "casual"
        fake_spec.user.emotion = "neutral"
        fake_spec.memories = []
        fake_spec.impulses = []
        fake_spec.emotional_reversals = []
        mock_circuit_cls.return_value.process.return_value = fake_spec
        async def fake_stream(*args, **kwargs):
            yield ("content", "hi")
        ctx.llm_client.generate_stream = fake_stream
        resp = cli.post("/chat/stream", json={"message": "hi"})
        body = resp.text
        assert "[TRACE]" in body
        assert "[DONE]" in body

    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_stream_with_tool_calls_loops(self, mock_circuit_cls, mock_retrieval, mock_embed, client):
        cli, ctx = client
        mock_embed.return_value = [0.1] * 1024
        mock_retrieval.return_value = ([], "", [], [])
        fake_spec = MagicMock()
        fake_spec.user.intent = "casual"
        fake_spec.user.emotion = "neutral"
        fake_spec.memories = []
        fake_spec.impulses = []
        fake_spec.emotional_reversals = []
        mock_circuit_cls.return_value.process.return_value = fake_spec
        round_count = [0]
        async def fake_stream(*args, **kwargs):
            round_count[0] += 1
            if round_count[0] == 1:
                yield ("tool_calls", {"calls": [{
                    "id": "tc1", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"x"}'}
                }], "reasoning_content": ""})
            else:
                yield ("content", "结果")
        ctx.llm_client.generate_stream = fake_stream
        with patch("app.api.chat.read_file", return_value="文件内容"):
            resp = cli.post("/chat/stream", json={"message": "读文件"})
            body = resp.text
            assert "[TRACE]" in body or "[CONTENT]" in body

    @patch("app.api.chat.local_embed_async", new_callable=AsyncMock)
    @patch("app.api.chat.run_chat_retrieval")
    @patch("app.api.chat.CircuitOrchestrator")
    def test_stream_error_emits_error_event(self, mock_circuit_cls, mock_retrieval, mock_embed, client):
        cli, ctx = client
        mock_embed.return_value = [0.1] * 1024
        mock_retrieval.return_value = ([], "", [], [])
        fake_spec = MagicMock()
        fake_spec.user.intent = "casual"
        mock_circuit_cls.return_value.process.return_value = fake_spec
        async def fake_stream(*args, **kwargs):
            raise RuntimeError("stream crash")
            yield ("content", "never")  # noqa
        ctx.llm_client.generate_stream = fake_stream
        resp = cli.post("/chat/stream", json={"message": "崩溃测试"})
        body = resp.text
        assert "[ERROR]" in body
