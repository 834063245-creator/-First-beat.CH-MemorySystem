"""测试 app/core/user_context.py — 用户上下文管理器。

覆盖：UserContextManager 的懒创建、移除、关闭、active_users。
"""
import pytest
from unittest.mock import MagicMock, patch
from app.core.user_context import UserContextManager, ctx_manager


# AppContext 在 get_context() 内部被 import from app.core.context
APPCTX_PATH = "app.core.context.AppContext"


class TestUserContextManager:
    def test_initial_active_users_empty(self):
        mgr = UserContextManager()
        assert mgr.active_users == []

    @patch(APPCTX_PATH)
    def test_get_context_creates_lazily(self, mock_appctx_cls):
        mock_ctx = MagicMock()
        mock_appctx_cls.return_value = mock_ctx
        mgr = UserContextManager()
        ctx = mgr.get_context("user_001", "/tmp/data")
        assert ctx is mock_ctx
        assert "user_001" in mgr.active_users
        mock_appctx_cls.assert_called_once_with(data_dir="/tmp/data")

    @patch(APPCTX_PATH)
    def test_get_context_returns_existing(self, mock_appctx_cls):
        mock_ctx = MagicMock()
        mock_appctx_cls.return_value = mock_ctx
        mgr = UserContextManager()
        ctx1 = mgr.get_context("user_001", "/tmp/data")
        ctx2 = mgr.get_context("user_001", "/tmp/data")
        assert ctx1 is ctx2
        assert mock_appctx_cls.call_count == 1

    @patch(APPCTX_PATH)
    def test_switch_user_closes_old_context(self, mock_appctx_cls):
        mock_ctx1 = MagicMock()
        mock_ctx2 = MagicMock()
        mock_appctx_cls.side_effect = [mock_ctx1, mock_ctx2]
        mgr = UserContextManager()
        mgr.get_context("user_001", "/tmp/data")
        mgr.get_context("user_002", "/tmp/data")
        mock_ctx1.close.assert_called_once()
        mock_ctx2.close.assert_not_called()
        assert "user_001" not in mgr.active_users
        assert "user_002" in mgr.active_users

    @patch(APPCTX_PATH)
    def test_remove_context_closes_and_removes(self, mock_appctx_cls):
        mock_ctx = MagicMock()
        mock_appctx_cls.return_value = mock_ctx
        mgr = UserContextManager()
        mgr.get_context("user_001", "/tmp/data")
        mgr.remove_context("user_001")
        mock_ctx.close.assert_called_once()
        assert "user_001" not in mgr.active_users

    def test_remove_nonexistent_no_error(self):
        mgr = UserContextManager()
        mgr.remove_context("no_such_user")  # 不抛异常

    @patch(APPCTX_PATH)
    def test_close_error_handled(self, mock_appctx_cls):
        mock_ctx = MagicMock()
        mock_ctx.close.side_effect = RuntimeError("close failed")
        mock_appctx_cls.return_value = mock_ctx
        mgr = UserContextManager()
        mgr.get_context("user_001", "/tmp/data")
        mgr.remove_context("user_001")  # 不抛异常

    @patch(APPCTX_PATH)
    def test_close_all_closes_everything(self, mock_appctx_cls):
        mock_ctx1 = MagicMock()
        mock_ctx2 = MagicMock()
        mock_appctx_cls.side_effect = [mock_ctx1, mock_ctx2]
        mgr = UserContextManager()
        mgr.get_context("user_001", "/tmp/data")
        mgr.get_context("user_002", "/tmp/data")
        mgr.close_all()
        mock_ctx1.close.assert_called_once()
        mock_ctx2.close.assert_called_once()
        assert mgr.active_users == []

    @patch(APPCTX_PATH)
    def test_close_all_handles_errors(self, mock_appctx_cls):
        mock_ctx = MagicMock()
        mock_ctx.close.side_effect = RuntimeError("close failed")
        mock_appctx_cls.return_value = mock_ctx
        mgr = UserContextManager()
        mgr.get_context("user_001", "/tmp/data")
        mgr.close_all()  # 不抛异常
        assert mgr.active_users == []


class TestGlobalCtxManager:
    """验证全局单例存在且可用。"""
    def test_global_instance_is_user_context_manager(self):
        assert isinstance(ctx_manager, UserContextManager)
