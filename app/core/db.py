"""DEPRECATED (Phase 3): SQLite 连接池已退役，Qdrant 替代。

保留兼容桩：close_all() 和 get_db() 供旧测试 teardown 使用。
Phase 5 清理测试后删除此文件。
"""
import warnings

def get_db(file_path: str):
    """已废弃。SQLite 存储已迁移至 Qdrant。"""
    warnings.warn(
        "get_db() is deprecated. SQLite storage migrated to Qdrant.",
        DeprecationWarning, stacklevel=2,
    )
    raise NotImplementedError("SQLite storage has been migrated to Qdrant.")


def close_all():
    """已废弃。SQLite 连接池已退役，no-op。"""
    pass


def close_db(file_path: str):
    """已废弃。no-op。"""
    pass
