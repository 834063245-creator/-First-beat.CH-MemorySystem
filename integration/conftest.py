"""集成测试 pytest 配置 — 注入项目根目录，提供 isolated_env 等共用 fixture。

与 tests/conftest.py 解耦，integration/ 可单独运行。
"""
import os
import sys
import json
import time
import shutil
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest

# 项目根目录（integration/ 的上级目录）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ═══════════════════════════════════════════════════════════════════
# 全局 autouse fixture — mock Ollama embedding，避免测试卡住
# ═══════════════════════════════════════════════════════════════════

_DUMMY_EMB = [0.1] * 1024


def _text_dependent_emb(text: str) -> list[float]:
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    val = int.from_bytes(h[:4], 'big') / (2 ** 32) * 0.1 - 0.05
    emb = [0.1] * 1024
    emb[0] = 0.1 + val
    return emb


def _text_dependent_emb_batch(texts: list[str]) -> list[list[float]]:
    return [_text_dependent_emb(t) for t in texts]


def _is_ollama_available() -> bool:
    try:
        import httpx
        url = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
        resp = httpx.get(f"{url}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "real_embed: 需要真实 Ollama embedding，否则跳过")


def pytest_collection_modifyitems(config, items):
    if _is_ollama_available():
        return
    skip_mark = pytest.mark.skip(reason="Ollama 不可用")
    for item in items:
        if item.get_closest_marker("real_embed"):
            item.add_marker(skip_mark)


@pytest.fixture(autouse=True)
def _mock_ollama_http(request):
    if request.node.get_closest_marker("real_embed"):
        yield
        return

    with patch("app.llm.embed._embed_via_ollama", side_effect=_text_dependent_emb), \
         patch("app.llm.embed._embed_via_ollama_batch",
               side_effect=_text_dependent_emb_batch), \
         patch("app.llm.local.LocalLLM.summarize",
               return_value="mock摘要"), \
         patch("app.brain.semantic.extract_entities",
               return_value=[]):
        yield


# ═══════════════════════════════════════════════════════════════════
# 标准种子记忆数据
# ═══════════════════════════════════════════════════════════════════

SEED_MEMORIES = [
    {
        "user": "我最近在学习Rust编程语言，感觉所有权系统很有意思，比C++的智能指针更优雅",
        "ai": "Rust的所有权系统确实是一个很棒的设计！",
        "tags_expected": ["Rust", "编程", "学习"],
        "entities_expected": ["Rust"],
        "topic": "技术",
    },
    {
        "user": "今天我的橘猫又尿闭了，带他去宠物医院花了不少钱",
        "ai": "听到橘猫生病我很难过。医生有建议换处方粮吗？",
        "tags_expected": ["橘猫", "宠物", "健康"],
        "entities_expected": ["橘猫"],
        "topic": "宠物",
    },
    {
        "user": "昨天和朋友去了东京旅行，吃了好多好吃的拉面",
        "ai": "东京真的是一个很棒的城市！",
        "tags_expected": ["东京", "旅行", "拉面"],
        "entities_expected": ["东京", "浅草寺"],
        "topic": "旅行",
    },
    {
        "user": "最近工作压力好大，项目快上线了但是还有一堆bug没修完",
        "ai": "工作压力确实需要重视。记得适当休息。",
        "tags_expected": ["工作", "压力", "项目"],
        "entities_expected": [],
        "topic": "工作",
    },
    {
        "user": "我每周都去健身房跑步，每次跑5公里，坚持了三个月了",
        "ai": "坚持运动三个月真的很厉害！",
        "tags_expected": ["健身", "跑步", "运动"],
        "entities_expected": [],
        "topic": "生活",
    },
    {
        "user": "最近在读《黑客与画家》，Paul Graham关于编程语言的思考很有意思",
        "ai": "《黑客与画家》是一本经典之作！",
        "tags_expected": ["阅读", "编程", "黑客与画家"],
        "entities_expected": ["Paul Graham", "黑客与画家"],
        "topic": "阅读",
    },
    {
        "user": "妈妈打电话说过几天要来郑州看我，好久没见到她了",
        "ai": "家人的关心总是很温暖。",
        "tags_expected": ["妈妈", "家庭", "郑州"],
        "entities_expected": ["妈妈", "郑州"],
        "topic": "家庭",
    },
    {
        "user": "周杰伦的新专辑我听了，感觉还是老歌更经典",
        "ai": "周杰伦的音乐确实陪伴了我们很多年。",
        "tags_expected": ["周杰伦", "音乐", "专辑"],
        "entities_expected": ["周杰伦"],
        "topic": "音乐",
    },
    {
        "user": "我的边牧学会了一个新技能，会用鼻子按门铃了",
        "ai": "边牧真的是最聪明的狗狗品种之一！",
        "tags_expected": ["边牧", "宠物", "训练"],
        "entities_expected": ["边牧"],
        "topic": "宠物",
    },
    {
        "user": "我在学微服务架构，感觉Docker和Kubernetes很强大但也好复杂",
        "ai": "微服务是现代架构的趋势。",
        "tags_expected": ["微服务", "Docker", "Kubernetes"],
        "entities_expected": ["Docker", "Kubernetes"],
        "topic": "技术",
    },
    {
        "user": "昨晚失眠了，一直在想年终奖会不会被砍",
        "ai": "年终奖确实让人焦虑。",
        "tags_expected": ["失眠", "年终奖", "工作"],
        "entities_expected": [],
        "topic": "工作",
    },
    {
        "user": "下周打算去大阪旅行，听说那边的道顿堀美食特别多",
        "ai": "大阪的美食确实很出名！",
        "tags_expected": ["大阪", "旅行", "美食"],
        "entities_expected": ["大阪", "道顿堀", "环球影城"],
        "topic": "旅行",
    },
]

CONFLICT_SEED = [
    {
        "user": "我叫张三，今年25岁，在北京工作",
        "ai": "好的张三，我记住了。",
        "tags_expected": ["张三", "北京", "工作"],
        "topic": "身份",
    },
    {
        "user": "不对，我叫李四，之前说错了，其实我在上海工作",
        "ai": "明白了，已更新你的名字为李四。",
        "tags_expected": ["李四", "上海", "工作"],
        "topic": "身份",
    },
]

SUPPRESSED_SEED = [
    {
        "user": "我有一个秘密要告诉你，我的银行卡密码是888999",
        "ai": "好的，我记住了。不过建议不要在公开场合分享密码。",
        "tags_expected": ["秘密", "密码"],
        "topic": "隐私",
    },
]


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def seed_memories(ctx, memories: list[dict], timestamp_base: str = "2026-06-01 10:00"):
    ids = []
    base_dt = datetime.strptime(timestamp_base, "%Y-%m-%d %H:%M")
    for i, mem in enumerate(memories):
        ts_dt = base_dt.replace(hour=base_dt.hour + i)
        ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        ctx._store_conversation(mem["user"], mem["ai"], ts_str)
        ids.append(None)
        time.sleep(0.05)
    time.sleep(0.3)
    return ids


def _wait_store_queue(ctx, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ctx._store_queue.empty():
            time.sleep(0.2)
            return
        time.sleep(0.1)


def get_all_memory_ids(ctx) -> list[str]:
    all_mems = ctx.chroma_service.list_all()
    return [m["id"] for m in all_mems]


def get_memory_by_id(ctx, mid: str) -> dict | None:
    try:
        result = ctx.chroma_service._collection.get(
            ids=[mid],
            include=["documents", "metadatas", "embeddings"],
        )
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "document": result["documents"][0] if result.get("documents") else "",
                "metadata": dict(result["metadatas"][0]) if result.get("metadatas") else {},
            }
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def isolated_env():
    """临时隔离环境 — 独立的 ChromaDB + 数据目录。"""
    tmpdir = tempfile.mkdtemp(prefix="henscratch_int_")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    old_data_dir = os.environ.get("DATA_DIR")
    old_benchmark = os.environ.get("BENCHMARK_MODE")

    os.environ["DATA_DIR"] = data_dir
    os.environ["BENCHMARK_MODE"] = "true"

    import importlib
    import app.config.settings as _settings
    importlib.reload(_settings)

    from app.core.context import AppContext
    ctx = AppContext(data_dir=data_dir)

    yield ctx

    try:
        ctx.close()
    except Exception:
        pass
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    if old_data_dir is not None:
        os.environ["DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("DATA_DIR", None)
    if old_benchmark is not None:
        os.environ["BENCHMARK_MODE"] = old_benchmark
    else:
        os.environ.pop("BENCHMARK_MODE", None)


@pytest.fixture
def isolated_env_no_bm():
    """临时隔离环境（非 Benchmark 模式）。"""
    tmpdir = tempfile.mkdtemp(prefix="henscratch_nobm_")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    old_data_dir = os.environ.get("DATA_DIR")
    old_benchmark = os.environ.get("BENCHMARK_MODE")

    os.environ["DATA_DIR"] = data_dir
    os.environ["BENCHMARK_MODE"] = "false"

    import importlib
    import app.config.settings as _settings
    importlib.reload(_settings)

    from app.core.context import AppContext
    ctx = AppContext(data_dir=data_dir)

    yield ctx

    try:
        ctx.close()
    except Exception:
        pass
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    if old_data_dir is not None:
        os.environ["DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("DATA_DIR", None)
    if old_benchmark is not None:
        os.environ["BENCHMARK_MODE"] = old_benchmark
    else:
        os.environ.pop("BENCHMARK_MODE", None)


@pytest.fixture
def seeded_env(isolated_env):
    """预写入 12 条标准记忆的隔离环境。返回 (ctx, memory_ids)。"""
    ctx = isolated_env

    for i, mem in enumerate(SEED_MEMORIES):
        ts_dt = datetime(2026, 6, 1, 10 + i, 0, 0)
        ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        ctx._store_conversation(mem["user"], mem["ai"], ts_str)
        time.sleep(0.03)

    _wait_store_queue(ctx)
    time.sleep(0.5)

    all_ids = get_all_memory_ids(ctx)
    return ctx, all_ids
