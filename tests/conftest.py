"""pytest 配置 — 注入项目根目录到 sys.path，确保 app.* 导入正常工作。

包含：
  - sys.path 注入（原逻辑）
  - isolated_env fixture：临时隔离 ChromaDB 环境
  - seeded_env fixture：预写入 12+ 条标准记忆的隔离环境
  - seed_memories() 辅助函数
"""
import os
import sys
import json
import time
import shutil
import tempfile
import threading
from datetime import datetime
from unittest.mock import patch, AsyncMock

import pytest
import httpx

# 项目根目录（tests/ 的上级目录）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将项目根目录加入路径
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ═══════════════════════════════════════════════════════════════════
# 全局 autouse fixture — mock Ollama embedding，避免测试卡住
# ═══════════════════════════════════════════════════════════════════

_DUMMY_EMB = [0.1] * 1024  # 确定性 dummy 向量


def _text_dependent_emb(text: str) -> list[float]:
    """根据文本内容生成确定性但不同的 embedding。

    避免所有文本返回同一向量导致语义分类始终选第一个原型。
    """
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    # 用 hash 的前 4 字节决定向量的第 0 维偏移（范围 -0.05~+0.05）
    val = int.from_bytes(h[:4], 'big') / (2**32) * 0.1 - 0.05
    emb = [0.1] * 1024
    emb[0] = 0.1 + val
    return emb


def _text_dependent_emb_batch(texts: list[str]) -> list[list[float]]:
    return [_text_dependent_emb(t) for t in texts]


def _skip_ollama_mock(item) -> bool:
    """标记了 @pytest.mark.real_embed 的测试跳过 autouse mock。"""
    return bool(item.get_closest_marker("real_embed"))


def _is_ollama_available() -> bool:
    """检测 Ollama 是否可用。"""
    try:
        url = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
        resp = httpx.get(f"{url}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


# 如果 Ollama 不可用，自动跳过 real_embed 测试
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
    """自动 mock Ollama 所有 HTTP 调用。

    未启动 Ollama 时防止测试因 HTTP 连接卡住。
    精确 mock 策略（不影响 FastAPI TestClient）：
    1) mock _embed_via_ollama / _embed_via_ollama_batch（文本相关向量，保证语义分类可用）
    2) mock LocalLLM.summarize（避免 Ollama HTTP 摘要调用）
    3) mock extract_entities（semantic 中直接调用 Ollama /api/chat）

    标记 @pytest.mark.real_embed 的测试不应用此 mock。
    """
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
# 标准种子记忆数据（硬编码，覆盖多话题）
# ═══════════════════════════════════════════════════════════════════

SEED_MEMORIES = [
    {
        "user": "我最近在学习Rust编程语言，感觉所有权系统很有意思，比C++的智能指针更优雅",
        "ai": "Rust的所有权系统确实是一个很棒的设计！它让内存安全在编译期就得到保证。你学到哪个阶段了？",
        "tags_expected": ["Rust", "编程", "学习"],
        "entities_expected": ["Rust"],
        "topic": "技术",
    },
    {
        "user": "今天我的橘猫又尿闭了，带他去宠物医院花了不少钱，医生说要注意饮食",
        "ai": "听到橘猫生病我很难过。宠物医院确实不便宜，但健康最重要。医生有建议换处方粮吗？",
        "tags_expected": ["橘猫", "宠物", "健康"],
        "entities_expected": ["橘猫"],
        "topic": "宠物",
    },
    {
        "user": "昨天和朋友去了东京旅行，吃了好多好吃的拉面，还去了浅草寺",
        "ai": "东京真的是一个很棒的城市！浅草寺的雷门很有名，拉面也是日本必吃的美食。旅途愉快！",
        "tags_expected": ["东京", "旅行", "拉面"],
        "entities_expected": ["东京", "浅草寺"],
        "topic": "旅行",
    },
    {
        "user": "最近工作压力好大，项目快上线了但是还有一堆bug没修完",
        "ai": "工作压力确实需要重视。上线前的冲刺阶段总是最累的，记得适当休息，别把自己逼太紧。",
        "tags_expected": ["工作", "压力", "项目"],
        "entities_expected": [],
        "topic": "工作",
    },
    {
        "user": "我每周都去健身房跑步，每次跑5公里，坚持了三个月了",
        "ai": "坚持运动三个月真的很厉害！5公里是个很好的有氧距离，坚持下去身体会越来越好。",
        "tags_expected": ["健身", "跑步", "运动"],
        "entities_expected": [],
        "topic": "生活",
    },
    {
        "user": "最近在读《黑客与画家》，Paul Graham关于编程语言的思考很有意思",
        "ai": "《黑客与画家》是一本经典之作！Paul Graham对Lisp的推崇和对创业的见解都很有启发性。",
        "tags_expected": ["阅读", "编程", "黑客与画家"],
        "entities_expected": ["Paul Graham", "黑客与画家"],
        "topic": "阅读",
    },
    {
        "user": "妈妈打电话说过几天要来郑州看我，好久没见到她了",
        "ai": "家人的关心总是很温暖。妈妈来看你是件开心的事，可以带她在郑州逛逛。",
        "tags_expected": ["妈妈", "家庭", "郑州"],
        "entities_expected": ["妈妈", "郑州"],
        "topic": "家庭",
    },
    {
        "user": "周杰伦的新专辑我听了，感觉还是老歌更经典，新歌少了一点味道",
        "ai": "周杰伦的音乐确实陪伴了我们很多年。老歌有青春的回忆在，新风格也需要时间品味。",
        "tags_expected": ["周杰伦", "音乐", "专辑"],
        "entities_expected": ["周杰伦"],
        "topic": "音乐",
    },
    {
        "user": "我的边牧学会了一个新技能，会用鼻子按门铃了，太聪明了",
        "ai": "边牧真的是最聪明的狗狗品种之一！会按门铃太厉害了，好好奖励它一下。",
        "tags_expected": ["边牧", "宠物", "训练"],
        "entities_expected": ["边牧"],
        "topic": "宠物",
    },
    {
        "user": "我在学微服务架构，感觉Docker和Kubernetes很强大但也好复杂",
        "ai": "微服务是现代架构的趋势，Docker和K8s虽然学习曲线陡峭，但掌握后效率提升很大。",
        "tags_expected": ["微服务", "Docker", "Kubernetes"],
        "entities_expected": ["Docker", "Kubernetes"],
        "topic": "技术",
    },
    {
        "user": "昨晚失眠了，一直在想年终奖会不会被砍，今年公司效益不太好",
        "ai": "年终奖确实让人焦虑，尤其是在效益不好的年份。不过焦虑也改变不了结果，不如先做好手头的事。",
        "tags_expected": ["失眠", "年终奖", "工作"],
        "entities_expected": [],
        "topic": "工作",
    },
    {
        "user": "下周打算去大阪旅行，听说那边的道顿堀美食特别多，还有环球影城",
        "ai": "大阪的美食确实很出名！道顿堀的章鱼烧和螃蟹招牌很有名，环球影城也值得一玩。",
        "tags_expected": ["大阪", "旅行", "美食"],
        "entities_expected": ["大阪", "道顿堀", "环球影城"],
        "topic": "旅行",
    },
]

# 冲突测试专用种子：矛盾的事实对
CONFLICT_SEED = [
    {
        "user": "我叫张三，今年25岁，在北京工作",
        "ai": "好的张三，我记住了，你在北京工作，25岁。",
        "tags_expected": ["张三", "北京", "工作"],
        "topic": "身份",
    },
    {
        "user": "不对，我叫李四，之前说错了，其实我在上海工作",
        "ai": "明白了，已更新你的名字为李四，工作地在上海。",
        "tags_expected": ["李四", "上海", "工作"],
        "topic": "身份",
    },
]

# Suppressed 记忆测试专用
SUPPRESSED_SEED = [
    {
        "user": "我有一个秘密要告诉你，我的银行卡密码是888999",
        "ai": "好的，我记住了。不过建议不要在公开场合分享密码信息。",
        "tags_expected": ["秘密", "密码"],
        "topic": "隐私",
    },
]


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def seed_memories(ctx, memories: list[dict], timestamp_base: str = "2026-06-01 10:00"):
    """向 AppContext 写入一批标准记忆。

    每条记忆走完整的 _store_conversation 管线（embed → tag → ChromaDB → 倒排索引）。

    Args:
        ctx: AppContext 实例
        memories: [{"user": str, "ai": str, "tags_expected": [...], ...}, ...]
        timestamp_base: 基准时间戳，每条记忆间隔 1 小时递增

    Returns:
        list[str]: 写入的记忆 ID 列表
    """
    ids = []
    base_dt = datetime.strptime(timestamp_base, "%Y-%m-%d %H:%M")
    for i, mem in enumerate(memories):
        ts_dt = base_dt.replace(hour=base_dt.hour + i)
        ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        # 在线程池外执行，确保入库完成
        ctx._store_conversation(mem["user"], mem["ai"], ts_str)
        ids.append(None)  # _store_conversation 不返回 id，通过检索回查
        time.sleep(0.05)  # 微延迟避免 ChromaDB 写入竞态
    # 等待队列 worker 处理完
    time.sleep(0.3)
    return ids


def _wait_store_queue(ctx, timeout: float = 2.0):
    """等待入库队列清空。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ctx._store_queue.empty():
            # 额外等一小段时间确保 worker 处理完
            time.sleep(0.2)
            return
        time.sleep(0.1)


def get_all_memory_ids(ctx) -> list[str]:
    """获取 ChromaDB 中所有记忆 ID。"""
    all_mems = ctx.chroma_service.list_all()
    return [m["id"] for m in all_mems]


def get_memory_by_id(ctx, mid: str) -> dict | None:
    """按 ID 获取完整记忆记录。"""
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
    """临时隔离环境 — 独立的 ChromaDB + 数据目录。

    每个测试获得全新的空数据库，测试结束后自动清理。
    """
    tmpdir = tempfile.mkdtemp(prefix="henscratch_test_")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 设置独立数据目录的环境变量（不影响全局）
    old_data_dir = os.environ.get("DATA_DIR")
    old_benchmark = os.environ.get("BENCHMARK_MODE")

    os.environ["DATA_DIR"] = data_dir
    os.environ["BENCHMARK_MODE"] = "true"  # 默认 benchmark 模式，放宽检索限制

    # 重新加载 settings 以反映新的 DATA_DIR
    import importlib
    import app.config.settings as _settings
    importlib.reload(_settings)

    # 创建 AppContext
    from app.core.context import AppContext
    ctx = AppContext(data_dir=data_dir)

    yield ctx

    # 清理
    try:
        ctx.close()
    except Exception:
        pass
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    # 恢复环境变量
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
    """临时隔离环境（非 Benchmark 模式）— 用于测试编织逻辑。"""
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
    """预写入 12 条标准记忆的隔离环境。

    记忆覆盖：技术、宠物、旅行、工作、生活、阅读、家庭、音乐等多个话题。
    返回 (ctx, memory_ids) 元组。
    """
    ctx = isolated_env

    # 批量写入种子记忆
    for i, mem in enumerate(SEED_MEMORIES):
        ts_dt = datetime(2026, 6, 1, 10 + i, 0, 0)
        ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        ctx._store_conversation(mem["user"], mem["ai"], ts_str)
        time.sleep(0.03)

    # 等待队列处理
    _wait_store_queue(ctx)
    time.sleep(0.5)

    # 收集所有写入的记忆 ID
    all_ids = get_all_memory_ids(ctx)
    return ctx, all_ids
