"""pytest 配置 — 注入项目根目录到 sys.path + 写入链路隔离 fixture。

包含：
  - sys.path 注入（原逻辑）
  - isolated_env fixture：临时隔离 ChromaDB 环境
  - seeded_env fixture：预写入 12+ 条标准记忆的隔离环境
  - seed_memories() 辅助函数
  - 各组件级隔离 fixture（链路一用）
"""
import importlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime

import pytest


# ── 注册自定义 mark ──
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要服务器在运行才能执行的集成测试")

# ── 项目根目录注入 ──
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── 固定种子，确保可复现 ──
SEED = 42


@pytest.fixture(scope="session", autouse=True)
def _fix_seed():
    """全局固定随机种子。"""
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import numpy as np
    np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════
# 写入链路隔离 fixture
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def temp_data_dir():
    """每个测试独立的临时数据目录，测试结束后自动清理。"""
    d = tempfile.mkdtemp(prefix="bench_w1_")
    yield d
    try:
        shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def isolated_chroma_service(temp_data_dir):
    """隔离的 ChromaService 实例，使用临时目录。"""
    from app.memory.chroma import ChromaService
    svc = ChromaService(
        persist_dir=os.path.join(temp_data_dir, "chroma"),
        collection_name=f"test_memories_{random.randint(0, 99999)}",
    )
    yield svc
    try:
        svc.close()
    except Exception:
        pass


@pytest.fixture
def isolated_ai_chroma_service(temp_data_dir):
    """隔离的 AI ChromaService 实例。"""
    from app.memory.chroma import ChromaService
    svc = ChromaService(
        persist_dir=os.path.join(temp_data_dir, "ai_chroma"),
        collection_name=f"test_ai_memories_{random.randint(0, 99999)}",
    )
    yield svc
    try:
        svc.close()
    except Exception:
        pass


@pytest.fixture
def isolated_chat_history(temp_data_dir):
    """隔离的 ChatHistory 实例。"""
    from app.memory.history import ChatHistory
    path = os.path.join(temp_data_dir, "chat_history.jsonl")
    ch = ChatHistory(path=path, max_memory=100)
    yield ch
    # 清理
    try:
        os.remove(path)
    except Exception:
        pass


@pytest.fixture
def isolated_inverted_index():
    """隔离的 InvertedIndex 实例。"""
    from app.memory.inverted import InvertedIndex
    idx = InvertedIndex()
    yield idx
    idx.clear()


@pytest.fixture
def isolated_entity_pair_tracker(temp_data_dir):
    """隔离的实体对跟踪器（Phase 3 兼容 shim）。

    Phase 3 已将 EntityPairTracker 功能迁至 Qdrant payload entity_co_counts。
    此 fixture 提供最小化兼容实现，供 E2E 测试使用。
    Phase 5 将移除独立的 entity_pair E2E 测试。
    """
    from types import SimpleNamespace

    # 简单内存存储模拟旧 EntityPairTracker API
    _pairs: dict[str, dict] = {}  # entity → {partner: {count, memory_ids}}
    _memory_pairs: dict[str, list] = {}  # memory_id → [(entity_a, entity_b)]

    class _CompatTracker:
        EXPAND_TOP_K = 5
        MAX_MEMORIES = 30

        def record(self, entity_a, entity_b, memory_id):
            if not entity_a or not entity_b or entity_a == entity_b:
                return
            for a, b in [(entity_a, entity_b), (entity_b, entity_a)]:
                if a not in _pairs:
                    _pairs[a] = {}
                if b not in _pairs[a]:
                    _pairs[a][b] = {"count": 0, "memory_ids": []}
                _pairs[a][b]["count"] += 1
                if memory_id not in _pairs[a][b]["memory_ids"]:
                    _pairs[a][b]["memory_ids"].append(memory_id)
            if memory_id not in _memory_pairs:
                _memory_pairs[memory_id] = []
            _memory_pairs[memory_id].append((entity_a, entity_b))

        def expand(self, entity_names):
            result = {}
            for ename in entity_names:
                if ename in _pairs:
                    related = _pairs[ename]
                    result[ename] = {k: v["count"] for k, v in related.items()}
            return result

        def get_memory_ids(self, entity_names):
            scored = {}
            for ename in entity_names:
                if ename in _pairs:
                    for partner, info in _pairs[ename].items():
                        for mid in info.get("memory_ids", []):
                            scored[mid] = scored.get(mid, 0) + info["count"]
            sorted_ids = sorted(scored.items(), key=lambda x: -x[1])
            return [mid for mid, _ in sorted_ids[:self.MAX_MEMORIES]]

        def remove_memory(self, memory_id):
            if memory_id in _memory_pairs:
                for a, b in _memory_pairs.pop(memory_id):
                    if a in _pairs and b in _pairs[a]:
                        _pairs[a][b]["count"] = max(0, _pairs[a][b]["count"] - 1)
                        if memory_id in _pairs[a][b]["memory_ids"]:
                            _pairs[a][b]["memory_ids"].remove(memory_id)

        def stats(self):
            total_pairs = sum(
                len(partners) for partners in _pairs.values()
            )
            return {"total_entities": len(_pairs), "total_pairs": total_pairs}

        def _load(self):
            data = {}
            for a, partners in _pairs.items():
                data[a] = {b: {"count": v["count"], "memory_ids": v["memory_ids"]}
                          for b, v in partners.items()}
            return data

    tracker = _CompatTracker()
    yield tracker


# ═══════════════════════════════════════════════════════════════
# 服务器可达性检测（W12 用）
# ═══════════════════════════════════════════════════════════════

_SERVER_URL = os.environ.get("BENCH_SERVER_URL", "http://localhost:8082")


def _is_server_alive(url: str, timeout: float = 2.0) -> bool:
    """检测服务器是否可达。"""
    try:
        import httpx
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{url}/v1/models")
            return resp.status_code in (200, 404, 405)  # 有响应即视为 alive
    except Exception:
        return False


@pytest.fixture(scope="session")
def server_url():
    """基准测试服务器地址。"""
    return _SERVER_URL


@pytest.fixture(scope="session")
def server_alive():
    """检测服务器是否在运行。"""
    return _is_server_alive(_SERVER_URL)


# ═══════════════════════════════════════════════════════════════════
# AppContext 级隔离 fixture（链路三跨轮测试用）
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def isolated_env():
    """临时隔离环境 — 独立的 ChromaDB + 数据目录 + 完整 AppContext。

    每个测试获得全新的空数据库，测试结束后自动清理。
    默认启用 BENCHMARK_MODE（快速入库，跳过 LLM 摘要）。
    """
    tmpdir = tempfile.mkdtemp(prefix="henscratch_test_")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    old_data_dir = os.environ.get("DATA_DIR")
    old_benchmark = os.environ.get("BENCHMARK_MODE")

    os.environ["DATA_DIR"] = data_dir
    os.environ["BENCHMARK_MODE"] = "true"

    import app.config.settings as _settings
    importlib.reload(_settings)

    from app.core.context import AppContext
    ctx = AppContext(data_dir=data_dir)

    yield ctx

    # 清理
    try:
        ctx.close()
    except Exception:
        pass
    try:
        from app.core.db import close_all; close_all()
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
    """临时隔离环境（非 Benchmark 模式）— 用于测试编织逻辑。"""
    tmpdir = tempfile.mkdtemp(prefix="henscratch_nobm_")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    old_data_dir = os.environ.get("DATA_DIR")
    old_benchmark = os.environ.get("BENCHMARK_MODE")

    os.environ["DATA_DIR"] = data_dir
    os.environ["BENCHMARK_MODE"] = "false"

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
        from app.core.db import close_all; close_all()
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


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def _wait_store_queue(ctx, timeout: float = 2.0):
    """等待入库队列清空。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ctx._store_queue.empty():
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
# 标准种子记忆（硬编码，覆盖多话题）
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


@pytest.fixture
def seeded_env(isolated_env):
    """预写入 12 条标准记忆的隔离环境。

    记忆覆盖：技术、宠物、旅行、工作、生活、阅读、家庭、音乐等多个话题。
    返回 (ctx, memory_ids) 元组。
    """
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


# ═══════════════════════════════════════════════════════════════════
# 链路四：记忆演化链路 fixture & 辅助函数
# ═══════════════════════════════════════════════════════════════════

# 演化测试用种子记忆 — 跨多天，覆盖多话题，部分高情绪强度
EVOLUTION_SEED_MEMORIES = [
    # 技术话题（近 7 天内）
    {"user": "我最近在学习Rust编程语言，感觉所有权系统很有意思", "ai": "Rust的所有权系统确实很棒！编译期内存安全是它最大优势。",
     "ts": "2026-05-30 10:00:00", "topic": "技术"},
    {"user": "今天用Python写了个爬虫，抓取了电商网站的数据", "ai": "爬虫是很好的实践！Scrapy框架很适合做电商数据抓取。",
     "ts": "2026-05-31 10:00:00", "topic": "技术"},
    {"user": "我在学微服务架构，Docker和Kubernetes很强大但也好复杂", "ai": "微服务是趋势，Docker和K8s虽然学习曲线陡但值得投入。",
     "ts": "2026-06-01 10:00:00", "topic": "技术"},
    {"user": "Python异步编程用asyncio还是trio？哪个更好？", "ai": "asyncio是标准库，生态更成熟；trio设计更优雅但生态较小。",
     "ts": "2026-06-02 10:00:00", "topic": "技术"},
    # 宠物话题（近 14 天内）
    {"user": "我的橘猫今天又尿闭了，带他去宠物医院花了不少钱", "ai": "听到橘猫生病我很难过。宠物医院的费用确实不便宜。",
     "ts": "2026-05-24 15:00:00", "topic": "宠物"},
    {"user": "边牧学会了一个新技能，会用鼻子按门铃了", "ai": "边牧真的是最聪明的狗狗！会按门铃太厉害了。",
     "ts": "2026-05-28 15:00:00", "topic": "宠物"},
    {"user": "今天给橘猫换了处方粮，希望不要再尿闭了", "ai": "处方粮对泌尿系统健康很有帮助，坚持吃会好转的。",
     "ts": "2026-06-01 15:00:00", "topic": "宠物"},
    # 旅行话题（远至 40 天前）
    {"user": "昨天和朋友去了东京旅行，吃了好多好吃的拉面", "ai": "东京的拉面确实很棒！浅草寺的雷门也很有名。",
     "ts": "2026-04-25 12:00:00", "topic": "旅行"},
    {"user": "下周打算去大阪旅行，道顿堀美食特别多", "ai": "大阪美食很出名！道顿堀的章鱼烧和环球影城都值得去。",
     "ts": "2026-04-28 12:00:00", "topic": "旅行"},
    # 工作话题（部分高情绪）
    {"user": "最近工作压力好大，项目快上线了但是还有一堆bug", "ai": "上线前的冲刺总是最累的，记得适当休息。",
     "ts": "2026-06-01 18:00:00", "topic": "工作"},
    {"user": "年终奖会不会被砍？今年公司效益不太好😭焦虑！", "ai": "年终奖确实让人焦虑，但焦虑也改变不了结果，先做好手头的事。",
     "ts": "2026-05-20 18:00:00", "topic": "工作"},
    {"user": "被同事坑了！！！太愤怒了！！！工作量翻倍！！！", "ai": "我感受到你的愤怒了，同事之间的信任问题确实很让人头疼。",
     "ts": "2026-05-22 19:00:00", "topic": "工作"},
    # 阅读/生活话题
    {"user": "在读《黑客与画家》，Paul Graham对编程语言的思考很有意思", "ai": "《黑客与画家》是经典！Paul Graham对Lisp的推崇很有启发性。",
     "ts": "2026-05-20 14:00:00", "topic": "阅读"},
    {"user": "今天去健身房跑了5公里，坚持三个月了", "ai": "坚持运动三个月真的很厉害！5公里是个很好的有氧距离。",
     "ts": "2026-05-25 07:00:00", "topic": "生活"},
]


@pytest.fixture
def seeded_env_evolution(isolated_env):
    """预写入 ≥14 条跨多天、多话题、含高情绪记忆的隔离环境。

    用于链路四演化测试。记忆时间跨度 ~40 天，覆盖：技术、宠物、旅行、工作、阅读、生活。
    返回 (ctx, memory_ids) 元组。
    """
    ctx = isolated_env

    all_mids = []
    for mem in EVOLUTION_SEED_MEMORIES:
        ctx._store_conversation(mem["user"], mem["ai"], mem["ts"])
        time.sleep(0.03)

    _wait_store_queue(ctx)
    time.sleep(0.5)

    all_ids = get_all_memory_ids(ctx)
    return ctx, all_ids


# ═══════════════════════════════════════════════════════════════════
# 链路四辅助函数
# ═══════════════════════════════════════════════════════════════════

def write_memories_with_topics(ctx, memories: list[dict], topic_label: str = ""):
    """写入一批记忆到 ChromaDB 并等待入库完成。

    Args:
        ctx: AppContext 实例
        memories: 列表，每项 {"user": str, "ai": str, "ts": str}
        topic_label: 可选标签，用于日志区分
    """
    for mem in memories:
        ctx._store_conversation(mem["user"], mem["ai"], mem["ts"])
        time.sleep(0.03)
    _wait_store_queue(ctx)
    time.sleep(0.3)


def force_consolidate_shallow(ctx):
    """手动触发浅巩固（不等待 4h 定时器）。"""
    if ctx.dmn is None:
        raise RuntimeError("DMN 未初始化，无法触发浅巩固")
    ctx.dmn.consolidate_shallow()


def force_consolidate_deep(ctx):
    """手动触发深巩固（不等待 24h 定时器）。"""
    if ctx.dmn is None:
        raise RuntimeError("DMN 未初始化，无法触发深巩固")
    ctx.dmn.consolidate_deep()


def get_memory_doc_hash(ctx, mid: str) -> str:
    """获取记忆 document 字段的 MD5 哈希。"""
    import hashlib
    try:
        result = ctx.chroma_service._collection.get(
            ids=[mid], include=["documents"],
        )
        if result["ids"] and result.get("documents"):
            doc = result["documents"][0] or ""
            return hashlib.md5(doc.encode("utf-8")).hexdigest()
    except Exception:
        pass
    return ""


def get_memory_meta(ctx, mid: str) -> dict:
    """获取记忆的完整 metadata 字典。"""
    try:
        result = ctx.chroma_service._collection.get(
            ids=[mid], include=["metadatas"],
        )
        if result["ids"]:
            return dict(result["metadatas"][0])
    except Exception:
        pass
    return {}


def write_cooccurrence_data(data_dir: str, user_data: dict, ai_data: dict):
    """预写入双共现矩阵数据（用于 M9 人格对称性测试）。

    Args:
        data_dir: 数据目录
        user_data: 用户共现矩阵 {tag: {related_tag: count}}
        ai_data: AI 共现矩阵 {tag: {related_tag: count}}
    """
    import json
    user_path = os.path.join(data_dir, "co_occurrence.json")
    ai_path = os.path.join(data_dir, "ai_co_occurrence.json")

    with open(user_path, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False)
    with open(ai_path, "w", encoding="utf-8") as f:
        json.dump(ai_data, f, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════
# 后台节律测试 fixture（非 Benchmark 模式，含 impulse + dmn）
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="class")
def isolated_env_background():
    """后台节律测试隔离环境 — 非 Benchmark 模式，含完整 impulse/dmn/consolidation。

    特点：
    - BENCHMARK_MODE=false，所有后台组件均创建
    - 立即停止 impulse 泊松线程（避免随机触发干扰测试）
    - 保留 dmn + consolidation 实例供手动调用
    - 测试结束后自动清理
    """
    tmpdir = tempfile.mkdtemp(prefix="henscratch_bg_")
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    old_data_dir = os.environ.get("DATA_DIR")
    old_benchmark = os.environ.get("BENCHMARK_MODE")

    os.environ["DATA_DIR"] = data_dir
    os.environ["BENCHMARK_MODE"] = "false"
    # 确保 Path B 开关打开（冲动消费路径）
    os.environ["IMPULSE_ACTIVE_PATH_B"] = "true"

    import app.config.settings as _settings
    importlib.reload(_settings)

    from app.core.context import AppContext
    ctx = AppContext(data_dir=data_dir)

    # 立即停止所有冲动泊松线程（避免随机触发干扰测试）
    if ctx.impulse_scheduler:
        ctx.impulse_scheduler.stop()
    # 给线程一点时间退出
    time.sleep(0.3)

    yield ctx

    # 清理
    try:
        ctx.close()
    except Exception:
        pass
    try:
        from app.core.db import close_all; close_all()
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

@pytest.fixture(scope="class")
def seeded_env_background(isolated_env_background):
    """后台节律测试环境 + 预写入 12 条种子记忆。

    记忆时间戳跨多天，覆盖多话题（技术/宠物/旅行/工作/生活/阅读/家庭/音乐）。
    用于冲动源（B1-B5）、巩固（B11-B12）、模式发现（B13）等需要记忆数据的测试。

    **重要**：非 benchmark 模式下 _store_conversation 会将 timestamp 覆写为 now。
    本 fixture 在写入后手动将 timestamp 修正为过去的时间，确保冲动源能正常筛选。

    Returns: (ctx, memory_ids)
    """
    ctx = isolated_env_background

    now_ts = time.time()
    written_ids = []

    for i, mem in enumerate(SEED_MEMORIES):
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ctx._store_conversation(mem["user"], mem["ai"], ts_str)
        time.sleep(0.03)

    _wait_store_queue(ctx)
    time.sleep(0.5)

    all_ids = get_all_memory_ids(ctx)

    # 修正时间戳：非 benchmark 模式入库会把 timestamp 设为 now，
    # 冲动源需要多样化时间戳：
    #   - 前 4 条留在"今天"（不同时段，供 emotion_trend/time_rhythm 使用）
    #   - 后续依次往回推，确保有"超过 1 小时前"的记忆
    for i, mid in enumerate(all_ids):
        if i < 4:
            # 今天的不同时段（当前时间往前偏移几小时）
            past_ts = now_ts - 3600 * (i + 1)  # 1h/2h/3h/4h 前
        else:
            # 过去 N 天
            past_ts = now_ts - 86400 * ((i - 4) * 2 + 1)
        try:
            ctx.chroma_service._collection.update(
                ids=[mid],
                metadatas=[{"timestamp": past_ts}],
            )
        except Exception:
            pass

    return ctx, all_ids


# ═══════════════════════════════════════════════════════════════════
# ControlledScheduler — 可控时钟的冲动调度器包装
# ═══════════════════════════════════════════════════════════════════

class ControlledScheduler:
    """可控时钟包装器 — 让测试可以显式推进时间，驱动疲劳衰减和 TTL 过期。

    用法::

        cs = ControlledScheduler(ctx.impulse_scheduler, start_time=1000000.0)
        cs.feed("测试内容", priority=30, source="测试源")
        cs.advance(minutes=15)    # 推进 15 分钟
        cs.decay_fatigue()         # 手动触发疲劳衰减
        assert cs.get_fatigue("测试源") < 0.3
    """

    def __init__(self, scheduler, start_time: float | None = None):
        self._scheduler = scheduler
        self._now = start_time if start_time is not None else time.time()
        self._original_time = time.time

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float = 0, minutes: float = 0, hours: float = 0):
        """推进模拟时钟。"""
        total = seconds + minutes * 60 + hours * 3600
        self._now += total

    def feed(self, content: str, priority: float, source: str, ttl: float | None = None):
        """注入冲动信号（使用模拟时钟打时间戳）。"""
        # 临时替换 time.time
        import app.background.impulse as _imp
        _orig = _imp.time.time
        _imp.time.time = lambda: self._now
        try:
            from app.config.settings import IMPULSE_TTL
            self._scheduler.feed_impulse(content, priority, source, ttl or IMPULSE_TTL)
        finally:
            _imp.time.time = _orig

    def decay_fatigue(self):
        """在模拟时钟下运行疲劳衰减。"""
        import app.background.impulse as _imp
        _orig = _imp.time.time
        mock_time = lambda: self._now
        _imp.time.time = mock_time
        try:
            self._scheduler._decay_fatigue()
        finally:
            _imp.time.time = _orig

    def get_fatigue(self, source: str) -> float:
        """获取指定源的疲劳度。"""
        return self._scheduler._source_fatigue.get(source, 0.0)

    def set_fatigue(self, source: str, value: float):
        """直接设置疲劳度（用于测试衰减）。"""
        self._scheduler._source_fatigue[source] = min(1.0, max(0.0, value))

    def set_last_fire(self, source: str, timestamp: float):
        """直接设置最后发射时间（用于测试衰减）。"""
        self._scheduler._source_fire_times[source] = [timestamp]

    def get_queue_size(self) -> int:
        """获取队列大小（线程不安全，测试中无竞争）。"""
        return self._scheduler._pq.qsize()

    def get_history(self) -> list[dict]:
        """获取历史记录。"""
        return self._scheduler.get_history()

    def get_next(self, test_mode: bool = True) -> dict | None:
        """获取下一个冲动（test_mode=True 跳过频率限制）。"""
        return self._scheduler.get_next(test_mode=test_mode)

    def inject_expired_impulse(self, content: str, priority: float, source: str,
                                 age_seconds: float, ttl: float = 300):
        """注入一条"已过期"的冲动（created_at 设为过去）。"""
        import app.background.impulse as _imp
        _orig = _imp.time.time
        # 先用当前模拟时间 feed（会正确计算 effective_priority 和 fatigue）
        _imp.time.time = lambda: self._now
        try:
            self._scheduler.feed_impulse(content, priority, source, ttl)
        finally:
            _imp.time.time = _orig
        # 然后推进时间使该冲动过期
        self.advance(seconds=age_seconds)

    def _patch_time_for_method(self, method_name: str):
        """上下文管理器：在方法调用期间临时替换 impulse 模块的 time.time。"""
        import app.background.impulse as _imp
        return _PatchTime(_imp, self._now)


class _PatchTime:
    """临时替换 impulse.time.time 的上下文管理器。"""
    def __init__(self, impulse_module, mock_now: float):
        self._mod = impulse_module
        self._mock_now = mock_now
        self._orig = None

    def __enter__(self):
        self._orig = self._mod.time.time
        self._mod.time.time = lambda: self._mock_now

    def __exit__(self, *args):
        self._mod.time.time = self._orig
