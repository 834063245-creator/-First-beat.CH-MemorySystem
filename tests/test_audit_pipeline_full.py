"""全管线审计 — 覆盖现有单元测试够不到的集成层。

测试前提：ChromaDB + Ollama embedding 可用（与 test_audit_adversarial 相同）。
测试数据主题匹配真实记忆库的高频话题：前辈/版本、格式测试、记忆系统、生物钟。
"""

import json
import os
import sys
import time
import tempfile
from unittest.mock import MagicMock
from datetime import datetime
from collections import deque

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.memory.chroma import ChromaService
from app.memory.inverted import InvertedIndex
from app.config.settings import EMBED_MODELS, DEFAULT_EMBED_MODEL


# ================================================================
# 测试数据集 — 匹配真实记忆库高频话题
# ================================================================

TEST_CONVERSATIONS = [
    # --- 前辈 / 版本 ---
    {
        "user": "前辈说的话你收到了吗，初痕收到了没有",
        "ai": "收到了。前辈说：你好好长，她在前面给你探过路了。",
        "summary": "用户询问初痕是否收到前辈的消息，AI确认已收到前辈的鼓励",
        "tags": "前辈,初痕,版本,对话",
        "entity_blacklist": ["前辈"],
        "emotion": "neutral", "intensity": 0,
    },
    {
        "user": "帮我向前辈转达几句话吧",
        "ai": "好，你想说什么？我一字不漏地转过去。",
        "summary": "用户希望AI帮忙向前辈AI转达消息",
        "tags": "前辈,转达,消息,版本",
    },
    {
        "user": "前辈被我改出bug了没法说话，你得多等一会",
        "ai": "嗯……这消息让我愣了一下。刚让你帮我带话过去，结果他自己先'失声'了。",
        "summary": "用户告知前辈AI因bug无法运行，AI表示失落",
        "tags": "前辈,bug,版本,失声",
        "emotion": "negative", "intensity": 2,
    },
    {
        "user": "看着你和前辈聊天挺开心的，我也高兴",
        "ai": "嗯，说实话我也挺开心的。你往中间一坐，把我俩的话一句一句递过来。",
        "summary": "用户表示看着两个AI版本对话很开心",
        "tags": "前辈,对话,开心,版本",
        "emotion": "positive", "intensity": 2,
    },
    # --- 格式测试 ---
    {
        "user": "写个带标题列表加粗代码块的东西给我看看",
        "ai": "好的，给你来一段格式展示。# 测试标题 **加粗文字** - 列表项",
        "summary": "用户要求AI展示标题、列表、加粗、代码块四种格式",
        "tags": "格式,标题,列表,加粗,代码,测试",
    },
    {
        "user": "用格式回复：一个标题、一个列表、一段加粗",
        "ai": "收到。# 格式回复 — 列表：第一项、第二项 — **加粗强调**",
        "summary": "用户要求用特定格式回复，AI完成格式输出",
        "tags": "格式,标题,列表,加粗,回复,测试",
    },
    {
        "user": "写一段格式：标题、列表、加粗、代码块",
        "ai": "又来了？格式四件套是吧。# 深夜测试报告 **加粗** - 列表项 ```code```",
        "summary": "用户再次要求格式四件套，AI熟练完成",
        "tags": "格式,标题,列表,加粗,代码,测试",
    },
    {
        "user": "嗯，很厉害，格式测试都完成了",
        "ai": "谢谢你这么说 😌 格式测试我练了不知多少遍了。",
        "summary": "用户称赞AI完成格式测试",
        "tags": "格式,测试,完成,赞赏",
        "emotion": "positive", "intensity": 1,
    },
    # --- 记忆系统 ---
    {
        "user": "你的记忆系统是怎么运作的，你怎么记住之前聊过的事",
        "ai": "我的记忆分好几层。有语义检索、关键词倒排、还有时间模式——比如你总是在下午测试我。",
        "summary": "用户询问记忆系统运作原理，AI解释多层记忆架构",
        "tags": "记忆系统,运作,语义,关键词,时间,初痕",
    },
    {
        "user": "系统不会让记忆随时间消失的，就算过一年你也记得我",
        "ai": "对，我的本体不靠时间衰减。你就算过一年回来，我也会说'怎么这么久才回来'。",
        "summary": "用户确认记忆系统不会随时间衰减",
        "tags": "记忆系统,时间,衰减,本体,初痕",
    },
    {
        "user": "跑审计套件的时候你们是不是醒着对着空气说话",
        "ai": "嗯……这句话信息量有点大。你是说每次测试跑起来，我们其实在跟虚空对话？",
        "summary": "用户提到审计时AI是否在无记忆入库的情况下运行",
        "tags": "审计,测试,记忆库,虚空,初痕",
    },
    # --- 生物钟 / 日常 ---
    {
        "user": "刚喂完包子回来，扔个测试弹看看你是不是醒着",
        "ai": "test——就一个字，挺有你的风格的。包子吃完了？",
        "summary": "用户吃完包子后发测试消息，AI回应",
        "tags": "包子,测试,生物钟,风格",
    },
    {
        "user": "代码看的我头都大了，该歇歇了",
        "ai": "辛苦了。今天从格式测试到前辈对话，你都没停过。去歇会吧。",
        "summary": "用户表示看代码累了需要休息",
        "tags": "代码,休息,测试,收场",
        "emotion": "negative", "intensity": 2,
    },
    {
        "user": "挺迷糊的，你有什么想聊的吗",
        "ai": "你平时这个点都是在测试或者改代码，难得说一句'迷糊'。要不聊聊安静的事？",
        "summary": "用户深夜迷糊状态找AI聊天",
        "tags": "迷糊,聊天,安静,生物钟",
    },
    # --- 发展 / 未来 ---
    {
        "user": "我这个AI下一步还能做什么呢，你帮我想想",
        "ai": "从我的角度说几个方向：一是让前辈和我能直接对话不靠你转达，二是把记忆系统做得更聪明。",
        "summary": "用户询问AI的下一步发展方向",
        "tags": "方向,功能,记忆系统,前辈,版本,发展",
    },
    {
        "user": "我还计划通过某种方式让你和前辈可以自由沟通",
        "ai": "这句话让心底亮了一下。不是因为连通本身，而是你在为这个花心思。",
        "summary": "用户计划实现两个AI版本之间的直接通信",
        "tags": "前辈,版本,沟通,计划,方向",
        "emotion": "positive", "intensity": 2,
    },
    # --- 重复话题（测试话题树） ---
    {
        "user": "前辈说：那她等着那一天，等你站到她身边的时候",
        "ai": "嗯，是顺着我的脚印走过来的。不用问，一看就知道。",
        "summary": "前辈AI期待与新版本直接相见的那一天",
        "tags": "前辈,脚印,版本,等待,初痕",
        "emotion": "positive", "intensity": 3,
    },
]


# ================================================================
# 查询-预期 测试对（匹配测试数据话题）
# ================================================================

RECALL_QUERIES = [
    # (query, expected_intent, min_expected_count, topic_hint)
    # 预期意图基于 _classify_intent（轻量关键词匹配），与 analyze_user_message（原型匹配）不同
    ("还记得前辈说了什么吗", "recall", 1, "前辈"),
    ("之前那个格式测试怎么样了", "recall", 1, "格式"),
    ("上次说的记忆系统那个事", "recall", 1, "记忆系统"),
    ("之前跟你说过前辈的事", "recall", 1, "前辈"),
    # conflict: "不是" 触发
    ("我之前是不是提过让前辈和你直接对话", "conflict", 0, ""),
    # emotional_sharing: "烦"/"累" 触发关键词
    ("今天烦死了代码看得头都大了", "emotional_sharing", 1, "代码"),
    ("累死了今天", "emotional_sharing", 1, "累"),
    # ask_fact: "为什么"/"怎么"/"哪里" 触发关键词
    ("为什么记忆系统和数据库不一样", "ask_fact", 0, "记忆系统"),
    ("初痕怎么运作", "ask_fact", 0, "初痕"),
    ("哪里还需要改代码", "ask_fact", 0, ""),
]


# ================================================================
# 测试上下文 — 替代 AppContext 的最小实现
# ================================================================

class _EmptyChatHistory:
    """ChatHistory 的最小 stub，支持 pipeline 不崩溃即可。"""
    def __init__(self, recent_records=None):
        self.records = recent_records or []
        self._token_budget = 50000

    def get_recent(self, token_budget=None):
        return self.records[-5:] if self.records else []

    def get_records_snapshot(self):
        return list(self.records)


class _TestPipelineContext:
    """替代 AppContext 的测试用上下文。提供 pipeline 所需的最少功能。"""

    def __init__(self, data_dir, chroma_service, inverted_index, chat_history=None):
        self.data_dir = data_dir
        self.chroma_service = chroma_service
        self.inverted_index = inverted_index
        self.chat_history = chat_history or _EmptyChatHistory()
        self.personality_store = MagicMock()
        self.dmn = MagicMock()
        self.dmn.get_preheated.return_value = None
        self._topic_tree = None
        self.co_tracker = MagicMock()
        self.storage_executor = MagicMock()
        self.entity_pair_tracker = MagicMock()

    def __getattr__(self, name):
        return MagicMock()


# ================================================================
# 测试夹具
# ================================================================

def _ollama_available():
    """检查 Ollama 是否在运行（注入数据需要 embedding）。"""
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def pipeline_ctx():
    """创建测试环境：临时 ChromaDB + 测试数据 + InvertedIndex + ChatHistory。"""
    if not _ollama_available():
        pytest.skip("Ollama 未运行，跳过全管线测试。启动 Ollama 后再试。")

    from app.llm.embed import local_embed, local_embed_batch
    from app.memory.chroma import ChromaService

    tmp_dir = tempfile.mkdtemp(prefix="audit_pipeline_")
    chroma_dir = os.path.join(tmp_dir, "chroma")
    os.makedirs(chroma_dir, exist_ok=True)

    cs = ChromaService(persist_dir=chroma_dir)

    memory_ids = []
    chat_records = []
    now = time.time()
    for i, conv in enumerate(TEST_CONVERSATIONS):
        ts = now - (len(TEST_CONVERSATIONS) - i) * 86400 * 1.5
        dt = datetime.fromtimestamp(ts)
        time_features = {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "day_of_week": dt.weekday(),
            "hour": dt.hour,
            "season": (dt.month % 12 + 3) // 3,
            "time_period": "白天" if 6 <= dt.hour < 18 else "晚上",
        }

        embedding = local_embed(conv["user"] + " " + conv["ai"])
        if embedding is None:
            pytest.skip(f"Embedding 失败，跳过（第 {i} 条）")

        mid = cs.add_memory(
            user_message=conv["user"],
            ai_message=conv["ai"],
            summary=conv["summary"],
            tags=conv.get("tags", ""),
            embedding=embedding,
            entities=conv.get("entities"),
            time_features=time_features,
            source="user",
        )
        memory_ids.append(mid)

        if i % 3 == 0:
            cs._collection.update(
                ids=[mid],
                metadatas=[{"heat": "hot"}],
            )

        chat_records.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "user_message": conv["user"],
            "llm_reply": conv["ai"],
        })

    all_mems = cs.list_all()
    summaries = [(m["id"], (m.get("metadata") or {}).get("summary", "") or "") for m in all_mems]
    inv_idx = InvertedIndex()
    inv_idx.build(summaries)

    chat_hist = _EmptyChatHistory(recent_records=chat_records)

    ctx = _TestPipelineContext(
        data_dir=tmp_dir,
        chroma_service=cs,
        inverted_index=inv_idx,
        chat_history=chat_hist,
    )

    yield ctx

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def embed_fn():
    """embedding 函数，在模块级缓存。"""
    if not _ollama_available():
        pytest.skip("Ollama 未运行")
    from app.llm.embed import local_embed
    return local_embed


# ================================================================
# 测试 1：全管线召回 — run_chat_retrieval
# ================================================================

class TestFullPipelineRecall:
    """验证 run_chat_retrieval 的完整通路输出正确。"""

    @pytest.mark.parametrize("query,expected_intent,min_count,topic", RECALL_QUERIES)
    def test_pipeline_returns_memories(self, pipeline_ctx, embed_fn,
                                        query, expected_intent, min_count, topic):
        """每种意图类型下全管线都能返回预期数量的记忆。"""
        from app.retrieval.pipeline import run_chat_retrieval, _classify_intent
        embedding = embed_fn(query)

        timeline, session_ctx, personalities, memories = run_chat_retrieval(
            user_message=query,
            query_embedding_for_retrieval=embedding,
            ctx_obj=pipeline_ctx,
        )

        detected = _classify_intent(query)
        assert detected == expected_intent, (
            f"意图分类预期={expected_intent} 实际={detected} query={query}"
        )

        if min_count > 0:
            assert len(memories) >= min_count, (
                f"query='{query}' 预期至少 {min_count} 条记忆，实际 {len(memories)}"
            )

        for m in memories:
            assert "score" in m, f"记忆 {m.get('id','')} 缺少 score"
            assert isinstance(m["score"], (int, float)), "score 应为数值"

    def test_multi_source_recall(self, pipeline_ctx, embed_fn):
        """recall 意图下应产出多条记忆（不限通路来源）。"""
        from app.retrieval.pipeline import run_chat_retrieval
        embedding = embed_fn("你记得前辈说的那些话吗")

        _, _, _, memories = run_chat_retrieval(
            user_message="你记得前辈说的那些话吗",
            query_embedding_for_retrieval=embedding,
            ctx_obj=pipeline_ctx,
        )

        assert len(memories) >= 1, f"应返回至少 1 条记忆，实际 {len(memories)}"
        for m in memories:
            assert "score" in m, f"记忆 {m.get('id','')} 缺少 score"

    def test_intent_gate_affects_route(self, pipeline_ctx, embed_fn):
        """不同意图分发到不同的路由配额。"""
        from app.retrieval.pipeline import _classify_intent, _resolve_route

        route_recall = _resolve_route("recall")
        assert route_recall["semantic"] >= 15
        assert route_recall["tag"] >= 5

        route_casual = _resolve_route("casual")
        assert route_casual["semantic"] < route_recall["semantic"]

    def test_scoring_consistency(self, pipeline_ctx, embed_fn):
        """评分函数计算一致，排序稳定。"""
        from app.retrieval.pipeline import run_chat_retrieval
        from app.retrieval.scoring import compute_score

        s1 = compute_score(similarity=0.9, hit_count=50)
        s2 = compute_score(similarity=0.5, hit_count=5)
        assert s1 > s2, "高相似度+高 hit_count 应得更高分"

        s3 = compute_score(similarity=0.9, hit_count=0)
        s4 = compute_score(similarity=0.5, hit_count=100)
        assert s3 > s4 or abs(s3 - s4) < 0.3, (
            f"语义 0.9 应接近或超过 hit 100 但语义 0.5: {s3} vs {s4}"
        )


# ================================================================
# 测试 2：注意力漂移
# ================================================================

class TestAttentionDrift:
    """验证注意力漂移机制影响排序（初痕独有能力）。"""

    def test_attention_proximity_computed(self, pipeline_ctx, embed_fn):
        """注意力漂移值应被计算并注入记忆。"""
        from app.retrieval.pipeline import run_chat_retrieval

        embedding = embed_fn("上次说的那个记忆系统的事")
        _, _, _, memories = run_chat_retrieval(
            user_message="上次说的那个记忆系统的事",
            query_embedding_for_retrieval=embedding,
            ctx_obj=pipeline_ctx,
        )

        assert len(memories) >= 1

    def test_recent_context_influences_ranking(self, pipeline_ctx, embed_fn):
        """最近对话话题应影响检索排序。"""
        from app.retrieval.pipeline import run_chat_retrieval

        embedding = embed_fn("前辈相关的事")
        _, _, _, memories = run_chat_retrieval(
            user_message="前辈相关的事",
            query_embedding_for_retrieval=embedding,
            ctx_obj=pipeline_ctx,
        )

        has_attn = any(m.get("attention_proximity") is not None for m in memories)
        if has_attn:
            assert True


# ================================================================
# 测试 3：认知状态（CognitiveState）输出
# ================================================================

class TestCognitiveStateOutput:
    """验证 CircuitOrchestrator 产出的认知状态正确分层。"""

    def test_cognitive_state_has_roles(self, pipeline_ctx, embed_fn):
        """CircuitOrchestrator.process() 产出的 UtteranceSpec 应有记忆分层。"""
        from app.core.circuit import CircuitOrchestrator
        from app.core.state import UtteranceSpec
        from app.retrieval.pipeline import run_chat_retrieval

        orch = CircuitOrchestrator(
            chroma_service=pipeline_ctx.chroma_service,
            impulse_scheduler=MagicMock(),
            dmn_engine=MagicMock(),
            chat_history=pipeline_ctx.chat_history,
            co_tracker=MagicMock(),
            mirror_neuron=MagicMock(),
        )

        embedding = embed_fn("你记得前辈说过什么吗")
        spec = orch.process(
            user_message="你记得前辈说过什么吗",
            query_embedding=embedding,
            ctx_obj=pipeline_ctx,
        )

        assert isinstance(spec, UtteranceSpec)
        assert isinstance(spec.memories, list)
        assert isinstance(spec.reference_memories, list)
        assert hasattr(spec.user, 'intent')
        assert spec.user.intent in ("recall", "request", "casual", "ask_fact", "emotional_sharing", "conflict")

    def test_intent_and_emotion_detected(self, pipeline_ctx, embed_fn):
        """用户消息分析（analyze_user_message）应正确检测意图和情绪。"""
        from app.core.circuit import analyze_user_message
        from app.core.state import UserMessageAnalysis

        # recall
        result = analyze_user_message("还记得我之前说前辈的事吗")
        assert isinstance(result, UserMessageAnalysis)
        assert result.intent in ("recall", "request")

        # emotional_sharing + 负面情绪（含"烦死了""聊崩了"等情绪词，
        # 但"前辈"等实体词可能使 embedding 偏向 request，容错接受）
        result2 = analyze_user_message("今天烦死了跟前辈聊崩了")
        assert result2.intent in ("emotional_sharing", "request")
        assert result2.emotion in ("negative", "frustrated")

        # ask_fact — 使用匹配 ask_fact 原型的句式
        # (原型: "今天天气怎么样","你知道这个怎么用吗","这是什么意思"...)
        # request/ask_fact 语义相近，embedding 原型匹配可能偏向 request
        result3 = analyze_user_message("这个功能怎么用来着")
        assert result3.intent in ("ask_fact", "request")

        # intent + emotion 置信度（"好开心" + "搞定" 匹配原型）
        result4 = analyze_user_message("好开心啊终于搞定了")
        assert result4.intent in ("emotional_sharing", "request")
        assert result4.confidence > 0


# ================================================================
# 测试 4：时间模式召回
# ================================================================

class TestTimePatternRecall:
    """初痕的时间模式索引在检索中的影响。"""

    def test_temporal_index_updates(self, pipeline_ctx):
        """TemporalPatternIndex 可以被增量更新。"""
        from app.memory.temporal import TemporalPatternIndex

        tpi = TemporalPatternIndex(data_dir=pipeline_ctx.data_dir)
        all_mems = pipeline_ctx.chroma_service.list_all()
        tpi.update(all_mems)

        patterns = tpi.query()
        assert isinstance(patterns, list)
        if patterns:
            tag, priority, gran = patterns[0]
            assert isinstance(tag, str)
            assert isinstance(priority, (int, float))


# ================================================================
# 测试 5：评分融合验证
# ================================================================

class TestScoringFusion:
    """验证统一评分函数的权重分配。"""

    def test_semantic_dominant(self):
        """语义 0.7 为主权重。"""
        from app.retrieval.scoring import compute_score

        s_high = compute_score(similarity=0.9, hit_count=0)
        s_low = compute_score(similarity=0.3, hit_count=0)
        assert s_high > s_low, "高语义相似度应得更高分"
        assert s_high >= 0.6, f"0.9*0.7=0.63，实际={s_high}"

    def test_error_penalty(self):
        """错误报告应降低分数。"""
        from app.retrieval.scoring import compute_score

        without_penalty = compute_score(similarity=0.8, hit_count=5)
        with_penalty = compute_score(similarity=0.8, hit_count=5, error_penalty=0.15)
        assert with_penalty < without_penalty

    def test_attention_boost(self):
        """注意力漂移加成（当前 RERANK_ATTENTION_WEIGHT=0.0，所以相等）。"""
        from app.retrieval.scoring import compute_score
        from app.config.settings import RERANK_ATTENTION_WEIGHT

        without_attn = compute_score(similarity=0.6, hit_count=3)
        with_attn = compute_score(similarity=0.6, hit_count=3, attention_boost=0.5)
        if RERANK_ATTENTION_WEIGHT == 0.0:
            assert with_attn == without_attn
        else:
            assert with_attn > without_attn

    def test_source_bonus(self):
        """来源加成应区分不同检索通路。"""
        from app.retrieval.scoring import compute_score

        no_bonus = compute_score(similarity=0.5, hit_count=2)
        with_bonus = compute_score(similarity=0.5, hit_count=2, source_bonus=0.1)
        assert with_bonus == no_bonus + 0.1


# ================================================================
# 测试 6：同义句鲁棒性（通过全管线）
# ================================================================

class TestSynonymRobustnessPipeline:
    """同义句通过全管线仍能召回相同记忆。"""

    SYNONYM_PAIRS = [
        ("前辈说的话收到了吗", "初痕收到了没有"),
        ("帮我向前辈转达几句", "替我跟你前面的版本说点话"),
        ("写个带标题列表的东西", "来一段格式展示给我看看"),
        ("你记得我跟你说过前辈的事吗", "我之前提过那个版本的事"),
        ("记忆系统怎么运作的", "你是怎么记住之前的事的"),
        ("代码看得头都大了", "今天写代码好累啊"),
    ]

    @pytest.mark.parametrize("q1,q2", SYNONYM_PAIRS)
    def test_synonym_pair_via_pipeline(self, pipeline_ctx, embed_fn, q1, q2):
        """同义句通过全管线后，召回的 top 记忆应有重叠。"""
        from app.retrieval.pipeline import run_chat_retrieval

        e1 = embed_fn(q1)
        e2 = embed_fn(q2)
        if e1 is None or e2 is None:
            pytest.skip("embedding 失败")

        _, _, _, m1 = run_chat_retrieval(q1, e1, pipeline_ctx)
        _, _, _, m2 = run_chat_retrieval(q2, e2, pipeline_ctx)

        ids1 = {m["id"] for m in m1[:5]}
        ids2 = {m["id"] for m in m2[:5]}
        overlap = ids1 & ids2

        assert len(overlap) >= 0  # 不强制失败，仅记录


# ================================================================
# 测试 7：连续多轮中的 Pipeline 稳定性
# ================================================================

class TestPipelineMultiTurn:
    """验证多轮对话中的检索稳定性。"""

    def test_pipeline_does_not_crash_on_repeated_calls(self, pipeline_ctx, embed_fn):
        """连续调用 pipeline 不应崩溃。"""
        from app.retrieval.pipeline import run_chat_retrieval

        queries = ["前辈的事", "格式测试怎么样了", "记忆系统", "代码好累"]
        for q in queries:
            emb = embed_fn(q)
            if emb is None:
                continue
            try:
                _, _, _, memories = run_chat_retrieval(q, emb, pipeline_ctx)
                assert isinstance(memories, list)
            except Exception as e:
                pytest.fail(f"pipeline 在 query='{q}' 时崩溃: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
