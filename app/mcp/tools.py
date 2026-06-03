"""MCP 工具定义 — 10 个记忆引擎工具（含 outputSchema）。

所有工具不返回内部 memory_id / chroma_id / 完整 document，
只返回摘要+标签+时间+热度+情绪方向。

每个工具包含 outputSchema（schema_versioned），客户端可据此校验。
"""

from app.models.schemas import MCP_SCHEMA_VERSION

_SCHEMA = {"type": "string", "const": MCP_SCHEMA_VERSION}

TOOLS: list[dict] = [
    {
        "name": "query_memories",
        "description": (
            "从初痕记忆库中语义检索相关记忆。给定查询文本返回最相关的记忆列表，含摘要、相关度分数、命中次数、相对时间、情绪方向。"
            " / Semantic search over the Chuchen memory store. Returns ranked memories with summary, relevance score, hit count, relative time, and emotion valence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询 / Search query"},
                "top_k": {"type": "integer", "description": "最多返回条数，默认 10 / Max results, default 10", "default": 10},
            },
            "required": ["query"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "query": {"type": "string"},
                "results": {"type": "array"},
                "total_found": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_recent_history",
        "description": (
            "返回最近 N 轮对话记录（用户+AI 的消息对），带时间戳。用于了解当前会话上下文。"
            " / Return the last N conversation turns (user+assistant pairs) with timestamps, for understanding current session context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "返回条数，默认 10 / Number of turns, default 10", "default": 10},
            },
            "required": [],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "items": {"type": "array"},
                "count": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_personality_tags",
        "description": (
            "返回用户人格标签或 AI 人格标签列表。type='user' 返回系统对用户的观察（兴趣/习惯/偏好），type='ai' 返回 AI 自己的表达习惯画像。"
            " / Return user or AI personality tags. type='user' for observed user traits (interests/habits/preferences), type='ai' for AI self-model expression habits."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "'user' 或 'ai'，默认 'user'", "default": "user", "enum": ["user", "ai"]},
                "top_k": {"type": "integer", "description": "最多返回条数，默认 15", "default": 15},
            },
            "required": [],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "tags": {"type": "array"},
            },
        },
    },
    {
        "name": "get_topic_tree",
        "description": (
            "返回当前话题树结构。话题树已逐步被 embedding 标签索引替代，但仍可用于探索标签间的统计共现关系。"
            " / Return the current topic tree (gradually being superseded by embedding-based tag index)."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "tree": {"type": "object", "nullable": True},
                "note": {"type": "string"},
            },
        },
    },
    {
        "name": "get_relationship",
        "description": (
            "返回当前用户与 AI 的关系状态：熟悉度、信任度、亲密度、互动模式。基于最近 30 轮对话计算，不落盘。"
            " / Return relationship state: familiarity, trust, closeness, interaction mode. Computed from last 30 turns, not persisted."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "familiarity": {"type": "number"},
                "trust": {"type": "number"},
                "closeness": {"type": "number"},
                "interaction_mode": {"type": "string"},
            },
        },
    },
    {
        "name": "get_memory_stats",
        "description": (
            "返回记忆库统计数据：总记忆数、热度分布（hot/warm/cool）、情绪分布（positive/negative/neutral/intimate）。"
            " / Return memory store statistics: total count, heat distribution, emotion distribution."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "total": {"type": "integer"},
                "total_hits": {"type": "integer"},
                "earliest": {"type": "string", "nullable": True},
                "latest": {"type": "string", "nullable": True},
                "heat_distribution": {"type": "object"},
                "emotion_distribution": {"type": "object"},
            },
        },
    },
    {
        "name": "get_pattern_observations",
        "description": (
            "返回模式发现层的最新观察结果（时间节律/情绪锚点/话题漂移）。"
            " / Return the latest pattern discovery observations."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "observations": {"type": "array"},
            },
        },
    },
    {
        "name": "run_engine",
        "description": (
            "运行初痕记忆引擎完整管线：意图选路→热温冷分层检索→触发链→共现→时间节律→话题树→"
            "人格对称性→模式发现调参→回路门控。返回结构化上下文（执行指令/人格/记忆/时间线/冲动/关系），"
            "外部 Agent 的 LLM 作为语言皮层读取此上下文生成回复。引擎全权决策，LLM 只管说话。"
            " / Run the full Chuchen memory engine pipeline. Returns structured context. "
            "The external agent's LLM acts as language cortex — engine decides, LLM speaks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "用户消息 / User message"},
                "include_impulses": {"type": "boolean", "description": "是否包含冲动/浮现念头，默认 true", "default": True},
            },
            "required": ["message"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "execute": {"type": "string"},
                "personality": {"type": "object"},
                "memories": {"type": "array"},
                "timeline_recent": {"type": "array"},
                "impulses": {"type": "array"},
                "impulse_raw": {"type": "array"},
                "relationship": {"type": "object"},
                "session_context": {"type": "string"},
                "mirror_prediction": {"type": "object"},
            },
        },
    },
    {
        "name": "store_turn",
        "description": (
            "存储一轮对话到初痕记忆库（ChatHistory + ChromaDB 异步入库）。调用 run_engine 拿到引擎上下文后，"
            "外部 LLM 生成回复，再调此工具将本轮对话写入记忆。入库包括摘要/标签/embedding，异步执行不阻塞。"
            " / Store a conversation turn to the memory store. Call after run_engine + external LLM response."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_message": {"type": "string", "description": "用户消息 / User message"},
                "ai_message": {"type": "string", "description": "AI 回复 / AI response"},
                "timestamp": {"type": "string", "description": "可选时间戳，默认为当前时间 / Optional timestamp, defaults to now"},
            },
            "required": ["user_message", "ai_message"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "schema_version": _SCHEMA,
                "status": {"type": "string"},
                "memory_id": {"type": "string"},
            },
        },
    },
]
