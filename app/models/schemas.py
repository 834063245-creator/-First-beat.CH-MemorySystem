from typing import Optional, List, Any
from pydantic import BaseModel, Field

# ── MCP Schema 版本 ──────────────────────────────────────────────
MCP_SCHEMA_VERSION = "1.0.0"


class ChatRequest(BaseModel):
    message: str
    debug: bool = False
    debug_include_prompt: bool = False
    test_mode: bool = False


class TraceItem(BaseModel):
    id: str
    summary: str
    timestamp: float
    source: str
    display_source: str
    hit_count: int
    tags: list[str]


class DebugInfo(BaseModel):
    retrieved_count: int = 0
    retrieved_ids: List[str] = []
    hit_counts: List[int] = []


class ChatResponse(BaseModel):
    response: str
    debug: Optional[DebugInfo] = None
    trace: Optional[List[TraceItem]] = None
    debug_info: Optional[dict] = None


class PromptBody(BaseModel):
    content: str


class MemoryListResponse(BaseModel):
    items: list
    total: int
    page: int
    per_page: int


class MemoryDeleteResponse(BaseModel):
    status: str
    id: str


class CorrectMemoryBody(BaseModel):
    corrected_summary: str


# ═══════════════════════════════════════════════════════════════
# MCP 工具输出 Schema (schema_versioned)
# ═══════════════════════════════════════════════════════════════

class PersonalityTagItem(BaseModel):
    content: str = ""
    hit_count: int = 0


class RelationshipOutput(BaseModel):
    familiarity: float = 0.0
    trust: float = 0.5
    closeness: float = 0.0
    interaction_mode: str = "casual"


class MemoryItem(BaseModel):
    id: str = ""
    time: str = ""
    relative_time: str = ""
    summary: str = ""
    source: str = ""
    hit_count: int = 0
    relevance: float = 0.0
    stale: bool = False


class TimelineItem(BaseModel):
    role: str = ""
    content: str = ""
    time: str = ""


class ImpulseRawItem(BaseModel):
    intent: str = ""
    target: str = ""


class MirrorPrediction(BaseModel):
    next_intents: list[str] = []


class RunEngineOutput(BaseModel):
    """MCP run_engine 返回结构（schema_versioned）。"""
    schema_version: str = MCP_SCHEMA_VERSION
    execute: str = ""
    personality: dict[str, list[PersonalityTagItem]] = Field(default_factory=dict)
    memories: list[MemoryItem] = Field(default_factory=list)
    timeline_recent: list[TimelineItem] = Field(default_factory=list)
    impulses: list[str] = Field(default_factory=list)
    impulse_raw: list[ImpulseRawItem] = Field(default_factory=list)
    relationship: RelationshipOutput = Field(default_factory=RelationshipOutput)
    session_context: str = ""
    mirror_prediction: MirrorPrediction = Field(default_factory=MirrorPrediction)


class QueryMemoriesOutput(BaseModel):
    """MCP query_memories 返回结构。"""
    schema_version: str = MCP_SCHEMA_VERSION
    query: str = ""
    results: list[MemoryItem] = Field(default_factory=list)
    total_found: int = 0


class RecentHistoryItem(BaseModel):
    user_message: str = ""
    llm_reply: str = ""
    timestamp: str = ""


class RecentHistoryOutput(BaseModel):
    """MCP get_recent_history 返回结构。"""
    schema_version: str = MCP_SCHEMA_VERSION
    items: list[RecentHistoryItem] = Field(default_factory=list)
    count: int = 0


class MemoryStatsOutput(BaseModel):
    """MCP get_memory_stats 返回结构。"""
    schema_version: str = MCP_SCHEMA_VERSION
    total: int = 0
    total_hits: int = 0
    earliest: Optional[str] = None
    latest: Optional[str] = None
    heat_distribution: dict = Field(default_factory=dict)
    emotion_distribution: dict = Field(default_factory=dict)


class PersonalityTagsOutput(BaseModel):
    """MCP get_personality_tags 返回结构。"""
    schema_version: str = MCP_SCHEMA_VERSION
    tags: list[PersonalityTagItem] = Field(default_factory=list)


class TopicTreeOutput(BaseModel):
    """MCP get_topic_tree 返回结构。"""
    schema_version: str = MCP_SCHEMA_VERSION
    tree: Optional[dict] = None
    note: str = ""


class PatternObservationsOutput(BaseModel):
    """MCP get_pattern_observations 返回结构。"""
    schema_version: str = MCP_SCHEMA_VERSION
    observations: list[dict] = Field(default_factory=list)


class StoreTurnOutput(BaseModel):
    """MCP store_turn 返回结构。"""
    schema_version: str = MCP_SCHEMA_VERSION
    status: str = "ok"
    memory_id: str = ""





