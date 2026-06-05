from typing import Optional, List, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    debug: bool = False
    debug_include_prompt: bool = False
    test_mode: bool = False
    benchmark_inject: bool = False


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


from dataclasses import dataclass, field


@dataclass
class WovenContext:
    """引擎编织后的记忆上下文。替代原来的 flat memory list。"""
    narratives: list[str] = field(default_factory=list)
    fact_memories: list[dict] = field(default_factory=list)
    stale_context: list[dict] = field(default_factory=list)  # v2.1: 被取代但不屏蔽的记忆
    background_notes: dict = field(default_factory=dict)
    should_speak: bool = False
    total_candidates: int = 0
    total_tokens: int = 0



