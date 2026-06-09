# First Beat Architecture

> Why this system is shaped the way it is, and how the pieces fit together.

---

## Table of Contents

1. [Core Design Decisions](#core-design-decisions)
2. [System Overview](#system-overview)
3. [Cognitive State Layer — The Engine/LLM Boundary](#cognitive-state-layer)
4. [Portrait System — PORTRAIT.md Unified Hub](#portrait-system)
5. [Retrieval Pipeline — 10-Path Parallel + Weaving](#retrieval-pipeline)
6. [Background Autonomous Rhythm](#background-autonomous-rhythm)
7. [Memory Lifecycle](#memory-lifecycle)
8. [AI Self-Expression Memory](#ai-self-expression-memory)
9. [Working Memory Digest](#working-memory-digest)
10. [User Feedback Loop](#user-feedback-loop)
11. [E2E Benchmark Suite](#e2e-benchmark-suite)
12. [Key Technology Choices](#key-technology-choices)
13. [Known Technical Debt](#known-technical-debt)
14. [Module Dependency Graph](#module-dependency-graph)

---

## Core Design Decisions

### Decision 1: Engine Decides → LLM Executes

**Nearly every AI memory system** works like this: the LLM calls memory retrieval tools, receives results, then judges for itself what's useful and how to use it.

**First Beat inverts this.** The engine makes every decision — what to retrieve, confidence levels, tone, whether to inject an impulse — and packages it into a single `UtteranceSpec` dataclass. The LLM's only job is translating that into natural language. The LLM doesn't own memory, doesn't call retrieval tools, doesn't decide.

**Why:**
- LLMs have context window limits. Offloading retrieval, ranking, and confidence scoring to the engine (pure algorithms) leaves the LLM's entire window for "speaking"
- Decisions are reproducible, auditable, and debuggable. Every engine step has logs and timing
- Swapping LLM providers doesn't affect memory quality. The engine's decisions don't depend on any specific model

**Cost:** The prompt injection structure depends on the base model's caching strategy (currently tuned for DeepSeek). Switching models requires restructuring the injection layer (see [Known Technical Debt](#known-technical-debt)).

### Decision 2: Tight Coupling, Not Loose

All functional modules (intent, emotion, personality, relationship, memory) live in a single process. Functions call each other directly — no RPC, no message queues, no microservice boundaries.

**Why:**
- Intent analysis and emotion perception exchange information in real time. Split them into separate services and all that remains is a data protocol — you lose the ability for them to sense and resonate with each other
- "Tight coupling" doesn't mean "can't scale" — it means "decisions flow freely across all contexts"

**Design target:** 1-to-1 service (one engine serves one user). Not multi-tenant. This constraint is an active choice, not an engineering limitation.

**Cost:** Module boundaries are blurry. `ConsolidationEngine` handles too many responsibilities (see [Known Technical Debt](#known-technical-debt)).

### Decision 3: Time Is a Skeleton, Not a Decay Factor

Most memory systems use exponential time decay — the older the memory, the lower the weight. First Beat uses no time-based decay function whatsoever.

**Why:**
- Old memories aren't necessarily less important. A memory from 3 months ago saying "I'm afraid of failure" matters more than yesterday's 10 messages about "what I ate"
- Time participates in organization (time-period indexing), association (co-occurrence matrix), and surfacing (impulse system's curiosity source) — but never in score decay
- Weight is driven by `hit_count` (behavior): memories that get recalled often naturally gain weight

### Decision 4: The Engine Has Its Own Rhythm. It Doesn't Wait.

10+ background daemon threads run their own Poisson rhythms. Consolidation, impulse generation, distillation, pattern discovery — all run whether the user is online or not.

**Why:**
- A real memory system isn't "I search when you ask" — it digests, organizes, and discovers patterns autonomously while you're away
- The impulse system lets the engine speak unprompted when it has something to say, rather than forever waiting passively

### Decision 5: Original Text Is Never Compressed

Original text is stored in ChromaDB. Summaries and embeddings are translations (for retrieval), not modifications (they don't replace the original).

**Why:**
- Summaries lose detail. Once the original is discarded, those details are gone forever
- Retrieval uses summaries + embeddings, but the LLM sees the original text in its prompt
- **Lesson learned:** v2.0's caching optimization mistakenly had the LLM see only summaries — single-session fact recall dropped from ~96% to 79%. Fixed by passing both summary + original text. This bug ran in production for two months undetected until the LongMemEval benchmark exposed it.

### Decision 6: Benchmark Mode Isolated from Cognitive Pipeline

Triggered by `BENCHMARK_MODE=true` environment variable. Doesn't affect normal code paths — all changes are behind a feature flag.

**Why:**
- Benchmarks test "retrieve raw text and feed it to the LLM." The system's cognitive layers (summarization, emotion, entity extraction, weaving, decay) are noise in that scenario
- Adaptation via feature flag rather than code changes or prompt tweaks
- Preserves full retrieval pipeline participation (10 parallel paths + BM25 fulltext + exhaustive fallback), only bypasses the cognitive filtering layer

---

## System Overview

```
Request-Response Pipeline (per chat)    Background Rhythm (fully autonomous)
──────────────────────────────          ──────────────────────────────
User message                            5 Impulse Sources → PriorityQueue
  │                                      Emotion trend / Time rhythm /
  ├─ Intent/Emotion (bge-m3 prototype)   Random roam / Curiosity / Behavior
  ├─ 10-path parallel retrieval          │
  ├─ Two-stage ranking                   Impulse Consumer (idle > 2 min)
  ├─ Weave context                         │
  │   ├─ Storyline detection             LLM generates → [inner voice]
  │   ├─ Cognitive layering              
  │   ├─ Conflict detection              Consolidation Engine (DMN merged)
  │   └─ Token budget                      ├─ Shallow 4h
  ├─ Circuit Orchestrator                  │   Topic tree / Duplicates /
  │   ├─ Gate decision                     │   Portrait shallow / Conflict
  │   ├─ Impulse injection                 ├─ Deep 24h
  │   ├─ Behavior prediction               │   Archival / Topic notes /
  │   ├─ Relationship assessment           │   Emotion dampening /
  │   └─ Portrait injection (s+d)          │   Portrait deep
  ├─ LLM generates reply                   ├─ AI (mirror ai_dmn)
  └─ Store (chat_history + ChromaDB            Shares triggers with user,
      + AI memory + Working memory             independent ConsolidationEngine
      + Portrait realtime update)          │
                                          Pattern Discovery 6h
                                            Temporal/Emotion/Topic/
                                            Rhythm/Trend

                                          Portrait System
                                            ├─ Realtime (after every turn, <100ms)
                                            ├─ Shallow (4h, extractors→LLM write)
                                            └─ Deep (24h, global scan+LLM synthesis)
```

Intersection points between the two layers:
- **Shared memory store** — request pipeline writes, background consolidation reads and reorganizes
- **Impulse injection** — background impulse signals are checked by CircuitOrchestrator, influencing LLM response direction
- **Bidirectional portrait flow** — background portrait system → portrait_stable/dynamic constant injection in system prompt, LLM responses → storage → relationship assessment → realtime portrait update → deep global synthesis
- **Shared working memory** — request pipeline writes conversation digest, next request loads it as context

---

## Cognitive State Layer

The cornerstone of the entire refactor — located in `app/core/state.py`.

### Old Architecture vs. New

```
Old: Engine → text note ("[HIGH CONFIDENCE] user likes coffee") → LLM figures it out
New: Engine → CognitiveState → LLM executes as directed
```

### MemoryDirective's Four-Tier Layering

| role | Meaning | What the LLM should do |
|------|---------|----------------------|
| `fact` | Engine high confidence | Can reference directly as fact |
| `reference` | Engine moderate confidence | Should hedge, verify |
| `background` | Contextually relevant | Use for tone, don't mention |
| `suppressed` | Engine filtered out | Not shown to LLM |

The LLM's prompt contains no text labels — confidence is a continuous `relevance` value, emotion is raw `emotional_intensity` + `valence`. No "high/medium/low" or "emotion: positive/negative" classifications. The LLM judges weights on its own.

### UtteranceSpec Construction

```
CircuitOrchestrator.process()
  ├─ 1. Intent analysis     → UserMessageAnalysis (intent + emotion + urgency)
  ├─ 2. Emotion analysis    → emotion_intensity supplement (exclamation/emoji/adverbs)
  ├─ 3. Retrieval           → 10-path parallel + weaving + cognitive layering
  ├─ 4. Gate decision       → GatingDecision (tone + formality + response_mode)
  ├─ 5. Impulse injection   → Check PriorityQueue, inject ImpulseDirective
  ├─ 6. Behavior prediction → mirror_prediction (Markov next intent/topic)
  ├─ 7. Relationship        → RelationshipState (familiarity + trust + closeness)
  ├─ 8. Portrait injection  → PortraitRenderer.render_stable() + render_dynamic()
  │                            stable (8 dims) → message[0] system prompt (prefix cache hits)
  │                            dynamic (4 dims) → message[N+1] system prompt (per-turn update)
  └─ 9. Personality         → [Phase 4 retiring] User tags + AI self-expression tags
                               │
                               ▼
                        UtteranceSpec → LLMClient.generate() → Reply
```

---

## Portrait System

Located in `app/portrait/`. PORTRAIT.md replaces the fragmented PersonalityStore + DistillEngine + personality_notes injection, providing a 12-dimension persistent cognitive portrait that is constantly injected into the LLM prompt.

### Core Problem & Design Flaws Fixed

**Old design:** Personality tags were semantically retrieved per query (`rerank_tags(top_k=3)`). When discussing topic A, personality traits distilled from topic B were invisible. Cognitive outputs (intent/emotion/relationship/behavior prediction) were computed and discarded every turn. Distillation outputs were isolated `{content, type, confidence}` fragment tags — never aggregated into a coherent picture of "what kind of person this is."

**New design:** The portrait is a 12-dimension structured Markdown document (PORTRAIT.md), unconditionally injected into every LLM prompt — no retrieval dependency. Cognitive conclusions accumulate incrementally: relationship snapshots update in realtime after each turn (<100ms, no LLM call); 4h/24h consolidation triggers engine feature extraction → LLM text synthesis → merge into portrait.

### 12 Dimensions

| Dim | User | AI | Update Tier | Description |
|-----|------|-----|-------------|-------------|
| Core traits | usr1 | ai1 | deep | Long-term stable personality/expression features |
| Current state | usr2 | ai2 | realtime | Current snapshot of emotion/energy/interests |
| Behavior rhythm | usr3 | ai3 | shallow | Time patterns / habit frequency |
| Relationship snapshot | usr4 | ai4 | realtime | familiarity/trust/closeness |
| Interest graph | usr5 | ai5 | shallow | Topic preferences / knowledge coverage |
| Emotion graph | usr6 | ai6 | deep | Long-term emotional trajectory / reaction patterns |

Stable portrait (8 dims: usr1/3/5/6 + ai1/3/5/6) injected into message[0] system prompt, hitting DeepSeek prefix cache (>95% hit rate). Dynamic portrait (4 dims: usr2/4 + ai2/4) injected into message[N+1] system prompt, updated in realtime after each turn.

### Three-Tier Update Rhythm

```
PortraitWriter
├─ realtime_update(utterance_spec, relationship_state)
│    After every turn (triggered in chat.py via loop.run_in_executor)
│    Updates only 4 dimensions (usr2/ai2 + usr4/ai4)
│    Engine feature extraction + rule-based synthesis, <100ms, no LLM
│
├─ shallow_update(app_context)
│    Mounted on DMN shallow consolidation rhythm (4h)
│    Extractors scan recent memories → change detection → LLM writes entries → merge into PORTRAIT.md
│    Covers usr3/ai3 + usr5/ai5 + fills new candidate entries for usr1/ai1
│
└─ deep_update(app_context)
     Mounted on DMN deep consolidation rhythm (24h)
     Global scan → LLM synthesis → merge/prune old entries
     Covers usr1/ai1 + usr6/ai6, minimum 20-turn threshold
```

### Portrait Injection Format

The LLM receives stable portrait as an independent paragraph at the end of message[0]'s system prompt, and dynamic portrait as an independent system message at position N+1:

```
message[0] system:  "You are First Beat..." [core personality + tool rules + stable portrait (8 dims)]
message[1] user:    conversation history...
message[2] assistant: ...
...
message[N] system:   [dynamic portrait (4 dims) + session_context + now_hint()]
message[N+1] user:  current user message
```

### Portrait & Retrieval Contact Point

Portrait does not go through retrieval — it is constantly injected into the prompt. The only retrieval contact point is in `pipeline.py`'s ranking stage: if a candidate memory's associated entities/tags show high relevance to portrait entries, it receives a +0.05 light boost. No portrait information is lost due to "not matched" — the portrait is already in the prompt.

### Module Structure

```
app/portrait/
├── __init__.py       # Module entry
├── manager.py        # PORTRAIT.md load/parse/write/entry CRUD
├── state.py          # PortraitEntry / EntryStatus / EntryStateMachine
├── extractors.py     # Feature extractors (tag_counter / entity_aggregator / ...)
├── renderer.py       # Render as stable/dynamic prompt sections
└── writer.py         # Three-tier updates (realtime / shallow / deep)
```

---

## Retrieval Pipeline

Located in `app/retrieval/pipeline.py`. Latency target < 500ms (including embedding).

### 10 Parallel Retrieval Paths

| Path | Method | Characteristics |
|------|--------|----------------|
| ① Semantic hot | ChromaDB (heat=hot) | High-activity memories first |
| ② Semantic cool | ChromaDB (warm/cool) | Low-activity fallback, sim≥0.3 |
| ③ BM25 fulltext | BM25Okapi | Full document index, exact keyword match |
| ④ Keyword | Inverted index (summaries) | AND → OR degradation |
| ⑤ Tag | Tag inverted index | Exact match ≥1 tag |
| ⑥ Entity | Entity name exact match | PERSON/LOCATION/ORG etc. |
| ⑦ Co-occurrence | Co-occurrence matrix expansion | Memories co-occurring with hits |
| ⑧ Time-triggered | TemporalPatternIndex | Historical patterns at current time |
| ⑨ Topic tree | Topic tree branch expansion | Same topic cluster memories |
| ⑩ Attention drift | Last 3 turns weighted embedding | Simulates attentional inertia |

Path ⑨ (v2.0) simulates human attentional inertia — when chatting about the same topic, related memories get auto-weighted. Path ⑩ (v2.2) solves keyword index missing matches on summaries by indexing full documents.

### Intent Gating

Different intents get different retrieval path quotas:

| intent | semantic | tag | entity | time_expand |
|--------|:--------:|:---:|:------:|:-----------:|
| casual | 10 | 5 | 0 | 0 |
| recall | 20 | 8 | 5 | 5 |
| ask_fact | 25 | 10 | 5 | 0 |
| emotional_sharing | 12 | 5 | 0 | 3 |
| conflict | 25 | 10 | 5 | 5 |

Benchmark mode quotas are 2-5× wider.

### Dedup and Ranking

1. Dedup by `id` across all paths
2. Two-stage ranking: embedding cosine + hit_count weighted + recency_weight soft degradation + portrait light boost (+0.05)
3. v2.1 soft degradation formula: `recency_weight = 1.0 - (days_ago / 90) × (1.0 - 0.15)`, floor 0.15
   - archived memories: cap 0.6
   - stale memories: cap 0.3
4. Source priority: semantic > bm25_fulltext > entity > keyword > tag > time > cooccurrence > attention
5. **Benchmark exhaustive fallback:** when BENCHMARK_MODE=true and ChromaDB memories ≤ 200, return all, zero retrieval loss
6. **Phase 4: Personality tag retrieval retired** — portrait constant injection replaces per-query `rerank_tags(top_k=3)` semantic recall. Portrait bypasses retrieval; it is unconditionally injected into every prompt.

### Weave Context

v2.0 introduced: replaces fixed TOP_K truncation. Zero LLM calls. Latency target < 150ms.

**Four-layer decision mechanism:**

```
Candidate memories (10-path retrieval results)
    │
    ├─ Preprocessing: remove stale + parse metadata
    │
    ├─ should_speak check
    │     └─ casual intent + candidates ≤3 → don't speak
    │
    ├─ Layer 1: Storyline weaving
    │     ├─ Cluster by entity/tag (across time)
    │     ├─ Calculate time span (≥1 day for storyline)
    │     └─ Extract emotional trend (continuation/reversal/sustained positive/sustained negative)
    │
    ├─ Layer 2: Cognitive layering
    │     ├─ fact: in storyline + semantic distance < 0.30 × source_boost
    │     ├─ reference / background: others by relevance
    │     └─ suppressed: engine filtered, not shown to LLM
    │
    └─ Layer 3: Token budget
          ├─ MAX_TOKENS = 20000 (soft limit)
          └─ Truncate by narrative summary, not hard cut
```

**Key design points:**

| Feature | Implementation |
|---------|---------------|
| Narrative detection | Cluster by entity+tag, extract cross-time patterns |
| Emotion trends | Detect valence shifts across mentions of same entity |
| Source awareness | semantic_hot(1.0) > entity(0.85) > keyword(0.7) > ... |
| Token control | Not a fixed count — 20000-token soft budget |
| Casual suppression | intent=casual + candidates≤3 → should_speak=False |

### V2 Prompt Injection Format

v2.0 uses JSON + tool role injection (replacing v1's plain-text memory paragraphs):

```json
{
  "id": "mem_003",
  "time": "2026-06-04 15:35",
  "relative_time": "1 day ago",
  "summary": "User calls themselves Plankton, likes coding late at night, drinks three cups of coffee daily",
  "document": "Full original text, never truncated",
  "source": "semantic_hot",
  "hit_count": 12,
  "relevance": 0.92,
  "stale": false,
  "emotional_intensity": 3,
  "emotion_valence": "positive"
}
```

**Key design principles:**
- The engine only filters (weave token budget), never truncates — `summary` and `document` are passed in full, no hardcoded `[:N]` truncation
- Emotion and confidence are raw values (`relevance: 0.92`, `emotional_intensity: 3`), not text labels — the LLM judges weights on its own
- Memories use `tool` role injection (API natively recognizes them as external facts), separated from `user/assistant` conversation history
- System prompt + conversation history form a stable prefix that hits DeepSeek's prompt cache

---

## Background Autonomous Rhythm

### Impulse System (6 threads — 5 sources + 1 consumer)

```
5 Impulse Sources              Impulse Consumer (1 thread)
──────────────────             ──────────────────────────
Emotion trend (10min)  ──┐     Polls PriorityQueue
Time rhythm (30min)    ──┤     │
Random roam (10min)    ──┼──→  ├─ Check idle (>2 min)
Curiosity (20min)      ──┤     ├─ Take highest priority
Behavior pattern (30min)──┘     ├─ LLM generates natural language
                                └─ Store as [inner voice]
```

**Internal inhibition:**
- Each source has fatigue (0~1), +0.15 per emission
- Fatigue half-life: 15 minutes
- Effective priority = base priority × (1 - fatigue)
- Effective priority < 2 → discard (suppress)
- Expired signals (> TTL) auto-discard

### Consolidation Engine

| Level | Interval | Trigger | Content |
|-------|----------|---------|---------|
| Shallow | 4h | DMN merged ticker | Topic tree rebuild, semantic duplicate detection, tag embedding index, portrait shallow update, personality symmetry, conflict detection, entity pair evolution, hot/cool transition |
| Deep | 24h | DMN merged ticker | Archival assessment, topic note generation, portrait deep update, emotion dampening (high-arousal old memories) |
| Idle | By idle duration | DMN worker | Level 1 warmup (rebuild retrieval cache), Level 2 review (>4h), Level 3 daily consolidation (>24h) |
| AI | Synced with user | DMN merged ticker shared trigger | AI side has a full ConsolidationEngine mirror (ai_dmn), sharing on_idle/shallow/deep triggers with user side; AI emotion dampening retains independent hourly timer |

### Phase 0b: AI Consolidation Mirror

The AI side no longer uses a standalone worker thread with simplified logic. In the DMN merged ticker, when the user side triggers shallow/deep consolidation, the AI side triggers synchronously — two ConsolidationEngine instances (`self.dmn` and `self.ai_dmn`) execute sequentially within the same ticker loop. AI memory ingestion receives full metadata parity with user memories: entity extraction (qwen2.5:3b), complete 10-field time features, session_continued flag, and dual-dimension emotion analysis based on full_text (user+AI).

### Distillation Engine (Zero LLM — Phase 4 Retiring)

`app/background/distill.py` — Pure statistical extraction, being gradually replaced by the portrait system. During the transition, idle distillation triggers are still executed, but outputs no longer affect portrait injection:
- Tag co-occurrence clustering
- Time pattern detection (time-period → topic correlation)
- Emotion correlation analysis
- Trend analysis

### Pattern Discovery (Zero LLM)

`app/analysis/pattern_discovery.py` — Runs every 6h, pure statistics, 5 modes:

- **Temporal patterns**: TemporalPatternIndex → current-period topic patterns → engine auto-tuning
- **Emotion anchors**: Russell coordinates → topic-emotion correlation
- **Topic drift**: Before/after topic distribution comparison
- **Interaction rhythm**: Session length and interval analysis
- **Trend detection**: Linear regression slope → formality_shift / emotional_dampening trends

Output written to `pattern_cache.json`, injected into prompt's `[Pattern Observations]` section.

### Thread Lifecycle Management

`app/background/lifecycle.py` — Unified registration/start/stop:
- Crash auto-restart (max 5 times/hour/thread)
- Graceful shutdown (stop_event)
- Thread liveness monitoring

---

## Memory Lifecycle

```
hot (new / emotional_intensity≥2)    warm (normal)           cool (cold)
     │                                   │                       │
     │ hit_count grows                    │ 14 days untouched     │
     │                                   ▼                       │
     │                                cool                       │
     │                                                           │
     │ emotion flip / fact update                                 │
     ▼                                                           ▼
  stale (superseded, soft degraded)                    archived
  recency_weight ≤ 0.3                                recency_weight ≤ 0.6
```

### State Transitions

| Transition | Trigger | Behavior |
|-----------|---------|----------|
| New → hot | `emotional_intensity >= 2` or high emotion | Initial heat = hot |
| New → warm | Above condition not met | Initial heat = warm |
| → cool | 14 days without hits | Checked during shallow consolidation |
| → stale | New memory semantically similar + emotion flip / fact update | Old memory marked `stale=True`, records `superseded_by` |
| → archived | Topic cluster median last-hit > 90 days | Marked archived |

### v2.1: Soft Degradation System

No memory is ever hard-blocked. All memories stay in the candidate pool, soft-degraded via `recency_weight`:

- **Normal memories**: 90-day linear decay to 0.15
- **Stale memories**: Cap 0.3, excluded from `fact_memories`, routed to `stale_context`
- **Archived memories**: Cap 0.6
- **Error-reported memories**: error_count ↑ → score ↓

Stale memories are injected into the LLM with `stale_reason` and `superseded_by`. The LLM can use them as background to understand change over time, but must not cite them as current facts.

### Emotion Decay (Independent of Consolidation Scheduler)

Triggers every 50 `increment_hit_count` calls. High-intensity memories untouched for 3 days have their intensity naturally decayed. Does not depend on the consolidation scheduler — happens naturally on the retrieval path.

---

## AI Self-Expression Memory

A separate ChromaDB collection (`ai_memories`) stores the AI's response style and expressive patterns. v2.3 achieves full metadata parity with user memories.

- **Write**: After each conversation, analyze the AI's reply and extract expression-style tags. AI memory ingestion now receives complete metadata: entity extraction (qwen2.5:3b), 10-field time features, session_continued flag, and dual-dimension emotion analysis from full_text (user+AI) — fully equivalent to the user side
- **Retrieve**: Path R10 — match historical expression styles to the current context
- **Consolidate**: AI has an independent ConsolidationEngine instance (ai_dmn) that shares on_idle/shallow/deep consolidation triggers with the user side in the DMN merged ticker. AI emotion dampening retains its own hourly timer
- **Inject**: AI portrait dimensions (ai1~ai6) are stored alongside user dimensions (usr1~usr6) in PORTRAIT.md, rendered by PortraitRenderer and injected into the system prompt — replacing the old `personality_notes_ai` top-5-by-created_at pattern

---

## Working Memory Digest

`app/memory/working.py` — Maintains conversation context incrementally, replacing full history injection.

```
Old approach: inject entire conversation history → token explosion
New approach: working memory digest (~3K tokens) + last 5 turns raw text (~2K tokens)
```

- **Incremental digest**: Updated after each turn by local LLM (zero API cost)
- **Topic shift detection**: Triggers full rewrite when topic overlap < 30%
- **Lock protection**: `RLock` prevents read/write races
- **Version number**: Incremented on each update for consistency checks

This doesn't replace ChromaDB memory retrieval — it specifically addresses the "what are we talking about right now" transient context problem.

---

## User Feedback Loop

`app/core/feedback.py` — External correction of memory quality.

| Action | Mechanism | Effect |
|--------|-----------|--------|
| User reports error | `log_error_report()` → error_reports.jsonl | Retrieval degraded; higher error_count → lower score |
| User corrects | New fact overrides old + stale marking | Corrected memory +0.3 boost, same-tag group +0.1 |
| User downvotes | Downvote signal | -0.3 penalty |
| Clear errors | `clear_memory_errors()` | Append clear marker, restore weight |

All feedback uses JSONL append (no file rewrite) for concurrency safety.

---

## E2E Benchmark Suite

`E2E/` — 6 test files, 89 check nodes, 5 chains. Real ChromaDB + real bge-m3 + real local LLM. No mocking.

| Chain | File | Nodes | Core Question |
|-------|------|:-----:|---------------|
| 1: Write | `test_write_path.py` | 12 | "Was it stored? Stored correctly?" |
| 2: Retrieve+Weave+Cognition | `test_link2_retrieve.py` | 35 | "Can we find it? Is the weaving correct? Is the reply reliable?" |
| 3: Cross-Turn Memory | `test_link3_cross_turn.py` | 9 | "Remembered across turns? Works with rephrased queries?" |
| 4: Memory Evolution | `test_link4_evolution.py` | 16 | "Does memory quality degrade over time?" |
| 5: Background Rhythm | `test_background.py` | 17 | "When no user is around, what's the system doing? Doing it right?" |

**Design principles:**
- Test data isolation: each case uses independent ChromaDB collection / temp directory
- Fixed random seed for reproducibility
- Five chains scored independently — **never combined into a single weighted number** (weighted totals mask problems)
- Fix what scores lowest

Full specification in `BENCHMARK_SPEC.md`.

---

## Key Technology Choices

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Embedding | bge-m3 via Ollama | 1024-dim, Chinese-friendly, local runtime, zero API cost |
| Vector store | ChromaDB | Local persistence, HNSW index, no external service |
| Full-text search | BM25Okapi (in-memory) | Benchmark-appropriate for <10K records; production-ready with disk index |
| Chinese tokenization | Character 2-gram + bge-m3 KeyBERT | No jieba dependency, comparable accuracy |
| Semantic core | bge-m3 prototype matching | Intent/emotion classification without trained classifiers, lazy cache |
| Emotion model | Russell circumplex (valence × arousal) | More nuanced than one-dimensional positive/negative |
| Entity extraction | qwen2.5:3b (Ollama) | Local runtime, async during ingestion only |
| Summarization | qwen2.5:3b (Ollama) | Local runtime, zero API cost (replaced original DeepSeek) |
| Main LLM | DeepSeek API (OpenAI-compatible) | Swappable via BASE_URL/API_KEY/MODEL env vars |
| Deployment | Docker + docker-compose | Ollama separate container + app container, with healthcheck |
| Data encryption | None (local 1-to-1 deployment) | No network transit; security boundary at host level |

---

## Known Technical Debt

### ConsolidationEngine Overloaded (P1)

`app/background/consolidation.py` — one class handles: idle consolidation, warmup cache, daily consolidation, conflict detection, topic notes, hot/cold scanning, topic tree rebuild, tag embedding index, personality symmetry, archival assessment.

**Suggested split:**
- `TopicNoteManager` — topic note read/write and expiry management
- `ConflictDetector` — three-layer conflict detection funnel logic
- `ArchivalManager` — archival assessment and execution
- `ConsolidationEngine` — keep core scheduling + warmup + idle triggers

### PersonalityStore / DistillEngine → Portrait Migration (P1 — In Progress)

Phase 4 transition: personality_store retains parameter compatibility but is no longer used in CircuitOrchestrator. DistillEngine is still executed on DMN idle triggers but its outputs no longer affect prompt injection (portrait has taken over). Distillation-related APIs (`/api/personalities`, `/api/distill/status`) have been removed. Full retirement requires:
- Replace personality distillation calls in consolidation.py with portrait_writer.shallow_update()
- Remove personality/ and background/distill.py modules
- Clean up PERSONALITY/DISTILL config constants in settings.py

### Prompt Injection Tied to DeepSeek Caching (P1 — Partially Improved)

v2.3 splits the system prompt into a stable prefix (message[0], cacheable) and a dynamic suffix (message[N+1], per-turn), making traditional prompt caching more friendly for conversational use. The stable segment contains core personality + tool rules + 8-dim portrait with >95% cache hit rate. However, the overall injection structure still depends on DeepSeek's caching strategy. Model migration requires re-evaluation.

### O(n²) Full Scans (P1 — Partially Fixed)

Semantic duplicate detection has been migrated from nested for-loops to ChromaDB query (O(n log n)). But `_check_conflicts` and `_assess_archival` still use `list_all()` full scans. Acceptable when memory < 5000; needs pagination or incremental processing beyond that.

### Observability (P1 — Foundation Exists)

`/api/status` endpoint provides aggregated snapshots. Next steps:
- Frontend dashboard consuming this endpoint
- Prometheus metrics (`bottleneck.py` data can be exposed as metrics)
- Critical path alerting (all-retrieval-empty, impulse queue backlog, ChromaDB write failure)

### No CI/CD Pipeline (P2)

No GitHub Actions or other CI. E2E tests must be run manually. Listed as a contributor need in AUTHOR_EN.md.

---

## Module Dependency Graph

```
app/
├── core/          ← Cognitive pipeline core
│   ├── state.py         Cognitive state data structures (MemoryDirective, UtteranceSpec)
│   ├── circuit.py       Circuit orchestrator (intent→retrieval→gate→injection→LLM)
│   ├── bottleneck.py    Full-pipeline latency monitoring
│   ├── feedback.py      Memory error reporting and correction
│   ├── conflict.py      Conflict detection and resolution (zero automatic adjudication)
│   └── context.py       AppContext service container
│
├── brain/         ← Semantic engine (zero model deps except bge-m3)
│   ├── semantic.py      7 public functions: tags/intent/emotion/negation/urgency/tokenize
│   ├── keywords.py      Intent/emotion keyword tables + intensifiers (single source of truth)
│   └── models.py        Semantic model loading
│
├── memory/        ← Storage layer
│   ├── chroma.py        ChromaDB wrapper (user + AI dual collections)
│   ├── inverted.py      Word/tag → memory ID inverted index
│   ├── cooccur.py       Co-occurrence matrix
│   ├── entity_pair.py   Entity pair co-occurrence tracking
│   ├── affinity.py      Topic affinity graph
│   ├── temporal.py      Time pattern index
│   ├── tree.py          Topic tree (hierarchical clustering)
│   ├── working.py       Working memory digest (incremental conversation context)
│   ├── history.py       Chat history management
│   └── tag_index.py     Tag embedding index
│
├── retrieval/     ← Retrieval pipeline
│   ├── pipeline.py      10-path retrieval + gating + weaving + benchmark exhaustive fallback
│   ├── scoring.py       Two-stage ranking (cosine + hit_count + recency_weight)
│   └── bm25_fulltext.py BM25 full-text index (in-memory)
│
├── analysis/      ← Analysis layer (zero LLM)
│   ├── emotion.py       Russell circumplex model
│   ├── pattern_discovery.py  5-mode pattern discovery (temporal/emotion/topic/rhythm/trend)
│   ├── entity.py        Entity extraction (qwen2.5:3b offline call)
│   ├── symmetry.py      Personality symmetry analysis (user/AI dual-matrix blind spot detection)
│   └── predictor.py     Behavior prediction (Markov chain)
│
├── portrait/      ← Cognitive portrait system (engine extraction + LLM synthesis)
│   ├── manager.py       PORTRAIT.md load/parse/write/entry CRUD
│   ├── state.py         PortraitEntry / EntryStatus / EntryStateMachine
│   ├── extractors.py    Feature extractors (tag_counter/entity_aggregator/...)
│   ├── renderer.py      Render stable (8 dims) / dynamic (4 dims) prompt sections
│   └── writer.py        Three-tier updates (realtime/shallow/deep)
│
├── personality/   ← Dual personality system (Phase 4 retiring, replaced by portrait/)
│   ├── store.py         Personality tag storage
│   └── behavior.py      Behavior pattern analysis
│
├── background/    ← Background autonomous rhythm
│   ├── consolidation.py  Consolidation engine (shallow/deep/idle three-level, user+AI dual instances)
│   ├── impulse.py        Impulse system (5 sources + consumer + internal inhibition)
│   ├── distill.py        Distillation engine (zero-LLM, Phase 4 retiring)
│   └── lifecycle.py      Thread lifecycle (crash restart + rate limiting)
│
├── llm/           ← LLM adaptation layer
│   ├── deepseek.py      Main LLM client (OpenAI-compatible API) + portrait injection
│   ├── embed.py         Local embedding (bge-m3 via Ollama)
│   └── local.py         Local LLM (qwen2.5:3b, summarization/entities)
│
├── api/           ← REST layer
│   ├── app.py           FastAPI factory
│   ├── chat.py          Chat endpoint + benchmark injection + admin reset + portrait realtime
│   │   ├─ /chat             Standard chat
│   │   ├─ /chat/stream      SSE streaming
│   │   ├─ /v1/chat/completions  OpenAI-compatible
│   │   ├─ /benchmark/inject  Direct benchmark ingestion (storage pipeline, no LLM)
│   │   └─ /admin/reset       Clear all memories and indices
│   ├── system.py        System/health/status endpoints
│   ├── memories.py      Memory query endpoints
│   ├── consolidation.py Consolidation status endpoint
│   ├── portrait.py      Portrait render endpoint
│   └── chat_history.py  Chat history endpoint
│
└── tools/         ← Utility layer
    ├── atomic.py         Atomic file writes
    ├── workspace.py      Workspace operations
    └── dispatch.py       Memory query dispatch
```

**Dependency direction:** api/ → core/ → brain/ + memory/ → retrieval/ + analysis/ + personality/ → background/ → llm/

**Circular dependency control:** `llm/` and `core/` avoid circular imports via TYPE_CHECKING deferred imports.

---

*Last updated: 2026-06-09*
