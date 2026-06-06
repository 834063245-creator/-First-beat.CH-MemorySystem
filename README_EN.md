# First Beat — A Self-Contained Memory Being

> First Beat is a closed-loop cognitive system. The engine makes decisions. The LLM is its mouth. Add an API key and it comes alive.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-237%2B%20collected-green.svg)]()
[![E2E](https://img.shields.io/badge/E2E-89%20nodes%20%E2%9C%93-brightgreen.svg)]()
[中文文档](README.md)

👉 [Quick Start](QUICKSTART.md) | 🔧 [Setup Guide](SETUP_EN.md) | [Environment Check](verify_env.py)

---

**Others bolt memory plugins onto LLMs. First Beat treats the LLM as its mouth.**

First Beat provides a self-contained memory infrastructure. The engine runs consolidation, impulse generation, distillation, and pattern discovery in the background — then speaks naturally through the LLM when the timing is right. What you build on top — a chat app, a desktop companion, an AI pet — is up to you. First Beat handles the memory and the voice.

**v2.2 current**: 10-path parallel retrieval + weave_context + v2.1 soft degradation + E2E 89 nodes all green. Iterating daily.

---

## Not a Memory Plugin — a Memory Being

**Most AI memory projects (Mem0, Zep, Letta, MemOS, Cognee, and others) do the same kind of thing**: they provide an SDK or API that lets developers embed "memory capability" into their own agents. And they do it well — Mem0 handles hundreds of millions of calls daily, Zep's temporal graph tops the benchmarks, and Letta's self-editing memory is an elegant design.

**First Beat chose a different direction**. It provides no SDK, exposes no integration API, and embeds into nothing. It is a complete, self-contained system — cognition, memory, emotion, impulse, and language output, all in a single process. The user adds an LLM API key, runs `python run.py`, and talks to it directly.

**The price of not being a "plugin"** is that if you want it to have Agent-like capabilities, you can only develop them from inside the closed loop — a workload far beyond what I can handle alone right now.

**I need people.** If you write Python, can read this architecture, and find this direction interesting — come help. Not hiring. Not founding a company. Just building something together. See [AUTHOR_EN.md](AUTHOR_EN.md).

Whether this direction is the right one — the path is far from paved. But it is genuinely not "yet another memory plugin."

### Design Choice: Tight Coupling

The entire pipeline is entangled. Functional areas can be identified — intent, emotion, personality, relationship — but they are not connected through modular interfaces. They interweave like a biological neural network. This isn't an engineering limitation; it's an intentional design choice. If you split intent analysis and emotion perception into two standalone services, what remains between them is just a data protocol — you lose the ability for them to sense and resonate with each other.

### The Closed Loop

The system has two layers: the request-response pipeline handles each conversation, and the autonomous background rhythm runs even when no user is present. The two layers interweave through shared memory storage and impulse injection — see [How It Works](#how-it-works) below for details.

### How It Compares

| | Mem0 | Zep | Letta | MemOS | First Beat |
|---|---|---|---|---|---|
| **What it is** | Memory API service | Temporal KG engine | Agent memory OS | Memory operating system | Self-contained memory being |
| **Integrator** | Developers (pip/SDK) | Developers (deploy/API) | Developers (Agent SDK) | Developers (deploy/API) | Anyone building a chat/companion/desktop-pet product |
| **End beneficiary** | Agent's chat users | Agent's chat users | Agent's chat users | Agent's chat users | Their product's end users |
| **How to use** | pip install → call API | Deploy service → call API | pip install → Agent SDK | Deploy → API call | pip install → add Key → run.py |
| **External interface** | SDK / REST API | MCP / REST API | Python SDK / ADE | Memory API | REST / OpenAI-compatible / SSE streaming |
| **Runs as** | Embedded in agent | Embedded in agent | Part of agent framework | Needs app layer on top | One process, full loop |
| **Background rhythm** | — | — | sleeptime compute | — | 10 threads: consolidation / impulse / distillation / patterns |
| **Unprompted speech** | — | — | — | — | Engine speaks when it wants to |
| **Coupling** | Loose (detached) | Loose | Medium (agent controls memory) | Loose | Tight — inseparable |

None of these differences are about right or wrong. Mem0 and Zep chose loose coupling — providing flexible APIs and SDKs so developers can embed memory into any architecture. First Beat chose tight coupling — providing an entire self-contained memory being that others use as infrastructure to build whatever product they want on top. Two different choices for two different kinds of users.

---

## LongMemEval / LoCoMo Scores

We eventually ran LongMemEval. 100-question curated subset, First Beat v2.1 + BENCHMARK_MODE, DeepSeek V4 Flash generation, DeepSeek-Chat judge.

**Raw score: 80.0%, corrected score: 92%.** Full experimental report: [`LONGMEMEVAL_REPORT.md`](LONGMEMEVAL_REPORT.md) (Chinese) / [`LONGMEMEVAL_REPORT_EN.md`](LONGMEMEVAL_REPORT_EN.md) (English).

But the score isn't the point. Here's what we found.

<details>
<summary><b>A personal note (click to expand)</b></summary>

I ran the benchmarks. I think I've earned the right to say this plainly.

LongMemEval has wrong answers. Not "maybe" — **verifiably wrong.** Ground truth labels that don't match the conversation data. LLM-as-Judge marking correct answers as wrong over phrasing differences. Swap the base model and your score jumps by dozens of percentage points. LoCoMo is even more absurd — it's a multimodal benchmark, but nobody actually has the image data. Everyone is running a crippled text-only version and pretending it counts.

**These benchmarks measure retrieval, not memory.** Cram conversations into a context window, or pull them out with a retriever, feed them to an LLM, let the LLM do the reasoning. It looks like a memory test. It's a reading comprehension test. Swap in a stronger base model and your score skyrockets — your memory system had nothing to do with it.

But here's the part that really gets me: **people are reporting suspiciously precise measurements from this broken ruler.**

I won't name names. But think about it — if the questions are mislabeled and the judge is unreliable, how exactly do you score 90%+? Tweaked prompts? A stronger base model? Or just stuffing everything into the context window so the memory system never actually touches the data?

**I went and read the official repos. Here's what's actually in them.**

**LongMemEval** (`xiaowu0162/LongMemEval`, ICLR 2025) has exactly four scripts:
- `evaluate_qa.py` — calls GPT-4o to judge your hypothesis
- `print_qa_metrics.py` — aggregates the scores into a table
- `run_generation.sh` — dumps ALL conversation sessions into the LLM's context window and lets it read the full text directly
- `run_retrieval.sh` — uses BM25 / Contriever / Stella to search for relevant snippets, feeds them to the LLM

Not a single line of code tells you "how to inject conversations into a memory system," "how to let the system consolidate over time," or "how to test what it remembers three months later." The design assumption is: **stuff data into an LLM context, or pull it out with a retriever. Memory is not involved.**

**LoCoMo** (`snap-research/LoCoMo`, ACL 2024) is even more explicit:
- `evaluate_gpts.sh` — dumps the entire 300+ turn conversation into GPT's context
- `evaluate_rag_gpts.sh` — runs RAG mode. But here's the catch: **the session_summary and observation fields ship with the dataset, pre-generated.** The retrieval step is pre-computed for you. You just take the results and feed them to the LLM
- `generate_session_summaries.sh` / `generate_observations.sh` — regenerate the above, if needed

Both official repos follow the exact same "evaluation" pipeline: **data → cram into LLM context window → LLM reads → LLM answers → GPT-4o scores.** At no point does "memory" enter the picture — no persistent storage, no passage of time, no retrieval pipeline, no cognitive filtering. What's being measured is the LLM's reading comprehension. Not any memory system's memory capability.

When you run a real memory system against these benchmarks, you're essentially being tested on "can you faithfully return the original conversation text to the LLM." The more your system does — summarization, emotion analysis, entity extraction, cognitive layering — the worse it scores, because all of that cognitive processing is noise in a test that only rewards raw text retrieval.

If you're choosing a memory system based on these numbers, good luck.

Score inflation is trivial. I could push both benchmarks past 95% with targeted tuning. I just won't tell you how — because none of those methods have anything to do with memory.

I built a cognitive memory engine. It accumulates understanding through conversation, tracks personality and emotional shifts, consolidates and distills autonomously, discovers patterns, and speaks when the moment is right. These things have no benchmark. I'd rather build a system that's genuinely remembering you than reshape my design around a crooked ruler.

You can have the ruler.

</details>

> The repo includes an audit suite (`scripts/audit.py`) covering 8 categories of regression tests. Not an industry-standard benchmark, but comprehensive functional verification of the entire system.

---

## How It Works

<details>
<summary><b>Expand architecture details</b></summary>

First Beat has two layers. The request-response pipeline handles each conversation — from user message in, to LLM reply out. The autonomous background rhythm runs when no user is present — consolidating memories, distilling personality, discovering patterns, generating impulses, and speaking unprompted when the timing is right. The two layers interweave through shared memory storage and impulse injection.

```
                       ┌─── Request-Response Pipeline ───┐
                       │                                  │
  User message         │                                  │       SSE streaming output
  ───────→ Embedding ──→ 10-path parallel ──→ weave_context ──→ CircuitOrchestrator
            (bge-m3)    retrieval            (4-layer         │
                        (semantic/BM25/tag/   decision engine) │
                         entity/attention/                     ├─ Intent (bge-m3 prototype match)
                         time/topic-tree/
                         co-occurrence)
                                                          ├─ Emotion (Russell circumplex)
                                                          ├─ Cognitive layering + gating
                                                          ├─ Impulse injection + mirror predict
                                                          ├─ Relationship state
                                                          └─ Output: UtteranceSpec
                                                                  │
                                                                  ▼
                                            LLM generates reply ←── LLMClient
                                            (prompt: memories + personality
                                             + impulses + execute directives)
                                                                  │
                                                                  ▼
                        ┌────────── Storage ──────────┐
                        │                              │
                        ├─ chat_history.append() ──→ JSONL conversation log
                        ├─ _enqueue_store_task() ──→ queue → worker
                        │                              │
                        │   ┌──────────────────────────┘
                        │   ▼
                        │   summary + tags + embedding → ChromaDB memory store
                        │   conflict detection ←── old vs new → auto-replace stale
                        │
                        └─ working_memory incremental update

  ┌───── Background Rhythm (10 threads, fully autonomous) ─────┐
  │                                                              │
  │  5 Impulse Sources (Poisson)    Impulse Consumer  Consolidation Engine  │
  │  ┌──────────────────┐      ┌──────────────┐    ┌──────────────┐        │
  │  │ Emotion trend     │      │              │    │ Shallow 4h    │        │
  │  │ Time rhythm       ├──→ PriorityQueue ──→ pop→LLM speak         │        │
  │  │ Random roam       │      │ store as      │    │ │ personality │        │
  │  │ Curiosity         │      │ [inner voice] │    │ │ distillation│        │
  │  │ Behavior pattern  │      └──────────────┘    │ Deep 24h      │        │
  │  └──────────────────┘                           │ │ cognitive   │        │
  │                                                  │ │ profile     │        │
  │                                                  └──────────────┘        │
  │  DMN idle detection ──→ triggers consolidation                           │
  │  AI consolidation ──→ AI expression analysis + distillation              │
  │  Pattern discovery ──→ multi-timescale stats (zero LLM calls)            │
  └──────────────────────────────────────────────────────────────────────────┘
```

### The Request-Response Pipeline: journey of a message

**① Embedding.** When a user message arrives, it is first converted into a 1024-dimensional vector by bge-m3 (Ollama, fully local). No external API calls at this stage.

**② Retrieval.** The embedding triggers 10 parallel retrieval paths — semantic hot, semantic cool, BM25 keyword, tag inverted index, entity match, attention drift, time-triggered, topic tree branching, and co-occurrence expansion. Paths run concurrently via ThreadPoolExecutor (max_workers=7), each independently recalling candidate memories. Candidates enter **weave_context** — a v2.0 4-layer decision engine replacing fixed TOP_K truncation: storyline weaving (cluster by entity/tag, detect cross-time narratives and emotional trends) → cognitive layering (fact / reference / background / suppressed) → token budget allocation (20000-token soft limit) → source priority ranking. Zero LLM calls, < 150ms latency. v2.1 soft degradation: 90-day linear decay + archived cap 0.6 + stale cap 0.3 — no memory is ever hard-blocked.

**③ Circuit Orchestration (CircuitOrchestrator).** This is the cognitive core. After receiving retrieval results, it executes in sequence:
- Intent analysis: bge-m3 semantically matches the user message against predefined intent prototypes — casual / question / emotional_sharing / request / command
- Emotion analysis: same bge-m3 prototype matching, mapped onto Russell's circumplex model (valence + arousal), producing emotion + intensity
- Cognitive layering: sorts memories into fact / reference / background tiers, deciding which are explicitly injected into the prompt and which serve as background context
- Gating: based on intent + emotion + memory confidence + relationship state, decides tone (warm/calm/humorous), formality (0~1), and response mode (empathize first / understand first / answer directly / confirm first)
- Impulse injection: checks the PriorityQueue for signals produced by background impulse sources, and injects any pending ones into UtteranceSpec
- Behavior prediction: Markov-chain probability table predicts what the user might say or need next
- Relationship assessment: combines familiarity, trust, closeness, and interaction_mode into a current relationship snapshot

All steps execute serially within a single method call. No microservices, no pipeline DAG — just a set of functions running one after another in a single thread. They don't communicate through JSON; they access the same in-memory data structures directly.

**④ LLM Generation.** UtteranceSpec is handed to LLMClient, which translates it into LLM-consumable format — memories formatted as tool-role JSON, impulses turned into natural language cues, gating decisions rendered as execution directives ("keep a warm tone, empathize before responding"), personality tags injected into the system prompt. It then calls the LLM API and streams the response. Tool calls (web search, file read/write, shell execution) are handled here — the LLM can invoke tools, results feed into the next generation round, up to two rounds.

**⑤ Storage.** After the reply is generated, two storage paths trigger in parallel. The synchronous path writes to chat_history.jsonl (conversation log) and triggers an incremental working_memory update (a lightweight summary of the most recent N turns). The asynchronous path places the message into an in-memory queue, consumed by a dedicated worker: local Ollama qwen2.5:3b generates a summary (zero API cost) → bge-m3 extracts semantic tags → entity extraction (reuses qwen2.5:3b) → emotion analysis → temporal feature annotation → write to ChromaDB. Conflict detection runs automatically during ingestion — if a new memory is semantically near-identical to an older one but more recent, the old one is marked stale and eventually replaced.

### The Autonomous Background: what happens when you're gone

The background pipeline does not depend on user messages. When the engine starts, 10 daemon threads begin running independently, each with its own Poisson rhythm or fixed interval.

**Impulse system (6 threads).** Five impulse sources run independently — Emotion Trend detects shifts in the user's emotional trajectory, Time Rhythm discovers behavioral patterns tied to specific times of day, Random Roam pulls old memories at random from the store, Curiosity explores topics that have never been discussed, and Behavior Pattern identifies paradigms in the user's behavior. Each source starts with a 120s cooldown (to let the system warm up), then triggers on its own Poisson distribution, producing (content, priority) signals that pass through fatigue suppression into a PriorityQueue. The sixth thread — the impulse consumer — polls the queue: when the user has been idle for over 2 minutes, it pulls the highest-priority signal, calls the LLM to turn it into natural language, and stores the result as `[inner voice]` in both chat_history and ChromaDB. This is how the engine "speaks unprompted" — it doesn't wait for the user to send a message. When it has something to say, it says it.

**Consolidation engine (2 threads).** Shallow consolidation triggers every 4 hours: rebuilds the topic tree, detects memory conflicts, and runs personality distillation (extracting behavior patterns, thinking patterns, preference patterns, communication patterns, and other tags from conversation). Includes a 60s startup cooldown to avoid resource contention with warmup and impulse sources. Deep consolidation triggers every 24 hours: extends shallow consolidation with cross-day pattern comparison, evolution trend detection, and cognitive profile refinement. A DMN thread handles idle detection — how long since the user last spoke, whether it's time to trigger consolidation.

**AI consolidation (1 thread).** Independent of user memory, this analyzes the AI's own expression patterns — what tone the AI uses in which emotional contexts, whether the AI's expressive habits are shifting. The resulting AI personality tags are stored separately from user personality tags and injected into the system prompt's "my own expressive habits" section during LLM generation.

**Pattern discovery (no standalone thread; triggered by consolidation).** Multi-timescale statistical pattern recognition — conversation frequency trends, topic drift velocity, emotional fluctuation cycles. Entirely statistical, zero LLM calls. Produces two outputs: tuning (auto-parameter suggestions like emotion dampening toggle, formality offset, proactive suppression) and observations (human-readable pattern descriptions injected into the LLM prompt as background context).

### Where the Two Layers Meet

The request-response pipeline and the background rhythm are not isolated — they intersect at several key points:

- **The memory store is shared.** The request pipeline writes new memories; the background consolidation pipeline reads and reorganizes them. What the user just discussed is pulled into the topic tree by shallow consolidation within hours.
- **Impulses inject into the request pipeline.** Signals produced by background impulse sources are checked during CircuitOrchestrator's impulse injection step — if there's a pending signal while the user is actively chatting, it gets injected into UtteranceSpec and influences the LLM's reply direction.
- **Personality tags flow both ways.** Tags distilled by background processes are injected into the system prompt during LLM generation, shaping how the LLM understands the user. New information in the LLM's replies gets stored, distilled, and fed back into updated personality tags.
- **Disable either direction, and the other degrades.** Without background consolidation, memories just pile up without being understood. Without the request pipeline, background impulses have no one to listen.

</details>

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** with bge-m3 model
- (Optional) **LLM API Key** — engine runs without one, just won't speak

```bash
# 1. Install Ollama and pull embedding model
ollama pull bge-m3

# 2. Clone & install
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd -First-beat.CH-MemorySystem

pip install -r requirements.txt

# 3. (Optional) Configure LLM Key (supports DeepSeek / OpenAI / SiliconFlow / etc.)
cp .env.example .env
# Edit .env, set LLM_API_KEY and LLM_BASE_URL

# 4. Start
python run.py
# → Service running at http://localhost:8082
```

### Verify

```bash
curl http://localhost:8082/health          # → {"status":"ok"}
python verify_env.py                        # One-click diagnostics
```

> Troubleshooting? See [SETUP_EN.md](SETUP_EN.md).

---

## API

The engine exposes REST and OpenAI-compatible endpoints. Connect any client (NextChat, Open WebUI, custom frontend, desktop companion shell), or call directly from your code.

### Chat

| Endpoint | Description |
|------|------|
| `POST /chat` | Chat (full response with trace) |
| `POST /chat/stream` | Chat (SSE streaming: reasoning + content + trace) |
| `POST /v1/chat/completions` | OpenAI-compatible endpoint |

### Management

| Endpoint | Description |
|------|------|
| `GET /health` | Health check |
| `GET /api/ping` | Heartbeat |
| `GET /api/user-active` | User activity heartbeat (frontend calls every 10s; used by impulse system for idle detection) |
| `GET /api/memories` | Memory list (semantic search, tag filter, pagination) |
| `GET /api/memories/stats` | Memory statistics |
| `GET /api/memories/{id}` | Memory detail (with conversation context) |
| `POST /api/memories/{id}/correct` | Correct memory summary |
| `DELETE /api/memories/{id}` | Delete memory |
| `POST /api/memories/feedback` | Report memory error |
| `GET /api/personalities` | Personality tags |
| `GET /api/consolidation/status` | Consolidation status |
| `GET /api/distill/status` | Distillation status |
| `GET /api/chat/history` | Chat history |
| `GET /api/prompt` | View/edit system prompt |

---

## Docker

```bash
docker compose up -d   # Ollama + Engine, one command
```

Pull the model on first run: `docker exec chuchen-ollama ollama pull bge-m3`

---

## Architecture

> See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions and module dependencies.

```
app/
├── core/          # Cognitive pipeline: orchestration · cognitive state · gating · context · bottleneck monitoring · feedback
├── brain/         # Semantic engine core — semantic.py (~240 lines, zero model deps)
│   ├── semantic.py        # 7 semantic functions: tags/intent/emotion/negation/urgency/tokenize/entities
│   ├── models.py          # Compatibility shim (delegates to semantic.py)
│   ├── keywords.py        # Keyword constants
│   └── metrics.py         # Training metrics persistence
├── memory/        # ChromaDB (user + AI dual collections) + working memory digest + inverted/co-occurrence/entity-pair/temporal indices
├── retrieval/     # 10-path parallel recall + weave_context 4-layer decision engine + v2.1 soft degradation
├── background/    # Autonomous: 4h/24h consolidation · 5-source impulse · distillation · 1h AI consolidation · lifecycle
├── analysis/      # Russell circumplex · entity extraction · pattern discovery · personality symmetry · behavior prediction
├── personality/   # Dual personality (user + AI, evolve independently)
├── llm/           # Local embedding (bge-m3) + LLM chat generation + local summarization (qwen2.5:3b)
├── api/           # REST endpoints: chat · memory management · personality · consolidation · distillation · feedback
├── tools/         # Atomic writes · tool dispatch · search · file operations
├── config/        # Central config
└── models/        # Pydantic schemas

tests/             # 237+ tests + E2E (6 files, 89 nodes, 5 chains) + audit suite
E2E/               # End-to-end full-chain regression (write / retrieve+weave+cognition / cross-turn / evolution / background rhythm)
```

---

## Design Principles

| # | Principle | Meaning |
|---|-----------|---------|
| 1 | **Raw text, never compressed** | Summaries and embeddings are translations, not alterations |
| 2 | **Time as skeleton** | Timestamps organize and surface — never used as decay factors |
| 3 | **Behavior is weight** | `hit_count` determines relevance. No artificial time decay |
| 4 | **The engine has its own rhythm** | Consolidation, impulse, distillation, pattern discovery run autonomously |
| 5 | **Engine decides, LLM executes** | The LLM owns no memory, calls no tools — it only speaks as directed |

---

## Tech Stack

- Python 3.11+ / FastAPI / ChromaDB (local persistence)
- Embedding: bge-m3 via Ollama (1024-dim)
- Semantic core: bge-m3 (keyword extraction / intent-emotion prototype matching)
- Entity extraction: Ollama qwen2.5:3b (async during ingestion)
- Summarization: Ollama qwen2.5:3b (reuses entity extraction model, zero API cost)
- Negation detection: whitelist + distance rules (no model)
- Urgency: 10-line ruleset (no model)
- BM25 tokenization: character 2-gram + rank-bm25
- Main LLM: configurable — DeepSeek / OpenAI / any OpenAI-compatible provider (1M context)
- Deployment: Windows / macOS / Linux, Docker optional

---

## Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `OLLAMA_EMBED_MODEL` | Yes | Embedding model, default `bge-m3` |
| `LOCAL_LLM_OLLAMA_URL` | Yes | Ollama endpoint, default `http://localhost:11434` |
| `LLM_API_KEY` | No | LLM API key (without it, engine won't speak) |
| `LLM_BASE_URL` | No | LLM API base URL, default `https://api.deepseek.com` |
| `LLM_MODEL` | No | Model name, default `deepseek-v4-flash` |
| `DEEPSEEK_API_KEY` | No | (Legacy — still works) Same as `LLM_API_KEY` |
| `LOCAL_LLM_ENABLED` | No | Enable local LLM (summarization + entity extraction), default `true` |
| `LOCAL_LLM_MODEL` | No | Local LLM model, default `qwen2.5:7b` (summarization reuses `qwen2.5:3b` internally) |
| `BOCHA_API_KEY` | No | Bocha search API key |
| `DATA_DIR` | No | Data directory, default `./data` |
| `USERS` | No | Multi-user auth JSON |
| `IMPULSE_ACTIVE_PATH_B` | No | Impulse system toggle, default `true` |

See `.env.example` for details.

---

## Audit

```bash
python scripts/audit.py              # Full 8 categories
python scripts/audit.py --quick      # Quick mode
python scripts/audit.py --category 1 # Single category
```

Reports saved in `audit/` directory.

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT License](LICENSE).

---

[📝 A Note from the Author](AUTHOR_EN.md)
