# First Beat — A Cognitive Memory Engine with Its Own Rhythm

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-320%20passed-green.svg)]()
[![MCP](https://img.shields.io/badge/MCP-10%20tools-orange.svg)]()
[中文文档](README.md)

👉 [Setup Guide](SETUP.md) | 🔧 [Environment Check](verify_env.py)

---

**Others bolt memory plugins onto LLMs. First Beat treats the LLM as its mouth.**

Doesn't generate text. Only does memory. First Beat is a standalone cognitive engine that serves memory to any AI Agent through the MCP protocol. The engine handles retrieval, consolidation, personality modeling, and cognitive decisions — the Agent's LLM is just the speaker.

---

## What Makes It Different

| | Mem0 | MemGPT / Letta | LangGraph | **First Beat (初痕)** |
|---|---|---|---|---|
| **Chinese README** | ❌ | ❌ | ❌ | **✅ Chinese + English** |
| **Chinese docs** | ❌ | ❌ | ❌ | **✅ Bilingual** |
| **Chinese tokenizer** | ❌ depends on English spaCy | ❌ none | ❌ none | **✅ ChuchuTok (Chinese char-level)** |
| **Custom Chinese models** | ❌ all GPT-based | ❌ all LLM-based | ❌ none | **✅ 4 ChuchuCNN models, 500KB each** |
| **Offline capable** | ❌ requires API | ❌ requires API | ❌ requires LLM | **✅ Ollama optional, models run locally** |
| Architecture | LLM extracts facts → stores in vector DB | LLM manages its own memory | State machine orchestrator | **Engine decides → LLM executes** |
| Retrieval | Semantic + BM25 + entity | Semantic + self-editing | (not provided) | **8-path parallel + 2-stage rerank** |
| Personality | ❌ | ❌ | ❌ | **User + AI dual personality, independent evolution** |
| Autonomous rhythm | ❌ | ❌ | ❌ | **5-source impulse, engine speaks unprompted** |
| Emotion analysis | ❌ | ❌ | ❌ | **Russell 2D circumplex + ChuchuCNN** |
| Pattern discovery | ❌ | ❌ | ❌ | **Multi-timescale + auto-tuning** |
| Background consolidation | ❌ | ❌ | ❌ | **4h/24h dual-cycle + distillation** |
| MCP protocol | ❌ | ❌ | ❌ | **✅ Native MCP Server** |
| Deployment | Cloud / self-host | Managed API | Python library | **pip install → python run.py** |
| Zero API key start | ❌ | ❌ | ❌ | **✅ Clone and run** |
| Custom models (not LLM) | ❌ | ❌ | ❌ | **✅ 4 classifiers, 2MB total** |

One sentence: other memory systems are passive tools for the LLM. First Beat is **an independent organ with its own heartbeat**. The engine runs consolidation, distillation, and impulse generation in the background — it doesn't wait for user input.

---

## How It Works

```
 Your AI Agent ─── MCP ──→ First Beat Engine (localhost:8082)
     │                          │
     │ ── run_engine("user said something") ──→
     │                          │  ① Intent / emotion analysis
     │                          │  ② 8-path parallel retrieval
     │                          │  ③ Cognitive state layering (fact / reference / background)
     │                          │  ④ Gating (suppress inappropriate impulses)
     │                          │
     │ ←── Structured Context ────  │
     │   {execute, memories,        │
     │    personality, impulses,     │
     │    relationship, mood}        │
     │                              │
 LLM generates reply                 │
     │                              │
     └── store_turn ──→ Persist ────┘

 Background (no user needed):
   Consolidation 4h/24h · 5-source impulse (Poisson) · Distillation · Pattern discovery
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** with bge-m3 model

```bash
# 1. Install Ollama and pull embedding model
ollama pull bge-m3

# 2. Clone & install
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd chuchen

pip install -r requirements.txt          # Full
# pip install -r requirements-lite.txt   # Lightweight

# 3. Start the engine
python run.py
# → Service running at http://localhost:8082
```

### Verify

```bash
curl http://localhost:8082/health          # → {"status":"ok"}
python verify_env.py                        # One-click diagnostics
```

> Troubleshooting? See [SETUP.md](SETUP.md).

---

## 10 MCP Tools

| Tool | Input | Output |
|------|-------|--------|
| **`run_engine`** | User message | Intent / emotion / memories / personality / impulses / relationship / execution directive |
| **`store_turn`** | User msg + AI reply | Persistence confirmation |
| **`query_memories`** | Query text | Semantic results with relevance, time, emotion |
| **`get_recent_history`** | N | Last N conversation turns |
| **`get_memory_stats`** | — | Total count, heat distribution, emotion distribution |
| **`get_personality_tags`** | Source (user/ai) | Personality tag list |
| **`get_topic_tree`** | — | Topic tree structure |
| **`get_relationship`** | — | Familiarity / trust / closeness / interaction mode |
| **`search_knowledge`** | Query text | Knowledge base search |
| **`get_pattern_observations`** | — | Pattern discoveries + auto-tuning records |

### run_engine Response Example

```json
{
  "execute": {
    "tone": "caring",
    "formality": 0.1,
    "response_mode": "soothe",
    "user_mood": "negative",
    "user_intent": "emotional_sharing"
  },
  "memories": [
    {"role": "fact",   "summary": "User has been under pressure lately", "time_hint": "Today", "emotional_context": "User seems down"},
    {"role": "reference", "summary": "User mentioned a project deadline last week", "time_hint": "Last week"}
  ],
  "personality": {
    "user": [{"content": "Tends to be emotional late at night", "hit_count": 8}],
    "ai":   [{"content": "Prefers empathy before advice", "hit_count": 12}]
  },
  "impulses": ["Something comes to mind — about their project"],
  "relationship": {
    "familiarity": 0.42,
    "trust": 0.68,
    "closeness": 0.35,
    "interaction_mode": "collaborator"
  }
}
```

---

## Connect Your AI Agent

Create `.claude/mcp.json` in your Agent's workspace (Claude Code), or use any MCP-compatible client:

```json
{
  "mcpServers": {
    "chuchen": {
      "url": "http://localhost:8082/mcp/jsonrpc"
    }
  }
}
```

For remote: replace with `https://your-server.com:8082/mcp/jsonrpc`.

Verify:

```bash
curl -X POST http://localhost:8082/mcp/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_memory_stats","arguments":{}},"id":"1"}'
```

---

## Docker

```bash
docker compose up -d   # Ollama + Engine, one command
```

Pull the model on first run: `docker exec chuchen-ollama ollama pull bge-m3`

---

## Architecture

```
app/
├── core/          # Cognitive pipeline: intent · gating · orchestration
├── memory/        # ChromaDB + working memory + inverted / co-occurrence / temporal indices
├── retrieval/     # 8-path parallel recall + BM25 / embedding two-stage rerank
├── background/    # Autonomous: 4h/24h consolidation · 5-source impulse · distillation
├── analysis/      # Russell circumplex · entity extraction · pattern discovery · personality symmetry
├── personality/   # Dual personality (user + AI, evolve independently)
├── mcp/           # MCP JSON-RPC server
├── llm/           # Local embedding (bge-m3) + DeepSeek / local LLM
├── api/           # REST admin endpoints
├── tools/         # Atomic writes · tool dispatch
├── brain/         # ChuchuCNN custom char-level CNN models
│   ├── model_intent/     # Intent classification (7 classes, 500KB)
│   ├── model_emotion/    # Emotion classification (5 classes, 500KB)
│   ├── model_urgency/    # Urgency 3-class (500KB)
│   └── model_negation/   # Negation detection (500KB)
├── config/        # Central config
├── models/        # Pydantic schemas
└── knowledge/     # Knowledge base management

backend/           # Legacy module shims (migrating to app/)
tests/             # 320+ tests, 5 layers: engine logic · gate · inverted index · thread safety · integration
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

## Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `OLLAMA_EMBED_MODEL` | Yes | Embedding model, default `bge-m3` |
| `LOCAL_LLM_OLLAMA_URL` | Yes | Ollama endpoint, default `http://localhost:11434` |
| `DEEPSEEK_API_KEY` | No | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | No | DeepSeek API base URL |
| `LOCAL_LLM_ENABLED` | No | Enable local LLM, default `true` |
| `LOCAL_LLM_MODEL` | No | Local LLM model, default `qwen2.5:7b` |
| `BOCHA_API_KEY` | No | Bocha search API key |
| `DATA_DIR` | No | Data directory, default `./data` |
| `DEPLOY_MODE` | No | `full` / `lite` |
| `USERS` | No | Multi-user auth JSON |
| `IMPULSE_ACTIVE_PATH_B` | No | Impulse system toggle, default `true` |

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT License](LICENSE).

---

[📝 A Note from the Author](AUTHOR_EN.md)
