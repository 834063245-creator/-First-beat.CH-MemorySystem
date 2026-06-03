# 初痕 · First Beat — Cognitive Memory Engine for AI Agents

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-214%20passed-green.svg)]()
[![MCP](https://img.shields.io/badge/MCP-10%20tools-orange.svg)]()
[中文文档](README.md)

👉 [Setup Guide](SETUP.md) | 🔧 [Environment Check](verify_env.py) | ⚡ [Lightweight Install](requirements-lite.txt)

**Doesn't generate text. Only does memory.** First Beat is a standalone cognitive engine that serves memory capabilities to any AI Agent through the MCP protocol. The engine handles retrieval, consolidation, personality modeling, and cognitive decisions — the Agent's LLM acts purely as its language cortex.

> Others bolt memory plugins onto LLMs. First Beat treats the LLM as its mouth.

---

## How It Works

```
  Your AI Agent ─── MCP ──→ First Beat Engine (localhost:8082)
      │                         │
      │ ── run_engine("user said something") ──→
      │                         │  Intent → Multi-path retrieval → Gate
      │                         │  Personality → Impulse → Emotion
      │                         │
      │ ←── Structured Context ────  │
      │   {execute, memories,        │
      │    personality, impulses,    │
      │    relationship, mood}       │
      │                             │
  LLM generates reply                │
      │                             │
      └── store_turn ──→ Persist ───┘
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

# Full install (includes torch/transformers, for local-only mode)
pip install -r requirements.txt

# Lightweight install (no torch/transformers, for MCP-only mode)
# pip install -r requirements-lite.txt

# 3. Start the engine
python run.py
# → Service running at http://localhost:8082
```

### Verify

```bash
curl http://localhost:8082/health
# → {"status":"ok"}

# Or run the full environment diagnostic
python verify_env.py
```

> Troubleshooting? See [SETUP.md](SETUP.md) for detailed diagnostics.  
> Lightweight install without PyTorch? Use `pip install -r requirements-lite.txt`.

---

## Docker Deployment

### One-Click Start (recommended)

```bash
docker compose up -d
# → Ollama (11434) + First Beat Engine (8082) start together
```

### Build & Run

```bash
docker build -t chuchen .
docker run -d -p 8082:8082 -v chuchen_data:/app/data chuchen
```

### docker-compose.yml

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama

  chuchen:
    build: .
    ports:
      - "8082:8082"
    environment:
      - LOCAL_LLM_OLLAMA_URL=http://ollama:11434
      - OLLAMA_EMBED_MODEL=bge-m3
    volumes:
      - chuchen_data:/app/data
    depends_on:
      ollama:
        condition: service_started
```

> On first run, pull the embedding model: `docker exec chuchen-ollama ollama pull bge-m3`

---

## 10 MCP Tools

| Tool | Input | Output |
|------|-------|--------|
| **`run_engine`** | User message | Full cognitive context: intent, emotion, retrieved memories, personality notes, impulse signals, relationship state, execution directives |
| **`store_turn`** | User msg + AI reply | Persistence confirmation |
| **`query_memories`** | Query text | Semantic search results with relevance, timestamps, emotion |
| **`get_recent_history`** | N | Last N conversation turns |
| **`get_memory_stats`** | — | Total count, heat distribution, emotion distribution |
| **`get_personality_tags`** | Source (user/ai) | Personality tag list |
| **`get_topic_tree`** | — | Topic tree structure |
| **`get_relationship`** | — | 4D relationship state (familiarity/trust/closeness/mode) |
| **`search_knowledge`** | Query text | Knowledge base search results |
| **`get_pattern_observations`** | — | Pattern discoveries + auto-tuning records |

### run_engine Response Example

```json
{
  "execute": {
    "tone": "warm",
    "formality": 0.3,
    "intimacy": 0.3,
    "response_mode": "question_first",
    "user_mood": "neutral",
    "user_intent": "emotional_sharing"
  },
  "memories": [
    {
      "role": "fact",
      "summary": "User is building an AI memory system",
      "time_hint": "Today",
      "emotional_context": "User seems positive"
    }
  ],
  "impulses": ["Follow up on the progress of their project"],
  "personality": {
    "user": [{"content": "Prefers deep technical discussions", "hit_count": 12}],
    "ai":  [{"content": "Replies tend toward analytical style", "hit_count": 8}]
  },
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

### Local Deployment

Create `.claude/mcp.json` in your Agent's workspace:

```json
{
  "mcpServers": {
    "chuchen": {
      "url": "http://localhost:8082/mcp/jsonrpc"
    }
  }
}
```

### Remote Deployment

```json
{
  "mcpServers": {
    "chuchen": {
      "url": "https://your-server.com:8082/mcp/jsonrpc"
    }
  }
}
```

### Verify Connection

Try calling `get_memory_stats` from your Agent. If it returns memory stats, the connection is working. Or test via curl:

```bash
curl -X POST http://localhost:8082/mcp/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_memory_stats","arguments":{}},"id":"1"}'
```

Any MCP-compatible Agent (Claude Code, Cursor, etc.) gains immediate access to all 10 tools.

---

## Architecture

```
chuchen/
├── app/
│   ├── core/          # Cognitive pipeline: intent · gating · orchestration
│   ├── memory/        # ChromaDB store + working memory + co-occurrence/temporal indices
│   ├── retrieval/     # 8-path parallel recall + BM25/embedding two-stage rerank
│   ├── background/    # Autonomous rhythms: 4h/24h consolidation · 5-source impulse · distillation
│   ├── analysis/      # Russell emotion circumplex · entity extraction · pattern discovery · personality symmetry
│   ├── personality/   # Dual-personality system (user + AI evolve independently)
│   ├── mcp/           # MCP JSON-RPC server
│   ├── llm/           # Local embedding (bge-m3) + DeepSeek/local LLM clients
│   ├── api/           # REST admin endpoints
│   ├── tools/         # Atomic writes · tool dispatch
│   ├── brain/         # ChuchuCNN custom char-level CNN — intent/emotion classification, 500KB, <5ms
│   ├── config/        # Central config · env defaults · path utils
│   ├── models/        # Pydantic schemas (ChatRequest, ChatResponse, etc.)
│   └── knowledge/     # Knowledge base management
├── backend/           # Legacy module shims (migrating to app/)
├── tests/             # 320+ unit tests, 5 layers: engine logic · gate · inverted index · thread safety · integration
├── scripts/           # Audit suite · report comparison
├── Dockerfile         # Container build
├── docker-compose.yml # Ollama + Engine one-click deployment
└── run.py             # Entry point
```

---

## Audit Suite

8 audit categories covering semantic retrieval, keyword search, temporal search, ranking, corrective feedback, personality consistency, working memory, and temporal rhythms. **Run after every retrieval logic change to ensure no regression.**

```bash
python scripts/audit.py           # Run all 8 categories
python scripts/audit.py --quick   # Quick mode
```

Compare two audit reports:

```bash
python scripts/compare_reports.py audit/report_before.json audit/report_after.json
```

Reports are saved to `audit/` (gitignored).

---

## Design Principles

| # | Principle | Meaning |
|---|-----------|---------|
| 1 | **Raw text, never compressed** | Summaries and embeddings are translations, not alterations. Originals are immutable |
| 2 | **Time as skeleton** | Timestamps organize, associate, and surface memories — they're never used as decay factors |
| 3 | **Behavior is weight** | `hit_count` determines relevance weight. No artificial time decay functions |
| 4 | **The engine has its own rhythm** | Consolidation, impulse, distillation, and pattern discovery run autonomously — no user feedback required |
| 5 | **Engine decides, LLM executes** | The LLM owns no memory, calls no retrieval tools — it only speaks as directed |

---

## Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `OLLAMA_EMBED_MODEL` | Yes | Embedding model, default `bge-m3` |
| `OLLAMA_NUM_THREADS` | No | CPU threads, default `4` |
| `LOCAL_LLM_OLLAMA_URL` | Yes | Ollama endpoint, default `http://localhost:11434` |
| `DEEPSEEK_API_KEY` | No | DeepSeek API key for enhanced working memory summaries |
| `DEEPSEEK_BASE_URL` | No | DeepSeek API base URL, default `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | No | DeepSeek model name, default `deepseek-chat` |
| `LOCAL_LLM_ENABLED` | No | Enable local LLM (Ollama summary), default `false` |
| `LOCAL_LLM_MODEL` | No | Local LLM model, default `qwen2.5:7b` |
| `LOCAL_LLM_TIMEOUT` | No | Local LLM timeout in seconds, default `30` |
| `BOCHA_API_KEY` | No | Bocha search API key for web search |
| `DATA_DIR` | No | Data directory, default `./data` |
| `DEPLOY_MODE` | No | `full` / `lite` |
| `USERS` | No | Multi-user auth (JSON), e.g. `{"admin":"changeme"}` |
| `DEBUG_INCLUDE_PROMPT` | No | Include prompt in debug output, default `false` |
| `IMPULSE_ACTIVE_PATH_B` | No | Enable impulse system (proactive chat), default `true` |
| `OLLAMA_MODELS` | No | Docker-only: Ollama model mount path |

See `.env.example` for details.

---

## Contributing

Issues and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This version is released under the [MIT License](LICENSE). Future versions and derivative works may adopt different licensing terms at the author's discretion.

---

[📝 A Note from the Author](AUTHOR_EN.md)
