# Contributing Guide

Thanks for your interest in First Beat. This project was built by one person using AI-assisted development (vibe coding) — I can't write code, but I designed every decision in this system. It has now grown beyond what I can manage alone. **I need you.**

---

## Quick Orientation: I want to help. Where do I start?

### 5-minute overview

Read in order:

1. [README_EN.md](README_EN.md) — what the project is, why it's different
2. [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) — 11 Mermaid architecture diagrams
3. [ARCHITECTURE.md](ARCHITECTURE.md) — detailed design decisions + known technical debt
4. [AUTHOR_EN.md](AUTHOR_EN.md) — the story behind this project

### Current priorities

| Priority | Task | Est. time | Requires |
|:--------:|------|:---------:|---------|
| 🔴 P0 | **Add GitHub Actions CI** — Create `.github/workflows/test.yml` to auto-run tests on push. Needs to cover `tests/` (56 files, 708 cases) and `E2E/` (5 files, 89 nodes). E2E depends on Ollama + bge-m3, may need pre-install on runner | 1 hour | GitHub Actions + Python |
| 🟡 P1 | **Split ConsolidationEngine** — `app/background/consolidation.py` (1,076 lines) does too much. Suggest extracting `TopicNoteManager`, `ConflictDetector`, `ArchivalManager` | 2-4 hours | Python · refactoring |
| 🟡 P1 | **O(n²) → incremental** — `_check_conflicts` and `_assess_archival` still use `list_all()` full scans. Needs pagination or incremental processing when memory > 5,000 | 2 hours | Python · algorithms |
| 🟢 P2 | **Prometheus metrics** — `app/core/bottleneck.py` tracks full-pipeline latency. Expose as metrics | 1 hour | Prometheus · FastAPI |
| 🟢 P2 | **Refactor tool dispatch** — `app/tools/dispatch.py` (812 lines) has tight coupling in registration/routing/execution. Consider splitting into registry + router + executor | 2-3 hours | Python · refactoring |
| 🟢 P2 | **Product builds** — Discord Bot / Telegram Bot / desktop companion. First Beat is infrastructure — build anything you want on top | open-ended | Whatever you want |

### Not sure where to start?

Open an Issue saying "I want to help, I know X." I'll point you to the best task right now.

---

## Dev Environment

### Minimal (run tests)

```bash
# 1. Clone
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd -First-beat.CH-MemorySystem

# 2. Install
pip install -r requirements.txt

# 3. Run tests
python -m pytest tests/ -v
```

### Full (run E2E + system)

```bash
# 1. Install Ollama and pull models
ollama pull bge-m3
ollama pull qwen2.5:3b

# 2. (Optional) Configure LLM
cp .env.example .env
# Edit .env, set LLM_API_KEY

# 3. Run E2E (real ChromaDB + bge-m3, no mocking)
python -m pytest E2E/ -v

# 4. Start
python run.py
# → http://localhost:8082
```

### Docker

```bash
docker compose up -d
docker exec firstbeat-ollama ollama pull bge-m3
docker exec firstbeat-ollama ollama pull qwen2.5:3b
```

---

## Project Structure (Key Entry Points)

```
app/
├── core/
│   ├── state.py         ← Cognitive state data structures. Start here.
│   ├── circuit.py       ← Circuit orchestrator. The "brain" of each conversation.
│   └── context.py       ← Service container. All modules wired here.
├── brain/
│   └── semantic.py      ← Semantic engine. Intent/emotion/tags. Zero model deps.
├── memory/
│   ├── chroma.py        ← ChromaDB wrapper. User + AI dual collections.
│   ├── working.py       ← Working memory digest. Incremental conversation context.
│   ├── inverted.py      ← Inverted index. Word/tag → memory ID.
│   ├── cooccur.py       ← Co-occurrence matrix. Entity/tag association strength.
│   ├── temporal.py      ← Temporal pattern index.
│   ├── tag_index.py     ← Multi-dimensional tag index.
│   ├── tree.py          ← Topic tree structure.
│   ├── entity_pair.py   ← Entity pair relationship graph.
│   └── affinity.py      ← Topic affinity computation.
├── retrieval/
│   ├── pipeline.py      ← 10-path parallel retrieval + weaving. Most complex file.
│   ├── scoring.py       ← Ranking formula + v2.1 soft degradation.
│   ├── bm25_fulltext.py ← BM25 full-text retrieval.
│   └── reranker.py      ← Re-ranking module.
├── background/
│   ├── consolidation.py ← Consolidation engine (⚠️ needs splitting — see tech debt)
│   ├── impulse.py       ← Impulse system (5 sources + consumer + fatigue suppression)
│   ├── distill.py       ← Distillation engine (zero-LLM profile extraction)
│   └── lifecycle.py     ← Thread lifecycle (crash restart + rate limiting)
├── analysis/
│   ├── emotion.py       ← Russell circumplex model
│   ├── entity.py        ← Entity extraction & analysis
│   ├── pattern_discovery.py ← Pattern discovery (6h, zero-LLM, 5 modes)
│   ├── predictor.py     ← Behavior prediction (Markov chain)
│   └── symmetry.py      ← Personality symmetry analysis
├── personality/
│   ├── behavior.py      ← Behavior pattern management
│   └── store.py         ← Dual personality storage (user + AI independent evolution)
├── tools/
│   ├── dispatch.py      ← Tool dispatch system (LLM tool call routing/registration/execution)
│   ├── search.py        ← Search tool
│   ├── workspace.py     ← File/workspace operations
│   └── atomic.py        ← Atomic write tool
├── llm/
│   ├── deepseek.py      ← Main LLM client (OpenAI-compatible)
│   ├── embed.py         ← Local bge-m3 embedding
│   └── local.py         ← Local qwen2.5:3b (summarization + entities)
└── api/
    └── chat.py          ← Chat endpoint + benchmark injection + admin

tests/                   # Unit + component tests (56 files, 708 cases, 53% line coverage, 98% module coverage)
E2E/                     # End-to-end full-chain regression (5 files, 89 nodes, 5 links)
scripts/                 # Audit suite + utility scripts
```

**Suggested reading order:** `state.py` → `circuit.py` → `pipeline.py` → `consolidation.py` → `impulse.py` → `dispatch.py`

---

## Code Style

- **Python**: PEP 8. In practice, the codebase is vibe-coded — style isn't perfectly uniform. Improve what you touch; don't stress about what you don't.
- **Comments**: Core logic uses Chinese comments. This is an intentional choice — we want Chinese-speaking developers to understand the codebase without translation barriers. When adding English comments, keep them alongside.
- **Commits**: Chinese or English both fine. Format: `feat:` / `fix:` / `docs:` / `perf:` / `refactor:` / `chore:`.
- **Tests**: Core logic changes must include tests. E2E/ for full-chain verification, tests/ for component-level.

---

## Pull Request Process

1. Fork the repo
2. Create a branch (`git checkout -b feat/what-you-do`)
3. Write code + tests
4. **Run tests before pushing:**
   ```bash
   python -m pytest tests/ -v
   python -m pytest E2E/ -v  # if you have Ollama
   ```
5. Commit and push
6. Create a PR describing what you did and why

---

## Context You Should Know

1. **This is a vibe-coded project.** The code was written by LLMs; the design decisions were made by a human. You'll see AI-generated artifacts (e.g. `__import__('time')` in some files, hardcoded thresholds). Fixing these is a welcome contribution.

2. **Designed for 1-to-1 service.** One engine serves one user. Not multi-tenant. PRs adding multi-user support should be discussed in an Issue first.

3. **Tight coupling is intentional.** If you think "this module should be split out" — read [ARCHITECTURE.md](ARCHITECTURE.md#决策-2紧耦合而非松耦合) first for the design rationale. (ConsolidationEngine is the known exception.)

4. **Benchmark mode.** `BENCHMARK_MODE=true` changes retrieval behavior (wider quotas, bypasses cognitive filtering). Keep this in mind when testing.

5. **E2E tests need Ollama.** They're not mocked — they call bge-m3 for real embeddings. Without GPU they'll be slow but won't crash.

---

## License

MIT License — see [LICENSE](LICENSE).

Your code is MIT too. You helped — the code belongs to the project, and the credit belongs to you.
