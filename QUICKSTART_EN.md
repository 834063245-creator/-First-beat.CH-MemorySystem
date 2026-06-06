# Quick Start · 3 Minutes

For people who just want it running. Follow along.

---

## Step 1: Install Ollama (1 min)

1. Go to [ollama.com](https://ollama.com), click **Download**, pick your OS
2. Open a terminal and pull the embedding model:

```bash
ollama pull bge-m3
```

You should see `success`.

---

## Step 2: Install First Beat (1 min)

```bash
# 1. Clone
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd -First-beat.CH-MemorySystem

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Step 3: Configure LLM Key (30 sec)

The engine needs an LLM to speak. Get an API key from [platform.deepseek.com](https://platform.deepseek.com), or use OpenAI / any OpenAI-compatible provider.

```bash
cp .env.example .env
# Open .env, fill in LLM_API_KEY=
# If using another provider, update LLM_BASE_URL and LLM_MODEL too
```

You can skip this — all background features (memory, consolidation, impulse, personality) still run. The engine just won't speak.

---

## Step 4: Start (30 sec)

```bash
# 1. Verify
python verify_env.py

# 2. Launch
python run.py
```

You'll see `Uvicorn running on http://0.0.0.0:8082`. Done.

---

## Verify

```bash
curl http://localhost:8082/health          # → {"status":"ok"}
```

Try chatting:

```bash
curl -X POST http://localhost:8082/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hi, my name is Alex, I like playing basketball"}'
```

---

## What Next?

1. **Chat**: `POST /chat` or `POST /chat/stream` (SSE streaming)
2. **OpenAI-compatible**: endpoint `POST /v1/chat/completions`
3. **Browse memories**: `GET /api/memories`, `GET /api/memories/stats`
4. **Run tests**: `python -m pytest tests/ -v`
5. **Run E2E**: `python -m pytest E2E/ -v` (needs Ollama)
6. **Run audit**: `python scripts/audit.py`
7. **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) · [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
8. **Troubleshooting**: [SETUP_EN.md](SETUP_EN.md)
9. **Docker**: `docker compose up -d`

---

> Still stuck? [Open an Issue](https://github.com/834063245-creator/-First-beat.CH-MemorySystem/issues/new) and paste the output of `python verify_env.py`.
