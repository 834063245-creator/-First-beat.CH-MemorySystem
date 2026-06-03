# First Beat · Setup & Troubleshooting Guide

## System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| Python | 3.11+ | 3.12+ |
| Memory | 4 GB | 8 GB+ |
| Disk | 2 GB (bge-m3 model ~1.2GB) | 5 GB+ |
| GPU | Not required (CPU inference) | NVIDIA GPU + CUDA |
| OS | Windows / macOS / Linux | — |

## Quick Install

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download the installer
# https://ollama.com/download/windows
```

### 2. Pull the Embedding Model

```bash
ollama pull bge-m3
```

Verify:

```bash
ollama list
# Should show: bge-m3:latest
```

### 3. Install Python Dependencies

```bash
# Clone the repo
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd -First-beat.CH-MemorySystem

# Full install (includes local embedding)
pip install -r requirements.txt

# Lightweight install (remote embedding / MCP-only mode)
pip install -r requirements-lite.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env — check OLLAMA_EMBED_MODEL and LOCAL_LLM_OLLAMA_URL
```

### 5. Start

```bash
python run.py
# Service running at http://localhost:8082
```

### 6. Verify

```bash
curl http://localhost:8082/health
# → {"status":"ok"}
```

Or run the environment diagnostic script:

```bash
python verify_env.py
```

---

## Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `Connection refused` to Ollama | Ollama service not running | Run `ollama serve` or start the Ollama app |
| `model 'bge-m3' not found` | Model not downloaded | `ollama pull bge-m3` |
| `ModuleNotFoundError: torch` | Full deps not installed | `pip install torch` or use `requirements-lite.txt` |
| Port 8082 already in use | Another service running | Kill the old process: `netstat -ano | findstr :8082` |
| `DEEPSEEK_API_KEY` 401 | Invalid key | Works without DeepSeek — summaries fall back to keyword truncation |
| Import `app.core.circuit` fails | Project root not in Python path | Make sure you run `python run.py` from the project root |
| ChromaDB write failure | Disk space or permission issue | Check `data/` directory permissions, ensure 500MB+ free space |

### Test Ollama Connection

```bash
curl http://localhost:11434/api/tags
# Should return a JSON list of models
```

### Check Python Version

```bash
python --version
# Should show Python 3.11.x or higher
```

---

## Docker Deployment (Optional)

```bash
docker compose up -d
# Ollama + First Beat engine start together
# Access: http://localhost:8082
```

On first run, pull the embedding model inside the Docker container:

```bash
docker exec -it chuchen-ollama ollama pull bge-m3
```

---

## Next Steps

- [README](README_EN.md) — Architecture overview and MCP tool documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guide
- [AUTHOR_EN.md](AUTHOR_EN.md) — The story behind the project

Still stuck? [Open an Issue](https://github.com/834063245-creator/-First-beat.CH-MemorySystem/issues/new).
