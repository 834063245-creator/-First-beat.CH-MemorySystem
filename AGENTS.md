# Agent 工作指南

> 详细规格见 `CLAUDE.md`（唯一权威）。Cursor 规则在 `.cursor/rules/`。

## 启动检查清单

1. 读 `CLAUDE.md` 当前 Phase 状态（Phase 5 已完成：ChromaDB 移除，Qdrant 唯一后端）
2. 改代码前确认你在数据流的哪一步（见 `CLAUDE.md §3`）
3. 改完后：`python scripts/check_conventions.py --quick`
4. 触及核心链路：`python -m pytest tests/ -q` 或对应 E2E

## 本机环境（Windows）

- 项目解释器是 **Python 3.14（`py` 启动器）**，已装 `fastapi`/`qdrant-client` 等；PATH 上的 `python` 是 MinGW，缺依赖。跑测试/脚本用 `py`。
- 本地无 Qdrant 服务器时，`QDRANT_URL` 留空即用嵌入式文件模式（`data/qdrant`）。
- Ollama 在 `D:\OLLAM\...\ollama.exe`（含 `bge-m3`/`qwen2.5:3b`/`qwen2.5:7b`）；E2E/real_embed 需先 `ollama serve`。

## 规则文件索引

| 规则 | 范围 |
|------|------|
| `chuhen-project-core.mdc` | 始终生效：架构、依赖方向、存储现状 |
| `chuhen-red-lines.mdc` | 始终生效：六条红线 |
| `chuhen-coding-conventions.mdc` | `app/**/*.py` |
| `chuhen-testing.mdc` | `tests/`, `E2E/`, `integration/` |
| `chuhen-storage-memory.mdc` | memory/retrieval/context 存储层 |

## 不要做的事

- 未经用户确认改红线区域（prompt 结构、后台线程、settings 承重墙）
- 重新引入 ChromaDB / `STORAGE_BACKEND` 开关（已在 Phase 5 移除，Qdrant 是唯一后端）
- 批量「清理」`except Exception`
- 未请求时创建 commit 或改大量 .md 文档

## Windows 注意

PowerShell 不支持 `&&`；用 `;` 或分开执行命令。
