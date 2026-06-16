# 贡献指南

> [English version](CONTRIBUTING_EN.md)

感谢你对初痕的关注。这个项目是我一个人用 AI 辅助（vibe coding）做出来的——我不会写代码，但我设计了这个系统的每一个决策。现在复杂度已经超出了我一个人能控制的范围。**我需要你。**

---

## 快速定位：我想帮忙，从哪开始？

### 5 分钟了解全貌

按顺序读：

1. [README.md](README.md) — 项目是什么、为什么不一样
2. [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) — 11 张 Mermaid 架构图，一眼看懂全链路
3. [ARCHITECTURE.md](ARCHITECTURE.md) — 详细设计决策 + 已知技术债
4. [AUTHOR.md](AUTHOR.md) — 这个项目背后的故事

### 当前最需要的帮助（按优先级排）

| 优先级 | 任务 | 预估时间 | 需要什么 |
|:------:|------|:--------:|---------|
| 🔴 P0 | **加 GitHub Actions CI** — 创建 `.github/workflows/test.yml`，push 自动跑测试。需要同时覆盖 `tests/`（62 文件 708 用例）和 `E2E/`（6 文件 89 节点）。注意：E2E 依赖 Ollama + bge-m3，可能需要在 runner 上预装 | 1 小时 | GitHub Actions + Python |
| 🟡 P1 | **拆分 ConsolidationEngine** — `app/background/consolidation.py`（1076 行）一个类管了太多事。建议拆出 `TopicNoteManager`、`ConflictDetector`、`ArchivalManager` | 2-4 小时 | Python · 重构经验 |
| 🟡 P1 | **O(n²) 改为增量** — `_check_conflicts` 和 `_assess_archival` 仍用 `list_all()` 全量扫描。记忆>5000条时需分页或增量 | 2 小时 | Python · 算法 |
| 🟢 P2 | **Prometheus metrics** — `app/core/bottleneck.py` 有全链路耗时数据，暴露为 metrics | 1 小时 | Prometheus · FastAPI |
| 🟢 P2 | **工具分发重构** — `app/tools/dispatch.py`（812 行）工具注册/路由/执行耦合较紧，可考虑拆分为 registry + router + executor | 2-3 小时 | Python · 重构经验 |
| 🟢 P2 | **产品化** — Discord Bot / 微信公众号 / 桌宠外壳。初痕是基础设施，上面搭什么都可以 | 不限 | 你想做什么就做什么 |

### 不确定从哪下手？

开一个 Issue，说"我想帮忙，我会 X"，我会告诉你当前最适合你的任务。

---

## 开发环境

### 最小配置（能跑测试）

```bash
# 1. 克隆
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd -First-beat.CH-MemorySystem

# 2. 安装依赖
pip install -r requirements.txt

# 3. 跑测试
python -m pytest tests/ -v
```

### 完整配置（能跑 E2E + 系统）

```bash
# 1. 安装 Ollama 并拉模型
ollama pull bge-m3
ollama pull qwen2.5:3b

# 2. （可选）配置 LLM
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 3. 跑 E2E（真实 ChromaDB + bge-m3，不 mock）
python -m pytest E2E/ -v

# 4. 启动
python run.py
# → http://localhost:8082
```

### Docker 开发环境

```bash
docker compose up -d
docker exec chuchen-ollama ollama pull bge-m3
docker exec chuchen-ollama ollama pull qwen2.5:3b
```

---

## 项目结构（关键入口）

```
app/
├── core/
│   ├── state.py         ← 认知状态数据结构。从这里开始读。
│   ├── circuit.py       ← 回路编排器。每次对话的"大脑"。
│   └── context.py       ← 服务容器。所有模块在这里装配。
├── brain/
│   └── semantic.py      ← 语义引擎。意图/情绪/标签，零模型依赖。
├── memory/
│   ├── chroma.py        ← ChromaDB 封装。用户+AI 双集合。
│   ├── working.py       ← 工作记忆摘要。增量对话脉络。
│   ├── inverted.py      ← 倒排索引。词/标签→记忆ID。
│   ├── cooccur.py       ← 共现矩阵。实体/标签关联强度。
│   ├── temporal.py      ← 时间模式索引。
│   ├── tag_index.py     ← 标签多维索引。
│   ├── tree.py          ← 话题树结构。
│   ├── entity_pair.py   ← 实体对关系图谱。
│   └── affinity.py      ← 话题亲和度计算。
├── retrieval/
│   ├── pipeline.py      ← 9 路并行检索 + 编织。整个系统最复杂的文件。
│   ├── scoring.py       ← 精排公式 + v2.1 软降权。
├── background/
│   ├── consolidation.py ← 巩固引擎（⚠️ 需要拆分，见技术债）
│   ├── impulse.py       ← 冲动系统（5 源 + 消费者 + 疲劳抑制）
│   └── lifecycle.py     ← 线程生命周期（崩溃重启 + 限流）
├── analysis/
│   ├── emotion.py       ← Russell 二维情绪环
│   ├── entity.py        ← 实体抽取与分析
│   ├── pattern_discovery.py ← 模式发现（6h，零 LLM，5 模式）
│   ├── predictor.py     ← 行为预测（马尔可夫链）
│   ├── symmetry.py      ← 人格对称性分析
│   ├── drift.py         ← 偏移率追踪（消费/节俭/漂移）
│   └── self_mirror.py   ← AI 自我镜像
├── tools/
│   ├── dispatch.py      ← 工具分发系统（LLM 工具调用路由/注册/执行）
│   ├── search.py        ← 搜索工具
│   ├── workspace.py     ← 文件/工作区操作
│   └── atomic.py        ← 原子写入工具
├── llm/
│   ├── deepseek.py      ← 主 LLM 客户端（OpenAI 兼容）
│   ├── embed.py         ← 本地 bge-m3 embedding
│   └── local.py         ← 本地 qwen2.5:3b（摘要+实体）
└── api/
    └── chat.py          ← 聊天端点 + benchmark 注入 + 管理

tests/                   # 单元测试 + 组件测试（62 文件，708 用例，行覆盖率 53%，模块覆盖率 98%）
E2E/                     # 端到端全链路回归（6 文件，89 节点，5 链路）
scripts/                 # 审计套件 + 工具脚本
```

**阅读顺序建议：** `state.py` → `circuit.py` → `pipeline.py` → `consolidation.py` → `impulse.py` → `dispatch.py`

---

## 代码风格

- **Python**：PEP 8。但实际情况是——代码是 vibe coding 出来的，风格不完全一致。提交时尽量改善你碰到的文件，不强求。
- **注释**：核心逻辑用中文注释。这是主动选择，不是英文不好——我们希望中文开发者不用翻墙读文档就能看懂。
- **Commit**：中文或英文都行。格式 `feat:` / `fix:` / `docs:` / `perf:` / `refactor:` / `chore:`。
- **测试**：改核心逻辑必须带测试。E2E 目录下加全链路验证，tests/ 下加组件验证。

---

## 提交流程

1. Fork 仓库
2. 创建分支 (`git checkout -b feat/what-you-do`)
3. 写代码 + 测试
4. **先跑测试：**
   ```bash
   python -m pytest tests/ -v
   python -m pytest E2E/ -v  # 如果你有 Ollama
   ```
5. 提交并 Push
6. 创建 Pull Request，描述你做了什么、为什么这么做

---

## 你需要知道的上下文

1. **这个项目是 vibe coding 产物。** 代码是 LLM 写的，设计决策是我做的。你看到的代码可能有 AI 生成的痕迹（比如某些地方的 `__import__` 写法、硬编码阈值）——修复它们是受欢迎的贡献。

2. **设计目标是 1 对 1 服务。** 一个引擎服务一个用户。不做多租户。做多用户支持的 PR 请先开 Issue 讨论。

3. **紧耦合是主动选择。** 如果你觉得"这个模块该拆出去"——先看 [ARCHITECTURE.md](ARCHITECTURE.md#决策-2紧耦合而非松耦合) 里的设计理由。当然，ConsolidationEngine 是已知该拆的例外。

4. **Benchmark 模式。** `BENCHMARK_MODE=true` 环境变量会改变检索行为（放宽配额、bypass 认知过滤）。测试时注意。

5. **E2E 测试需要 Ollama。** 不是 mock——真的调 bge-m3 做 embedding。如果你没有 GPU，E2E 的 embedding 部分会很慢但不会挂。

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。

你写的代码也是 MIT。你帮了忙，代码属于这个项目，荣誉属于你。
