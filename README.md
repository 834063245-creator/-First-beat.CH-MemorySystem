# 初痕 · First Beat — 有自主节律的认知记忆引擎

> 初痕是一个独立运行的认知记忆引擎，不做文本生成，只做记忆。


[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-258%20passed-green.svg)]()
[English](README_EN.md)

👉 [快速安装](SETUP.md) | 🔧 [环境诊断](verify_env.py)

---

**别人给 LLM 加记忆插件；初痕让 LLM 当自己的语言皮层。**

不做文本生成。只做记忆。初痕是一个独立运行的认知记忆引擎。引擎负责检索、巩固、人格建模和认知决策，LLM 只负责说话。

---

## 和别的方案有什么不同

### 初痕独特的地方

| 能力 | 说明 |
|------|------|
| **自主节律** | 后台 4h/24h 双周期巩固、5 源冲动系统，引擎会主动开口——不依赖用户发消息 |
| **人格建模** | 用户 + AI 双人格独立演化，从对话中蒸馏形成认知画像 |
| **情绪分析** | Russell 二维情绪环 + bge-m3 语义原型匹配，纯本地零 LLM 调用 |
| **事实时序推理** | 双路径冲突检测，追踪情绪翻转和事实更新 |
| **模式发现** | 多时间尺度模式识别 + 自动调参 |
| **零 API Key 启动** | Clone 即跑，全部模型本地推理，不需要任何外部服务注册 |
| **语义引擎** | bge-m3 驱动：关键词抽取 / 意图分类 / 情绪分析 / 否定检测 / 紧急度，纯函数零模型文件 |

### 别人更强的地方

| 维度 | 初痕现状 | 更强的方案 |
|------|----------|------------|
| **Benchmark 跑分** | 无 LongMemEval/LoCoMo 分数 | Mem0 / Zep 有公开 benchmark 背书 |
| **知识图谱** | 不自建 | Obsidian / Mem0 / Zep / Cognee |
| **社区生态** | 个人项目 | Mem0 34K+ star，有商业公司维护 |
| **生产成熟度** | 自用为主 | Mem0 / Zep 提供 SaaS 托管 |
| **多模态** | 文本为主 | 部分方案支持图片/音频 |
| **时序推理** | 情绪驱动，不追踪中性事实变化 | Zep 原生时序图 |

### 设计定位差异

| | Mem0 | Letta (MemGPT) | **初痕** |
|---|---|---|---|
| 思路 | LLM 提取事实 → 存向量库 | LLM 管理自身内存 | **引擎决策 → LLM 执行** |
| LLM 角色 | 记忆的主人 | 记忆的管家 | 引擎的语言皮层 |
| 检索方式 | 语义 + BM25 + 实体 | 语义 + 自编辑窗口 | **8 路并行检索 + 两级精排** |
| 部署 | 云端 SaaS / 自建 | 自托管 Agent 运行时 | **pip install → python run.py** |
| 中文原生 | ❌ | ❌ | **✅ ChuchuTok + 全中文文档** |

核心差异就一句话：别人的记忆系统是 LLM 的被动工具，初痕是**有自己节律的独立器官**。引擎在后台自己跑巩固、蒸馏、冲动，不需要等用户发消息。

---

### 为什么没有 LongMemEval / LoCoMo 跑分

你可能注意到 Mem0 等系统会晒 LongMemEval、LoCoMo 等 benchmark 分数。初痕没有，原因很简单：

**这些 benchmark 测的是"事实召回"，不是"认知引擎"。**

它们的测试方式是：冷注入大量事实 → 问问题 → 测召回率。这本质上是在测一个**键值数据库的检索精度**，不是在测一个**有自主节律的认知引擎**的能力。

初痕的设计目标不是"存得多、查得准"，而是：
- 在对话中自然积累认知（不是冷注入）
- 理解用户的人格和情绪变化
- 在后台自主巩固、蒸馏、发现模式
- 在合适的时机主动开口

这些能力无法用"你问一条事实，我答一条事实"的 benchmark 来度量。

更直接地说：如果初痕的目标只是在 LongMemEval 上跑高分，我完全可以写一个专门的检索器去做这件事。但那就不是初痕了——它只是一个向量数据库，不是我想要的那个"认识用户"的东西。

事实上我已经做过一个了。早期有个叫 **Jarvis** 的实验版（现在躺在 D 盘里），SQLite + FAISS 向量检索 + LLM 提取事实，能做冷注入、事实召回、知识图谱，跑 benchmark 应该能拿个不错的分数。但它本质上还是"存进去、查出来"，没有自主节律，没有人格建模，没有冲动系统。它记性很好，但它不认识你。

初痕是在否定了 Jarvis 那条路之后从头做的。

**如果你有兴趣帮着跑 benchmark，非常欢迎。** 引擎的接口是标准的，外部评测工具可以直接调用。提 Issue 或 PR 都可以，我会全力配合。先谢过了。

> 仓库里包含了我自己用的审计套件（`scripts/audit.py`），覆盖语义检索、关键词、时间检索、排序等 8 类场景的回归测试。虽然不是 LongMemEval 那样的行业标准，但对整个系统的功能完整性做了全面验证，可作为参考。

---

## 工作原理

引擎启动后独立运行，不依赖用户在线：
- ① 意图/情绪分析 → ② 8 路并行检索 → ③ 认知状态分层 → ④ 门控决策
- 后台：巩固 4h/24h · 冲动 5源泊松 · 蒸馏 · 模式发现

---

## 快速启动

### 你需要

- **Python 3.11+**
- **Ollama** + bge-m3 模型

```bash
# 1. 安装 Ollama 并拉取 Embedding 模型
ollama pull bge-m3

# 2. 克隆 & 安装
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd chuchen

pip install -r requirements.txt

# 3. 启动引擎
python run.py
# → 服务运行于 http://localhost:8082
```

### 验证

```bash
curl http://localhost:8082/health          # → {"status":"ok"}
python verify_env.py                        # 一键诊断
```

> 安装遇到问题？查阅 [SETUP.md](SETUP.md)。

---

## Docker

```bash
docker compose up -d   # Ollama + 引擎一键启动
```

首次启动后拉模型：`docker exec chuchen-ollama ollama pull bge-m3`

---

## 架构

```
app/
├── core/          # 认知管线：意图分析 · 门控决策 · 回路编排
├── memory/        # ChromaDB 记忆库 + 工作记忆 + 倒排/共现/时间索引
├── retrieval/     # 8 路并行检索 + BM25/Embedding 两级精排
├── background/    # 后台节律：4h/24h 巩固 · 5 源冲动 · 蒸馏
├── analysis/      # Russell 情绪环 · 实体提取 · 模式发现 · 人格对称性
├── personality/   # 双人格系统（用户 + AI 独立演化）
├── llm/           # 本地 embedding (bge-m3) + DeepSeek/本地 LLM
├── api/           # REST 管理端点
├── tools/         # 原子写入 · 工具分发
├── brain/         # 语义引擎核心 semantic.py（~240 行，零模型依赖）
│   ├── semantic.py        # 7 个语义函数：标签/意图/情绪/否定/紧急度/分词/实体
│   ├── models.py          # 兼容外壳（调 semantic.py）
│   ├── keywords.py        # 关键词常量
│   └── metrics.py         # 训练指标持久化
├── config/        # 中央配置
└── models/        # Pydantic schemas

backend/           # 旧模块桥接层（逐步迁移至 app/）
tests/             # 320+ 测试，5 层覆盖
```

---

## 设计哲学

| # | 原则 | 含义 |
|---|------|------|
| 1 | **原文不加工** | 原文永不压缩。摘要和 Embedding 是翻译，不是加工 |
| 2 | **时间即骨架** | 时间参与组织、关联、浮现，不是衰减因子 |
| 3 | **行为即权重** | hit_count 决定权重，无时间衰减函数 |
| 4 | **引擎自有节律** | 巩固/冲动/蒸馏/模式发现独立运行，不依赖用户在线 |
| 5 | **引擎决策 → LLM 执行** | LLM 不拥有记忆、不调检索工具，只按引擎指令说话 |

---

## 环境变量

| 变量 | 必填 | 说明 |
|------|:----:|------|
| `OLLAMA_EMBED_MODEL` | 是 | Embedding 模型名，默认 `bge-m3` |
| `LOCAL_LLM_OLLAMA_URL` | 是 | Ollama 地址，默认 `http://localhost:11434` |
| `DEEPSEEK_API_KEY` | 否 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | 否 | DeepSeek API 地址 |
| `LOCAL_LLM_ENABLED` | 否 | 启用本地 LLM，默认 `true` |
| `LOCAL_LLM_MODEL` | 否 | 本地 LLM 模型名，默认 `qwen2.5:7b` |
| `BOCHA_API_KEY` | 否 | 博查搜索 API Key |
| `DATA_DIR` | 否 | 数据目录，默认 `./data` |
| `DEPLOY_MODE` | 否 | `full` |
| `USERS` | 否 | 多用户认证 JSON |
| `IMPULSE_ACTIVE_PATH_B` | 否 | 冲动系统开关，默认 `true` |

详见 `.env.example`。

---

## 贡献

欢迎提 Issue 和 PR。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

[MIT License](LICENSE)。

---

[📝 作者的话](AUTHOR.md)
