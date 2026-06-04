# 初痕 · 3 分钟快速上手

给完全不懂技术的人准备的。跟着走，3 分钟跑起来。

---

## 第一步：安装 Ollama（1 分钟）

1. 打开 [ollama.com](https://ollama.com)，点 **Download**，选你的系统
2. 安装完成后，打开终端（Windows 按 `Win+R`，输入 `cmd` 回车）
3. 输入这行命令，等它下载完模型：

```bash
ollama pull bge-m3
```

看到 `success` 字样就对了。

---

## 第二步：安装初痕（1 分钟）

终端里继续输入：

```bash
# 1. 下载项目
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git
cd -First-beat.CH-MemorySystem

# 2. 安装依赖（等一分钟）
pip install -r requirements.txt
```

---

## 第三步：配置 LLM Key（30 秒）

引擎需要 LLM 才能说话。去 [platform.deepseek.com](https://platform.deepseek.com) 注册拿一个 API Key，或者用 OpenAI / 硅基流动等其他厂商。

```bash
cp .env.example .env
# 用记事本打开 .env，找到 LLM_API_KEY= 这一行，把 Key 填进去
# 如果用其他厂商，同时修改 LLM_BASE_URL 和 LLM_MODEL
```

不填也能启动——引擎所有后台功能（记忆、巩固、冲动、人格建模）照常运行，只是不会说话。

---

## 第四步：启动（30 秒）

```bash
# 1. 检查环境
python verify_env.py

# 2. 启动引擎
python run.py
```

看到 `Uvicorn running on http://0.0.0.0:8082` 就成功了。

---

## 验证：能用了没？

打开另一个终端，输入：

```bash
curl http://localhost:8082/health
```

返回 `{"status":"ok"}` 就说明引擎在跑了。

试着跟它说话：

```bash
curl -X POST http://localhost:8082/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好，我叫小明，我喜欢打篮球"}'
```

---

## 接下来干嘛？

1. **跟引擎聊天**：`POST /chat` 或 `POST /chat/stream`（流式）
2. **接 OpenAI 兼容客户端**：端点 `POST /v1/chat/completions`
3. **看记忆**：`GET /api/memories`、`GET /api/memories/stats`
4. **跑测试**：`python -m pytest tests/ -v`
5. **跑审计**：`python scripts/audit.py`
6. **看架构**：[README](README.md)
7. **出问题了？** [排查指南](SETUP.md)

---

> 还是不行？[提个 Issue](https://github.com/834063245-creator/-First-beat.CH-MemorySystem/issues/new)，把 `python verify_env.py` 的截图贴上。
