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

> 如果 `pip install -r requirements.txt` 报错（特别是 torch 装不上），用轻量版：
> ```bash
> pip install -r requirements-lite.txt
> ```

---

## 第三步：启动（1 分钟）

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

---

## 接下来干嘛？

1. **把你的 AI Agent 接上来**：在 `.claude/mcp.json` 里配置，[看这里](README.md#接入-ai-agent)
2. **跑测试**：`python -m pytest tests/ -v`
3. **看架构**：[README](README.md)
4. **出问题了？** [排查指南](SETUP.md)

---

> 还是不行？[提个 Issue](https://github.com/834063245-creator/-First-beat.CH-MemorySystem/issues/new)，把 `python verify_env.py` 的截图贴上。
