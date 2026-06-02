# 贡献指南

感谢你对初痕的关注！这是一个个人项目，但欢迎任何形式的参与。

## 项目理念

初痕的目标是让 AI 真正"认识"用户——不只是存储对话，而是通过独立的认知节律（巩固、冲动、蒸馏、模式发现）让记忆活起来。在贡献前，请先了解[设计哲学](README.md#设计哲学)。

## 如何贡献

### 报告 Bug

1. 在 Issues 中搜索，确认是否已有相同的报告
2. 新建 Issue，包含：
   - 环境信息（OS、Python 版本、Docker 版本）
   - 复现步骤
   - 预期行为 vs 实际行为
   - 相关日志

### 提出功能建议

1. 先搜索现有 Issues，避免重复
2. 在 Issue 中描述：
   - 你想要解决什么问题
   - 你期望的解决方案
   - 为什么这个方案适合初痕的架构

### 提交代码

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feat/your-feature`)
3. 确保现有测试通过：`python -m pytest tests/ -v`
4. 如果你的改动涉及新功能，请添加测试
5. 提交时使用中文或英文，格式：`feat: xxx` / `fix: xxx` / `docs: xxx`
6. Push 并创建 Pull Request

## 开发环境

```bash
# 1. 克隆
git clone https://github.com/834063245-creator/-First-beat.CH-MemorySystem.git && cd -First-beat.CH-MemorySystem

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置
cp .env.example .env
# 编辑 .env，填入 Ollama 配置（可选，默认即可运行）

# 4. 运行测试
python -m pytest tests/ -v

# 5. 启动开发服务器
python run.py
```

## 代码风格

- Python：遵循 PEP 8
- 注释：核心逻辑用中文注释，API 文档用英文

## 项目结构

参见 [README.md](README.md#项目结构) 中的完整目录树。

关键入口：
- `run.py` — 启动入口
- `app/api/app.py` — FastAPI 应用工厂
- `app/core/circuit.py` — 认知编排器
- `app/memory/chroma.py` — ChromaDB 记忆存储
- `app/retrieval/pipeline.py` — 检索管线

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。
