# 初痕 — L2+ 认知型 AI 记忆系统
# 镜像只含核心管线，Ollama 由 docker-compose 或宿主机提供
FROM python:3.11-slim

WORKDIR /app

# 系统依赖（仅 chromadb 构建需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目代码（.dockerignore 已排除 data/log/static）
COPY . .

# 暴露 FastAPI 端口
EXPOSE 8082

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8082/health')" || exit 1

# 启动
CMD ["python", "run.py"]
