# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 20566f47

"""Configuration — 优先从环境变量读取，.env 文件可选。

这是新的配置中心。旧 backend/config.py 仍保留供旧模块使用，
新代码统一从此处导入。
"""
import json
import math
import os
from dotenv import load_dotenv

# 加载 .env 文件：优先找根目录，回退到 backend/.env
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if not os.path.exists(_env_path):
    _env_path = os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env")
load_dotenv(_env_path)

# ============================================================
# Embedding 提供者切换（Phase 0-5 过渡期）
# ============================================================
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")  # "vllm" 启用新后端, "ollama" 回退

# ============================================================
# vLLM 推理服务 — 替代 Ollama (Phase 1+)
# ============================================================
# vLLM 实例1: embedding（已统一切到 qwen_embed，vLLM embed 仅保留兼容）
VLLM_EMBED_URL = os.getenv("VLLM_EMBED_URL", "http://localhost:8001")
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "Qwen/Qwen2.5-7B")
VLLM_EMBED_TIMEOUT = int(os.getenv("VLLM_EMBED_TIMEOUT", "30"))

# vLLM 实例2: qwen2.5:3b 摘要 + 实体抽取
VLLM_CHAT_URL = os.getenv("VLLM_CHAT_URL", "http://localhost:8002")
VLLM_CHAT_MODEL = os.getenv("VLLM_CHAT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
VLLM_CHAT_TIMEOUT = int(os.getenv("VLLM_CHAT_TIMEOUT", "60"))

# ============================================================
# Embedding (Ollama GPU) — 回退保留
# ============================================================
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen_embed")  # v3: 已切到 qwen_embed，不再用 Ollama

# ============================================================
# LLM API（主模型生成回答）
# ============================================================
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 保留旧名导出，不破坏现有代码引用
DEEPSEEK_API_KEY = LLM_API_KEY
DEEPSEEK_BASE_URL = LLM_BASE_URL
DEEPSEEK_MODEL = LLM_MODEL

# ============================================================
# 博查搜索 API
# ============================================================
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")

# ============================================================
# 多用户认证（内测用）
# ============================================================
_USERS_DEFAULT = {
    "admin": "changeme",
}
USERS_RAW = os.getenv("USERS")
try:
    USERS: dict[str, str] = json.loads(USERS_RAW) if USERS_RAW else _USERS_DEFAULT
except (json.JSONDecodeError, TypeError):
    USERS = _USERS_DEFAULT

# ============================================================
# 数据根目录
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录
_DATA_DEFAULT = os.path.join(BASE_DIR, "data")
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", _DATA_DEFAULT))
AUTH_TOKEN_PATH = os.path.join(DATA_DIR, "auth_tokens.json")

# 用户 → 数据目录映射
_USER_DIRS_RAW = os.getenv("USER_DATA_DIRS")
try:
    USER_DATA_DIRS: dict[str, str] = json.loads(_USER_DIRS_RAW) if _USER_DIRS_RAW else {}
except (json.JSONDecodeError, TypeError):
    USER_DATA_DIRS = {}
if not USER_DATA_DIRS:
    USER_DATA_DIRS["admin"] = DATA_DIR
    # 自动发现 data/users/ 下的子目录
    _users_dir = os.path.join(DATA_DIR, "users")
    if os.path.isdir(_users_dir):
        for _u in os.listdir(_users_dir):
            _up = os.path.join(_users_dir, _u)
            if os.path.isdir(_up) and _u not in USER_DATA_DIRS:
                USER_DATA_DIRS[_u] = _up

# ============================================================
# 本地 LLM（Ollama 摘要生成）
# ============================================================
LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "true").lower() == "true"
LOCAL_LLM_OLLAMA_URL = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
LOCAL_LLM_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", "30"))

# ============================================================
# 本地推理模式 — llama.cpp + CVEC 残差注入（替代 DeepSeek API）
# ============================================================
# LOCAL_LLM_MODE=true → 引擎走本地 qwen2.5 + CVEC steering，不走 DeepSeek API
LOCAL_LLM_MODE = os.getenv("LOCAL_LLM_MODE", "false").lower() == "true"

# qwen2.5 GGUF 路径（本地推理用）
_QWEN_GGUF_DEFAULT = "D:/ollama_models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730"
QWEN_GGUF_PATH = os.getenv("QWEN_GGUF_PATH", _QWEN_GGUF_DEFAULT)

# CVEC steering 开关（本地模式下默认开启）
STEERING_ENABLED = os.getenv("STEERING_ENABLED", str(LOCAL_LLM_MODE)).lower() == "true"

# CVEC steering 全局强度倍率（1.0=默认强度，调小减弱，调大增强）
STEERING_STRENGTH = float(os.getenv("STEERING_STRENGTH", "1.0"))

# MinGW DLL 目录（Windows llama-cpp-python 运行时依赖）
MINGW_BIN_DIR = os.getenv("MINGW_BIN_DIR", "D:/mingw64/bin")

# 直接向量注入模式 — 绕过文本中转，模块结构化数值直出残差向量
# STEERING_DIRECT=true 时走 app/llm/steering_direct.py 的 build_steering_trajectory()
# STEERING_DIRECT=false 时走原有的 build_steering_segments() 文本路径
# 默认关闭（实验特性），开启需同时 LOCAL_LLM_MODE=true + STEERING_ENABLED=true
STEERING_DIRECT = os.getenv("STEERING_DIRECT", "false").lower() == "true"

# ============================================================
# AuraSDK — 零 embedding 事实召回引擎（Rust 核心 + Python 绑定）
# ============================================================
# SDR 稀疏哈希 (256k bits, 512 active) + MinHash n-gram + 倒排索引
# 确定性编码, <1ms 召回, ~3MB 内存, 不依赖任何 embedding 模型
# 替代 9 路检索管线的事实召回部分 (CLAUDE.md §10)
AURA_ENABLED = os.getenv("AURA_ENABLED", "true").lower() == "true"
AURA_DATA_DIR = os.getenv("AURA_DATA_DIR", os.path.join(DATA_DIR, "aura"))
# 默认 Level: Domain — 通用领域知识
AURA_DEFAULT_LEVEL = os.getenv("AURA_DEFAULT_LEVEL", "Domain")
# 召回 top-k
AURA_RECALL_TOP_K = int(os.getenv("AURA_RECALL_TOP_K", "10"))

# 注：OLLAMA_MODELS 环境变量仅在 Ollama 服务端进程生效，
# Python 端设置无效。保留此变量供子进程 spawn 时继承。
_OLLAMA_MODELS = os.getenv("OLLAMA_MODELS")
if _OLLAMA_MODELS:
    os.environ["OLLAMA_MODELS"] = _OLLAMA_MODELS

# ============================================================
# Qdrant 向量数据库 — 唯一存储后端（Phase 5：ChromaDB 已移除）
# ============================================================
# QDRANT_URL 为空 → 本地嵌入式文件模式（persist_dir）；
# 设为 http(s):// → 连接 Qdrant 服务器（docker-compose 显式注入）。
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # None = 本地开发无认证
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

# Qdrant 量化配置 (Phase 4 启用)
QDRANT_ON_DISK = os.getenv("QDRANT_ON_DISK", "true").lower() == "true"
QDRANT_QUANTIZATION = os.getenv("QDRANT_QUANTIZATION", "scalar_int8")  # Phase 4: 默认启用 int8 量化
QDRANT_QUANTIZATION_QUANTILE = float(os.getenv("QDRANT_QUANTIZATION_QUANTILE", "0.99"))

# Qdrant HNSW 参数
QDRANT_HNSW_M = int(os.getenv("QDRANT_HNSW_M", "16"))
QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "100"))
QDRANT_HNSW_EF = int(os.getenv("QDRANT_HNSW_EF", "64"))

# Qdrant Embedding 缓存 (Phase 4)
QDRANT_EMB_CACHE_MAX = int(os.getenv("QDRANT_EMB_CACHE_MAX", "20000"))  # LRU 上限
QDRANT_EMB_CACHE_BATCH = int(os.getenv("QDRANT_EMB_CACHE_BATCH", "500"))  # 分批 scroll 大小

# ============================================================
# Qdrant 本地持久化目录（QDRANT_URL 为空时使用）
# ============================================================
QDRANT_PERSIST_DIR = os.path.join(DATA_DIR, "qdrant")
MEMORIES_COLLECTION = "memories"

# ============================================================
# 检索 & 存储参数
# ============================================================
DEFAULT_TOP_K = 5
MAX_MEMORIES_IN_PROMPT = 12

TIME_PERIOD_MAP = {
    (0, 5): "深夜", (6, 8): "早晨", (9, 11): "上午",
    (12, 13): "中午", (14, 17): "下午", (18, 20): "傍晚",
    (21, 23): "晚上",
}

# 时间段标签反向映射（标签 → 小时范围），供 dispatch.py 工具描述等统一引用
TIME_PERIOD_LABELS = {
    "深夜": (0, 5),
    "早晨": (6, 8),
    "上午": (9, 11),
    "中午": (12, 13),
    "下午": (14, 17),
    "傍晚": (18, 20),
    "晚上": (21, 23),
}

TOPIC_KEYWORDS = {
    "情感": ["爱", "喜欢", "感情", "关系", "陪伴", "想你", "在乎", "依赖", "珍惜"],
    "技术": ["代码", "编程", "bug", "优化", "开发", "部署", "Rust", "Python", "AI", "算法"],
    "生活": ["吃饭", "睡觉", "作息", "健康", "运动", "健身", "咖啡", "茶"],
    "宠物": ["猫", "狗", "橘猫", "边牧", "宠物", "兽医", "尿闭"],
    "工作": ["公司", "项目", "leader", "重构", "微服务", "绩效", "年终奖"],
    "旅行": ["旅行", "日本", "东京", "大阪", "酒店", "机票", "景点"],
    "阅读": ["读", "书", "小说", "文章", "黑客与画家", "DDIA"],
    "音乐": ["歌", "音乐", "专辑", "周杰伦", "演唱会"],
    "家庭": ["妈", "爸", "妹妹", "家人", "老家", "郑州"],
}

# ============================================================
# 共现 & 时间触发
# ============================================================
CO_OCCURRENCE_MAX_PAIRS = 10000
CO_OCCURRENCE_CLEANUP_RATIO = 0.2
CO_OCCURRENCE_MIN_COUNT = 2
TIME_TRIGGERED_MAX = 5
CONTEXT_ROUNDS = 10

# ============================================================
# 语义重排序参数
# ============================================================
RERANK_LN_MAX = math.log(501)
RERANK_SEMANTIC_WEIGHT = 0.7     # 与文档一致
RERANK_ATTENTION_WEIGHT = 0.0    # 调用方按需传入注意力偏移量
RERANK_HIT_WEIGHT = 0.3          # hit_count 对数归一化后的评分权重
ATTENTION_WINDOW = 3

# ============================================================
# Embedding 模型注册中心
# ============================================================
DEFAULT_EMBED_MODEL = "qwen_embed"
EMBED_MODELS = {
    "qwen_embed": {
        "dimension": 3584,
        "collection": "memories",
        "provider": "local",  # qwen_embed 纯 Python+numpy，不走 HTTP
    },
    # 保留旧条目供 backfill 兼容
    "bge-m3": {
        "dimension": 1024,
        "collection": "memories",
        "provider": "legacy",
    },
}
EMBED_BACKFILL_MARKER = os.path.join(DATA_DIR, ".embed_model_backfill_done")

# ============================================================
# AI 人格系统 (画像基础设施)
# ============================================================
AI_QDRANT_DIR = os.path.join(DATA_DIR, "ai_qdrant")
AI_COLLECTION = "ai_memories"

# ============================================================
# 画像系统 (Portrait System) — Phase 4 退役完成
# ============================================================
PORTRAIT_FILE_PATH = os.getenv("PORTRAIT_FILE_PATH",
                                os.path.join(DATA_DIR, "PORTRAIT.md"))
PORTRAIT_SHALLOW_HOURS = 4
PORTRAIT_DEEP_HOURS = 24
PORTRAIT_DEEP_MIN_TURNS = 20     # 深巩固最低对话轮数门槛
PORTRAIT_REALTIME_DIMS = [2, 4]  # 实时更新维度（用户+AI 两侧）

# DMN 后台任务触发阈值（两轮对话间隔超过此值触发巩固+预热）
DMN_IDLE_TRIGGER_HOURS = 1

# ============================================================
# 时间线近端检索
# ============================================================
# legacy — 当前无 .py 引用，保留供可能的旧模块兼容
TIMELINE_RECENT_COUNT = 5
WORK_MEMORY_TOKEN_BUDGET = 200000

# ============================================================
# ChatHistory
# ============================================================
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.jsonl")
CHAT_HISTORY_MAX_MEMORY = 500

# ============================================================
# 文件存储路径
# ============================================================
CO_OCCURRENCE_FILE = os.path.join(DATA_DIR, "co_occurrence.db")
STORE_FAILURES_PATH = os.path.join(DATA_DIR, "store_failures.jsonl")

# ============================================================
# 偏移率追踪 (Drift Velocity) — Part A
# ============================================================
DRIFT_DECISION_LOG = os.path.join(DATA_DIR, "drift_decisions.jsonl")

# ============================================================
# Debug 模式
# ============================================================
DEBUG_INCLUDE_PROMPT = os.getenv("DEBUG_INCLUDE_PROMPT", "false").lower() == "true"

# ============================================================
# ============================================================
# 后台空闲处理（替代原 DMN 命名）
# ============================================================
IDLE_PREHEAT_QUERIES = 3       # 预热查询数
IDLE_LEVEL2_HOURS = 4          # 触发 Level 2 的空闲阈值（小时）
IDLE_LEVEL3_HOURS = 12         # 触发 Level 3 的空闲阈值（小时）

# ============================================================
# 引擎独立巩固节律
# ============================================================
CONSOLIDATION_SHALLOW_INTERVAL = 14400
CONSOLIDATION_DEEP_INTERVAL = 86400
ARCHIVAL_THRESHOLD_DAYS = 30

# ============================================================
# 自主触发冲动系统
# ============================================================
IMPULSE_MAX_PER_HOUR = 4
IMPULSE_MIN_INTERVAL = 600
IMPULSE_IDLE_MINUTES = 2
IMPULSE_HEARTBEAT_IDLE = 15
IMPULSE_TTL = 7200
IMPULSE_ACTIVE_PATH_B = os.getenv("IMPULSE_ACTIVE_PATH_B", "true").lower() == "true"

# ============================================================
# Benchmark 模式 — 禁用认知过滤，最大化事实召回
# ============================================================
BENCHMARK_MODE = os.getenv("BENCHMARK_MODE", "false").lower() == "true"

if BENCHMARK_MODE:
    CONSOLIDATION_SHALLOW_INTERVAL = 999_999_999   # 实质上禁用
    CONSOLIDATION_DEEP_INTERVAL = 999_999_999
    WORK_MEMORY_TOKEN_BUDGET = 200_000             # benchmark 不限制工作记忆

# ============================================================
# 共享停用词表
# ============================================================
STOP_WORDS = frozenset({"的", "了", "在", "是", "我", "有", "和", "就", "不",
                         "人", "都", "一", "一个", "上", "也", "很", "到", "说",
                         "要", "去", "你", "会", "着", "没有", "看", "好", "自己",
                         "这", "他", "她", "它", "们", "那", "什么", "怎么", "这个",
                         "那个", "吗", "啊", "吧", "呢", "哦", "嗯", "哈", "嘛",
                         "可以", "知道", "这样", "就是", "还是", "因为", "所以",
                         "但是", "如果", "虽然", "而且", "或者", "然后", "已经",
                         "时候", "现在", "今天", "明天", "昨天"})
