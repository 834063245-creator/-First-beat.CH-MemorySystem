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
# Embedding (Ollama GPU)
# ============================================================
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

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

# 注：OLLAMA_MODELS 环境变量仅在 Ollama 服务端进程生效，
# Python 端设置无效。保留此变量供子进程 spawn 时继承。
_OLLAMA_MODELS = os.getenv("OLLAMA_MODELS")
if _OLLAMA_MODELS:
    os.environ["OLLAMA_MODELS"] = _OLLAMA_MODELS

# ============================================================
# ChromaDB 持久化
# ============================================================
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma")
CHROMA_COLLECTION_NAME = "memories"

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
CONTEXT_WINDOW_SIZE = 10          # 保留供未来使用（当前无任何 .py 引用）
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
DEFAULT_EMBED_MODEL = "bge-m3"
EMBED_MODELS = {
    "bge-m3": {
        "dimension": 1024,
        "collection": "memories",
        "provider": "ollama",
    },
}
EMBED_BACKFILL_MARKER = os.path.join(DATA_DIR, ".embed_model_backfill_done")

# ============================================================
# AI 人格系统 (画像基础设施)
# ============================================================
AI_CHROMA_DIR = os.path.join(DATA_DIR, "ai_chroma")
AI_COLLECTION = "ai_memories"

# Phase 4: 人格库 & 蒸馏已退役，由画像系统替代
# 以下配置保留供旧模块（DistillEngine/PersonalityStore/ConsolidationEngine）过渡期使用
PERSONALITY_COLLECTION = "personality_tags"
PERSONALITY_CHROMA_DIR = os.path.join(DATA_DIR, "personality_chroma")
DISTILL_STATE_PATH = os.path.join(DATA_DIR, "distill_state.json")
AI_DISTILL_STATE_PATH = os.path.join(DATA_DIR, "ai_distill_state.json")
DISTILL_IDLE_HOURS = 1
PERSONALITY_DEDUP_THRESHOLD = 0.85

# ============================================================
# 画像系统 (Portrait System)
# ============================================================
PORTRAIT_FILE_PATH = os.getenv("PORTRAIT_FILE_PATH",
                                os.path.join(DATA_DIR, "PORTRAIT.md"))
PORTRAIT_SHALLOW_HOURS = 4
PORTRAIT_DEEP_HOURS = 24
PORTRAIT_DEEP_MIN_TURNS = 20     # 深巩固最低对话轮数门槛
PORTRAIT_REALTIME_DIMS = [2, 4]  # 实时更新维度（用户+AI 两侧）
BEHAVIOR_CHROMA_DIR = os.path.join(DATA_DIR, "behavior_chroma")
BEHAVIOR_COLLECTION = "behavior_patterns"

# ============================================================
# 时间线近端检索
# ============================================================
# legacy — 当前无 .py 引用，保留供可能的旧模块兼容
TIMELINE_RECENT_COUNT = 5
WORK_MEMORY_TOKEN_BUDGET = 50000

# ============================================================
# ChatHistory
# ============================================================
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.jsonl")
CHAT_HISTORY_MAX_MEMORY = 500

# ============================================================
# 文件存储路径
# ============================================================
CO_OCCURRENCE_FILE = os.path.join(DATA_DIR, "co_occurrence.json")
STORE_FAILURES_PATH = os.path.join(DATA_DIR, "store_failures.jsonl")

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
# 部署模式 & 轻量版开关
# ============================================================
DEPLOY_MODE = os.getenv("DEPLOY_MODE", "full")
IS_LITE = DEPLOY_MODE == "lite"

# 轻量版功能开关
LITE_DISABLE_BACKGROUND_TASKS = True        # 禁用后台巩固 + 空闲回顾
LITE_DISABLE_IMPULSE = True                 # 禁用冲动调度器 + 独立开口
LITE_WORK_MEMORY_BUDGET = 5000 if IS_LITE else 50000

# ============================================================
# Benchmark 模式 — 禁用认知过滤，最大化事实召回
# ============================================================
BENCHMARK_MODE = os.getenv("BENCHMARK_MODE", "false").lower() == "true"

if BENCHMARK_MODE:
    DEPLOY_MODE = "lite"
    IS_LITE = True
    CONSOLIDATION_SHALLOW_INTERVAL = 999_999_999   # 实质上禁用
    CONSOLIDATION_DEEP_INTERVAL = 999_999_999
    LITE_DISABLE_BACKGROUND_TASKS = True
    LITE_DISABLE_IMPULSE = True
    LITE_WORK_MEMORY_BUDGET = 100_000              # benchmark 不限制工作记忆

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
