"""Configuration — 优先从环境变量读取，.env 文件可选。"""
import json
import math
import os
from dotenv import load_dotenv

# 加载 .env 文件：兼容新旧路径
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if not os.path.exists(_env_path):
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_env_path)

# ============================================================
# Embedding (Ollama GPU)
# ============================================================
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

# ============================================================
# DeepSeek API (主模型生成回答)
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-v4-flash"

# ============================================================
# 博查搜索 API
# ============================================================
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")

# ============================================================
# 多用户认证（内测用）
# ============================================================
# USERS 可通环境变量覆盖，默认三个内测账户
_USERS_DEFAULT = {
    "admin": "changeme",
}
USERS_RAW = os.getenv("USERS")
try:
    USERS: dict[str, str] = json.loads(USERS_RAW) if USERS_RAW else _USERS_DEFAULT
except (json.JSONDecodeError, TypeError):
    USERS = _USERS_DEFAULT

# ============================================================
# 数据根目录（可通过 DATA_DIR 环境变量覆盖，用于隔离测试数据）
# ============================================================
DATA_DIR = os.getenv("DATA_DIR", "./data")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_TOKEN_PATH = os.path.join(DATA_DIR, "auth_tokens.json")

# 用户 → 数据目录映射（原地继承现有实例数据）
USER_DATA_DIRS: dict[str, str] = {
    "admin": os.path.join(BASE_DIR, "data"),
}

# ============================================================
# 本地 LLM（Ollama 摘要生成）
# ============================================================
LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "true").lower() == "true"
LOCAL_LLM_OLLAMA_URL = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://localhost:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
LOCAL_LLM_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", "30"))

# Ollama 模型路径（需要早于 Ollama 启动前设置）
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
DEFAULT_TOP_K = 5           # 语义检索默认top_k
MAX_MEMORIES_IN_PROMPT = 12  # 最终注入 LLM 的记忆条数上限（上限）


# 时段映射
TIME_PERIOD_MAP = {
    (0, 5): "深夜", (6, 8): "早晨", (9, 11): "上午",
    (12, 13): "中午", (14, 17): "下午", (18, 20): "傍晚",
    (21, 23): "晚上",
}

# 话题关键词
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
CO_OCCURRENCE_MAX_PAIRS = 10000    # 超过此数量触发淘汰
CO_OCCURRENCE_CLEANUP_RATIO = 0.2  # 淘汰时移除最旧的 20%
CO_OCCURRENCE_MIN_COUNT = 2        # count >= 此值的活跃关系在淘汰中受保护
TIME_TRIGGERED_MAX = 5       # 时间触发上限
CONTEXT_WINDOW_SIZE = 10
CONTEXT_ROUNDS = 10       # 动态上下文读取轮数

# ============================================================
# 语义重排序参数
# ============================================================
RERANK_BETA = 0.5               # 后向兼容，维护旧代码
RERANK_LN_MAX = math.log(501)   # ≈ 6.217 — ln(501)，hit_count 封顶值
RERANK_SEMANTIC_WEIGHT = 0.7    # 语义权重（与 app/config/settings.py 一致）
RERANK_ATTENTION_WEIGHT = 0.0   # 调用方按需传入注意力偏移量
RERANK_HIT_WEIGHT = 0.3         # 替代 RERANK_BETA
ATTENTION_WINDOW = 3           # 取最近几条用户消息算注意力中心

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

# Embedding 模型回填标记文件（首次启动写入，避免重复扫描）
EMBED_BACKFILL_MARKER = os.path.join(DATA_DIR, ".embed_model_backfill_done")

# AI 人格系统
AI_CHROMA_DIR = os.path.join(DATA_DIR, "ai_chroma")
AI_COLLECTION = "ai_memories"
AI_DISTILL_STATE_PATH = os.path.join(DATA_DIR, "ai_distill_state.json")

# ============================================================
# 人格库 & 蒸馏
# ============================================================
PERSONALITY_COLLECTION = "personality_tags"
PERSONALITY_CHROMA_DIR = os.path.join(DATA_DIR, "personality_chroma")
DISTILL_STATE_PATH = os.path.join(DATA_DIR, "distill_state.json")
DISTILL_IDLE_HOURS = 1           # 触发蒸馏的空闲时间阈值（小时）
PERSONALITY_DEDUP_THRESHOLD = 0.85   # 蒸馏去重余弦相似度阈值

# ============================================================
BEHAVIOR_CHROMA_DIR = os.path.join(DATA_DIR, "behavior_chroma")
BEHAVIOR_COLLECTION = "behavior_patterns"

# ============================================================
# 时间线近端检索 & 触发链 & 时间线扩展
# ============================================================
TIMELINE_RECENT_COUNT = 5           # 时间线近端取最近几条（token_budget 为 None 时的兜底）
WORK_MEMORY_TOKEN_BUDGET = 50000   # 50K tokens，覆盖 ~100 轮对话

# ============================================================
# ChatHistory
# ============================================================
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.jsonl")
CHAT_HISTORY_MAX_MEMORY = 500       # 内存中缓存的最近对话条数

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
# 知识库
# ============================================================
KNOWLEDGE_COLLECTION = "knowledge"
# ============================================================
# 后台空闲处理
# ============================================================
IDLE_PREHEAT_QUERIES = 3     # 预热查询数
IDLE_LEVEL2_HOURS = 4        # 触发 Level 2 的空闲阈值（小时）
IDLE_LEVEL3_HOURS = 12       # 触发 Level 3 的空闲阈值（小时）

# ============================================================
# 引擎独立巩固节律（不依赖用户空闲）
# ============================================================
CONSOLIDATION_SHALLOW_INTERVAL = 14400   # 浅巩固间隔（秒），默认 4 小时
CONSOLIDATION_DEEP_INTERVAL = 86400      # 深巩固间隔（秒），默认 24 小时
ARCHIVAL_THRESHOLD_DAYS = 90             # 记忆归档阈值（天）

# ============================================================
# 自主触发冲动系统
# ============================================================
IMPULSE_MAX_PER_HOUR = 4           # 每小时主动发言上限
IMPULSE_MIN_INTERVAL = 600         # 两次主动发言最小间隔（秒）
IMPULSE_IDLE_MINUTES = 2           # 用户无消息多久后才允许触发（分钟）
IMPULSE_HEARTBEAT_IDLE = 15        # 无打字心跳多久后视为空闲（秒）
IMPULSE_TTL = 7200                 # 每个冲动的存活时间（秒），默认2小时

# Path B 开关：关闭后冲动仅通过聊天注入（Path A），不独立开口
IMPULSE_ACTIVE_PATH_B = os.getenv("IMPULSE_ACTIVE_PATH_B", "true").lower() == "true"

# ============================================================
# 部署模式 & 轻量版开关
# ============================================================
DEPLOY_MODE = os.getenv("DEPLOY_MODE", "full")  # "full" | "lite"
IS_LITE = DEPLOY_MODE == "lite"

# 轻量版功能开关（仅 IS_LITE 时生效）
LITE_DISABLE_BACKGROUND_TASKS = True        # 禁用后台巩固 + 空闲回顾
LITE_DISABLE_IMPULSE = True    # 禁用冲动调度器 + 独立开口
LITE_WORK_MEMORY_BUDGET = 5000 if IS_LITE else 50000

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
