"""
配置管理模块
从 .env 文件加载所有配置项
"""

from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

class Config:
    """应用配置类"""
    
    # ==================== Qwen LLM 配置 ====================
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    QWEN_MODEL = os.getenv("QWEN_MODEL_NAME", "qwen-turbo")
    
    # 温度参数
    TEMP_CONVERSATION = float(os.getenv("QWEN_TEMPERATURE_CONVERSATION", "0.0"))
    TEMP_SCORING_POS = float(os.getenv("QWEN_TEMPERATURE_SCORING_POS", "0.2"))
    TEMP_SCORING_NEG = float(os.getenv("QWEN_TEMPERATURE_SCORING_NEG", "0.0"))
    TEMP_SEMANTIC = float(os.getenv("QWEN_TEMPERATURE_SEMANTIC", "0.4"))
    TEMP_ROUTER = float(os.getenv("QWEN_TEMPERATURE_ROUTER", "0.0"))
    
    # ==================== 向量嵌入配置 ====================
    QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v2")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "db5")
    
    # ==================== Qwen TTS 配置 ====================
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "qwen")
    QWEN_TTS_MODEL = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-flash")
    QWEN_TTS_VOICE = os.getenv("QWEN_TTS_VOICE", "Marcus")
    QWEN_TTS_LANGUAGE = os.getenv("QWEN_TTS_LANGUAGE", "Chinese")
    QWEN_TTS_STREAM = os.getenv("QWEN_TTS_STREAM", "true").lower() == "true"
    USE_GTTS_FALLBACK = os.getenv("USE_GTTS_FALLBACK", "true").lower() == "true"
    
    # ==================== RAG 检索配置 ====================
    RAG_MMR_K = int(os.getenv("RAG_MMR_K", "4"))
    RAG_MMR_FETCH_K = int(os.getenv("RAG_MMR_FETCH_K", "20"))
    RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))
    ENABLE_HISTORY_DEDUP = os.getenv("ENABLE_HISTORY_DEDUP", "true").lower() == "true"
    MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "10"))
    
    # 混合检索（可选）
    ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
    HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
    HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))
    
    # 重排序（可选）
    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
    RERANKING_MODEL = os.getenv("RERANKING_MODEL", "cohere")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
    COHERE_RERANK_TOP_N = int(os.getenv("COHERE_RERANK_TOP_N", "3"))
    
    # ==================== 数据库配置 ====================
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_TABLE_NAME = os.getenv("SUPABASE_TABLE_NAME", "interactions")
    
    # ==================== 智能体配置 ====================
    USE_WEB_SEARCH = os.getenv("USE_WEB_SEARCH", "true").lower() == "true"
    WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
    ENABLE_SMART_ROUTING = os.getenv("ENABLE_SMART_ROUTING", "true").lower() == "true"
    ROUTING_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTING_CONFIDENCE_THRESHOLD", "0.7"))
    
    # Tavily（可选）
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
    
    # ==================== 应用配置 ====================
    APP_NAME = os.getenv("APP_NAME", "Trocaz_Pigeon")
    APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
    APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"
    
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))
    MAX_INTIMACY_SCORE = int(os.getenv("MAX_INTIMACY_SCORE", "6"))
    
    # ==================== 功能开关 ====================
    FEATURE_QWEN_TTS = os.getenv("FEATURE_QWEN_TTS", "true").lower() == "true"
    FEATURE_SMART_AGENT = os.getenv("FEATURE_SMART_AGENT", "true").lower() == "true"
    FEATURE_HYBRID_RAG = os.getenv("FEATURE_HYBRID_RAG", "false").lower() == "true"
    FEATURE_GIFT_SYSTEM = os.getenv("FEATURE_GIFT_SYSTEM", "true").lower() == "true"
    FEATURE_VOICE_SELECTION = os.getenv("FEATURE_VOICE_SELECTION", "true").lower() == "true"
    
    # ==================== 备用配置 ====================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ENABLE_OPENAI_FALLBACK = os.getenv("ENABLE_OPENAI_FALLBACK", "false").lower() == "true"
    FALLBACK_VECTOR_DB_PATH = os.getenv("FALLBACK_VECTOR_DB_PATH", "db7")
    
    # ==================== 日志配置 ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    ENABLE_API_LOGGING = os.getenv("ENABLE_API_LOGGING", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """验证必需配置"""
        required_configs = {
            'DASHSCOPE_API_KEY': cls.DASHSCOPE_API_KEY,
            'SUPABASE_URL': cls.SUPABASE_URL,
            'SUPABASE_KEY': cls.SUPABASE_KEY,
        }
        
        missing = [key for key, value in required_configs.items() if not value]
        
        if missing:
            raise ValueError(f"缺少必需配置: {', '.join(missing)}")
        
        return True
    
    @classmethod
    def print_config(cls):
        """打印当前配置（调试用）"""
        print("=" * 50)
        print("当前配置")
        print("=" * 50)
        print(f"Qwen Model: {cls.QWEN_MODEL}")
        print(f"TTS Provider: {cls.TTS_PROVIDER}")
        print(f"TTS Voice: {cls.QWEN_TTS_VOICE}")
        print(f"Vector DB: {cls.VECTOR_DB_PATH}")
        print(f"Smart Agent: {cls.FEATURE_SMART_AGENT}")
        print(f"Voice Selection: {cls.FEATURE_VOICE_SELECTION}")
        print("=" * 50)

# 创建全局配置实例
config = Config()

# 自动验证（可选，注释掉以跳过）
# config.validate()

