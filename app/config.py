import os
from dotenv import load_dotenv
load_dotenv()

# HuggingFace 国内镜像：Reranker 走 sentence-transformers 下载权重，
# 必须在导入 huggingface_hub 之前把镜像地址写进环境变量才生效。
_hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").strip()
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

def _bool_env(key: str, default: str) -> bool:
    return os.getenv(key, default).strip().lower() in ("1", "true", "yes", "on")

class Settings:
    """全局配置管理"""
    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2:7b-instruct")

    # Embedding
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "bge-m3")
    EMBED_MODEL_PATH: str = os.getenv("EMBED_MODEL_PATH", "BAAI/bge‑m3‑zh")
    EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu")

    # Reranker
    USE_RERANKER: bool = _bool_env("USE_RERANKER", "true")
    RERANK_MODEL_PATH: str = os.getenv("RERANK_MODEL_PATH", "BAAI/bge‑reranker‑base‑zh")
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "3"))
    # CrossEncoder 打分阈值：低于该分数视为不相关，直接不交给LLM（bge-reranker 的 logit < 0 表示不相关）
    RERANK_SCORE_THRESHOLD: float = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.0"))
    RETRIEVE_INIT_K: int = int(os.getenv("RETRIEVE_INIT_K", "8"))

    # 混合检索 (BM25 + 向量) 权重，两者之和建议为1
    USE_HYBRID_SEARCH: bool = _bool_env("USE_HYBRID_SEARCH", "true")
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))
    # RRF 融合常数，越大越弱化排名差异
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # Qdrant
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_data")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_demo")

    # 文件上传
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    # LLM参数
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    # 中文场景下 1 个汉字≈1个token，上下文窗口必须留足，否则Ollama会截断prompt
    N_CTX: int = int(os.getenv("N_CTX", "8192"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # 检索参数
    # RETRIEVER_K: 最终交给LLM的分片数量
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
    # 单次检索拼给LLM的上下文字符上限，防止prompt超出N_CTX被截断
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "3000"))

    # 检索不到任何相关内容时的固定回复
    NO_ANSWER_TEXT: str = os.getenv("NO_ANSWER_TEXT", "文档未找到相关信息")

    # 多轮对话记忆
    SESSION_HISTORY_MAX_TURNS: int = int(os.getenv("SESSION_HISTORY_MAX_TURNS", "5"))

settings = Settings()
