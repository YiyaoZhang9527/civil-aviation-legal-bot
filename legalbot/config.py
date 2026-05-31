"""项目路径与基础配置。"""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LAW_DATA_DIR = DATA_DIR / "法律数据"
INDEX_DIR = DATA_DIR / "indexs"
TRACE_DIR = DATA_DIR / "trace"
SUMMARY_CACHE_DIR = DATA_DIR / "summaries"

DEFAULT_TOP_K = 8
MAX_RETRIES = 3
MAX_SUBQUESTIONS = 4
MAX_REFLEXION_ITERATIONS = 2
MAX_CLARIFICATION_ATTEMPTS = 3

INDEX_SUFFIX = ".indexs.md"
ANCHOR_SUFFIX = ".anchors.json"

VECTOR_MODEL_NAME = "thenlper/gte-large-zh"
VECTOR_MODEL_PATH = BASE_DIR / "models" / "gte-large-zh"
VECTOR_CACHE_FILE = INDEX_DIR / ".vector_cache.npz"

RETRIEVAL_MODE = "tree"  # "tree" | "flat" | "keyword_hints"

TREE_TOP_LAWS = 30
TREE_TOP_CHAPTERS = 15
TREE_VECTOR_CACHE_FILE = INDEX_DIR / ".tree_vector_cache.npz"
TREE_ENABLED = True  # 三级树检索：法→章→条逐层剪枝

HYBRID_WEIGHT_BM25 = 0.35
HYBRID_WEIGHT_VECTOR = 0.35
HYBRID_WEIGHT_HINT = 0.30

HYBRID_FUSION_MODE = "rrf"
RRF_K = 60

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_PATH = BASE_DIR / "models" / "bge-reranker-v2-m3"
RERANKER_ENABLED = True

VECTOR_SCORE_THRESHOLD = 0.3

INDEX_MAX_WORKERS = 8  # 索引构建并行线程数（LLM 调用是 I/O 瓶颈）

