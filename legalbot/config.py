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
RRF_K = 15

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_PATH = BASE_DIR / "models" / "bge-reranker-v2-m3"
RERANKER_ENABLED = True
RERANKER_MIN_SCORE = 0.1                   # 精排后过滤阈值，低于此分数的证据不传给LLM

VECTOR_SCORE_THRESHOLD = 0.3

INDEX_MAX_WORKERS = 8  # 索引构建并行线程数（LLM 调用是 I/O 瓶颈）

# ── 一致性改进 flags ──
QUERY_GATE_ENABLED = True                 # A1: 原始query保底检索，LLM改写仅做扩展
CROSS_ENCODER_CITATION = True             # C1: cross-encoder替代LLM引用校验
CROSS_ENCODER_CITATION_THRESHOLD = 0.3    # C1: supported判定阈值
# A2: 关键词→法规硬路由 + 树检索Level0前置过滤 + hint_scores修复（代码层，无独立flag）
CONFIDENCE_CUTOFF_ENABLED = False         # B3: 多信号一致性截断（测试显示降低精度）
CONFIDENCE_MIN_SIGNALS = 2                # B3: 至少几路信号一致才纳入
LEXICAL_REFLEXION_ENABLED = False         # E1: 词法支撑自检+答案格式后处理（测试显示引入错误正反馈）

LEVEL0_CE_ENABLED = False                 # Level 0 CE 粗筛（信号质量不够，暂关闭）

