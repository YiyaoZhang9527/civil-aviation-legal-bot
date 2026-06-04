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
LLM_TIMEOUT = 60                           # LLM API 超时秒数
HISTORY_TURNS = 6                          # 多轮对话历史保留轮数

INDEX_SUFFIX = ".indexs.md"
ANCHOR_SUFFIX = ".anchors.json"

VECTOR_MODEL_NAME = "thenlper/gte-large-zh"
VECTOR_MODEL_PATH = BASE_DIR / "models" / "gte-large-zh"
VECTOR_CACHE_FILE = INDEX_DIR / ".vector_cache.npz"

RETRIEVAL_MODE = "tree"  # "tree" | "flat" | "keyword_hints"

TREE_TOP_LAWS = 30
TREE_TOP_CHAPTERS = 30
TREE_VECTOR_CACHE_FILE = INDEX_DIR / ".tree_vector_cache.npz"
TREE_ENABLED = True  # 三级树检索：法→章→条逐层剪枝
TREE_ARTICLE_CANDIDATES = 30              # 条级候选扩展上限
TREE_CHAPTER_EXPANSION = 30              # 章级扩展上限
TREE_HINT_BOOST = 0.3                    # hint 匹配加成分数
TREE_ORPHAN_TOP_LAWS = 5                 # 孤儿文章纳入的 top 法规数
TREE_CHAPTER_PER_LAW = 10                 # Level 1 每部法规最多章节数（排第1的法规不限）

HYBRID_WEIGHT_BM25 = 0.35
HYBRID_WEIGHT_VECTOR = 0.35
HYBRID_WEIGHT_HINT = 0.30

HYBRID_FUSION_MODE = "rrf"
RRF_K = 15

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_PATH = BASE_DIR / "models" / "bge-reranker-v2-m3"
RERANKER_ENABLED = True
RERANKER_MIN_SCORE = 0.05                   # 精排后过滤阈值（0.1太严，过滤掉了正确结果）
RERANKER_MAX_CHARS = 512                   # 精排输入截断字符数
BM25_RECALL_K = 50                         # BM25 召回量

VECTOR_SCORE_THRESHOLD = 0.3

INDEX_MAX_WORKERS = 8  # 索引构建并行线程数（LLM 调用是 I/O 瓶颈）

# ── 一致性改进 flags ──
QUERY_GATE_ENABLED = True                 # A1: 原始query保底检索，LLM改写仅做扩展
CROSS_ENCODER_CITATION = True             # C1: cross-encoder替代LLM引用校验
CROSS_ENCODER_CITATION_THRESHOLD = 0.25   # C1: supported判定阈值（消融实验0.20-0.25略优）
CROSS_ENCODER_PARTIAL_THRESHOLD = 0.15   # supported/partial 分界线
CROSS_ENCODER_BATCH = 60                  # CE 推理批大小
CROSS_ENCODER_MAX_CHARS = 512             # CE 输入截断字符数
REFLEXION_CONFIDENCE_THRESHOLD = 0.7      # 自检快速通过的置信度门槛
# A2: 关键词→法规硬路由 + 树检索Level0前置过滤 + hint_scores修复（代码层，无独立flag）
CONFIDENCE_CUTOFF_ENABLED = False         # B3: 多信号一致性截断（测试显示降低精度）
CONFIDENCE_MIN_SIGNALS = 2                # B3: 至少几路信号一致才纳入
LEXICAL_REFLEXION_ENABLED = False         # E1: 词法支撑自检+答案格式后处理（测试显示引入错误正反馈）

SYNTHESIS_JSON_MODE = True                # SynthesisAgent 结构化JSON输出（每条claim带node_id）
SET_MEMBERSHIP_CHECK = True               # Set-Membership 确定性校验（claim引用必须存在于证据集）
CLAIM_LEVEL_CITATION = False              # Claim→Evidence 校验方向（消融实验：新方向太激进，旧方向evidence→claim更均衡）

LEVEL0_CE_ENABLED = False                 # Level 0 CE 粗筛（信号质量不够，暂关闭）
TREE_LAW_PRIOR_WEIGHT = 0.2               # Level 0→1 法规分数先验权重（解决 Level 0→1 信号断裂）
TREE_ADAPTIVE_K_ENABLED = False           # Level 1 自适应截断（消融实验显示负收益，关闭）
TREE_GENERIC_ARTICLE_PENALTY = 0.5        # 通用条文降权系数（消融v3: 0.5最优，1.0=关闭）
TREE_EARLY_ARTICLE_PENALTY = 0.6            # 位置降权系数：前 N 条（总则候选）额外降权
TREE_EARLY_ARTICLE_THRESHOLD = 5             # 视为"总则候选"的最大条号
TREE_EARLY_SHORT_TEXT_LIMIT = 200            # 长度门控：仅当条文文本 ≤ N 字时才视为"短总则"降权（长条款即使在前 5 条也不动）
KEYWORD_ROUTING_ENABLED = True            # 确定性关键词路由（law_routes.json + fallback 规则，消融v3: +18pp）

WRRF_ENABLED = False                      # WRRF 多query加权合并（消融实验显示负收益，关闭）
WRRF_QUERY_DECAY = 0.3                    # 改写query权重衰减系数
MAX_EVIDENCE_PER_LAW = 3                  # 每部法规最多证据数（防垄断）
WRRF_PRIMARY_TOP = 4                      # 原始query保底 top 数

RETRIEVAL_TOP_K = 10                      # 每个 query 检索条数
EVIDENCE_LOAD_LIMIT = 12                  # 回读法条原文的最大证据数
SYNTHESIS_EVIDENCE_LIMIT = 12             # 传给 LLM 的最大证据数
SYNTHESIS_EVIDENCE_TRUNCATE = 2000        # 传给 LLM 的单条证据截断字符数
CITATION_LLM_TRUNCATE = 1500              # LLM 校验输入截断字符数
CROSSREF_MAX_ITEMS = 5                    # 交叉引用最大补全数

EVAL_EVIDENCE_MAX_CHARS = 2000             # 评测：加载证据文本截断字符数
EVAL_JUDGE_MAX_CHARS = 500                # 评测：传给评测LLM的单条证据截断字符数

RELEVANCE_GATE_ENABLED = True              # 证据相关性门控：生成前独立评估证据是否回答了问题
SYNTHESIS_REFUSAL_FALLBACK = True          # 拒答兜底：LLM 答非所问时列出检索到的法规，避免对用户说"未找到"
SYNTHESIS_FALLBACK_MAX_ITEMS = 5           # 兜底最多列几条证据
NUMBER_GUARD_ENABLED = True                # D 类防御：答案中 evidence 未支撑的具体数字自动加 [待核实] 后缀

# ── 答案渲染（用户可读） ──
ANSWER_EXCERPT_MAX_CHARS = 200             # 每条 claim 下的原文摘录最大字符数
ANSWER_FALLBACK_EXCERPT_MAX_CHARS = 80     # 兜底路径的摘录最大字符数

# ── Evidence 排序（按 CE 状态） ──
EVIDENCE_SORT_BY_CE = True                 # 把 supported evidence 排前（避免 LLM 被总则类带偏）
EVIDENCE_SORT_MIN_SUPPORTED_CONF = 0.5     # 低于此置信度的 supported 视为 partial（防 CE 阈值过松污染）

# ── Parser ──
PARSER_STRIP_MD_TABLE = True               # 剥离 markdown 表格符号（|  | ），否则 article 正则匹配失败

