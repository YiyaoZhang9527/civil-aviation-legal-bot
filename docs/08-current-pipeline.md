# 当前实现管线说明

本文档记录当前代码实际运行方式，避免与目标设计混淆。

## 1. 当前已实现 Agent
当前实现 10 个 Agent：

- `SubjectAgent`
- `IssueAgent`
- `ClarificationAgent`
- `RewriteAgent`
- `DecompositionAgent`
- `RetrievalAgent`
- `CitationAgent`
- `ConflictAgent`
- `SynthesisAgent`
- `ReflexionAgent`

`LegalOrchestrator` 是编排器，不算 Agent。

## 2. 当前请求链路
```text
用户问题
 -> LegalOrchestrator.answer()
 -> SubjectAgent.extract_subjects()
 -> IssueAgent.identify_issues()
 -> ClarificationAgent.should_clarify()
 -> RewriteAgent.rewrite_queries()
 -> DecompositionAgent.decompose()
     ├─ needs_decomposition=True: 每个 SubProblem 独立 retrieve，合并去重
     └─ needs_decomposition=False: 走原逻辑
 -> RetrievalAgent.retrieve()
     -> search_index_tree() [三阶段]
         ├─ Phase 0: _direct_read_articles(article_hints)
         │   精确读取 RewriteAgent 指定的法条，score=1.0
         ├─ Phase 1: TreeRetriever.search() 或 _flat_search()
         │   ├─ 树路径: 法→章→条 逐层 BM25+Vector+RRF（需 TREE_ENABLED=True）
         │   └─ 扁平路径: BM25+Vector+RRF+CrossEncoder 精排
         └─ 合并: 直接读结果优先，(law_id, node_id) 去重
 -> CitationAgent.verify()
 -> ConflictAgent.check()
 -> SynthesisAgent.compose_answer()
 -> ReflexionAgent.evaluate()  [最多 2 轮循环]
     ├─ quality=pass: break
     └─ quality=gap: 补搜 → 重验 → 重生成
 -> AnswerResult
```

## 3. 前置 Agent 当前职责
`SubjectAgent.extract_subjects()` 是 LLM 驱动，负责：

- 事实抽取
- 主体判断
- 关系判断
- 不确定事实识别

`IssueAgent.identify_issues()` 是 LLM 驱动，负责：

- 法律问题识别
- 法律争点识别
- 法律领域判断
- 缺失事实整理

`ClarificationAgent.should_clarify()` 负责：

- 判断主体或关键事实缺失时是否先追问

`RewriteAgent.rewrite_queries()` 是 LLM 驱动，负责：

- 问题改写
- 检索 query 生成
- `law_hints` 生成
- `article_hints` 生成
- `sub_questions` 生成

最终组合成：

```python
RetrievalPlan(
    intent,
    legal_issues,
    facts,
    queries,
    law_hints,
    article_hints,
    need_clarification,
    clarification,
    sub_questions,
)
```

## 4. DecompositionAgent 当前行为
`DecompositionAgent.decompose()` 负责：

- 接收 RewriteAgent 输出的 `sub_questions`
- 若 <= 1 个子问题，直接返回 `needs_decomposition=False`（不调 LLM）
- 若 >= 2 个子问题，LLM 为每个子问题生成独立检索策略（queries + law_hints + article_hints）
- 上限 `MAX_SUBQUESTIONS = 4`

分解后由 `_retrieve_decomposed()` 为每个 SubProblem 独立执行检索，结果按 `(law_id, node_id)` 去重合并。

## 5. ReflexionAgent 当前行为
`ReflexionAgent.evaluate()` 负责：

- 快速路径：所有 citation 都是 supported 且 confidence >= 0.7 → 直接返回 pass（不调 LLM）
- 否则：LLM 评估答案质量，输出 gaps + 补搜 queries/hints
- 最多循环 `MAX_REFLEXION_ITERATIONS = 2` 轮
- 每轮补搜结果与已有 evidence 合并去重，上限 12 条

## 6. 当前已删除的旧逻辑
以下问题库式逻辑已经删除：

- 固定 `TOPIC_HINTS`
- 固定 topic 路由
- 固定法律结论模板
- 固定 query rewrite
- 针对个别问题的 if 规则

## 7. CitationAgent 当前行为
`CitationAgent` 已升级为 LLM 驱动校验：

1. `extract_claims`
   - 从用户问题和法律争点中抽取待验证法律主张
2. `verify`
   - 将 claim 与读取到的 evidence 对齐
   - 输出 `supported / partial / unsupported`
   - 输出 `quote`、`reason`、`confidence`

`SynthesisAgent` 会接收引用校验结果，并优先使用 supported/partial 的 evidence。

## 8. 当前不足
还需要进一步把 unsupported evidence 从最终上下文中完全剔除，并把 citation check 作为可审计日志持久化。

## 9. 索引构建管线

### 9.1 构建命令
```bash
python run.py --build-index
```

### 9.2 构建流程
```
data/法律数据/*.txt
  → parse_law_text()         # 规则解析 → LawDocument 树
  → enhance_summaries(doc)   # LLM 增强 law/chapter 摘要（文件哈希缓存）
  → render_index_markdown()  # → data/indexs/*.indexs.md
  → anchor_map               # → data/indexs/*.anchors.json
```

### 9.3 LLM 摘要增强
- `summary_generator.py`：LLM 为法级/章级生成语义范围摘要
- 缓存目录：`data/summaries/`，按源文件 MD5 哈希缓存
- LLM 不可用时退回规则摘要（`parser._summary_from_text()`）

### 9.4 树检索状态
- `TreeRetriever` 已实现但 **默认禁用**（`TREE_ENABLED=False`）
- 原因：`text2vec-base-chinese` CMTEB 检索得分仅 38.79，聚类得分 37.66，层级区分度不足
- 升级方案（按优先级排序）：
  1. **`gte-large-zh`**（推荐）：CMTEB 检索 72.49 / 聚类 53.07，模型仅 0.65GB，综合最优
  2. `gte-base-zh`：检索 71.71 / 聚类 53.86，仅 0.20GB，资源受限场景首选
  3. `bge-large-zh-v1.5`：检索 70.46 / 聚类 48.99，1.3GB，生态成熟但性价比低于 GTE
- 升级步骤：换模型 → 重建向量缓存 → 改 `TREE_ENABLED=True`
- 禁用时自动降级为扁平 BM25+Vector+CrossEncoder 管线

## 10. 检索配置
```python
# config.py 关键参数
DEFAULT_TOP_K = 5
TREE_TOP_LAWS = 30          # 树检索法级候选数
TREE_TOP_CHAPTERS = 15      # 树检索章级候选数
TREE_ENABLED = False         # 树检索开关
HYBRID_WEIGHT_BM25 = 0.35   # 扁平检索 BM25 权重
HYBRID_WEIGHT_VECTOR = 0.35 # 扁平检索向量权重
HYBRID_WEIGHT_HINT = 0.30   # 扁平检索 hint 权重
RERANKER_ENABLED = True      # Cross-Encoder 精排开关
```
