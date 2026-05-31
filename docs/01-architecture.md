# 劳动法机器人 Multi-Agent Agentic RAG 架构

## 1. 目标
系统目标不是“能回答”，而是“可审计、可回放、可扩展、可校验”。

核心要求：
- 先检索法律索引树，再读取原文
- 所有法律结论都要经过引用校验
- 复杂问题可拆分、可重试、可回退
- 多 Agent 分工明确，状态统一管理

## 2. 总体分层
### 2.1 Orchestrator 层
负责一次请求的全局编排，不直接生成法律结论。

职责：
- 创建请求上下文
- 管理 `CaseState`
- 决定走哪条 agent 路由
- 控制重试、回退、追问、转人工
- 汇总最终结果

### 2.2 Agent 层
每个 Agent 只负责一个明确任务。

当前代码已经实现 10 个 Agent：

- `SubjectAgent`：LLM 驱动，判断法律主体关系，例如谁怀孕、谁被辞退、谁请假
- `IssueAgent`：LLM 驱动，识别法律关系和争点
- `ClarificationAgent`：判断关键事实缺失时是否应追问
- `RewriteAgent`：LLM 驱动，把口语问题改写成法律检索 query
- `DecompositionAgent`：LLM 驱动，复杂问题拆分为独立子问题并分别检索
- `RetrievalAgent`：执行 LLM 生成的检索计划，检索索引树并回读原文
- `CitationAgent`：LLM 驱动，抽取 claim 并做 claim-evidence 对齐校验
- `ConflictAgent`：识别 evidence 是否来自多部法律
- `SynthesisAgent`：LLM 驱动，基于 evidence 生成最终答案
- `ReflexionAgent`：LLM 驱动，答案质量自检与补搜

### 2.3 Service 层
提供确定性业务能力。

- `IndexBuilderService`：构建索引树，集成 LLM 摘要增强
- `SummaryGenerator`：LLM 驱动的法级/章级语义摘要生成，文件哈希缓存
- `IndexSearchService`：三级树检索 + 扁平检索双路径
- `LawReaderService`：锚点定位 + 原文回读
- `CitationVerifierService`
- `TraceStore`

### 2.4 Tool 层
Agent 通过工具访问服务，工具必须是稳定 JSON contract。

## 3. 运行主链路
当前已实现主链路：

1. `LegalOrchestrator.answer`
2. `SubjectAgent.extract_subjects`
3. `IssueAgent.identify_issues`
4. `ClarificationAgent.should_clarify`
5. `RewriteAgent.rewrite_queries`
6. `DecompositionAgent.decompose`（<=1 子问题跳过，>=2 独立检索合并）
7. `RetrievalAgent.retrieve`
8. `search_index_tree`（三阶段：Phase 0 直接读条 → Phase 1 树/扁平检索 → 合并去重）
   - **Phase 0**：`_direct_read_articles(article_hints)` — 精确读取指定法条，score=1.0
   - **Phase 1**：`TreeRetriever.search()`（启用时）或 `_flat_search()`（BM25+Vector+RRF+CrossEncoder）
   - **合并**：直接读结果优先，按 `(law_id, node_id)` 去重
9. `read_law_node`
10. `CitationAgent.verify`，包括 claim 抽取和 evidence 支持性判断
11. `ConflictAgent.check`
12. `SynthesisAgent.compose_answer`
13. `ReflexionAgent.evaluate`（最多 2 轮，gap 时补搜重验重生成）
14. 返回 `AnswerResult`

## 4. 状态对象
建议全局使用强类型状态对象。

### 4.1 CaseState
```python
CaseState(
    request_id: str,
    original_question: str,
    normalized_question: str,
    plan: RetrievalPlan,
    evidence: list[Evidence],
    citation_checks: list[CitationCheck],
    conflicts: list[Conflict],
    trace: list[TraceEvent],
    final_answer: str,
    reflexion_iteration: int,
    reflexion_trace: list[dict],
)
```

当前 `RetrievalPlan`：

```python
RetrievalPlan(
    intent: str,
    legal_issues: list[str],
    facts: dict,
    queries: list[str],
    law_hints: list[str],
    article_hints: list[str],
    need_clarification: bool,
    clarification: str,
    sub_questions: list[str],
)
```

### 4.2 Evidence
```python
Evidence(
    law_id: str,
    law_title: str,
    node_id: str,
    article: str,
    text: str,
    score: float,
    source_file: str,
    source_anchor: str,
    verified: bool,
)
```

### 4.3 CitationCheck
```python
CitationCheck(
    claim: str,
    law_id: str,
    node_id: str,
    status: str,
    reason: str,
)
```

## 5. 索引构建管线

### 5.1 构建流程
```
source_path (.txt)
  → parse_law_text() → LawDocument 树（规则解析）
  → enhance_summaries(doc, llm) → LLM 增强法级/章级摘要
  → render_index_markdown(doc) → .indexs.md
  → anchor_map → .anchors.json
```

### 5.2 LLM 摘要增强
`summary_generator.py` 负责：
- 为法级节点生成 ~150 字语义范围摘要
- 为章级节点生成 ~100 字语义范围摘要
- 文件哈希缓存：源文件不变则跳过 LLM 调用
- 缓存目录：`data/summaries/`，JSON 格式

### 5.3 三级树索引
`tree_retrieval.py` 的 `TreeRetriever`：
- 法级 → 章级 → 条级，每级独立 BM25 + Vector 索引
- 父子映射：`chapter_to_law`、`article_to_chapter`
- 向量缓存：`data/indexs/.tree_vector_cache.npz`
- 当前状态：`TREE_ENABLED = False`（text2vec-base-chinese CMTEB 检索 38.79，层级区分度不足）
- 推荐升级：`gte-large-zh`（CMTEB 检索 72.49，0.65GB）> `gte-base-zh`（71.71，0.20GB）> `bge-large-zh-v1.5`（70.46，1.3GB）

## 6. 决策规则
- 无命中，不输出确定法律结论
- 引用未通过，不进入最终生成
- 多意图问题必须拆分
- 宽泛问题先追问，追问失败再检索
- 同一结论必须能回链到具体法条
- article_hints 指定法条优先直接读取（Phase 0），不依赖检索命中率
- 树检索不可用时自动降级为扁平检索
