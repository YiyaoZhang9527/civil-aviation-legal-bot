# 劳动法机器人 Claude Code 实施计划

## 1. 目录骨架
建议实现：

```text
app/
  orchestrator/
  agents/
  services/
  tools/
  guardrails/
  trace/
  eval/
data/
  法律数据/
  indexs/
docs/
```

## 2. 实施顺序
1. 构建索引树生成器
2. 构建索引树检索服务
3. 构建原文条文读取服务
4. 构建引用校验服务
5. 构建 Orchestrator
6. 构建各 Agent
7. 接入 trace
8. 接入评测集

## 3. 函数级任务拆分
### 3.1 IndexBuilderService
- `parse_law_text`
- `split_sections`
- `build_tree`
- `emit_indexs_md`
- `emit_anchor_map`

### 3.2 IndexSearchService
- `load_all_indexes`
- `search_tree`
- `score_hits`
- `rerank_hits`

### 3.3 LawReaderService
- `resolve_node_path`
- `read_source_text`
- `trim_to_article`
- `attach_context`

### 3.4 CitationVerifierService
- `extract_claims`
- `align_claim_to_article`
- `verify_support`
- `flag_conflict`

### 3.5 Orchestrator
- `create_request`
- `dispatch_agents`
- `merge_outputs`
- `retry_or_escalate`
- `finalize_response`

## 4. 验收标准
- 能从原文生成 `indexs.md`
- 能通过索引树定位到条文
- 能校验法律依据
- 能对多意图问题拆分后合成
- 能记录完整 trace

