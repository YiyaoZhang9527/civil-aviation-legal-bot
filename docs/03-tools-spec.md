# 劳动法机器人 Tools 规范

## 1. 设计原则
- 工具必须是确定性动作
- 工具输入输出统一为 JSON
- 工具不直接输出最终法律结论
- 工具返回必须可审计

## 2. 核心工具
### 2.1 search_law_index_tree
```json
{
  "query": "试用期最长多久",
  "top_k": 5,
  "filters": {
    "law_ids": [],
    "node_types": ["law", "chapter", "article"]
  }
}
```

返回：
```json
{
  "hits": [
    {
      "law_id": "labor_contract_law",
      "law_title": "中华人民共和国劳动合同法",
      "node_id": "article:19",
      "title": "第十九条",
      "score": 0.91,
      "summary": "规定试用期上限。"
    }
  ]
}
```

### 2.2 read_law_node
```json
{
  "law_id": "labor_contract_law",
  "node_id": "article:19",
  "include_context": true
}
```

返回：
```json
{
  "law_id": "labor_contract_law",
  "node_id": "article:19",
  "article": "第十九条",
  "text": "……",
  "source_file": "data/法律数据/中华人民共和国劳动合同法.txt",
  "source_anchor": "第一章/第二章/第十九条"
}
```

### 2.3 rerank_law_hits
```json
{
  "query": "试用期最长多久",
  "candidates": []
}
```

### 2.4 decompose_question
```json
{
  "question": "没签合同怎么办，试用期怎么算，社保要补吗"
}
```

### 2.5 rewrite_query
```json
{
  "query": "试用期最长多久",
  "reason": "初次命中不足",
  "previous_hits": []
}
```

### 2.6 extract_claims
```json
{
  "answer_draft": "..."
}
```

### 2.7 verify_citation
```json
{
  "claim": "三年以上固定期限和无固定期限劳动合同，试用期不得超过六个月。",
  "law_id": "labor_contract_law",
  "node_id": "article:19"
}
```

### 2.8 check_conflicts
```json
{
  "evidence_bundle": []
}
```

### 2.9 ask_clarification
```json
{
  "question": "请说明是劳动合同、工伤、社保还是监察类问题",
  "options": ["劳动合同", "工伤保险", "社会保险", "劳动监察"]
}
```

### 2.10 direct_read_articles
当 `article_hints` 包含明确法条引用时（如 "劳动合同法第39条"），直接按法名+条号读取原文，跳过检索。

```json
{
  "article_hints": ["劳动合同法第39条", "劳动合同法第40条"]
}
```

返回：
```json
{
  "evidence": [
    {
      "law_id": "中华人民共和国劳动合同法",
      "node_id": "article:39",
      "article": "第三十九条",
      "text": "……",
      "score": 1.0,
      "source": "direct_read"
    }
  ]
}
```

### 2.11 tree_search
三级树检索（法→章→条逐层剪枝），需要 `TREE_ENABLED=True`。

```json
{
  "query": "用人单位单方解除劳动合同",
  "top_k": 5
}
```

内部流程：
1. 法级 BM25 + Vector → top 30 法
2. 章级（候选法内）BM25 + Vector → top 15 章
3. 条级（候选章内）BM25 + Vector → RRF 融合 → top_k 条

## 3. 工具错误码
- `NOT_FOUND`
- `AMBIGUOUS`
- `INVALID_INPUT`
- `CITATION_FAILED`
- `CONFLICT_DETECTED`
- `NEED_CLARIFICATION`

