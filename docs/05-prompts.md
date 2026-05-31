# 劳动法机器人 Prompts 规范

## 1. 总原则
- 每个 Agent 独立 prompt
- 每个 prompt 输出格式固定
- 所有结构化输出优先 JSON
- 不暴露内部思维链给最终用户

## 2. QueryAgent Prompt
职责：
- 归一化问题
- 识别意图
- 判断是否宽泛、多意图、需要追问

## 3. DecompositionAgent Prompt（已实现）
职责：
- 接收 RewriteAgent 输出的 sub_questions
- 若 <= 1 个子问题，直接返回（不调 LLM）
- 若 >= 2 个子问题，LLM 为每个生成独立检索策略
- 输出每个子问题的 queries、law_hints、article_hints

## 4. RetrievalAgent Prompt
职责：
- 先索引树，后原文
- 给出候选节点和检索理由

## 5. CitationAgent Prompt
职责：
- 从答案草稿中提取可验证 claim
- 对齐具体法条
- 输出支持/不支持/部分支持

## 6. ConflictAgent Prompt
职责：
- 判断特别法优先、上位法优先、地方规则优先
- 输出冲突路径

## 7. SynthesisAgent Prompt
职责：
- 仅基于 verified evidence 写最终答案
- 必须包含：
  - 结论
  - 法律依据
  - 适用条件
  - 风险提示

## 8. ReflexionAgent Prompt（已实现）
职责：
- 评估答案是否完整覆盖所有法律争点
- 快速路径：所有 citation supported 且 confidence >= 0.7 → pass
- 否则 LLM 评估：输出 gaps + 补搜 queries/hints
- 最多 2 轮循环

## 9. Summary Generator Prompts（索引构建阶段）

### 9.1 法级摘要 Prompt
```
你是法律索引专家。请为以下法律生成一段约 150 字的范围摘要。
要求：
1. 描述该法的适用主体和场景
2. 列出核心规范领域
3. 不要复述条文内容，只做范围概括
4. 输出纯文本，不要列表或标题

法律名称：{law_title}
前 10 条原文：{sample_text}
```

### 9.2 章级摘要 Prompt
```
你是法律索引专家。请为以下章节生成一段约 100 字的范围摘要。
要求：
1. 描述该章节覆盖的法律领域
2. 列出核心规范点
3. 不要复述条文内容，只做范围概括
4. 输出纯文本，不要列表或标题

法律：{law_title}
章节：{chapter_title}
章内条文：{chapter_text}
```
