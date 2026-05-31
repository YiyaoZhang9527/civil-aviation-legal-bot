# 劳动法机器人技术选型说明

这份文档说明本项目采用的核心技术，以及每种技术在系统中的职责。

## 1. Multi-Agent
### 采用原因
- 法律问答不只是单轮检索问题
- 需要把理解、拆分、检索、校验、冲突处理、生成分开
- 便于审计和回放

### 在系统中的位置
当前已实现：
- `QueryAgent`
- `RetrievalAgent`
- `CitationAgent`
- `ConflictAgent`
- `SynthesisAgent`

目标拆分：
- `SubjectAgent`
- `IssueAgent`
- `RewriteAgent`
- `ClarificationAgent`
- `DecompositionAgent`

### 作用
- 统一分工
- 降低单个 prompt 的职责负担
- 让复杂问题可以分阶段处理

## 2. Agentic RAG
### 采用原因
- 法律答案必须先检索证据，再生成结论
- 不能直接凭模型记忆回答
- 遇到不足时要能重写、补搜、重试

### 核心流程
1. 问题归一
2. 索引树召回
3. 候选重排
4. 原文读取
5. 引用校验
6. 答案合成

### 在系统中的位置
- 作为整个法律问答的主流程

## 3. ReAct
### 采用原因
- 允许模型在工具调用过程中逐步决策
- 适合检索型任务
- 可用于 agent 内部执行循环

### 在系统中的位置
- 工具调用层的执行范式

### 作用
- 让模型按“思考 -> 行动 -> 观察 -> 再行动”推进
- 适合复杂法律检索链路中的局部决策

## 4. ReWOO
### 采用原因
- 适合需要先规划、再执行的复杂问题
- 比纯 ReAct 更适合多步骤任务

### 在系统中的位置
- 复合问题和跨章节问题的规划层

### 作用
- 先输出步骤计划
- 再按步骤调用工具

## 5. Reflexion
### 采用原因
- 检索和生成后仍可能遗漏信息
- 需要自检与补搜机制

### 在系统中的位置
- 质量门控后的补充审查环节

### 作用
- 评估当前答案是否完整
- 给出缺失项和补搜建议

## 6. PageIndex 风格检索树
### 采用原因
- 法律文档本身有天然层级结构
- 目录树比纯 chunk 更适合条文定位

### 在系统中的位置
- `indexs.md`

### 作用
- 先查“哪部法、哪一章、哪一条”
- 再回读原文

## 7. Structured Indexing
### 采用原因
- 仅靠向量检索不够稳定
- 法律文档需要明确锚点

### 在系统中的位置
- `node_id`
- `source_anchor`
- `summary`
- `keywords`

### 作用
- 提高召回精度
- 降低定位歧义

## 8. Retrieval + Rerank
### 采用原因
- 先粗召回，再精排，比一次性检索更稳

### 在系统中的位置
- `search_law_index_tree`
- `rerank_law_hits`

### 作用
- 先找候选
- 再按相关性重排

## 9. Query Rewrite
### 采用原因
- 用户问题常有省略、口语化、歧义

### 在系统中的位置
- `QueryAgent`
- 检索失败回退分支

### 作用
- 同义归一
- 省略补全
- 范围收窄

## 10. Question Decomposition
### 采用原因
- 复合问题不能一次性硬答

### 在系统中的位置
当前由 `QueryAgent.plan()` 的 `sub_questions` 字段承载；目标是拆成独立 `DecompositionAgent`。

### 作用
- 把一个复杂问题拆成多个子问题
- 分别检索后再合成

## 11. Citation Verification
### 采用原因
- 法律场景不能只看“像对”
- 必须逐句对齐法条

### 在系统中的位置
- `CitationAgent`
- `CitationVerifierService`

### 作用
- 提取 claim
- 映射到具体条文
- 判断支持/不支持/部分支持

## 12. Conflict Resolution
### 采用原因
- 多部法律之间可能存在适用冲突

### 在系统中的位置
- `ConflictAgent`

### 作用
- 识别特别法/普通法关系
- 处理上位法/下位法关系
- 给出适用路径

## 13. Trace Logging
### 采用原因
- 法律系统必须可审计

### 在系统中的位置
- `TraceStore`

### 作用
- 记录每次检索、重写、校验、回退
- 便于回放和评测

## 14. JSON Tool Schema
### 采用原因
- 工具输入输出必须稳定
- 适合 Claude Code 和 agent runtime 调用

### 在系统中的位置
- 所有 tools

### 作用
- 减少参数漂移
- 降低工具调用错误

## 15. 本地法律文本管线
### 采用原因
- 第一版以本地法律数据为基础
- 可控、稳定、可复现

### 在系统中的位置
- `data/法律数据/*.txt`
- `data/indexs/*.md`

### 作用
- 作为事实来源
- 作为检索与引用的基础

## 16. 技术组合结论
本项目的核心技术组合是：

- **Multi-Agent** 负责分工
- **Agentic RAG** 负责检索-生成闭环
- **ReAct / ReWOO / Reflexion** 负责局部推理与执行
- **PageIndex 风格索引树** 负责法律结构化检索
- **Citation Verification** 负责法律正确性
- **Trace Logging** 负责可审计性
