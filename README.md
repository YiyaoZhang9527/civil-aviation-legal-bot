# 智能民航法律问答系统

基于 Multi-Agent RAG 架构的民航法律智能问答系统，覆盖 130+ 部中国民航法规，提供专业、可靠的法律问答服务。

## 架构

```
Query → SubjectAgent → IssueAgent → ClarificationAgent → RewriteAgent
      → DecompositionAgent → RetrievalAgent → CitationAgent → ConflictAgent
      → SynthesisAgent → ReflexionAgent → Answer
```

三级树检索：法级 → 章级 → 条级，逐层剪枝。BM25 + Dense Vector + Cross-Encoder RRF 融合。

## 质量指标（三层独立评测，2026-06-04）

| 指标 | 数值 |
|------|------|
| F1 | 83.3% |
| 忠实率 | 66.5% |
| 幻觉率 | 1.5% |
| 完整率 | 92.2% |
| 引用有效率 | 92.6% |

## 运行

构建法律索引：

```bash
python run.py --build-index
```

单次提问：

```bash
python run.py "航班延误赔偿标准是什么？"
```

连续对话 CLI：

```bash
python run.py --chat
```

模块方式运行：

```bash
python -m legalbot "无人机飞行需要什么审批？"
```

## 测试

```bash
pytest -q
```

三层评测：

```bash
python tests/test_30questions.py
python tests/test_citation_validity.py
python tests/test_faithfulness.py
python tests/test_summary.py
```

## 技术栈

- **LLM**: OpenAI API 兼容接口（DeepSeek / OpenAI / 自定义）
- **向量模型**: `thenlper/gte-large-zh`（Apache 2.0）
- **精排模型**: `BAAI/bge-reranker-v2-m3`（Apache 2.0）
- **检索**: BM25 + Dense Vector + Cross-Encoder RRF 融合
- **Python 3.12**，所有依赖均为宽松许可证（MIT/BSD/Apache 2.0），可商用

## 核心特性

- **证据相关性门控**：生成前独立评估证据是否回答了问题，不相关时拒答
- **数字幻觉检测**：答案中未被证据支撑的具体数字自动标注 `[待核实]`
- **Reflexion 自检循环**：LLM 评估答案质量，不足则补搜重试
- **149 单测**：覆盖核心管线各组件

## 迭代记录 #1（2026-06-07）

（本轮迭代运行中...）
