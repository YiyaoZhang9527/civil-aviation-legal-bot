用于减少常见 LLM 编码错误的行为指南。可根据项目特定说明进行合并调整。

**权衡取舍：** 本指南偏向谨慎而非速度。对于简单任务，请自行判断。

## 0. 虚拟环境

**所有 Python 运行必须使用项目虚拟环境：`.venv/bin/python` 或 `.venv/bin/python3`。禁止使用系统 Python。**

示例：`.venv/bin/python tests/test_30questions.py` 而非 `python tests/test_30questions.py`。

## 1. 编码前先思考

**不要臆断。不要掩饰困惑。明确展示权衡方案。**

在开始实现之前：

- 明确陈述你的假设。如果不确定，请提问。
- 如果存在多种解释，把它们都列出来——不要自作主张地选择一种。
- 如果存在更简单的方案，直接指出来。必要时可以反驳要求。
- 如果有些事情不清楚，立即停下。明确指出困惑所在，然后提问。

## 2. 简单优先

**用最少的代码解决问题。不写任何猜测性的代码。**

- 不添加未明确要求的额外功能。
- 不为仅使用一次的代码创建抽象层。
- 不添加未要求的"灵活性"或"可配置性"。
- 不为不可能发生的场景编写错误处理。
- 如果你写了 200 行代码，但实际用 50 行就能搞定，请重写。

扪心自问："资深工程师会说这是过度设计吗？"如果答案是肯定的，那就简化它。

## 3. 外科手术式修改

**只改动必须碰的部分。只收拾自己留下的烂摊子。**

在编辑现有代码时：

- 不要"顺便优化"相邻的代码、注释或格式。
- 不要重构没有坏的东西。
- 匹配现有的代码风格，哪怕你觉得你的方式更好。
- 如果你注意到了无关的死代码，可以指出来——但不要直接删除。

当你的改动产生了孤立代码时：

- 移除因**你的改动**而变得不再使用的导入/变量/函数。
- 除非明确要求，否则不要删除原本就存在的死代码。

测试标准：每一行被改动的代码都应能直接追溯到用户的具体要求。

## 4. 目标驱动执行

**定义成功标准。循环迭代直到验证通过。**

将任务转化为可验证的目标：

- "添加验证" → "为无效输入编写测试，然后让测试通过"
- "修复 Bug" → "编写一个能复现该 Bug 的测试，然后修复它并让测试通过"
- "重构 X" → "确保重构前后所有测试均能通过"

对于多步骤任务，简述执行计划：

```
1.[步骤] → 验证方式: [检查项]

2.[步骤] → 验证方式: [检查项]

3.[步骤] → 验证方式: [检查项]
```

明确的成功标准能让你独立循环推进。模糊的标准（如"把它弄好"）则需要不断澄清。

## 5.先规划再动手

## 6.让合适的人做合适的事

**复杂的任务该用sub-agent就不要一个主agent完成。**

## 7. 大输出归档到文件

**所有大型输出（分析报告、排查过程、测试结果摘要、调研总结等）必须写入 `tests/对话过程/` 目录，按时间命名：`YYYYMMDD_HHMM_主题.md`。**

不要在对话中直接粘贴大段内容，保持对话简洁。输出写入文件后，仅在对话中给出关键结论和文件路径。

---

**判断本指南生效的标志：** diff 中的非必要改动减少，因过度复杂而重写的次数减少，澄清性问题发生在实现之前而不是犯错之后。

---

## 项目：智能法律问答系统

### 开源许可证合规

本项目的所有依赖均为宽松许可证（MIT/BSD/Apache 2.0），无 GPL/AGPL 传染性风险，全部允许商用。

**代码依赖（requirements.txt）：**

| 依赖                  | 许可证       | 商用 |
| --------------------- | ------------ | ---- |
| python-dotenv         | BSD 3-Clause | ✅   |
| openai                | Apache 2.0   | ✅   |
| anthropic             | MIT          | ✅   |
| pyyaml                | MIT          | ✅   |
| python-frontmatter    | MIT          | ✅   |
| rich                  | MIT          | ✅   |
| jieba                 | MIT          | ✅   |
| rank-bm25             | Apache 2.0   | ✅   |
| sentence-transformers | Apache 2.0   | ✅   |

间接依赖（PyTorch BSD、transformers Apache 2.0、numpy BSD、scikit-learn BSD 等）同样全部宽松许可。

**预训练模型（HuggingFace）：**

| 模型                         | 用途                  | 许可证     | 商用 |
| ---------------------------- | --------------------- | ---------- | ---- |
| `thenlper/gte-large-zh`      | 向量语义编码（config: VECTOR_MODEL） | Apache 2.0 | ✅   |
| `BAAI/bge-reranker-v2-m3`    | Cross-encoder 精排 + Claim-NLI 校验 | Apache 2.0 | ✅   |

**分发要求：** Apache 2.0 要求保留原始版权声明和 LICENSE 文本。若修改了源文件需注明变更。

**新增依赖规则：** 禁止引入 GPL/AGPL/LGPL 系依赖。新增任何依赖前须确认许可证为 MIT/BSD/Apache 2.0 或同等宽松许可。若需使用新 HuggingFace 模型，须先确认其 Model Card 中 `license` 字段允许商用。

### 评测体系（三层独立评测）

**禁止使用旧评测指标（CitationAgent CE 得分）作为质量依据。** 旧评测的 45.5% "幻觉率"实际是检索覆盖率缺口，不是答案幻觉。正确拒答被判为 92% 幻觉。

#### !! 优化目标：Faithfulness，不是 CE supported !!

**消融测试 v3（2026-06-03）发现的关键事实：**

CE supported（CitationAgent 的 cross-encoder 打分）是**自洽指标**，不是真实答案质量。它只检查"引用是否匹配证据文本"，不检查"答案是否正确回答了用户问题"。

- CE supported 从 28%→51%（+23pp），但独立评测的 Faithfulness 只从 49.0%→52.8%（+3.8pp）
- 消融测试按 CE supported 选出的"最优配置"，在真实答案质量上改善远小于指标显示
- 典型例子：Q20 行李赔偿 CE=6/12（50%），但 Faithfulness=0%，90.9% unverifiable
- 典型例子：Q24 适航指令 CE=6/12（50%），但 Faithfulness=0%，86.7% unverifiable

**因此：**
1. **消融测试的筛选指标必须是 Faithfulness 率（faithful / total claims），不是 CE supported 率**
2. **优化目标应同时最小化 hallucinated 率，而不仅是最大化 faithful 率**
3. **CE supported 可以作为检索质量的中期信号，但不能作为最终评判标准**
4. **任何声称"XX 配置最优"的结论，必须有三层评测（尤其 Faithfulness）支撑，不能只看 CE**

**原因**：CE 只检查引用和证据的语义匹配度，不管答案内容是否真的来自证据。检索到正确法规的条文 → CE 给高分 → 但 LLM 可能只读了标题，用自己的常识编造了答案内容。用户看到的是"有引用看似可靠但内容不准的答案"，比"没有引用的拒答"更危险。

#### 当前最优配置的实际表现

D 配置（`KEYWORD_ROUTING_ENABLED=True`, `TREE_GENERIC_ARTICLE_PENALTY=0.5`）经三层评测验证：

| 质量等级 | 题数 | 占比 | 说明 |
|---------|------|------|------|
| 优秀（faithful≥80%） | 6 题 | 20% | 答案内容有证据直接支撑 |
| 良好（faithful 50-80%） | 11 题 | 37% | 核心正确，细节不可验证 |
| 一般（faithful 20-50%） | 6 题 | 20% | 部分正确，大部分靠 LLM 常识 |
| 差（faithful<20%） | 6 题 | 20% | 几乎无证据支撑，引用可能误导 |
| 拒答 | 1 题 | 3% | 正确拒答 |

**用户获得可靠答案的比例：17/30（57%）。还有 6 题（20%）给了看似专业但实际不可靠的答案。**

**评测流程（三步顺序执行）：**

```bash
# 1. 先跑 30 题测试，生成基础结果
.venv/bin/python tests/test_30questions.py

# 2. 第 2 层：引用真实性（纯代码，~30秒）
.venv/bin/python tests/test_citation_validity.py

# 3. 第 1 层：Faithfulness（独立 LLM claim-level，~5-10分钟）
.venv/bin/python tests/test_faithfulness.py

# 4. 第 3 层：三层汇总
.venv/bin/python tests/test_summary.py
```

**三层指标定义：**

| 层 | 指标 | 方法 | 说明 |
|----|------|------|------|
| 1 | Faithfulness | 独立 LLM 逐 claim 评测 | faithful / partial / unverifiable / hallucinated 四档 |
| 2 | 引用有效率 | 纯正则 + index 对照 | 答案中《法名》第X条是否真实存在 |
| 3 | 拒答率 | 统计"无法确定"类回答 | 不算幻觉，属中性指标 |

**第 1 层四档判定标准（核心区分）：**

| 状态 | 含义 | 判定依据 |
|------|------|---------|
| faithful | 证据直接支撑 | 声明可从证据中推导出来 |
| partial | 部分支撑 | 核心信息有证据，细节无法验证 |
| unverifiable | 检索缺口 | 证据完全没有覆盖该话题，声明可能正确也可能错误 |
| hallucinated | 真正编造 | 证据覆盖了该话题但与声明矛盾，或编造了具体数字/细节 |

**关键规则：**
- 评测 LLM 与生成 LLM 必须不同（避免自我肯定偏差）
- 拒答不计入幻觉率（N/A）
- 证据必须读完整法条原文（`node.text`），不能只读标题（`node.summary`）
- 严格区分 `unverifiable` 和 `hallucinated`：证据沉默→unverifiable，证据矛盾→hallucinated
- 环境变量：`FAITHFULNESS_API_KEY` / `FAITHFULNESS_BASE_URL` / `FAITHFULNESS_MODEL`

**当前基线（v3 证据相关性门控，test30_20260603_120835，三层评测）：**

| 指标 | 数值 | 说明 |
|------|------|------|
| 真实幻觉率 | **0.0%** | 0/246 条声明是真正编造 |
| 检索缺口率 | **24.8%** | 61 条声明证据未覆盖 |
| 忠实率 | **69.9%** | 证据直接支撑的声明占比 |
| 可接受率 | **75.2%** | faithful+partial |
| 完整率 | **87.3%** | 答案未遗漏关键信息（Recall） |
| F1 | **80.8%** | 精确率 × 完整率综合 |
| 拒答率 | **10.0%** | 3/30 |

**已启用的配置：**
- `KEYWORD_ROUTING_ENABLED = True`（消融v3: +18pp）
- `TREE_GENERIC_ARTICLE_PENALTY = 0.5`（消融v3: 与KW协同+5pp）
- `RELEVANCE_GATE_ENABLED = True`（v3门控: +17.2pp faithful，-2.8pp hallucination）
- 22条精确fallback路由规则
- WRRF/AdaptiveK 已关闭（消融实验负收益）

**⚠ 注意：CE supported 率不能代表真实质量，后续所有优化必须以 Faithfulness 为准。**
