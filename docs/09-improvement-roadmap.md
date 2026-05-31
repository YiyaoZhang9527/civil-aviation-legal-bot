# 改进路线图

本文档记录已识别的改进项、技术方案和执行计划。

---

## 一、Citation Verifier 本地化 — 中文法律 NLI 模型

### 1.1 问题

当前 `CitationAgent.verify()` 完全依赖 LLM（DeepSeek）做引用校验：

- 每批 15 条 evidence 需要一次 LLM 调用，32 条证据就要 2~3 次调用
- 按 token 计费，成本累积快
- LLM 裁判本身也会幻觉，存在"裁判不可靠"的问题
- 延迟高，单次校验可能需要 3~10 秒

### 1.2 目标

用本地 NLI 模型替代 LLM 做引用校验，实现：

- 推理延迟从秒级降到毫秒级（~100ms / 条）
- 零 API 成本
- 判断确定性更高（cross-encoder 不会自回归幻觉）
- 专门适配中国法律文本

### 1.3 方案：Auto-GDA 式蒸馏 + 中文 DeBERTa 微调

#### 技术路线

```
DeepSeek (Teacher)
    ↓ 生成合成训练数据
(evidence, claim, supported/partial/unsupported) × N 条
    ↓
Erlangshen-DeBERTa-v2-186M-Chinese (Student)
    ↓ 微调
法律专用 NLI 模型 → 替换 citation.py 中的 LLM 调用
```

#### 分步执行计划

**Step 1：训练数据生成（1~2 天）**

利用现有 DeepSeek API，为每部法律的每个法条生成 NLI 训练样本。

```
输入：法条原文（evidence）
      ↓
LLM 生成 N 条 claims（法律主张/结论）
      ↓
LLM 标注每条 claim 相对 evidence 的关系：
  - supported：法条直接支持该主张
  - partial：法条部分相关
  - unsupported：法条不支持（含反向/无关主张）
      ↓
输出：(evidence_text, claim, label) 三元组
```

数据量规划：

| 法律           | 法条数 | 每条生成 claims | 合计样本 |
|----------------|--------|----------------|----------|
| 劳动法         | ~107   | 5~8            | ~600     |
| 劳动合同法     | ~98    | 5~8            | ~550     |
| 劳动争议调解仲裁法 | ~54 | 4~6            | ~250     |
| 社会保险法     | ~98    | 4~6            | ~450     |
| 其余 34 部     | ~800   | 3~5            | ~3000    |
| **合计**       |        |                | **~5000** |

目标：生成 5000+ 条高质量 (evidence, claim, label) 样本。

实现：新建 `scripts/generate_nli_data.py`
- 遍历 `data/法律数据/*.txt`，逐法条调用 DeepSeek
- Prompt 设计：给法条原文，要求生成 N 条"能被该法条支持的主张"和 N 条"不能被该法条支持的主张"
- 正负样本比例控制在 4:3:3（supported : partial : unsupported）
- 输出格式：JSONL，每行 `{"evidence": "...", "claim": "...", "label": "supported"}`
- 人工抽样检查 200 条，准确率 >= 90% 才进入下一步

**Step 2：模型微调（1~2 天）**

基座模型：`IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese`（186M 参数，Apache 2.0）

```
训练配置：
  - 模型：Erlangshen-DeBERTa-v2-186M
  - 输入：[CLS] evidence [SEP] claim [SEP]
  - 输出：3 分类（supported / partial / unsupported）
  - 框架：transformers + PyTorch
  - 训练参数：
    - epochs: 3~5
    - batch_size: 16
    - learning_rate: 2e-5
    - max_length: 512（法条 + 主张一般不超过 512 token）
    - 8:1:1 划分 train/val/test
  - 硬件：单卡 GPU（3090/4090 即可，186M 模型很小）
  - 预计训练时间：30 分钟 ~ 1 小时
```

验收标准：
- Test set accuracy >= 85%
- Test set macro-F1 >= 0.82
- supported 类 F1 >= 0.85（最关键，不能把支持的判成不支持）

**Step 3：集成到 CitationAgent（0.5 天）**

修改 `legalbot/citation.py`：

```python
# 新增：本地 NLI 模型推理
class LocalNLICitationVerifier:
    """用本地微调 NLI 模型替代 LLM 做引用校验。"""
    def __init__(self, model_path: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict(self, evidence_text: str, claim: str) -> tuple[str, float]:
        """返回 (status, confidence)。"""
        inputs = self.tokenizer(evidence_text, claim, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        label_id = probs.argmax().item()
        label_map = {0: "supported", 1: "partial", 2: "unsupported"}
        return label_map[label_id], probs[0][label_id].item()
```

调用方式：

```
CitationAgent.verify():
  1. extract_claims() → 仍用 LLM（claim 抽取需要语义理解）
  2. 对每条 (evidence, claim) 对 → 用本地 NLI 模型打分 → 毫秒级
  3. 不再需要分批，32 条 evidence × N 个 claims 全部本地推理
```

性能对比预期：

| 维度         | 当前 LLM        | 改进后 NLI      |
|--------------|-----------------|-----------------|
| 32 条校验延迟 | 6~30 秒        | < 1 秒         |
| API 成本     | ¥0.01~0.05/次  | ¥0             |
| 判断稳定性   | 不稳定（LLM 幻觉） | 确定性输出     |
| 可解释性     | 有 reason       | 需补充 quote 逻辑 |

**Step 4：兜底策略**

- NLI 模型置信度 < 0.6 时，fallback 到 LLM 校验（保底）
- `config.py` 新增 `CITATION_USE_LOCAL_NLI = True`（可关闭，回退到 LLM）
- 新增 `CITATION_NLI_MODEL_PATH` 配置项

### 1.4 文件变更清单

| 文件 | 操作 |
|------|------|
| `scripts/generate_nli_data.py` | 新建 — 训练数据生成脚本 |
| `scripts/finetune_nli.py` | 新建 — 微调脚本 |
| `legalbot/citation.py` | 修改 — 新增 LocalNLICitationVerifier，verify() 改用本地模型 |
| `legalbot/config.py` | 修改 — 新增 CITATION_USE_LOCAL_NLI / CITATION_NLI_MODEL_PATH |
| `models/legal-nli/` | 新建 — 存放微调后的模型文件 |
| `data/nli_training_data/` | 新建 — 存放生成的训练数据 |

### 1.5 许可证合规

| 组件 | 许可证 | 商用 |
|------|--------|------|
| Erlangshen-DeBERTa-v2-186M-Chinese | Apache 2.0 | ✅ |
| 训练数据（自行生成） | 自有 | ✅ |
| transformers / PyTorch | Apache 2.0 / BSD | ✅ |

### 1.6 风险

| 风险 | 缓解措施 |
|------|----------|
| NLI 模型在法律文本上精度不够 | 人工标注 200 条作为 test set，F1 < 0.82 则增加训练数据 |
| 法条 + claim 超过 512 token | 截断 evidence 到 450 token，保留 claim 62 token |
| 法律领域外的 claim（用户闲聊） | NLI 只在 evidence 非空时启用，无 evidence 跳过 |

---

## 二、民航法规问答系统扩展

### 2.1 目标

将现有劳动法机器人架构复用到民航法规领域，构建民航法规智能问答系统。

### 2.2 民航法规数据范围（初步）

| 类别 | 法规名称 | 法条规模 |
|------|----------|----------|
| 法律 | 民用航空法 | ~215 条 |
| 行政法规 | 民用航空器适航管理条例 | ~35 条 |
| 行政法规 | 民用航空安全保卫条例 | ~60 条 |
| 行政法规 | 通用航空飞行管制条例 | ~45 条 |
| 行政法规 | 民用航空器国籍登记条例 | ~25 条 |
| 规章（CCAR） | CCAR-121 大型飞机公共航空运输承运人运行合格审定规则 | ~600 条 |
| 规章（CCAR） | CCAR-135 小型航空器商业运输运营人运行合格审定规则 | ~300 条 |
| 规章（CCAR） | CCAR-91 一般运行和飞行规则 | ~200 条 |
| 规章（CCAR） | CCAR-67 中国民用航空人员医学标准和体检合格证管理规则 | ~80 条 |
| 规章（CCAR） | CCAR-25 运输类飞机适航标准 | ~500 条 |
| 规章（CCAR） | CCAR-145 民用航空器维修单位合格审定规定 | ~100 条 |
| 规章（CCAR） | CCAR-66 民用航空器维修人员执照管理规则 | ~60 条 |
| 规章（CCAR） | CCAR-27/29 正常/运输类旋翼航空器适航规定 | ~300 条 |
| 规范性文件 | 各类咨询通告（AC）、管理程序（AP） | 数量庞大 |
| 国际公约 | 芝加哥公约及附件 | ~200 条 |

**总计约 2800+ 法条**，规模是劳动法的 3~4 倍。

### 2.3 架构复用策略

劳动法机器人的 Multi-Agent Agentic RAG 架构天然支持领域迁移。需要调整的部分：

```
可完全复用（不改代码）：
  ✅ 全部 Agent（Subject → Issue → Clarification → Rewrite → Decomposition
     → Retrieval → Citation → Conflict → Synthesis → Reflexion）
  ✅ Orchestrator 编排逻辑
  ✅ 对话管理（ConversationManager）
  ✅ 检索引擎（BM25 + Vector + RRF + CrossEncoder + 树检索）
  ✅ Citation 校验框架（改完 NLI 本地化后直接复用）
  ✅ 会话存储、日志系统

需要领域适配（改配置和 prompt）：
  🔧 Prompt 模板（SubjectAgent 的领域关键词、IssueAgent 的法律争点框架）
  🔧 Clarification 的缺失事实判断标准
  🔧 ConflictAgent 的法律优先级规则
  🔧 compose_answer 的回答格式模板

需要重建（数据驱动）：
  🔧 法律文本数据 → data/法律数据/
  🔧 索引 → data/indexs/
  🔧 向量缓存 → data/.vector_cache.npz
  🔧 摘要缓存 → data/summaries/
```

### 2.4 分步执行计划

**Phase 0：项目结构（1 天）**

```
民航法规机器人/
  ├── legalbot/           # 直接复用，不改代码
  ├── data/
  │   ├── 法律数据/       # 放入民航法规 txt 文件
  │   ├── indexs/         # 由 build-index 自动生成
  │   ├── trace/
  │   └── summaries/
  ├── models/
  │   ├── gte-large-zh/       # 复用
  │   ├── bge-reranker-v2-m3/ # 复用
  │   └── legal-nli/          # 复用微调后的模型
  ├── scripts/
  │   └── generate_nli_data.py  # 需要为民航法规重新生成训练数据
  ├── docs/
  ├── .env
  ├── config.py           # 继承基础配置
  └── run.py
```

两种组织方式（待定）：

**方案 A：独立仓库** — 民航法规机器人是独立项目，复制 legalbot/ 目录，各自维护
**方案 B：同一仓库多领域** — 一个仓库，通过 `--domain labor|aviation` 参数切换法律数据目录

推荐方案 A，理由：法律数据相互独立，部署互不影响，prompt 差异大。

**Phase 1：数据准备（3~5 天）**

1. 收集民航法规文本（CCAR 规章全文、行政法规、国际公约中文版）
2. 格式化为统一的 txt 文件（复用 `parser.py` 的解析规则）
3. 法规文本需要按"编/章/节/条"层级组织（CCAR 规章通常结构清晰）
4. 民航法规的特殊性：
   - CCAR 编号体系（如 CCAR-121-R6）需要在 `law_id` 中体现
   - 修订版本追踪（R5/R6/R7）需要在元数据中标注
   - 交叉引用密集（CCAR-121 引用 CCAR-25、CCAR-145 等），crossref.py 需要适配

**Phase 2：索引构建（1~2 天）**

```bash
python run.py --build-index
```

自动执行：
- `parse_law_text()` → LawDocument 树
- `enhance_summaries()` → LLM 增强摘要（民航领域术语需要 prompt 适配）
- `render_index_markdown()` → indexs.md
- 向量缓存构建

民航法规的摘要增强 prompt 需要调整：
- 劳动法关键词：加班费、劳动合同、工伤、社保……
- 民航关键词：适航、运行合格审定、飞行标准、维修执照、安保……

**Phase 3：Prompt 领域适配（2~3 天）**

修改各 Agent 的 system prompt，注入民航领域知识：

| Agent | 适配内容 |
|-------|----------|
| SubjectAgent | 民航主体识别：航空公司、飞行员、维修人员、空管、机场、旅客、适航审定部门 |
| IssueAgent | 民航法律争点框架：运行合格、适航标准、人员资质、安全保卫、责任赔偿 |
| ClarificationAgent | 民航缺失事实标准：航空器类型、运行种类、持证情况、事故/事件性质 |
| ConflictAgent | 民航法律优先级：法律 > 行政法规 > CCAR 规章 > 规范性文件；CCAR 之间特别优于一般 |
| SynthesisAgent | 回答格式适配民航场景：引用 CCAR 编号、条款号、修订版本 |

**Phase 4：测试与调优（3~5 天）**

测试用例：

```
Q1: "航空公司未按规定进行 C 检，继续运营有什么后果？"
    → 应引用 CCAR-121 相关维修检查条款 + CCAR-145

Q2: "飞行员体检不合格但继续飞行，违反了哪些规定？"
    → 应引用 CCAR-67 医学标准 + CCAR-91/121 运行规则

Q3: "通用航空企业想开展载客飞行业务需要什么资质？"
    → 应引用 CCAR-135 运行合格审定 + CCAR-91 基础规则

Q4: "航班延误 8 小时，航空公司拒绝赔偿怎么办？"
    → 应引用民用航空法第 130 条等消费者保护条款
```

调优方向：
- 检索精度（调整 BM25/Vector 权重）
- 树检索层级是否合适（民航法规"编-章-节-条"可能比劳动法更深）
- CrossEncoder reranker 在民航文本上的表现

### 2.5 NLI 模型的民航法规适配

复用劳动法的 NLI 微调方案，但需要：

1. 为民航法规单独生成 NLI 训练数据（~3000 条）
2. 方案选择：
   - **方案 A：独立模型** — 训练一个 `legal-nli-aviation`，与 `legal-nli-labor` 分开
   - **方案 B：联合模型** — 将劳动法 + 民航法 NLI 数据合并训练一个通用法律 NLI 模型

推荐方案 B，理由：
- 186M 参数足够覆盖两个法律领域
- 蕴含关系判断是通用的（"这条法律是否支持这个主张"），不依赖具体法律领域
- 减少模型维护成本

训练数据合并：5000（劳动法）+ 3000（民航法）= 8000 条 → 联合训练一个模型。

### 2.6 时间估算

| 阶段 | 内容 | 时间 |
|------|------|------|
| Phase 0 | 项目结构搭建 | 1 天 |
| Phase 1 | 民航法规数据收集与格式化 | 3~5 天 |
| Phase 2 | 索引构建 + 摘要增强 | 1~2 天 |
| Phase 3 | Prompt 领域适配 | 2~3 天 |
| Phase 4 | 测试与调优 | 3~5 天 |
| Phase 5 | NLI 模型联合训练 | 1~2 天 |
| **合计** | | **11~18 天** |

---

## 三、已完成的改进项

| 改进项 | 状态 | 完成日期 |
|--------|------|----------|
| CrossRef ghost Evidence 修复（law_id="" → 真实值） | ✅ 已完成 | 2026-05-27 |
| compose_answer 证据排序+标签（supported 前 / 未检中 / unsupported 后） | ✅ 已完成 | 2026-05-27 |
| CitationAgent 改为 evidence-centric 校验（每条都检查） | ✅ 已完成 | 2026-05-27 |
| 树检索激活（TREE_ENABLED=True + gte-large-zh） | ✅ 已完成 | 2026-05-27 |
| 日志双语化（English stage + 中文描述） | ✅ 已完成 | 2026-05-27 |
| LLM 重试机制（429/500/502/503 指数退避） | ✅ 已完成 | 2026-05-27 |
| 澄清追问次数限制（MAX_CLARIFICATION_ATTEMPTS=3） | ✅ 已完成 | 2026-05-27 |
| 日志内存泄漏防护（entries 上限 500） | ✅ 已完成 | 2026-05-27 |
| ConflictAgent 重写（假多法检测 → 法律优先级排序） | ✅ 已完成 | 2026-05-27 |
