# 消融测试 EVIDENCE_SORT_BY_CE 结果

**测试时间:** 2026-06-05 12:32 - 13:32
**测试文件:** test30_20260605_123203.json
**基线文件:** test30_20260604_180203.json
**评测模型:** deepseek-v4-flash

## 核心结论

**当前配置（EVIDENCE_SORT_BY_CE=True）显著优于关闭（=False）。** 关闭后所有核心指标均下降。
- faithful 下降 12.6 pp
- F1 下降 11.3 pp
- 检索缺口率上升 15.8 pp
- 拒答率上升 3.3 pp

**推荐：保持 EVIDENCE_SORT_BY_CE = True 不变。** 该参数在三种典型场景下保护 LLM 不被大量未支持的总则类证据带偏（详见 agents.py 第 770-774 行注释）。

## 三层指标对比表（30 题测试）

| 指标 | 当前值 (True) | 修改后 (False) | 变化 |
|------|--------------:|---------------:|-----:|
| **忠实率 faithful** | 0.6654 | 0.5391 | **-12.6 pp** |
| **真实幻觉率 hallucinated** | 0.0152 | 0.0195 | +0.4 pp |
| **完整率 Recall** | 0.9217 | 0.9053 | -1.6 pp |
| **引用有效率** | 0.9256 | 0.9123 | -1.3 pp |
| **拒答率** | 0.0000 | 0.0333 | +3.3 pp |
| **F1** | 0.8333 | 0.7200 | **-11.3 pp** |
| 检索缺口率 unverifiable | 0.2243 | 0.3828 | +15.8 pp |
| CE supported 率（系统自评） | 0.5028 | 0.4158 | -8.7 pp |

**用户提供的 100 题基线（仅供对比，量纲不同）：**
- 准确率(LeMAJ): 0.940
- faithful: 0.276
- hallucinated: 0.070
- F1: 0.783
- 完整率: 0.924
- 引用有效率: 0.883
- 拒答率: 0.020

注：100 题基线是 100 题不同方法（quality_20260605_101848.json），与本次 30 题三层评测方法不同，**不能直接对比**。本次 30 题测试在两种配置下使用完全相同方法（test_30questions + citation_validity + faithfulness + summary），可直接对比。

## 关键发现

### 1. 忠实率显著下降 (-12.6 pp)

关闭 CE 排序后，LLM 被大量未通过 CE 校验的总则类证据（如"91.1 目的"、"121.3 适用范围"等占位总则）带偏。
- 9 个未支持证据中可能只有 2-3 条是真正的"答案"条款（CE supported），其余是占位总则
- LLM 看到 75% 都是"目的/依据"，会倾向判"证据不足"或瞎编内容
- 这正是 agents.py 第 770-774 行注释中描述的反模式

### 2. 检索缺口率上升 15.8 pp

- True: 59/263 = 22.4%
- False: 98/256 = 38.3%

差距巨大。证据没有被正确排序时，LLM 容易"忽略"真正的答案条款，转而引用无关的总则条款。

### 3. 拒答率上升 3.3 pp

- True: 0/30
- False: 1/30（Q02 拒答）

Q02 燃油量要求是 CCAR-121 核心条款，关闭排序后 LLM 误判为"证据不足"。

## 决策

**不修改配置，保持 EVIDENCE_SORT_BY_CE = True。**

理由：
1. 5/7 核心指标下降，没有指标上升
2. faithful 下降 12.6 pp 是巨大退化（>10pp 视为配置不当）
3. F1 下降 11.3 pp
4. 与 agents.py 注释中描述的"9 条 unsupported vs 3 条 supported"反模式高度吻合

## 测试产物

- 测试输出: tests/test30_20260605_123203.csv (172 KB)
- 完整 JSON: tests/test30_20260605_123203.json (185 KB)
- 引用校验: tests/test30_20260605_123203_citation_validity.json
- 忠实度评测: tests/test30_20260605_123203_faithfulness.json
- 三层汇总: tests/test30_20260605_123203_summary.json

## 备注

本次测试发现的关键工程问题：
- worktree 没有 .env 文件，需要 symlink 主目录的 .env 才能加载 API key
- worktree 没有 models/ 目录（gte-large-zh 651MB），需要 symlink 才能避免 HuggingFace HEAD 请求超时
- 这两个问题导致首轮测试在 12:17 启动后被卡住 14 分钟，重启时加 `HF_HUB_OFFLINE=1` + symlink models 才解决
