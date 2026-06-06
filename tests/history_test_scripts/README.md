# 废弃的测试脚本

这些是早期测试脚本，已被新的**分离式架构**取代。

## 废弃原因分类

### 旧消融（test_ablation*.py）
被 `ablation_runner.py` + `ablation_analyzer.py` 取代。

### 旧 combo / comprehensive（test_combo / test_comprehensive / test_full_answer）
| 问题 | 旧脚本 | 新方案 |
|------|--------|--------|
| 数据/评测混在一起 | 跑题 + 算指标 + 写报告都在一个文件 | `ablation_runner.py` 只跑题存数据，`ablation_analyzer.py` 独立评测 |
| 单 worker 顺序跑 | 顺序 8 组 × 30 题 | `ablation_runner.py` 2 worker 并行 |
| 不可重算 | 跑完只存 CSV，无法换指标重算 | 每题一 JSON，任意指标任意次数重算 |
| 难扩展 | 改一个指标要改全脚本 | analyzer 加 metric 函数即可 |
| 已被替代 | `test_comprehensive.py` 综合测试 | `eval_quality.py`（更先进） |

### 旧一致性（test_consistency.py）
10x 同一题一致性——单题波动评测，已被新评测体系替代。

## 文件清单

| 文件 | 原用途 | 替代方案 |
|------|--------|----------|
| `test_ablation.py` | 8 题 × 6 组合消融 | `ablation_runner.py` |
| `test_ablation2.py` | 8 题 × 2 组合 + 阈值扫描 | `ablation_runner.py` |
| `test_ablation3.py` | 30 题 × 8 组合消融 | `ablation_runner.py` |
| `test_combo.py` | 5 flag 组合 × 3 轮 | `ablation_runner.py` |
| `test_comprehensive.py` | 一键全面测试 | `analyze_static.py` + `analyze_llm.py` |
| `test_comprehensive_unit.py` | comprehensive 的单测 | （随 comprehensive 走）|
| `test_consistency.py` | 10x 同题一致性 | 已弃用 |
| `test_full_answer.py` | 完整答案测试（旧版）| `test_30questions.py` |
| `test_citation_validity.py` | 引用有效性（评测）| `analyze_static.py` |
| `test_faithfulness.py` | 忠实率评测 | `analyze_llm.py` |
| `test_summary.py` | 三层汇总 | `analyze_static.py`（refusal）+ `analyze_llm.py`（F1）|
| `eval_quality.py` | LeMAJ + MQS | `analyze_llm.py` |
| `test_eval_quality_unit.py` | eval_quality 的单测 | （随 eval_quality 走）|

## 当前活跃测试

### 主测试（执行类）
- `test_30questions.py` - 30 题（支持 `--workers N` 并行）
- `test_100questions.py` - 100 题（支持 `--workers N` 并行）

### 测评（合并为 2 个）
- `analyze_static.py` - 静态分析（无 LLM）：引用有效性 + 拒答率
- `analyze_llm.py` - LLM 分析（含 3 个裁判）：Faithfulness + LeMAJ + MQS + F1 + recall

### 消融框架（数据/评测分离）
- `ablation_grids.py` - 参数网格
- `ablation_runner.py` - 主控（2 worker 并行）
- `run_ablation_worker.py` - worker 进程
- `ablation_analyzer.py` - 后期分析（也合并了基础指标 + 引用 + LLM 评测）

### 单元测试（passing）
- test_synthesis_fallback, test_reflexion_best_so_far, test_tree_early_article_penalty
- test_article_label, test_evidence_sort, test_llm_json_fallback, test_number_guard
- test_plan_ab, test_decomposition_reflexion
- test_gold_coverage, test_gold_coverage_unit

### 集成测试（慢但活跃）
- test_cases.py - 10 个口语化用例

## 又被合并的 4 个旧脚本（2026-06-06）

合并理由：5 个评测脚本功能重复、维护成本高、文档难写。

| 旧脚本 | 合并到 |
|--------|--------|
| `test_citation_validity.py` | `analyze_static.py` |
| `test_summary.py` | `analyze_static.py`（refusal 部分）+ `analyze_llm.py`（F1/recall 聚合）|
| `test_faithfulness.py` | `analyze_llm.py` |
| `eval_quality.py` | `analyze_llm.py` |
| `test_eval_quality_unit.py` | （随 eval_quality 走）|
