# 劳动法机器人工程文档

本目录定义 Multi-Agent Agentic RAG 法律问答系统的工程设计。

阅读顺序：
1. `01-architecture.md`：总体架构、Agent 分工、状态对象
2. `02-flowcharts.md`：总流程、函数级流程、失败回退
3. `03-tools-spec.md`：工具 JSON 协议
4. `04-indexs-spec.md`：每部法律的 `indexs.md` 检索树规范
5. `05-prompts.md`：各 Agent prompt 边界
6. `06-implementation-plan.md`：Claude Code 可执行任务拆分
7. `07-tech-stack.md`：采用的技术与用途说明
8. `08-current-pipeline.md`：当前代码实际运行管线

## 关键设计原则
- 采用多 Agent，但必须由 Orchestrator 统一调度
- Agent 负责决策，Tool 负责确定性动作，Service 负责业务逻辑
- 法律回答必须先有 evidence，再有 answer
- 引用校验失败时不允许输出该法律结论
- Web search 只能作为官方来源补充，不直接作为最终法律依据
- Memory 只能复用检索路径和 query rewrite，不直接复用法律结论

## 旧项目借鉴点
参考项目：
`/Users/zhangjing/Documents/学校食堂机器人/FrequentlyAskedQuestions_法律版`

可借鉴：
- 策略路由
- query rewrite
- 多意图拆分
- strict tool schema
- 质量门控
- trace/log

需要规避：
- `main.py` / `agent.py` / `planner.py` / `tools.py` 横向耦合
- LLM 自评分替代法律引用校验
- 网络搜索过早兜底
- 记忆库直接复用答案
- prompt 过度集中
