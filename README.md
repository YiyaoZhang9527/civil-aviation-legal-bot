# 劳动法机器人

本项目是一个本地可运行的劳动法 Multi-Agent Agentic RAG 原型。

## 运行方式
先构建法律索引：

```bash
python run.py --build-index
```

直接提问：

```bash
python run.py "试用期最长多久？"
```

进入连续对话 CLI：

```bash
python run.py --chat
```

对话命令：
- `/exit`：退出
- `/quiet`：关闭分阶段日志
- `/logs`：开启分阶段日志

也可以用模块方式运行：

```bash
python -m legalbot "没签劳动合同怎么办？"
```

## 测试
```bash
pytest -q
```

## 当前能力
- 解析 `data/法律数据/*.txt`
- 生成 `data/indexs/*.indexs.md`
- 使用多 Agent 编排完成检索、引用校验和答案生成
- 支持 10 个口语化法律问题回归测试
