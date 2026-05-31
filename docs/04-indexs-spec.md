# 劳动法机器人 indexs.md 规范

## 1. 文件职责
每部法律单独生成一个 `indexs.md`，作为该法律的检索树。

原则：
- 只存结构、摘要、关键词、锚点
- 不存最终回答
- 不代替原文全文

## 2. 推荐目录
```text
data/
  法律数据/
    中华人民共和国劳动合同法.txt
  indexs/
    中华人民共和国劳动合同法.indexs.md
```

## 3. 节点层级
- law
- chapter
- section
- article
- clause
- item

## 4. 节点字段
每个节点至少包含：
- `node_id`
- `title`
- `type`
- `summary`
- `keywords`
- `source_file`
- `source_anchor`
- `children`

## 5. 示例
```markdown
# 中华人民共和国劳动合同法

- node_id: law:labor_contract_law
- type: law
- source_file: data/法律数据/中华人民共和国劳动合同法.txt
- summary: 调整劳动合同订立、履行、变更、解除、终止及法律责任。
- keywords: 劳动合同, 用人单位, 劳动者, 解除, 终止
- children: chapter:1, chapter:2

## 第一章 总则

- node_id: chapter:1
- type: chapter
- summary: 立法目的、适用范围、基本原则。
- children: article:1, article:2

### 第一条

- node_id: article:1
- type: article
- source_anchor: 第一条
- keywords: 立法目的, 劳动关系, 权益保护
- summary: 说明制定本法的目的。
```

## 6. 生成规则
- `source_anchor` 由原文标题抽取
- `node_id` 在同一法律内唯一
- 条/款/项嵌套要稳定
- 构建阶段生成锚点映射表，运行时不靠正则猜定位

## 7. LLM 摘要增强
索引构建时，`summary_generator.py` 通过 LLM 增强 `law` 和 `chapter` 级节点的 summary：

### 7.1 法级摘要
- 输入：法名 + 条文全文
- 输出：~150 字语义范围描述（覆盖该法适用主体、核心场景、主要约束）
- 示例：`"适用范围覆盖所有与企业、个体经济组织建立劳动关系的劳动者，核心规范包括合同订立形式、试用期限制、解雇保护、经济补偿标准等"`

### 7.2 章级摘要
- 输入：法名 + 章标题 + 章内条文
- 输出：~100 字章节范围描述

### 7.3 缓存机制
- 缓存目录：`data/summaries/`
- 缓存键：源文件 MD5 哈希
- 缓存格式：JSON，包含 `file_hash`、`nodes`（node_id → summary 映射）
- 源文件未变化时跳过 LLM 调用，直接从缓存加载
- LLM 不可用时使用规则摘要兜底（`parser._summary_from_text()`）

## 8. 三级树索引
`tree_retrieval.py` 为每级构建独立索引：

| 级别 | 编码文本格式 | BM25 | Vector | 映射 |
|------|------------|------|--------|------|
| 法级 | `{法名} {法summary}` | yes | yes | — |
| 章级 | `{法名} {章标题} {章summary}` | yes | yes | `chapter_to_law` |
| 条级 | `{法名} {章标题} {条标题} {条summary}` | yes | yes | `article_to_chapter` |

- 向量缓存：`data/indexs/.tree_vector_cache.npz`（numpy npz，allow_pickle=False）
- 缓存签名：基于条级 `(law_id, node_id)` 列表的 MD5，数据变化时自动重建

