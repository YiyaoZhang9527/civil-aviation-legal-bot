"""用LLM分析129部法规，生成完整的关键词→法规路由表。

用法: source .venv/bin/activate && python scripts/generate_law_routes.py
输出: legalbot/law_routes.json
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legalbot.retrieval import IndexRepository
from legalbot.llm import LLMClient

OUTPUT_PATH = PROJECT_ROOT / "legalbot" / "law_routes.json"

PROMPT_TEMPLATE = """你是一个中国民航法规专家。根据以下法规信息，提取该法规的关键词路由信息。

法规标题: {title}
法规ID: {law_id}
前几条正文:
{preview}

请输出JSON格式（不要markdown代码块，直接输出JSON）：
{{
  "domains": ["该法规覆盖的领域，用2-4个短词描述"],
  "keywords": ["能标识该法规的关键词或短语，至少5个，包含简称、部号、核心主题词"],
  "question_patterns": ["用户可能提问的模式，包含口语化表达，至少3个"]
}}

要求：
1. keywords要包含常见的简称和口语表达（如"无人机"而非仅"民用无人驾驶航空器"）
2. keywords要包含CCAR-XX部号（如果适用）
3. question_patterns要模拟真实用户提问，用日常语言
4. 输出纯JSON，不要任何额外文字"""


def get_doc_preview(doc, max_chars=800):
    """提取法规前几条正文作为预览。"""
    parts = []
    total = 0
    for child in doc.root.children[:5]:
        text = child.summary or child.text or child.title
        if text:
            parts.append(text[:200])
            total += len(text[:200])
            if total >= max_chars:
                break
    return "\n".join(parts)[:max_chars]


def main():
    print("加载法规索引...")
    docs = IndexRepository.documents()
    print(f"共 {len(docs)} 部法规\n")

    client = LLMClient()

    results = {}
    failed = []

    for i, doc in enumerate(docs):
        title = doc.title[:80]
        law_id = doc.law_id
        preview = get_doc_preview(doc)

        print(f"[{i+1}/{len(docs)}] {law_id[:40]}...", end=" ", flush=True)

        prompt = PROMPT_TEMPLATE.format(
            title=title,
            law_id=law_id,
            preview=preview,
        )

        try:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.strip()

            # 清理markdown代码块
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed = json.loads(content)
            results[law_id] = {
                "title": title,
                "domains": parsed.get("domains", []),
                "keywords": parsed.get("keywords", []),
                "question_patterns": parsed.get("question_patterns", []),
            }
            print(f"OK ({len(parsed.get('keywords', []))}组关键词)")
        except Exception as e:
            print(f"失败: {e}")
            failed.append(law_id)
            # 失败时用标题做基础路由
            results[law_id] = {
                "title": title,
                "domains": [],
                "keywords": [title[:10]],
                "question_patterns": [],
            }

        time.sleep(0.5)  # 避免API限流

    # 保存
    output = {
        "_meta": {
            "generated_by": "generate_law_routes.py",
            "total_laws": len(docs),
            "failed": failed,
        },
        "routes": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n完成! 成功: {len(docs) - len(failed)}/{len(docs)}")
    print(f"输出: {OUTPUT_PATH}")
    if failed:
        print(f"失败法规: {failed}")


if __name__ == "__main__":
    main()
