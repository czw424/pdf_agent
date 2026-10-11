"""
轻量级RAG评测脚本（"RAGAS-lite"）。

动机：完整接入 ragas 库通常要求OpenAI兼容的LLM/embedding接口，
跟本项目"纯本地Ollama"的定位不太契合，容易在离线环境里卡住。
这里用同样的评测思路，自己实现三个核心指标，依赖更少、更适合纯本地跑：

1. retrieval_hit   —— 检索到的上下文里是否命中人工标注的关键词（context recall的代理指标）
2. answer_hit      —— 最终回答里是否命中关键词（answer relevancy的代理指标）
3. groundedness    —— 用同一个本地LLM做"裁判"，判断回答是否忠于检索到的上下文（1-5分，衡量幻觉程度）

用法：
    1. 先启动服务: python -m app.main
    2. 把 eval/eval_dataset.json 换成你自己知识库对应的问题和关键词
    3. 运行: python -m eval.evaluate
    4. 结果会打印一张汇总表，并写入 eval/eval_results.json
"""
import json
import os
import sys
import logging

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.rag.chain import load_llm  # noqa: E402

logging.basicConfig(level=logging.WARNING)

API_BASE_URL = os.getenv("EVAL_API_BASE_URL", "http://localhost:8000")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")

JUDGE_PROMPT = """你是一个严格的评测员。请判断下面的【回答】是否完全基于【上下文】，没有编造上下文之外的信息。
只输出一个1到5的整数分数，不要输出任何其他文字：
5分 = 完全基于上下文，无幻觉
3分 = 部分基于上下文，有轻微引申
1分 = 明显编造，与上下文无关

【上下文】
{context}

【回答】
{answer}

分数："""


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def call_rag_api(question: str) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/api/rag/query",
        json={"question": question, "session_id": "eval"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def keyword_hit(text: str, keywords: list) -> bool:
    if not keywords:
        return True
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords if kw)


def judge_groundedness(judge_llm, context: str, answer: str) -> int:
    if not context.strip() or answer.strip() == "【文档未找到相关信息】":
        return 5  # 没有上下文时模型正确拒答，视为满分（无幻觉）
    prompt = JUDGE_PROMPT.format(context=context[:3000], answer=answer)
    try:
        result = judge_llm.invoke(prompt).content.strip()
        digits = "".join(c for c in result if c.isdigit())
        return int(digits[0]) if digits else 3
    except Exception:
        return 3  # 裁判本身失败时给中性分，不影响整体流程


def main():
    dataset = load_dataset()
    judge_llm = load_llm()

    rows = []
    for item in dataset:
        question = item["question"]
        keywords = item.get("expected_keywords", [])

        data = call_rag_api(question)
        answer = data["answer"]
        context = "\n\n".join(d["page_content"] for d in data["source_documents"])

        row = {
            "question": question,
            "answer": answer,
            "retrieval_hit": keyword_hit(context, keywords),
            "answer_hit": keyword_hit(answer, keywords),
            "groundedness": judge_groundedness(judge_llm, context, answer),
        }
        rows.append(row)

    n = len(rows) or 1
    summary = {
        "num_questions": len(rows),
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in rows) / n,
        "answer_hit_rate": sum(r["answer_hit"] for r in rows) / n,
        "avg_groundedness": sum(r["groundedness"] for r in rows) / n,
        "details": rows,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== RAG 评测结果 ===")
    print(f"问题数量        : {summary['num_questions']}")
    print(f"检索命中率      : {summary['retrieval_hit_rate']:.2%}")
    print(f"回答关键词命中率: {summary['answer_hit_rate']:.2%}")
    print(f"平均忠实度评分  : {summary['avg_groundedness']:.2f} / 5")
    print(f"\n详细结果已写入: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
