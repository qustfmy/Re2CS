#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评测脚本：统计每条样本的事实正确性与幻觉，并输出总计与逐条结果到 JSON

用法示例：
python eval_fact_count_v2.py \
  --input /home/fmy/project/DPO-Summary/data/deepseek/python/enhanced_data_full_0_9999.json \
  --output /home/fmy/project/DPO-Summary/evaluate/ProCon/fact_halluc/resultscores_0_999.json \
  --limit 100 \
  --offset 0
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Union

from openai import OpenAI
from tqdm import tqdm  # 进度条

SYSTEM_PROMPT = (
    "You are a knowledgeable C/C++ code expert. "
    "You are here to help your colleagues with abstractive code summarization task. "
    "Your answers must be concise and substantial. Follow your instructions strictly. "
    "Your colleagues would appreciate it if you give a short and accurate answer."
)

# 产出严格 JSON，避免后处理困难
USER_PROMPT_TEMPLATE = """Below we have a C/C++ code of a function and a docstring for that function (delimited with XML tags).
We need to decide whether this function docstring gives a factual high-level summary of the code.
Patiently go over each statement from this function docstring.
Then return a strict JSON object with the following fields ONLY:

- "factual_incorrect": boolean  // true if the docstring contains ANY factual mistakes wrt the code
- "has_hallucinations": boolean // true if the docstring mentions ANY statement that cannot be extracted from the given code or general context
- "wrong_details": string[]     // short quotes or paraphrases of wrong statements (may be empty)
- "hallucinations": string[]    // short quotes or paraphrases of hallucinated statements (may be empty)

Rules:
- Focus on high-level summary consistency, not completeness.
- If the docstring merely omits details, do NOT mark it as incorrect or hallucinated.
- Respond with **JSON only**. No extra text.

<code>
{code}
</code>

<docstring>
{doc}
</docstring>
"""

def build_user_prompt(code: str, doc: str) -> str:
    return USER_PROMPT_TEMPLATE.format(code=code.strip(), doc=doc.strip())

def call_openai_eval(
    client: OpenAI,
    code: str,
    doc: str,
    model: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """调用 OpenAI Chat Completions，强制 JSON 输出并解析为 dict。"""
    user_prompt = build_user_prompt(code, doc)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        # enable_thinking=False
        # 需要可重复性可加 seed=0
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # ← 放这里
        #extra_body={"enable_thinking": False},
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 理论上不会发生，因为已设置 response_format=json_object
        data = {
            "factual_incorrect": False,
            "has_hallucinations": False,
            "wrong_details": [],
            "hallucinations": [],
            "raw": text,
        }
    return data

def normalize_records(
    raw: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """统一为列表；若是 dict，则取其 values 保持原字段。"""
    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict):
        return list(raw.values())
    else:
        raise ValueError("输入 JSON 顶层必须是列表或字典。")

# === 新增：空字段健壮性处理函数（仅用于将空/非字符串值安全转换为 ''） ===
def safe_str(val: Any) -> str:
    """
    将 None、非字符串、只包含空白的值统一转为空字符串；
    若为字符串则去除首尾空白返回。
    """
    if isinstance(val, str):
        s = val.strip()
        return s if s else ""
    return ""  # 非字符串或 None 统一视为空

def evaluate_file(
    input_path: str,
    output_path: str,
    model: str,
    max_tokens: int,
    limit: int = None,     # 新增：数量上限
    offset: int = 0,       # 新增：起始偏移
) -> Dict[str, Any]:
    # 建议改为使用环境变量：os.getenv("OPENAI_API_KEY")
    client = OpenAI(
        #api_key="PHrhcuFEefw1I0XkmZ7yiU6wMWAQ5olPAMJkIv4CVRaIwapF8OW4G3PXPX7l2eAZ5ikzu_E45mB55JuDtL7K9Q",
        api_key="lwdeU-BRrHUnSvsVJSv2GpqFN6N-oLnOu141jjh8MHnSJG4I3-QW7Nh4fnSUT8GsYhBJTdyYfeVgLjUVZE4kxg",
        base_url="https://api.modelarts-maas.com/v1"
    )
    #lwdeU-BRrHUnSvsVJSv2GpqFN6N-oLnOu141jjh8MHnSJG4I3-QW7Nh4fnSUT8GsYhBJTdyYfeVgLjUVZE4kxg

    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_records = normalize_records(raw)

    # 应用 offset/limit 窗口
    if offset < 0:
        offset = 0
    if limit is not None and limit < 0:
        limit = None

    records = all_records[offset : (offset + limit) if limit is not None else None]

    per_item: List[Dict[str, Any]] = []
    num_fact_issues = 0
    num_hallucinations = 0
    num_both = 0

    # 使用 tqdm 显示进度
    for i, rec in enumerate(tqdm(records, desc="Evaluating", unit="item")):
        # === 新增：对原数据字段做空值与类型健壮化处理 ===
        code = safe_str(rec.get("code", ""))
        summary = safe_str(rec.get("step5_final", ""))  # dingwei1
        # 若原始数据自带 idx 就用原 idx，否则用窗口内序号 + 全局偏移
        idx = rec.get("idx", offset + i)

        if not code or not summary:
            # 若字段缺失或为空字符串，给出空结果但不调用模型
            result = {
                "idx": idx,
                "factual_incorrect": False,
                "has_hallucinations": False,
                "wrong_details": [],
                "hallucinations": [],
                "factual_correctness_score": 1,  # 1=通过；0=有问题
                "hallucination_score": 1,        # 1=无幻觉；0=有幻觉
                "note": "missing code or summary; skipped model call",
            }
        else:
            out = call_openai_eval(
                client=client,
                code=code,
                doc=summary,
                model=model,
                max_tokens=max_tokens,
            )
            factual_incorrect = bool(out.get("factual_incorrect", False))
            has_hallucinations = bool(out.get("has_hallucinations", False))

            # 统计
            if factual_incorrect:
                num_fact_issues += 1
            if has_hallucinations:
                num_hallucinations += 1
            if factual_incorrect and has_hallucinations:
                num_both += 1

            result = {
                "idx": idx,
                "factual_incorrect": factual_incorrect,
                "has_hallucinations": has_hallucinations,
                "wrong_details": out.get("wrong_details", []),
                "hallucinations": out.get("hallucinations", []),
                # 简单 0/1 得分：通过=1，不通过=0
                "factual_correctness_score": 0 if factual_incorrect else 1,
                "hallucination_score": 0 if has_hallucinations else 1,
            }

        per_item.append(result)

    total = len(per_item)
    # 干净样本数 = 两者都没有
    num_clean = sum(
        1
        for x in per_item
        if not x["factual_incorrect"] and not x["has_hallucinations"]
    )

    summary = {
        "total": total,
        "offset": offset,              # 新增：记录窗口信息
        "limit": limit,                # 新增：记录窗口信息
        "num_with_factual_issues": num_fact_issues,
        "num_with_hallucinations": num_hallucinations,
        "num_with_both": num_both,
        "num_clean": num_clean,
    }

    output = {
        "summary": summary,
        "per_item": per_item,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output

def main():
    parser = argparse.ArgumentParser(
        description="统计 Factual correctness 与 Hallucinations，并把得分写入 JSON。"
    )
    parser.add_argument("--input", default="/home/fmy/project/DPO-Summary/data/deepseek/java/java_data_train_0_4999.json", help="输入数据 JSON 文件路径")
    parser.add_argument("--output", default="/home/fmy/project/DPO-Summary/evaluate/ProCon/fact_halluc/scores_9999.json", help="输出结果 JSON 文件路径")
    parser.add_argument(
        "--model",
        default="deepseek-v3.2-exp",
        help="OpenAI Chat Completions 模型名（默认与原代码一致）",
    )
    #DeepSeek-V3
    #deepseek-r1-250528
    #deepseek-v3.2-exp
    #qwen3-235b-a22b
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20000,
        help="LLM 回复的最大 tokens（默认 600）",
    )
    # 新增：数量控制
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多评测多少条样本（默认全部）。"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="从第几条开始评测（默认 0）。"
    )

    args = parser.parse_args()

    # # 简单检查 API Key（建议启用）
    # if not os.getenv("OPENAI_API_KEY"):
    #     raise RuntimeError(
    #         "缺少 OPENAI_API_KEY 环境变量。请先 `export OPENAI_API_KEY=...` 再运行。"
    #     )

    evaluate_file(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        max_tokens=args.max_tokens,
        limit=args.limit,
        offset=args.offset,
    )

if __name__ == "__main__":
    main()
