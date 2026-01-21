#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评测脚本：统计每条样本的事实正确性与幻觉，并输出总计与逐条结果到 JSON / 支持 JSONL

用法示例：
python eval_fact_count_v2.py \
  --input /path/to/data.jsonl \
  --output /path/to/results.json \
  --limit 100 \
  --offset 0
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

from openai import OpenAI
from tqdm import tqdm  # 进度条

# —— 改为语言无关（支持 Python/C/C++ 等）——
SYSTEM_PROMPT = (
    "You are a knowledgeable code expert. "
    "You help with abstractive function-level code summarization checking. "
    "Be concise and substantial. Follow instructions strictly. "
    "Return short and accurate answers."
)

# 产出严格 JSON，避免后处理困难
USER_PROMPT_TEMPLATE = """Below we have a piece of code for a function and a docstring/summary for that function (delimited with XML tags).
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
- Respond with JSON only. No extra text.

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
        response_format={"type": "json_object"}
        # extra_body={"enable_thinking": False},
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
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

# === 新增：空字段健壮性处理（非字符串视为空） ===
def safe_str(val: Any) -> str:
    if isinstance(val, str):
        s = val.strip()
        return s if s else ""
    return ""

# === 新增：JSON/JSONL 装载 ===
def load_input_records(input_path: str) -> List[Dict[str, Any]]:
    """
    - .json  : 兼容原有结构（list 或 dict）
    - .jsonl : 每行一个 JSON 对象
    """
    if input_path.lower().endswith(".jsonl"):
        recs: List[Dict[str, Any]] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"第 {ln} 行 JSON 解析失败: {e}")
                recs.append(obj)
        return recs
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return normalize_records(raw)

# === 新增：从 prompt 中去除硬提示并提取代码 ===
_SYS_BLOCK_RE = re.compile(
    r"<\|im_start\|>system\s*You are Qwen, created by Alibaba Cloud\. You are a helpful assistant\.\s*<\|im_end\|>\s*",
    flags=re.DOTALL
)
_USER_LINE_RE = re.compile(
    r"^\s*Please describe in simple english the purpose of the following Python code snippet:\s*",
    flags=re.IGNORECASE | re.MULTILINE
)
_IM_END_RE = re.compile(r"<\|im_end\|>", flags=re.DOTALL)

def strip_hard_prompt(text: str) -> str:
    """移除固定的 Qwen 系统/用户硬提示标记与说明行。"""
    if not isinstance(text, str):
        return ""
    t = _SYS_BLOCK_RE.sub("", text)
    # 去掉 <|im_start|>user 标记（若存在）
    t = t.replace("<|im_start|>user", "")
    # 去掉英文说明行
    t = _USER_LINE_RE.sub("", t)
    # 去掉任何残留的 <|im_end|>
    t = _IM_END_RE.sub("", t)
    return t.strip()

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

def extract_code_from_prompt(prompt: str) -> str:
    """
    1) 去硬提示
    2) 优先提取 ``` 代码块；若无，则返回剩余全文（通常就是代码）
    """
    cleaned = strip_hard_prompt(prompt or "")
    m = _CODE_BLOCK_RE.search(cleaned)
    if m:
        return m.group(1).strip()
    return cleaned

def evaluate_file(
    input_path: str,
    output_path: str,
    model: str,
    max_tokens: int,
    limit: int = None,
    offset: int = 0,
) -> Dict[str, Any]:

    # —— 不要把密钥写死；改为环境变量 —— 
    # client = OpenAI(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url=os.getenv("OPENAI_BASE_URL", "https://api.modelarts-maas.com/v1"),
    # )
    client = OpenAI(
        #api_key="PHrhcuFEefw1I0XkmZ7yiU6wMWAQ5olPAMJkIv4CVRaIwapF8OW4G3PXPX7l2eAZ5ikzu_E45mB55JuDtL7K9Q",
        api_key="lwdeU-BRrHUnSvsVJSv2GpqFN6N-oLnOu141jjh8MHnSJG4I3-QW7Nh4fnSUT8GsYhBJTdyYfeVgLjUVZE4kxg",
        base_url="https://api.modelarts-maas.com/v1"
    )
    all_records = load_input_records(input_path)

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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for i, rec in enumerate(tqdm(records, desc="Evaluating", unit="item")):
        # 支持两种数据形态：
        # 1) 原始：{code, summary}
        # 2) JSONL：{prompt, lable/label, predict}
        raw_code = safe_str(rec.get("code", ""))
        prompt = safe_str(rec.get("prompt", ""))
        code = raw_code if raw_code else extract_code_from_prompt(prompt)

        # 优先评测 predict（你的需求），若无再退回 summary
        predict = safe_str(rec.get("predict", ""))
        summary = predict if predict else safe_str(rec.get("summary", ""))

        # 若原始数据自带 idx 就用原 idx，否则用窗口内序号 + 全局偏移
        idx = rec.get("idx", offset + i)

        # 附带保存 label（可选，不参与计算）
        label = safe_str(rec.get("label", rec.get("lable", "")))

        if not code or not summary:
            result = {
                "idx": idx,
                "factual_incorrect": False,
                "has_hallucinations": False,
                "wrong_details": [],
                "hallucinations": [],
                "factual_correctness_score": 1,  # 1=通过；0=有问题
                "hallucination_score": 1,        # 1=无幻觉；0=有幻觉
                "note": "missing code or predict/summary; skipped model call",
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
                "factual_correctness_score": 0 if factual_incorrect else 1,
                "hallucination_score": 0 if has_hallucinations else 1,
            }

        # 附带原字段，便于追溯（不影响统计）
        result["predict"] = predict
        if label:
            result["label"] = label
        if raw_code:
            result["code_from"] = "code"
        elif prompt:
            result["code_from"] = "prompt"
        else:
            result["code_from"] = "none"

        per_item.append(result)

    total = len(per_item)
    num_clean = sum(
        1
        for x in per_item
        if not x["factual_incorrect"] and not x["has_hallucinations"]
    )

    summary = {
        "total": total,
        "offset": offset,
        "limit": limit,
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
        description="统计 Factual correctness 与 Hallucinations，并把得分写入 JSON（支持 JSONL）。"
    )
    parser.add_argument("--input", required=False,
                        default="/home/fmy/project/DPO-Summary/data/deepseek/java/java_data_train_0_4999.json",
                        help="输入数据 JSON 或 JSONL 文件路径")
    parser.add_argument("--output", required=False,
                        default="/home/fmy/project/DPO-Summary/evaluate/ProCon/fact_halluc/scores_9999.json",
                        help="输出结果 JSON 文件路径")
    parser.add_argument(
        "--model",
        default="deepseek-v3.2-exp",
        help="OpenAI Chat Completions 模型名",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="LLM 回复的最大 tokens",
    )
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

    # # 建议：使用环境变量
    # export OPENAI_API_KEY=...
    # export OPENAI_BASE_URL=https://api.modelarts-maas.com/v1

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
