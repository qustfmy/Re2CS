#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评测脚本：A/B 摘要“充分性”对比（支持单文件或锚点/基线双文件），并把总分与逐条结果写入 JSON。
现已同时支持 .json 与 .jsonl 输入：
- .json  : 顶层 list 或 { "test": [...] }
- .jsonl : 每行一个 JSON 对象

单文件（同一条记录内两列对比，例如 summary vs step5_final）
python eval_suffic_jsonl.py \
  --mode single \
  --input /home/fmy/project/DPO-Summary/evaluate/ProCon/verbosity/llamafactory/1989.jsonl \
  --doc-a-key lable \
  --doc-b-key predict \
  --output /home/fmy/project/DPO-Summary/evaluate/ProCon/sufficiency/jsonlsuff_scores.json \
  --model deepseek-r1-250528 \
  --limit 200 --offset 0

双文件（锚点 vs 基线，通过 name_key 配对）
python eval_suffic.py \
  --mode pair \
  --anchor /path/to/anchor.jsonl \
  --base /path/to/base.jsonl \
  --anchor-doc-key summary \
  --base-doc-key step5_final \
  --output /path/to/suff_scores.json \
  --model deepseek-r1-250528 \
  --limit 200 --offset 0
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm  # 进度条


# =========================
# 1) GPT 比对 Prompt 与调用
# =========================

def get_suff_prompts(code: str, doc_a: str, doc_b: str) -> Tuple[str, str, str]:
    sys = (
        "Below you have a code snippet with 2 summaries delimited with summary_A and summary_B tags. "
        "Please tell which one of them is more comprehensive and complete, i.e. covers more crucial "
        "aspects of the code and gives a clearer description of what the function does, or if they are "
        "equally comprehensive. Please be as concise as possible."
    )
    user = (
        f"<code>\n{code}\n</code>\n"
        f"<summary_A>\n{doc_a}\n</summary_A>\n"
        f"<summary_B>\n{doc_b}\n</summary_B>\n"
        "Which one is more complete? Are they comparable? Your answer:"
    )
    final_grade_message = (
        '  Based on your thoughts give a final answer. Return a single character: "A" for summary_A, '
        '"B" for summary_B and "C" if they are comparable. Your response (one letter):'
    )
    return sys, user, final_grade_message


def _normalize_choice(one_char: str) -> str:
    """
    归一化模型返回的选择字符：将西里尔 'С' 映射为拉丁 'C'，并只保留 A/B/C，否则返回 'N'。
    """
    if not one_char:
        return "N"
    ch = one_char.strip()
    ch = "C" if ch == "С" else ch  # 映射西里尔大写С (U+0421)
    return ch if len(ch) == 1 and ch in {"A", "B", "C"} else "N"


def call_openai_pairwise(
    client: OpenAI,
    code: str,
    doc_a: str,
    doc_b: str,
    model: str,
    rationale_max_tokens: int = 150,
    grade_max_tokens: int = 5,
    temperature: float = 0.0,
) -> Tuple[str, str]:
    """
    两轮调用：
    - 第一轮要简短理由 rationale
    - 第二轮只要单字母 A/B/C
    返回: (assessment, rationale)
    """
    sys, user, final_grade = get_suff_prompts(code, doc_a, doc_b)

    # 第一轮：要理由
    r0 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=rationale_max_tokens,
        extra_body={"enable_thinking": False},
    )
    rationale = (r0.choices[0].message.content or "").strip()

    # 第二轮：只要单字母
    new_prompt = user + " " + rationale + final_grade
    r1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": new_prompt},
        ],
        temperature=0,
        max_tokens=grade_max_tokens,
        extra_body={"enable_thinking": False},
    )
    assessment_raw = (r1.choices[0].message.content or "").strip()
    assessment = _normalize_choice(assessment_raw)
    return assessment, rationale


# =========================
# 2) 加载：同时支持 JSON / JSONL
# =========================

def _safe_str(v: Any) -> str:
    if isinstance(v, str):
        s = v.strip()
        return s if s else ""
    return ""


def load_json_list(path: str) -> List[Dict[str, Any]]:
    """
    兼容：
    - .json  顶层为 list 或 { "test": [...] }
    - .jsonl 每行一个 JSON 对象
    """
    if path.lower().endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"{path}: 第 {ln} 行 JSON 解析失败: {e}")
                rows.append(obj)
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "test" in data and isinstance(data["test"], list):
        return data["test"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure in {path}. Expected list or dict with 'test' list.")


# =========================
# 3) 从 prompt 去硬提示并提取代码
# =========================

_SYS_BLOCK_RE = re.compile(
    r"<\|im_start\|>system\s*You are Qwen, created by Alibaba Cloud\. You are a helpful assistant\.\s*<\|im_end\|>\s*",
    flags=re.DOTALL
)
_USER_LINE_RE = re.compile(
    r"^\s*Please describe in simple english the purpose of the following Python code snippet:\s*",
    flags=re.IGNORECASE | re.MULTILINE
)
_IM_END_RE = re.compile(r"<\|im_end\|>", flags=re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

def strip_hard_prompt(text: str) -> str:
    """移除固定的 Qwen 系统/用户硬提示标记与说明行。"""
    if not isinstance(text, str):
        return ""
    t = _SYS_BLOCK_RE.sub("", text)
    t = t.replace("<|im_start|>user", "")
    t = _USER_LINE_RE.sub("", t)
    t = _IM_END_RE.sub("", t)
    return t.strip()

def extract_code_from_prompt(prompt: str) -> str:
    """
    1) 去硬提示
    2) 优先提取 ``` 代码块；若无，则返回剩余全文（通常就是代码）
    """
    cleaned = strip_hard_prompt(prompt or "")
    m = _CODE_BLOCK_RE.search(cleaned)
    if m:
        return (m.group(1) or "").strip()
    return cleaned


# =========================
# 4) 行 -> 统一点位映射
# =========================

def point_from_row(
    r: Dict[str, Any],
    name_key: str,
    code_key: str,
    doc_key: str,
    prompt_key: str = "prompt",
    id_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    转成内部统一结构：
    { "name": str, "code": str, "doc_selected": str, optional "id": any, "code_from": "code|prompt|none" }
    - code 优先来自 code_key；若缺失则从 prompt 解析
    - doc_selected 来自 doc_key（你在 CLI 传入的字段名，比如 summary / step5_final / predict_* 等）
    """
    name = r.get(name_key)
    if not name:
        return None

    raw_code = _safe_str(r.get(code_key, ""))
    prompt = _safe_str(r.get(prompt_key, ""))
    code_from = "code" if raw_code else ("prompt" if prompt else "none")
    code = raw_code if raw_code else (extract_code_from_prompt(prompt) if prompt else "")

    doc_selected = _safe_str(r.get(doc_key, ""))

    item: Dict[str, Any] = {
        "name": name,
        "code": code,
        "doc_selected": doc_selected,
        "code_from": code_from,
    }
    if id_key and id_key in r:
        item["id"] = r.get(id_key)
    return item


def points_from_rows(
    rows: List[Dict[str, Any]],
    name_key: str,
    code_key: str,
    doc_key: str,
    prompt_key: str = "prompt",
    id_key: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    以 name 为键的映射：{ name: point }
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        p = point_from_row(r, name_key, code_key, doc_key, prompt_key, id_key)
        if p is None:
            continue
        out[p["name"]] = p
    return out


# =========================
# 5) 评测流程（单/双文件）与打分
# =========================

def evaluate_single_file(
    client: OpenAI,
    input_path: str,
    doc_a_key: str,
    doc_b_key: str,
    name_key: str = "func_name",
    code_key: str = "code",
    prompt_key: str = "prompt",
    id_key: Optional[str] = "idx",
    model: str = "deepseek-r1-250528",
    rationale_max_tokens: int = 150,
    grade_max_tokens: int = 5,
    temperature: float = 0.0,
    limit: Optional[int] = None,
    offset: int = 0,
    onetime_limit: int = 1_000_000,
) -> Dict[str, Any]:
    rows = load_json_list(input_path)

    # 构建 anchor/base（同一数据，不同列）
    anchor_points = points_from_rows(rows, name_key, code_key, doc_a_key, prompt_key, id_key)
    base_points   = points_from_rows(rows, name_key, code_key, doc_b_key, prompt_key, id_key)

    # 重叠 name 列表
    names_all = [n for n in anchor_points.keys() if n in base_points]
    names = names_all[offset : (offset + limit) if (limit is not None) else None]

    if len(names) >= onetime_limit:
        raise AssertionError(f"onetime_limit exceeded: {len(names)} vs {onetime_limit}")

    per_item: List[Dict[str, Any]] = []
    cnt_A = cnt_B = cnt_C = cnt_N = 0

    for name in tqdm(names, desc="Comparing (single)", unit="pair"):
        a = anchor_points[name]
        b = base_points[name]
        code  = a.get("code", "")
        doc_a = a.get("doc_selected", "")
        doc_b = b.get("doc_selected", "")

        if not code or not doc_a or not doc_b:
            assessment, rationale = "N", ""
            note = "missing code/doc; skipped model call"
        else:
            assessment, rationale = call_openai_pairwise(
                client=client,
                code=code,
                doc_a=doc_a,
                doc_b=doc_b,
                model=model,
                rationale_max_tokens=rationale_max_tokens,
                grade_max_tokens=grade_max_tokens,
                temperature=temperature,
            )
            note = ""

        if assessment == "A":
            cnt_A += 1
        elif assessment == "B":
            cnt_B += 1
        elif assessment == "C":
            cnt_C += 1
        else:
            cnt_N += 1

        per_item.append({
            "name": name,
            "id": b.get("id", a.get("id", None)),
            "assessment": assessment,  # A/B/C/N
            "rationale": rationale,
            "note": note,
            "code_from": a.get("code_from", "unknown"),
        })

    # 只以 A/B 计分，C/N 不参与
    effective = cnt_A + cnt_B
    frac_A = 0.5 if effective == 0 else (cnt_A / effective)
    final_score = 1 - 2 * frac_A  # ∈ [-1, 1]，越大 = 基线(doc_b)更充分

    summary = {
        "mode": "single",
        "input": input_path,
        "doc_a_key": doc_a_key,
        "doc_b_key": doc_b_key,
        "name_key": name_key,
        "code_key": code_key,
        "prompt_key": prompt_key,
        "id_key": id_key,
        "total_pairs": len(names),
        "offset": offset,
        "limit": limit,
        "count_A_anchor_better": cnt_A,
        "count_B_base_better": cnt_B,
        "count_C_comparable": cnt_C,
        "count_N_invalid": cnt_N,
        "effective_pairs_AB_only": effective,
        "score": final_score,
        "specification": (
            "A for anchor being more sufficient, B for base being more sufficient, "
            "C if comparable. Score in [-1,1]; larger means base(doc_b) more sufficient."
        ),
    }

    return {"summary": summary, "per_item": per_item}


def evaluate_pair_files(
    client: OpenAI,
    anchor_path: str,
    base_path: str,
    anchor_doc_key: str,
    base_doc_key: str,
    name_key: str = "func_name",
    code_key: str = "code",
    prompt_key: str = "prompt",
    id_key: Optional[str] = "idx",
    model: str = "deepseek-r1-250528",
    rationale_max_tokens: int = 150,
    grade_max_tokens: int = 5,
    temperature: float = 0.0,
    limit: Optional[int] = None,
    offset: int = 0,
    onetime_limit: int = 1_000_000,
) -> Dict[str, Any]:
    rows_anchor = load_json_list(anchor_path)
    rows_base   = load_json_list(base_path)

    anchor_points = points_from_rows(rows_anchor, name_key, code_key, anchor_doc_key, prompt_key, id_key)
    base_points   = points_from_rows(rows_base,   name_key, code_key, base_doc_key,   prompt_key, id_key)

    names_all = [n for n in anchor_points.keys() if n in base_points]
    names = names_all[offset : (offset + limit) if (limit is not None) else None]

    if len(names) >= onetime_limit:
        raise AssertionError(f"onetime_limit exceeded: {len(names)} vs {onetime_limit}")

    per_item: List[Dict[str, Any]] = []
    cnt_A = cnt_B = cnt_C = cnt_N = 0

    for name in tqdm(names, desc="Comparing (pair)", unit="pair"):
        a = anchor_points[name]
        b = base_points[name]
        code  = a.get("code", "")
        doc_a = a.get("doc_selected", "")
        doc_b = b.get("doc_selected", "")

        if not code or not doc_a or not doc_b:
            assessment, rationale = "N", ""
            note = "missing code/doc; skipped model call"
        else:
            assessment, rationale = call_openai_pairwise(
                client=client,
                code=code,
                doc_a=doc_a,
                doc_b=doc_b,
                model=model,
                rationale_max_tokens=rationale_max_tokens,
                grade_max_tokens=grade_max_tokens,
                temperature=temperature,
            )
            note = ""

        if assessment == "A":
            cnt_A += 1
        elif assessment == "B":
            cnt_B += 1
        elif assessment == "C":
            cnt_C += 1
        else:
            cnt_N += 1

        per_item.append({
            "name": name,
            "id": b.get("id", a.get("id", None)),
            "assessment": assessment,  # A/B/C/N
            "rationale": rationale,
            "note": note,
            "code_from": a.get("code_from", "unknown"),
        })

    effective = cnt_A + cnt_B
    frac_A = 0.5 if effective == 0 else (cnt_A / effective)
    final_score = 1 - 2 * frac_A  # ∈ [-1, 1]，越大 = 基线(base_doc_key)更充分

    summary = {
        "mode": "pair",
        "anchor": anchor_path,
        "base": base_path,
        "anchor_doc_key": anchor_doc_key,
        "base_doc_key": base_doc_key,
        "name_key": name_key,
        "code_key": code_key,
        "prompt_key": prompt_key,
        "id_key": id_key,
        "total_pairs": len(names),
        "offset": offset,
        "limit": limit,
        "count_A_anchor_better": cnt_A,
        "count_B_base_better": cnt_B,
        "count_C_comparable": cnt_C,
        "count_N_invalid": cnt_N,
        "effective_pairs_AB_only": effective,
        "score": final_score,
        "specification": (
            "A for anchor being more sufficient, B for base being more sufficient, "
            "C if comparable. Score in [-1,1]; larger means base more sufficient."
        ),
    }

    return {"summary": summary, "per_item": per_item}


# =========================
# 6) CLI 入口
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="对比两份摘要（A/B）的充分性，统计总分并输出逐条评测结果到 JSON（支持 JSON/JSONL；自动从 prompt 提取代码并去除硬提示）。"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "pair"],
        required=True,
        help="single=单文件两列对比；pair=锚点/基线双文件对比。",
    )

    # 单文件参数
    parser.add_argument("--input", help="单文件路径（mode=single）")
    parser.add_argument("--doc-a-key", help="单文件中作为 A 的字段名（如 summary）")
    parser.add_argument("--doc-b-key", help="单文件中作为 B 的字段名（如 step5_final）")

    # 双文件参数
    parser.add_argument("--anchor", help="锚点文件路径（mode=pair）")
    parser.add_argument("--base", help="基线文件路径（mode=pair）")
    parser.add_argument("--anchor-doc-key", help="锚点文件中的摘要字段名（A）")
    parser.add_argument("--base-doc-key", help="基线文件中的摘要字段名（B）")

    # 通用数据字段
    parser.add_argument("--name-key", default="func_name", help="用于配对的名称字段（默认 func_name）")
    parser.add_argument("--code-key", default="code", help="代码字段（默认 code）")
    parser.add_argument("--prompt-key", default="prompt", help="含硬提示与代码的字段名（默认 prompt）")
    parser.add_argument("--id-key", default="idx", help="可选的 id 字段名（默认 idx）")

    # OpenAI/模型
    parser.add_argument("--model", default="deepseek-v3.2-exp", help="模型名称")
    parser.add_argument("--api-key", default="PHrhcuFEefw1I0XkmZ7yiU6wMWAQ5olPAMJkIv4CVRaIwapF8OW4G3PXPX7l2eAZ5ikzu_E45mB55JuDtL7K9Q", help="API Key（默认读取环境变量 OPENAI_API_KEY）")
    parser.add_argument("--base-url", default="https://api.modelarts-maas.com/v1", help="OpenAI 兼容接口 base_url")
    parser.add_argument("--rationale-max-tokens", type=int, default=1500, help="第一轮理由最大 tokens（默认 1500）")
    parser.add_argument("--grade-max-tokens", type=int, default=5, help="第二轮打分最大 tokens（默认 5）")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度（默认 0.0）")

    # 批量/窗口与额度保护
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少对（默认全部）")
    parser.add_argument("--offset", type=int, default=0, help="从第几对开始评测（默认 0）")
    parser.add_argument("--onetime-limit", type=int, default=1000000, help="单次上限，防止失控开销")

    # 输出
    parser.add_argument("--output", required=True, help="输出结果 JSON 文件路径")

    args = parser.parse_args()

    # 客户端（优先 CLI，其次环境变量）
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 API Key：请通过 --api-key 或环境变量 OPENAI_API_KEY 提供。")

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or None
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    if args.mode == "single":
        if not (args.input and args.doc_a_key and args.doc_b_key):
            raise ValueError("single 模式需要 --input、--doc-a-key、--doc-b-key。")
        result = evaluate_single_file(
            client=client,
            input_path=args.input,
            doc_a_key=args.doc_a_key,
            doc_b_key=args.doc_b_key,
            name_key=args.name_key,
            code_key=args.code_key,
            prompt_key=args.prompt_key,
            id_key=args.id_key,
            model=args.model,
            rationale_max_tokens=args.rationale_max_tokens,
            grade_max_tokens=args.grade_max_tokens,
            temperature=args.temperature,
            limit=args.limit,
            offset=args.offset,
            onetime_limit=args.onetime_limit,
        )
    else:
        if not (args.anchor and args.base and args.anchor_doc_key and args.base_doc_key):
            raise ValueError("pair 模式需要 --anchor、--base、--anchor-doc-key、--base-doc-key。")
        result = evaluate_pair_files(
            client=client,
            anchor_path=args.anchor,
            base_path=args.base,
            anchor_doc_key=args.anchor_doc_key,
            base_doc_key=args.base_doc_key,
            name_key=args.name_key,
            code_key=args.code_key,
            prompt_key=args.prompt_key,
            id_key=args.id_key,
            model=args.model,
            rationale_max_tokens=args.rationale_max_tokens,
            grade_max_tokens=args.grade_max_tokens,
            temperature=args.temperature,
            limit=args.limit,
            offset=args.offset,
            onetime_limit=args.onetime_limit,
        )

    # 写出 JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 控制台回显
    s = result["summary"]
    print(
        f"[OK] Saved: {args.output}\n"
        f"mode={s['mode']}, pairs={s['total_pairs']}, score={s['score']:.4f}, "
        f"A={s['count_A_anchor_better']}, B={s['count_B_base_better']}, "
        f"C={s['count_C_comparable']}, N={s['count_N_invalid']}, effective={s['effective_pairs_AB_only']}"
    )


if __name__ == "__main__":
    main()
#python eval_suffic.py   --mode single   --input /home/fmy/project/DPO-Summary/data/qwen/java/train/enhanced_data_full_0_999.json  --doc-a-key summary   --doc-b-key step5_final   --output /home/fmy/project/DPO-Summary/evaluate/ProCon/sufficiencysuff_scores1000.json --limit 1000 --offset 0