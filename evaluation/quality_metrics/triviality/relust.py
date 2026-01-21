#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
两种输入模式的“计 1 次数”统计脚本：

A) scores.json（来自批量 Triviality 打分脚本的输出）
   - 自动识别顶层含 "results" 列表
   - 逐条读取 item["scores"]，统计每个字段 == 1.0 的次数
   - 可用 --per-file 按 source_file 细分

B) JSONL（通用逐行 JSON）
   - 使用 --prefix（默认 "verbosity__"）过滤键
   - 键值转 float 后与 1.0 比较（容差 --eps）
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def _isclose1(x: float, eps: float) -> bool:
    try:
        return math.isfinite(x) and abs(float(x) - 1.0) < eps
    except Exception:
        return False


def count_from_scores_json(path: str, eps: float, per_file: bool):
    """
    读取形如：
    {
      "results": [
        {
          "source_file": "a.json",
          "idx": 0,
          "func_name": "...",
          "scores": {"summary": 1.0, "step1_explanation": 0.3, ...}
        },
        ...
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    total = len(results)

    field_counts: Dict[str, int] = defaultdict(int)
    per_file_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rec in results:
        scores = rec.get("scores", {}) or {}
        src = rec.get("source_file") or "<unknown>"
        for k, v in scores.items():
            if v is None:
                continue
            if _isclose1(v, eps):
                field_counts[k] += 1
                if per_file:
                    per_file_counts[src][k] += 1

    return total, field_counts, per_file_counts


def count_from_jsonl(path: str, eps: float, prefix: str):
    """
    通用 JSONL 统计：仅统计键名以 prefix 开头的项里，值 == 1.0 的次数。
    """
    field_counts: Dict[str, int] = defaultdict(int)
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            for k, v in rec.items():
                if not isinstance(k, str) or not k.startswith(prefix):
                    continue
                try:
                    if _isclose1(float(v), eps):
                        field_counts[k] += 1
                except Exception:
                    pass

    return total, field_counts


def main():
    ap = argparse.ArgumentParser(description="统计各字段中取值为 1.0 的出现次数")
    ap.add_argument("--input", required=True, help="输入文件路径：scores.json 或 JSONL 文件")
    ap.add_argument("--eps", type=float, default=1e-12, help="判断等于 1.0 的容差（默认 1e-12）")
    ap.add_argument("--prefix", type=str, default="verbosity__",
                    help="JSONL 模式下用于过滤键名的前缀（默认 verbosity__）")
    ap.add_argument("--per-file", action="store_true",
                    help="在 scores.json 模式下按 source_file 细分统计")
    args = ap.parse_args()

    in_path = args.input
    suffix = Path(in_path).suffix.lower()

    # 尝试判断是否为 scores.json（顶层有 results）
    is_scores_json = False
    if suffix == ".json":
        try:
            with open(in_path, "r", encoding="utf-8") as f:
                probe = json.load(f)
            is_scores_json = isinstance(probe, dict) and "results" in probe
        except Exception:
            is_scores_json = False

    if is_scores_json:
        total, counts, per_file_counts = count_from_scores_json(in_path, args.eps, args.per_file)
        print(f"总记录数: {total}")
        print("每个字段 1 分出现次数:")
        for k in sorted(counts.keys()):
            print(f"  {k}: {counts[k]}")

        if args.per_file:
            print("\n按 source_file 细分：")
            for src in sorted(per_file_counts.keys()):
                print(f"- {src}")
                for k in sorted(per_file_counts[src].keys()):
                    print(f"    {k}: {per_file_counts[src][k]}")
    else:
        # 走 JSONL 逻辑（或无法解析成 scores.json 的普通 JSON 当作 JSONL 逐行尝试）
        total, counts = count_from_jsonl(in_path, args.eps, args.prefix)
        print(f"总记录数: {total}")
        print(f"每个字段 1 分出现次数（前缀={args.prefix!r}）：")
        for k in sorted(counts.keys()):
            print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()

#每个字段 1 分出现次数:
#   step2_summary: 3
#   step3_optimised: 13
#   summary: 49


#  python  step2_summary: 717
#   step3_optimised: 2229
#   step5_final: 241
#   summary: 15994


#python ./relust.py --input java_scores_164999111.json