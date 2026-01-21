from typing import Dict, Optional

class Verbosity():
    def __init__(self, name: str = "Verbosity", column: Optional[str] = None):
        self._name = name
        self._column = column  # 指定 JSON 字段名，如 "step1_explanation"

    def get_name(self) -> str:
        return self._name

    @staticmethod
    def _score(text: str) -> float:
        text = text or ""
        l = len(text)
        # 与原逻辑一致，顺便修正 >= 550 的边界写法
        len_score = (l > 250) * ((l < 550) * (l - 250) / 300 + (l >= 550))

        exact_rep_score = 0.0
        for substr_len, bump in ((30, 0.4), (50, 1.0)):
            if l >= substr_len:
                # 覆盖最后一个起始位置的 off-by-one 修正
                substrs = [text[i : i + substr_len] for i in range(0, l - substr_len + 1)]
                substrs = [s for s in substrs if len(s.split()) > 3]
                if len(substrs) != len(set(substrs)):
                    exact_rep_score = max(exact_rep_score, bump)

        return float(min(1.0, len_score + exact_rep_score))

    def compute(self, doc: str, other_columns: Dict) -> float:
        # 优先从 JSON 记录中读取指定字段；没有则回退到 doc
        text = other_columns.get(self._column, doc) if self._column else doc
        return self._score(text)


import json
from pathlib import Path
from typing import Iterable, Dict, List, Union

def iter_json_records(path: Union[str, Path]) -> Iterable[Dict]:
    """
    迭代给定路径中的 JSON 记录：
    - .jsonl：逐行 json.loads
    - .json ：json.load；若为列表则逐条返回，为 dict 则单条返回
    """
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for rec in data:
                yield rec
        elif isinstance(data, dict):
            yield data
        else:
            raise ValueError("Unsupported JSON top-level type (expected list or dict).")
    else:
        raise ValueError("Unsupported file extension. Use .json or .jsonl")

def evaluate_file_for_fields(
    input_path: Union[str, Path],
    fields: List[str] = ("step1_explanation", "step2_summary"),
) -> List[Dict]:
    """
    对输入文件逐条记录计算给定字段的冗长分，返回一个结果列表。
    结果中包含 idx（若存在）、func_name（若存在）以及每个字段的分数。
    """
    metrics = {field: Verbosity(name=f"Verbosity[{field}]", column=field) for field in fields}
    results = []

    for rec in iter_json_records(input_path):
        row_out = {}
        # 方便定位记录：尝试带上 idx / func_name
        if "idx" in rec:
            row_out["idx"] = rec["idx"]
        if "func_name" in rec:
            row_out["func_name"] = rec["func_name"]

        # 逐字段计算
        for field, metric in metrics.items():
            # compute 的 doc 参数不再依赖，传空字符串即可；实际会从 rec[field] 读
            row_out[f"verbosity__{field}"] = metric.compute("", rec)

        results.append(row_out)

    return results

# —— 使用示例 ——
# results = evaluate_file_for_fields("eval_data.json")
# print(results)
# 结果示例：
# [{'idx': 0, 'func_name': 'split_phylogeny',
#   'verbosity__step1_explanation': 1.0, 'verbosity__step2_summary': 0.53},
#  {'idx': 1, 'func_name': 'ensure_dir',
#   'verbosity__step1_explanation': 0.91, 'verbosity__step2_summary': 0.48}]


results = evaluate_file_for_fields(
    "/home/fmy/project/DPO-Summary/data/qwen/ruby/train/ruby_train_0_24999.json",
    fields=["summary", "step1_explanation", "step2_summary", "step3_optimised", "step5_final"]
)

# /home/fmy/project/DPO-Summary/data/qwen/java/human_train_java_0_164999.json
# /home/fmy/project/DPO-Summary/data/qwen/php/train/php_train_0_242999.json
# /home/fmy/project/DPO-Summary/data/qwen/python/train/python_human_0_249999.json
# /home/fmy/project/DPO-Summary/data/qwen/ruby/train/ruby_train_0_24999.json
# /home/fmy/project/DPO-Summary/data/qwen/go/train/go_train_0_169999.json
# /home/fmy/project/DPO-Summary/data/qwen/javascript/train/javascript_train_0_58024.json
tmp="/home/fmy/project/DPO-Summary/evaluate/ProCon/verbosity/ruby/verbosity_scores_python.jsonl"
import csv

def save_results(results: List[Dict], output_path: Union[str, Path]):
    """
    根据文件后缀将 results 写入 .jsonl / .json / .csv
    - .jsonl：每行一个 JSON
    - .json ：整个列表一次性写入
    - .csv  ：按并集字段写表头
    """
    p = Path(output_path)
    suffix = p.suffix.lower()

    p.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".jsonl":
        with p.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elif suffix == ".json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    elif suffix == ".csv":
        # 取所有键的并集，保证列齐全与顺序稳定
        fieldnames = []
        seen = set()
        for r in results:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
    else:
        raise ValueError("Unsupported output extension. Use .jsonl / .json / .csv")

save_results(results, tmp)

import json
from collections import defaultdict

def count_ones(jsonl_path: str):
    counts = defaultdict(int)
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            for k, v in rec.items():
                if k.startswith("verbosity__"):
                    try:
                        if abs(float(v) - 1.0) < 1e-12:
                            counts[k] += 1
                    except (ValueError, TypeError):
                        pass
    return total, counts

if __name__ == "__main__":
    #path = "/home/fmy/project/DPO-Summary/evaluate/ProCon/verbosity_java_scores_0_164999.jsonl"
    total, counts = count_ones(tmp)
    print(f"总记录数: {total}")
    print("每个字段 1 分出现次数:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
