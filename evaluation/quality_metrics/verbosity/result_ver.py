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
    path = "/home/fmy/project/DPO-Summary/evaluate/ProCon/verbosity/verbosity_scores_0_164999.jsonl"
    total, counts = count_ones(path)
    print(f"总记录数: {total}")
    print("每个字段 1 分出现次数:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
