#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量计算 Triviality 分数（针对 JSON 数据集）
用法示例：
  python score_triviality_json.py     --inputs "/home/fmy/project/DPO-Summary/data/deepseek/java/enhanced_data_java_full_0_164999.json"     --output java_scores_164999111.json     --fields summary step1_explanation step2_summary step3_optimised step4_evaluation step5_final    
 --len-limit 150
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ========================
#  Minimal DatasetScheme
# ========================
class DatasetScheme:
    # 你的 JSON 中字段就是 func_name 和 code，因此直接映射
    NAME_KEY = "func_name"
    CODE_KEY = "code"

# ========================
#  Triviality（内联版）
# ========================
import itertools
import copy
import re
from collections import defaultdict

# CODE_ABBREVIATIONS：若可导入则用之，否则回退为空
try:
    from proconsul.evaluation.code_abbreviations import CODE_ABBREVIATIONS  # 可选
except Exception:
    CODE_ABBREVIATIONS = []

class Triviality():
    def __init__(self, name: str = "Triviality", trivial_len_limit: int = 150) -> None:
        super().__init__()
        self._name = name
        self.abbrs, self.common_doc_words, self.stopwords = self._get_common_wordsets()
        self.trivial_len_limit = trivial_len_limit

    def get_name(self) -> str:
        return self._name

    @staticmethod
    def _get_common_wordsets():
        abbrs = CODE_ABBREVIATIONS or []
        if isinstance(abbrs, list):
            try:
                abbrs = {d['word']: [d['abbrs'][i]['abbr'] for i in range(len(d['abbrs']))] for d in abbrs}
                abbrs = {w + 's': l + [abb + 's' for abb in l] for w, l in abbrs.items()} | abbrs
                if 'destination' in abbrs:
                    abbrs['destination'] += ['dst']
            except Exception:
                abbrs = {}
        abbrs  = defaultdict(lambda: [], abbrs)

        common_doc_words = ['return', 'get', 'given', 'input', 'output', 'value', 'information', 'to',
                            'current', 'param', 'parameter', 'example', 'true', 'false', 'see',
                            'function', 'name', 'pre', 'post', 'set', 'assign', 'assigned']
        common_doc_words += [w + 's' for w in common_doc_words]

        # 尝试使用 spacy/nltk；失败则回退到简化停用词
        try:
            import spacy
            from nltk.corpus import stopwords as nltk_stop
            stopwords = list (spacy.load("en_core_web_sm").Defaults.stop_words | set(nltk_stop.words('english')))
        except Exception:
            basic_sw = {
                'a','an','the','of','to','and','in','on','for','with','at','by','from','as',
                'is','are','be','this','that','these','those','it','its','you','your','we',
                'they','he','she','or','if','then','else','when','where','how','what','which',
                'who','whom','into','out','up','down','over','under','about','above','below',
                'between','before','after','again','further','once'
            }
            stopwords = list(basic_sw)
        stopwords = {w for w in stopwords if (len(w) > 1 or w in ['a'])}
        return abbrs, common_doc_words, stopwords

    @staticmethod
    def _split_fun_name(name: str) -> List[str]:
        presplits = name.split('_')
        splits = []
        for presplit in presplits:
            split_pos = []
            for i in range(len(presplit)):
                if i == 0:
                    split_pos += [i]
                elif presplit[i].isupper() and presplit[i - 1].islower():
                    split_pos += [i]
                elif presplit[i].isupper() and i != len(presplit) - 1 and presplit[i + 1].islower():
                    split_pos += [i]
            splits += [presplit[split_pos[i]:split_pos[i + 1]] for i in range(len(split_pos) - 1)]
            if len(split_pos) > 0:
                splits += [presplit[split_pos[-1]:]]
        for spl in copy.deepcopy(splits):
            for gr in itertools.groupby(spl, str.isdigit):
                splits += [''.join(list(gr[1]))]
        return list({spl.lower() for spl in splits}) + [name.lower()]

    def compute(self, summ: str, other_columns: Dict) -> float:
        fun_name, code = other_columns[DatasetScheme.NAME_KEY], other_columns[DatasetScheme.CODE_KEY]
        if len(summ.strip()) > self.trivial_len_limit:
            return 0
        splitted_fun_name = set(self._split_fun_name(fun_name))
        regex = re.compile('[^a-zA-Z0-9]')
        summ = summ.replace('\'s', ' ')
        splitted_doc = {w.lower() for w in regex.sub(' ', summ).split()}.difference(set(self.stopwords))
        if len(splitted_doc) < 2:
            return 1
        pre_fun_name = ' '.join(code.split('(')[0].split(' ')[:-1])
        in_brackets = code.split(')')[0].split('(')[-1]
        args_types = pre_fun_name + ' ' + in_brackets
        presplitted_args = {s for s in regex.sub(' ', args_types).split() if len(s) > 2}
        splitted_args = set(itertools.chain(*[self._split_fun_name(s) for s in presplitted_args]))

        score = 0
        digit2str = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
        for w in splitted_doc:
            if w in digit2str:
                w = digit2str[w]
            max_score = 0
            for spl in splitted_fun_name | splitted_args:
                score_mult = 1 if spl in splitted_fun_name else 0.6
                cur_score = 0
                if spl in digit2str:
                    spl = digit2str[spl]
                if w == spl:
                    cur_score = 1
                elif spl in self.abbrs[w]:
                    cur_score = 0.9
                elif len(spl) >= 3 and w.startswith(spl):
                    cur_score = 0.8
                elif len(w) >= 4 and w in spl:
                    cur_score = 0.8
                elif len(os.path.commonprefix([w, spl])) >= 3:
                    cur_score = len(os.path.commonprefix([w, spl])) / max(len(spl), len(w))
                elif w in self.common_doc_words:
                    cur_score = 0.5
                elif len(w) >= 4 and w[:4] in spl:
                    cur_score = 0.5
                elif w.startswith(spl):
                    cur_score = 0.3
                else:
                    continue
                max_score = max(max_score, cur_score * score_mult)
            score += max_score
        score /= (len(splitted_doc) - 0.5)
        adjusted_score = (score > 0.6) * ((score - 0.6) / 0.25) * (score < 0.85) + (score >= 0.85)
        return adjusted_score

# ========================
#  批量评分逻辑
# ========================

DEFAULT_FIELDS = [
    "summary",
    "step1_explanation",
    "step2_summary",
    "step3_optimised",
    "step4_evaluation",
    "step5_final",
]

def _iter_records_from_json(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
    elif isinstance(obj, dict):
        values_are_dicts = all(isinstance(v, dict) for v in obj.values()) if obj else False
        if values_are_dicts:
            for it in obj.values():
                yield it
        else:
            yield obj

def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return []
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            items = []
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return items

def safe_compute_triviality(metric: Triviality, text: Optional[str], func_name: Optional[str], code: Optional[str]) -> Optional[float]:
    try:
        summ = (text or "").strip()
        other_cols = {
            DatasetScheme.NAME_KEY: func_name or "",
            DatasetScheme.CODE_KEY: code or "",
        }
        score = metric.compute(summ, other_cols)
        return float(score) if score is not None else None
    except Exception:
        return None

def score_file(path: str, metric: Triviality, fields: List[str]) -> List[Dict[str, Any]]:
    data = _load_json_file(path)
    out_rows: List[Dict[str, Any]] = []
    for rec in _iter_records_from_json(data):
        code = rec.get("code")
        func_name = rec.get("func_name")
        idx = rec.get("idx")

        row: Dict[str, Any] = {
            "source_file": os.path.basename(path),
            "idx": idx,
            "func_name": func_name,
            "scores": {},
        }
        for field in fields:
            val = rec.get(field)
            score = safe_compute_triviality(metric, val, func_name, code)
            row["scores"][field] = score

        row["status"] = {
            f: ("missing" if rec.get(f) in (None, "") else ("ok" if row["scores"][f] is not None else "error"))
            for f in fields
        }
        out_rows.append(row)
    return out_rows

def main():
    parser = argparse.ArgumentParser(description="Batch compute Triviality scores for JSON datasets.")
    parser.add_argument("--inputs", type=str, required=True, help="输入文件的 glob 模式，例如 'data/*.json'")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 文件路径，例如 'scores.json'")
    parser.add_argument("--fields", type=str, nargs="*", default=DEFAULT_FIELDS,
                        help=f"需要评分的字段名列表（默认：{', '.join(DEFAULT_FIELDS)}）")
    parser.add_argument("--len-limit", type=int, default=150, help="Triviality 的 trivial_len_limit（默认 150）")
    args = parser.parse_args()

    metric = Triviality(trivial_len_limit=args.len_limit)

    files = sorted(glob.glob(args.inputs))
    if not files:
        raise SystemExit(f"未找到匹配的输入文件：{args.inputs}")

    all_results: List[Dict[str, Any]] = []
    for fp in files:
        all_results.extend(score_file(fp, metric, args.fields))

    result_payload: Dict[str, Any] = {
        "metric": "Triviality",
        "trivial_len_limit": args.len_limit,
        "fields": args.fields,
        "count": len(all_results),
        "results": all_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存打分到：{out_path.resolve()}")

if __name__ == "__main__":
    main()

#python score_triviality_json.py     --inputs "/home/fmy/project/DPO-Summary/data/qwen/php/train/php_train_0_242999.json"     --output php_scores_242999.json     --fields summary step1_explanation step2_summary step3_optimised step4_evaluation step5_final     --len-limit 150