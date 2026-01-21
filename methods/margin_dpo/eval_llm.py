import torch
import json
import os
import re
import sys
import math
import xml.sax.saxutils
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

import nltk
from nltk.translate.bleu_score import SmoothingFunction

# ==========================================
# 1. 配置路径
# ==========================================
base_model_path = "/data/hugginface_model/Qwen/Qwen2.5-Coder-1.5B-Instruct/"
lora_adapter_path = "/home/fmy/project/DPO-Summary/ma-enhan/qwen-dpo-funnel-lora/adapter"
valid_data_path = "/home/fmy/project/DPO-Summary/data/qwen/python/valid/python_valids_0_13913.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("[1] Load tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[2] Load base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

print("[3] Load LoRA adapter on top of base...")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("[4] Merge LoRA weights into base weights...")
merged_model = model.merge_and_unload()
model = merged_model
model.eval()

# 放 input 到模型所在 device（比固定 cuda 更稳）
_model_device = next(model.parameters()).device

print("Loading SentenceBERT...")
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)

# ==========================================
# 2. 原版 BLEU 实现（完全参考你贴的 bleu.py / compblue 逻辑）
#    这里保留你第二版自带的原版 BLEU实现，用于 bleuFromMaps
# ==========================================
nonorm = 0
preserve_case = False
eff_ref_len = "shortest"

normalize1 = [
    ('<skipped>', ''),
    (r'-\n', ''),
    (r'\n', ' '),
]
normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

normalize2 = [
    (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 '),
    (r'([^0-9])([\.,])', r'\1 \2 '),
    (r'([\.,])([^0-9])', r' \1 \2'),
    (r'([0-9])(-)', r'\1 \2 ')
]
normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]


def normalize(s):
    """Normalize and tokenize text. Lifted from NIST mteval-v11a.pl (as in original bleu.py)."""
    if nonorm:
        return s.split()
    if type(s) is not str:
        s = " ".join(s)

    for (pattern, replace) in normalize1:
        s = re.sub(pattern, replace, s)

    s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})

    s = " %s " % s
    if not preserve_case:
        s = s.lower()

    for (pattern, replace) in normalize2:
        s = re.sub(pattern, replace, s)

    return s.split()


def count_ngrams(words, n=4):
    counts = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def cook_refs(refs, n=4):
    refs = [normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return ([len(ref) for ref in refs], maxcounts)


def cook_test(test, item, n=4):
    (reflens, refmaxcounts) = item
    test = normalize(test)

    result = {}
    result["testlen"] = len(test)

    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens)) / len(reflens)
    elif eff_ref_len == "closest":
        min_diff = None
        for reflen in reflens:
            if min_diff is None or abs(reflen - len(test)) < min_diff:
                min_diff = abs(reflen - len(test))
                result['reflen'] = reflen

    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]

    result['correct'] = [0] * n
    counts = count_ngrams(test, n)
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result


def score_cooked(allcomps, n=4, ground=0, smooth=1):
    totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}

    for comps in allcomps:
        for key in ['testlen', 'reflen']:
            totalcomps[key] += comps[key]
        for key in ['guess', 'correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]

    logbleu = 0.0
    all_bleus = []
    for k in range(n):
        correct = totalcomps['correct'][k]
        guess = totalcomps['guess'][k]
        addsmooth = 0
        if smooth == 1 and k > 0:
            addsmooth = 1

        logbleu += math.log(correct + addsmooth + sys.float_info.min) - math.log(guess + addsmooth + sys.float_info.min)

        if guess == 0:
            all_bleus.append(-10000000)
        else:
            all_bleus.append(math.log(correct + sys.float_info.min) - math.log(guess))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)

    brevPenalty = min(0, 1 - float(totalcomps['reflen'] + 1) / (totalcomps['testlen'] + 1))

    for i in range(len(all_bleus)):
        if i == 0:
            all_bleus[i] += brevPenalty
        all_bleus[i] = math.exp(all_bleus[i])

    return all_bleus  # [cumulative, bleu1, bleu2, bleu3, bleu4] in 0..1


def bleu(refs, candidate, ground=0, smooth=1, n=4):
    refs = cook_refs(refs, n=n)
    test = cook_test(candidate, refs, n=n)
    return score_cooked([test], n=n, ground=ground, smooth=smooth)


def splitPuncts(line):
    # 和你原版 compblue 的 splitPuncts 一致
    return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))


def bleuFromMaps(m1, m2, n=4):
    score = [0] * (n + 1)  # 5
    num = 0.0
    for key in m1:
        if key in m2:
            bl = bleu(m1[key], m2[key][0], n=n)  # returns 0..1
            score = [score[i] + bl[i] for i in range(len(bl))]
            num += 1
    return [s * 100.0 / num for s in score]  # 转成百分比（和 compblue 一样）


# ==========================================
# 3. NLTK sentence BLEU（完全按你第一个脚本的口径）
# ==========================================
def nltk_sentence_bleu(hypothesis, reference, order=4):
    cc = SmoothingFunction()
    # 注意：这里保持和你第一段脚本一致的调用方式
    return nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method4)


# ==========================================
# 4. 推理与评估
#    BLEU 计算方式改成：与第一个脚本完全一致
# ==========================================
with open(valid_data_path, 'r', encoding='utf-8') as f:
    valid_data = json.load(f)

valid_data = valid_data[0:50]
print(f"Evaluating {len(valid_data)} samples...")

results = []
total_sbert = 0.0

# ===== 这三个就是第一段脚本的 bleu_compuate 逻辑累加器 =====
bleu_score_sum = 0.0
bleu_nltk_sum = 0.0
bleu_num = 0

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def build_qwen_prompt(code: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        "Please describe in simple english the purpose of the following Python code snippet:\n"
        f"{code}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


for idx, entry in enumerate(tqdm(valid_data)):
    code = entry['code']
    reference = entry['step2_summary']  # ground truth

    prompt = build_qwen_prompt(code)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,   # 关键：不要让 tokenizer 再自动加别的特殊符号
    )
    inputs = {k: v.to(_model_device) for k, v in inputs.items()}

    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,   # transformers 允许 list[int] / int
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    # 只取 <|im_end|> 前面的内容（确保不是把后续对话也算进去）
    generated_summary = generated_text.split("<|im_end|>")[0].strip()

    print("SUMMARY:", generated_summary)

    # ===== 原版一致的文本预处理：splitPuncts + lower =====
    ref_proc = splitPuncts(reference.strip().lower())
    hyp_proc = splitPuncts(generated_summary.strip().lower())

    # ====== BLEU：改成与第一个脚本 bleu_compuate 完全一致 ======
    predictionMap_one = {}
    goldMap_one = {}
    predictionMap_one[idx] = [hyp_proc]
    goldMap_one[idx] = [ref_proc]

    dev_bleu = round(bleuFromMaps(goldMap_one, predictionMap_one)[0], 2)  # 每条先 round 2 位
    bleu_score_sum += dev_bleu
    bleu_nltk_sum += (nltk_sentence_bleu(predictionMap_one[idx], goldMap_one[idx]) * 100)
    bleu_num += 1

    # ===== SentenceBERT cosine（保持你第二版的做法）=====
    emb1 = sbert_model.encode(generated_summary, convert_to_tensor=True)
    emb2 = sbert_model.encode(reference, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(emb1, emb2).item()
    total_sbert += cos_sim

    # ===== 详细保存（可选）：这里保存每条样本的“原版 bleu(...)”分项 =====
    # 不影响最终 bleu_score_final（最终已经按第一段脚本口径计算）
    bl = bleu([ref_proc], hyp_proc)        # 0..1
    bl_pct = [x * 100.0 for x in bl]

    results.append({
        "id": str(idx),
        "generated": generated_summary,
        "reference": reference,
        "bleu_cumulative_raw": bl_pct[0],  # 原版实现的未 round 版本（百分比）
        "bleu_1_raw": bl_pct[1],
        "bleu_2_raw": bl_pct[2],
        "bleu_3_raw": bl_pct[3],
        "bleu_4_raw": bl_pct[4],
        "bleu_v1style_per_sample": dev_bleu,  # 与第一段脚本一致的 per-sample round(.,2) 值
        "sbert": cos_sim
    })


# ==========================================
# 5. 汇总输出（与第一个脚本一致的最终打印）
# ==========================================
bleu_score_final = round(bleu_score_sum / bleu_num, 4)
bleu_nltk_final = bleu_nltk_sum / bleu_num
avg_sbert = total_sbert / len(valid_data)

print("\nFinal Evaluation Scores:")
print(f"| BLEU | NLTK-BLUE |Sentence-BERT |")
print(f"|{bleu_score_final:^6.4f}|{bleu_nltk_final:^13.4f}|{avg_sbert:^13.4f}|")

with open("eval_results_detailed.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
