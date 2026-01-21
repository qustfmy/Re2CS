import torch
import torch.nn.functional as F
import numpy as np

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

import json
from datasets import Dataset

from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # Qwen2 / Qwen2.5 常见最小可用：q_proj、v_proj（PEFT 里也有默认映射思路）:contentReference[oaicite:1]{index=1}
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)


def load_records(path: str):
    """兼容 JSONL 或 JSON array 两种格式"""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            # JSON array
            return json.load(f)
        else:
            # JSONL
            return [json.loads(line) for line in f if line.strip()]

def build_prompt_from_code(code: str) -> str:
    return (
        "Please describe the purpose of the following code:\n"
        + code
    )


def _clean_text(x):
    if x is None:
        return None
    if isinstance(x, (int, float, bool)):
        x = str(x)
    if not isinstance(x, str):
        return None
    x = x.strip()
    return x if x else None

def to_dpo_row(rec: dict, model, tokenizer):
    chosen = _clean_text(rec.get("step2_summary"))
    rejected = _clean_text(rec.get("summary"))
    code = _clean_text(rec.get("code"))

    # 任何一个为空都直接丢弃（否则 tokenize 必炸）
    if not chosen or not rejected or not code:
        return None

    prompt = "Please describe the purpose of the following code:\n" + code

    margin = calculate_funnel_margin(
        y_w_text=chosen,
        y_l_text=rejected,
        code_text=prompt,
        model=model,
        tokenizer=tokenizer,
    )

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "margin": float(margin),
        "idx": rec.get("idx", -1),
        "func_name": rec.get("func_name", ""),
    }





# ========== 1) 加载模型与分词器 ==========
model_name = "/data/hugginface_model/Qwen/Qwen2.5-Coder-1.5B-Instruct/"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else None,
).to(device)

ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else None,
).to(device)


# ========== 2) 三层漏斗动态 Margin（你原来的逻辑保留） ==========
def calculate_funnel_margin(y_w_text, y_l_text, code_text, model, tokenizer):
    # --- 层级1: 准确性检测（示例：你之后可以换成真实代码相似度/embedding cosine） ---
    score_acc_w = 0.9
    score_acc_l = 0.4
    acc_diff = score_acc_w - score_acc_l
    if acc_diff > 0.15:
        return float(1.5 * acc_diff)

    # --- 层级2: 流畅度（Virk：早期 token 几何平均概率） ---
    def get_virk_conf(text):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits  # [B, T, V]
            probs = F.softmax(logits, dim=-1)

            token_probs = []
            # 前 10 个 next-token 概率
            for i in range(min(10, inputs.input_ids.size(1) - 1)):
                p = probs[0, i, inputs.input_ids[0, i + 1]].item()
                token_probs.append(max(p, 1e-12))

            return float(np.exp(np.mean(np.log(token_probs))))

    conf_w = get_virk_conf(y_w_text)
    conf_l = get_virk_conf(y_l_text)
    if abs(conf_w - conf_l) > 0.2:
        return float(0.8 * (conf_w - conf_l))

    # --- 层级3: 简洁性 ---
    len_w = len(y_w_text)
    len_l = len(y_l_text)
    return float(0.2 * np.tanh((len_l - len_w) / 20.0))


class MarginAwareDPOTrainer(DPOTrainer):
    def __init__(self, sft_weight=0.0, *args, **kwargs):
        # 初始化父类
        super().__init__(*args, **kwargs)
        # 保存 SFT 权重
        self.sft_weight = sft_weight

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        margin = 0.0,
    ):
        """
        计算 DPO Loss (带 Margin) + SFT Loss
        """
        device = chosen_logps.device
        if not torch.is_tensor(margin):
            margin = torch.tensor(margin, device=device, dtype=chosen_logps.dtype)
        else:
            margin = margin.to(device=device, dtype=chosen_logps.dtype)

        # 1. 计算 DPO Logits
        chosen_logratios = chosen_logps - (not self.reference_free) * ref_chosen_logps
        rejected_logratios = rejected_logps - (not self.reference_free) * ref_rejected_logps
        
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = logratios - (not self.reference_free) * ref_logratios

        # 2. 加入 Margin
        z = self.beta * logits - margin

        # 3. 计算原始 DPO Loss (Sigmoid)
        ls = getattr(self, "label_smoothing", 0.0)
        dpo_losses = -F.logsigmoid(z) * (1 - ls) - F.logsigmoid(-z) * ls

        # 4. ✅ 加入 SFT Loss
        # SFT 目标是最大化 chosen 序列的概率，即最小化 -log(P(chosen))
        # chosen_logps 已经是 log(P(chosen|x))
        sft_losses = -chosen_logps
        
        # 最终 Loss = DPO_Loss + coeff * SFT_Loss
        if self.sft_weight > 0:
            total_losses = dpo_losses + self.sft_weight * sft_losses
        else:
            total_losses = dpo_losses

        # 计算 Rewards (用于日志)
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        return total_losses, chosen_rewards, rejected_rewards, sft_losses

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        margin = batch.get("margin", 0.0)

        # 注意：这里接收 4 个返回值，增加了 sft_losses
        losses, chosen_rewards, rejected_rewards, sft_losses = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            margin=margin,
        )

        loss = losses.mean()

        # 日志指标
        metrics["loss"] = loss.detach()
        metrics["rewards/chosen"] = chosen_rewards.mean().detach()
        metrics["rewards/rejected"] = rejected_rewards.mean().detach()
        metrics["rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().detach()
        metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).mean().detach()
        
        # ✅ 记录 SFT Loss 方便观察
        if self.sft_weight > 0:
            metrics["loss/sft"] = sft_losses.mean().detach()
            metrics["loss/dpo"] = (losses - self.sft_weight * sft_losses).mean().detach()

        return loss, metrics
    
# ====== 用法示例：从文件读取 ======
data_path = "/home/fmy/project/DPO-Summary/data/qwen/python/train/python_human_0_249999.json"   # 改成你的路径
records = load_records(data_path)

N = 2000
rows = []
bad = 0

for r in records:
    row = to_dpo_row(r, model, tokenizer)
    if row is None:
        bad += 1
        continue
    rows.append(row)
    if len(rows) >= N:
        break

train_dataset = Dataset.from_list(rows)
print("loaded:", len(rows), "dropped:", bad)


print("loaded:", len(rows), "dropped:", bad)
train_dataset = Dataset.from_list(rows)
print(train_dataset.column_names)
print(train_dataset[0])
# train_dataset=train_dataset[0:50000]
# ========== 5) 训练：beta 放到 DPOConfig 里；processing_class 用 tokenizer ==========
SFT_LOSS_WEIGHT = 0.5 

training_args = DPOConfig(
    output_dir="./qwen-dpo-funnel-sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # 建议调大梯度累积，因为 SFT 需要较稳的梯度
    learning_rate=5e-6,
    logging_steps=1,
    bf16=(device == "cuda"),
    remove_unused_columns=False,
    beta=0.1,
)

trainer = MarginAwareDPOTrainer(
    sft_weight=SFT_LOSS_WEIGHT,  # ✅ 传入权重
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

trainer.model.save_pretrained("./qwen-dpo-funnel-sft/adapter")
tokenizer.save_pretrained("./qwen-dpo-funnel-sft/adapter")
# import os, json

# meta = {
#     "data_path": data_path,
#     "total_records": total,
#     "valid_records": len(rows),
#     "dropped_records": len(bad_ids),
# }

# os.makedirs(save_dir, exist_ok=True)
# with open(os.path.join(save_dir, "data_stats.json"), "w", encoding="utf-8") as f:
#     json.dump(meta, f, ensure_ascii=False, indent=2)
# print("[SAVE] wrote:", os.path.join(save_dir, "data_stats.json"))

