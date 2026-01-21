from __future__ import annotations
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
import concurrent.futures

STEP1_PROMPT = """
You are a senior software engineer.Carefully read the following code and produce a detailed, technical explanation
using exactly this structure (use bullet points where helpful): 1. Functional Overview - One sentence describing the core purpose of the code in plain, precise language. 2. Dependencies - List imported modules, external libraries, and key helper functions the code relies on. - Briefly note what each dependency is used for. 3. Detailed Behaviour - Describe input types and shapes (arguments, data structures, important defaults). - Explain step-by-step how the code transforms inputs into outputs. - Highlight any control flow (loops, branches, early returns, exception handling). 4. Usage Example - If the public API or main entry function is clear, provide one realistic example of how a developer would call it (in code). - Include sample argument values and show the expected kind of result.
5. Considerations - Edge Cases: list non-obvious inputs or states and how the code handles them. - Performance: mention time/space complexity and any potential bottlenecks. - Robustness & Pitfalls: mention assumptions, error conditions, and common ways a user of this code could misuse it. Constraints: - Do not rewrite or copy the code itself. - Do not speculate beyond what can be reasonably inferred from the code.
### Code
{code}
"""

STEP2_PROMPT = """
You previously produced a detailed technical explanation of this Python function.
Now generate a single concise technical summary sentence that a professional
developer could read in isolation and still understand the function’s purpose.
Follow these rules:
1. Describe the core functionality in active voice.
2. Use appropriate domain-specific terminology (e.g., I/O, caching, batching, async, etc.).
3. Avoid implementation details (no mention of loops, specific variables, or low-level steps).
4. The summary must be under 18 words.
5. Base your summary only on the behaviour described in your previous explanation
   and what is directly implied by the code; do not invent functionality.
6. Do not mention that you are summarizing or referencing previous steps; just output the sentence.
Output:
- Only the final one-sentence summary, nothing else.
"""


STEP3_PROMPT = """
Here is an official reference summary of the same function:{truth_summ}
Here is your previous concise summary:{step2_summ}
Revise your previous concise summary so that it:
- Maximizes agreement with the official reference summary on behaviour and scope.
- Preserves any additional correct nuances from your original concise summary.
- Corrects any inaccuracies, over-generalizations, or missing key aspects.
- Remains a single sentence under 18 words.
- Uses clear, professional technical language in active voice.
- Does not mention that it is revised or based on a reference.

Output:
- Only the final revised one-sentence summary, nothing else.
"""


STEP4_PROMPT = """
You will see:
- A Python function
- One reference summary (intended ground truth)
- Two candidate summaries

Your task:
1. Treat all three summaries as candidates to be scored:
   - Reference (intended ground truth)
   - Candidate A (STEP2)
   - Candidate B (STEP3)
2. Score each one on:
   - Accuracy (1–5): faithfulness to actual code behaviour and edge cases.
   - Fluency  (1–5): clear, natural, unambiguous technical language.
   - Coverage (1–5): captures core points and important constraints.
3. Briefly justify each candidate’s scores (1–2 sentences).
4. Return a Markdown table with:
   Candidate, Accuracy, Fluency, Coverage, Average, Comment.

Scoring guide (1–5):
- 5: Excellent, no significant issues.
- 4: Minor issues only.
- 3: Mixed; some important aspects are unclear or missing.
- 2: Major issues; not reliable alone.
- 1: Misleading or mostly incorrect.

Be concise; comments 1–2 sentences.

### Code
{code}

### Summaries
- Reference: {truth_summ}
- Candidate A: {step2_summ}
- Candidate B: {step3_summ}

Output format:

| Candidate   | Accuracy | Fluency | Coverage | Average | Comment |
|------------|----------|---------|----------|---------|---------|
| Reference  |          |         |          |         |         |
| Candidate A|          |         |          |         |         |
| Candidate B|          |         |          |         |         |
"""


STEP5_PROMPT = """
You have:
- The Python code.
- A reference summary.
- Two candidate summaries with evaluation scores and comments in a Markdown table.

Using all this information, craft a final polished one-sentence summary of the code.

Requirements:
1. Maximize accuracy and coverage of the actual code behaviour.
2. Prefer wording that is as good or better than the highest-scoring candidate.
3. The summary must be:
   - One sentence
   - Under 15 words
   - In active voice
   - Clear, precise, and professional.
4. Do not mention evaluations, scores, or that this is a summary; just state the behaviour.

Output:
- Only the final one-sentence summary, nothing else.
"""

def load_dataset(file_path: str, *, output_dir: str | Path = "./cache") -> List[Dict[str, str]]:
    cache_file = Path(output_dir) / f"{Path(file_path).stem}.pkl"
    if cache_file.is_file():
        try:
            with cache_file.open("rb") as fp:
                print(f"🔄  Loading cache {cache_file}")
                return pickle.load(fp)
        except Exception:
            print("⚠️  Cache unreadable – rebuilding …")

    data: List[Dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            js = json.loads(line)
            data.append({
                "code": " ".join(js["code_tokens"]).replace("\n", " "),
                "summary": " ".join(js["docstring_tokens"]).replace("\n", " "),
                "func_name": js.get("func_name", ""),
            })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as fp:
        pickle.dump(data, fp)
    print(f"💾  Cached dataset → {cache_file}")
    return data

# ──────────────────────────────────────────────────────────────────────────────
# LLM helper with retry mechanism
# ──────────────────────────────────────────────────────────────────────────────
import re

def _call_llm(
    client: OpenAI,
    *,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        content = resp.choices[0].message.content or ""

        import re
        content = re.sub(r"\s+", " ", content).strip()

        return content
    except Exception as e:
        print(f"❌  Error with model {model}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Five-step chain with concurrent requests
# ──────────────────────────────────────────────────────────────────────────────

from typing import Dict, List
from openai import OpenAI  

CODE_MAX_CHARS = 2000  #

def run_chain(
    code: str,
    truth_summary: str,
    *,
    client: OpenAI,
    model_chat: str,
    model_reasoner: str,
) -> Dict[str, str]:
    """
    Run a 5-step summarization & evaluation chain over a code snippet.

    model_chat:    
    model_reasoner: 
    """
    convo: List[Dict[str, str]] = []
    results: Dict[str, str] = {}

    truncated_code = code[:CODE_MAX_CHARS]

    # --- Step 1: 结构化技术解释 ---
    convo.append({
        "role": "user",
        "content": STEP1_PROMPT.format(code=truncated_code),
    })
    step1_resp = _call_llm(
        client,
        messages=convo,
        model=model_chat,
        max_tokens=256,
    )
    results["step1"] = step1_resp
    convo.append({"role": "assistant", "content": step1_resp or ""})

    # --- Step 2: 基于上一轮解释的简洁摘要（<18 词） ---
    convo.append({
        "role": "user",
        "content": STEP2_PROMPT,
    })
    step2_resp = _call_llm(
        client,
        messages=convo,
        model=model_chat,
        max_tokens=64,
    )
    results["step2"] = step2_resp
    convo.append({"role": "assistant", "content": step2_resp or ""})

    # --- Step 3: 用官方参考摘要对齐修正 ---
    convo.append({
        "role": "user",
        "content": STEP3_PROMPT.format(
            truth_summ=truth_summary,
            step2_summ=step2_resp,
        ),
    })
    step3_resp = _call_llm(
        client,
        messages=convo,
        model=model_chat,
        max_tokens=64,
    )
    results["step3"] = step3_resp
    convo.append({"role": "assistant", "content": step3_resp or ""})

    # --- Step 4: 打分评估（Accuracy / Fluency / Coverage） ---
    convo.append({
        "role": "user",
        "content": STEP4_PROMPT.format(
            code=truncated_code,
            truth_summ=truth_summary,
            step2_summ=step2_resp,
            step3_summ=step3_resp,
        ),
    })
    step4_resp = _call_llm(
        client,
        messages=convo,
        model=model_reasoner,
        max_tokens=3096,
    )
    results["step4"] = step4_resp
    convo.append({"role": "assistant", "content": step4_resp or ""})

    # --- Step 5: 最终精炼的一句话摘要（<15 词） ---
    convo.append({
        "role": "user",
        "content": STEP5_PROMPT,
    })
    step5_resp = _call_llm(
        client,
        messages=convo,
        model=model_reasoner,
        max_tokens=2048,
    )
    results["step5"] = step5_resp

    return {
        "step1_explanation": results.get("step1", ""),
        "step2_summary": results.get("step2", ""),
        "step3_optimised": results.get("step3", ""),
        "step4_evaluation": results.get("step4", ""),
        "step5_final": results.get("step5", ""),
    } 


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Five‑step Python code summariser (v6)")
    p.add_argument("--data", required=True, help="CodeSearchNet‑style JSONL file")
    p.add_argument("--output-dir", default="./data/kimi/java", help="Where to save outputs")
    p.add_argument("--start-index", type=int, default=0, help="Row index to start (default: 0)")
    p.add_argument("--max-samples", type=int, default=10, help="Max rows to process (default: 1000)")
    p.add_argument("--model_chat", default="DeepSeek-V3", help="Model for Steps 1‑4 (default: DeepSeek-V3)")
    p.add_argument("--model_reasoner", default="DeepSeek-R1", help="Model for Step 5 (default: DeepSeek-R1)")
    p.add_argument("--api-key", default="", help="fmy OpenAI API key (or env)")
    p.add_argument("--api-base", default="https://api.modelarts-maas.com/v1", help="API base URL")
    return p.parse_args(argv)

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("❌  Provide an API key via --api-key or OPENAI_API_KEY env var")

    client = OpenAI(api_key=api_key, base_url=args.api_base)

    full_data = load_dataset(args.data, output_dir=args.output_dir)
    if not full_data:
        sys.exit("❌  Empty dataset – exiting")

    start, end = args.start_index, args.start_index + args.max_samples
    data_slice = full_data[start:end]
    print(f"✅  Processing rows {start}–{end-1} (n={len(data_slice)})")

    enriched: List[Dict[str, str]] = []

    for idx, rec in enumerate(data_slice, start=start):
        print(f"… Row {idx}")
        res = run_chain(
            rec["code"],
            rec["summary"],
            client=client,
            model_chat=args.model_chat,
            model_reasoner=args.model_reasoner,
        )
        rec.update(res)
        rec["idx"] = idx
        enriched.append(rec)

    slice_id = f"{start}_{end-1}"
    out_path = Path(args.output_dir) / f"enhanced_data_full_{slice_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(enriched, fp, ensure_ascii=False, indent=2)
    print(f"💾  Saved → {out_path.resolve()}")

    print("\n🖨️  Example record:\n" + json.dumps(enriched[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
