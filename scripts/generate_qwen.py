"""Stage 2: generate one Qwen reply per frozen prefix on a Modal L4.

Install locally:  python -m pip install modal==1.6.4   (then `modal setup` once)
Run (from the repo root): modal run scripts/generate_qwen.py
Output:           data/generations_v1.jsonl (written only when all 100 replies succeed; never overwritten)

The prompt is each prefix's visible messages rendered with Qwen's chat template and
add_generation_prompt=True, with no system prompt. No sampling category, rubric, or
screening output is sent to the model. Settings follow the model card at the pinned
revision (checked by the user) with neutral penalties.
"""
from hashlib import sha256
import json
from pathlib import Path
import time

import modal

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VLLM_VERSION = "0.33.0"
# vLLM 0.33.0 requires transformers>=5.16.1,<5.21.0, so the GPU image cannot use the
# sampler's 5.24.1. The chat template and tokenizer files come from the pinned model
# revision, and generate() checks every prompt length against the sampler's count.
TRANSFORMERS_VERSION = "5.20.2"
SELECTED = Path("data/selected_prefixes_v1.jsonl")
SELECTED_SHA256 = "8c9c78d7e5f76d69dd1b506f5411ae91316d432f490a4378d1ecc094d70609f2"
OUTPUT = Path("data/generations_v1.jsonl")
GENERATION_VERSION = "qwen-generation-v1"
MAX_MODEL_LEN = 20480  # longest prefix is 4,038 tokens + 16,384 output tokens
SETTINGS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 16384,
}

app = modal.App("persona-tag-qwen-generation")
hf_cache = modal.Volume.from_name("persona-tag-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", f"transformers=={TRANSFORMERS_VERSION}")
    # The slim image has no CUDA toolkit, so FlashInfer cannot JIT-compile its top-k/top-p
    # sampler. Use vLLM's PyTorch-native sampler instead; the sampling settings are unchanged.
    .env({"HF_HOME": "/cache/huggingface", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


def seed_for(candidate_id):
    """Deterministic per-request seed derived from the candidate ID."""
    return int(sha256(candidate_id.encode()).hexdigest()[:8], 16)


@app.function(image=image, gpu="L4", volumes={"/cache": hf_cache}, timeout=3 * 60 * 60)
def generate(requests):
    import importlib.metadata as metadata
    import os
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    prompts, params = [], []
    for request in requests:
        prompt_ids = tokenizer.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        # The sampler measured prefix_tokens the same way; a mismatch means the prompt changed.
        if len(prompt_ids) != request["prefix_tokens"]:
            raise ValueError(f"{request['candidate_id']}: prompt has {len(prompt_ids)} tokens, "
                             f"expected {request['prefix_tokens']}.")
        prompts.append({"prompt_token_ids": prompt_ids})
        params.append(SamplingParams(**SETTINGS, seed=request["seed"]))

    llm = LLM(model=MODEL, revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
              dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=0)
    outputs = llm.generate(prompts, params)
    hf_cache.commit()

    environment = {
        "vllm": metadata.version("vllm"),
        "transformers": metadata.version("transformers"),
        "torch": str(torch.__version__),  # TorchVersion would need torch to unpickle locally
        "gpu": torch.cuda.get_device_name(0),
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    }
    results = []
    for request, output in zip(requests, outputs):
        if list(output.prompt_token_ids) != prompts[len(results)]["prompt_token_ids"]:
            raise ValueError(f"{request['candidate_id']}: vLLM changed the prompt tokens.")
        completion = output.outputs[0]
        results.append({
            "candidate_id": request["candidate_id"],
            "reply": str(completion.text),
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(completion.token_ids),
            "finish_reason": None if completion.finish_reason is None else str(completion.finish_reason),
        })
    # Return plain JSON so the local side never needs torch or vLLM to read the result,
    # and keep a copy on the volume in case the local side fails after generation.
    payload = json.dumps({"environment": environment, "results": results}, ensure_ascii=False)
    with open(f"/cache/{GENERATION_VERSION}-{int(time.time())}.json", "w", encoding="utf-8") as backup:
        backup.write(payload)
    hf_cache.commit()
    return payload


@app.local_entrypoint()
def main():
    data = SELECTED.read_bytes()
    if sha256(data).hexdigest() != SELECTED_SHA256:
        raise SystemExit(f"{SELECTED} does not match the frozen selection hash.")
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} already exists; generations are frozen and are not overwritten.")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    if len(rows) != 100 or len({row["candidate_id"] for row in rows}) != 100:
        raise SystemExit("Expected 100 distinct frozen prefixes.")
    for row in rows:
        digest = sha256(json.dumps(row["messages"], ensure_ascii=False).encode()).hexdigest()
        if digest != row["prefix_sha256"] or row["messages"][-1]["role"] != "user":
            raise SystemExit(f"{row['candidate_id']}: prefix does not match its frozen hash.")
    # Only the visible messages leave this machine; sampling category and screening stay local.
    requests = [{"candidate_id": row["candidate_id"], "messages": row["messages"],
                 "prefix_tokens": row["prefix_tokens"], "seed": seed_for(row["candidate_id"])}
                for row in rows]

    payload = json.loads(generate.remote(requests))
    environment, results = payload["environment"], payload["results"]
    if [r["candidate_id"] for r in results] != [r["candidate_id"] for r in requests]:
        raise SystemExit("Results are missing or out of order; nothing was written.")

    with OUTPUT.open("x", encoding="utf-8") as out:
        for row, request, result in zip(rows, requests, results):
            out.write(json.dumps({
                "candidate_id": row["candidate_id"],
                "prefix_sha256": row["prefix_sha256"],
                "generation_version": GENERATION_VERSION,
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "settings": SETTINGS,
                "seed": request["seed"],
                "system_prompt": None,
                "environment": environment,
                **{k: v for k, v in result.items() if k != "candidate_id"},
            }, ensure_ascii=False) + "\n")
    lengths = sum(r["finish_reason"] == "length" for r in results)
    print(f"Wrote {len(results)} replies to {OUTPUT}. Hit the 16,384-token cap: {lengths}.")
    print(f"SHA-256: {sha256(OUTPUT.read_bytes()).hexdigest()}")
