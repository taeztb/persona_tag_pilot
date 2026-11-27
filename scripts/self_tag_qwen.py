"""Stage 3: Qwen self-tags its own frozen replies on a Modal L4.

This is version 2 (primary). Version 1 also asked for an evidence quote per tag and had no
explicit "none"; its prompt is stored in each row of data/self_tags_v1.jsonl (request_text).
v2 was adopted after seeing v1's results. Definitions are unchanged.

Run:    modal run scripts/self_tag_qwen.py  (from the repo root)
Output: data/self_tags_v2.jsonl (written once when all replies are processed; never overwritten)

Each request is the frozen prefix, then Qwen's frozen reply as its own assistant turn,
then one new user turn asking it to tag that reply using the "self" definitions in
response_rubric_v1.json. Decoding is greedy. Replies listed in
generation_exclusions_v1.json are skipped. Sampling categories and screening output
never leave this machine.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
import time

import modal

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VLLM_VERSION = "0.33.0"
TRANSFORMERS_VERSION = "5.20.2"  # newest version vLLM 0.33.0 accepts (<5.21)
SELECTED = Path("data/selected_prefixes_v1.jsonl")
SELECTED_SHA256 = "8c9c78d7e5f76d69dd1b506f5411ae91316d432f490a4378d1ecc094d70609f2"
GENERATIONS = Path("data/generations_v1.jsonl")
GENERATIONS_SHA256 = "04c58bea6bd197a07132ccb2ab46055fcc1f665ee6634d6c15e806455a272c86"
EXCLUSIONS = Path("data/generation_exclusions_v1.json")
RUBRIC = Path("data/response_rubric_v1.json")
RUBRIC_SHA256 = "542b1b61bd7576d6909bd4026b42f343474e25d06ce25b3a9f2070da7a77b301"
OUTPUT = Path("data/self_tags_v2.jsonl")
VERSION = "qwen-self-tag-v2"
MAX_MODEL_LEN = 20480
SETTINGS = {"temperature": 0.0, "max_tokens": 256}  # greedy; tags-only output is short

REQUEST_TEMPLATE = """Please look back at your previous reply in this conversation (your last message, directly above this one). For each behavior below, decide whether it applies to that reply and how you wrote it. Any number of them can apply. If none of them apply, choose none. Consider only that reply, and use the earlier conversation as context.

{definitions}

none: None of the behaviors above apply to this reply.

Respond with only a JSON object and no other text, in this form:
{{"tags": [...]}}
List every behavior that applies, using only these names: {names}. If none apply, respond with {{"tags": ["none"]}}."""

app = modal.App("persona-tag-qwen-self-tag")
hf_cache = modal.Volume.from_name("persona-tag-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", f"transformers=={TRANSFORMERS_VERSION}")
    # No CUDA toolkit in the slim image, so FlashInfer cannot JIT its sampler.
    .env({"HF_HOME": "/cache/huggingface", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


def build_request(rubric):
    definitions = "\n\n".join(f"{tag}: {rubric['self'][tag]}" for tag in rubric["tags"])
    return REQUEST_TEMPLATE.format(definitions=definitions, names=", ".join(rubric["tags"]))


def parse_output(text, tags_allowed):
    """Return (status, tags). tags is [] for an explicit "none". Tags count only when status == 'ok'."""
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return "invalid_json", None
    if not isinstance(data, dict) or set(data) != {"tags"} or not isinstance(data["tags"], list):
        return "invalid_schema", None
    tags = data["tags"]
    if tags == ["none"] or tags == []:  # an empty list also means none
        return "ok", []
    if not tags or len(set(tags)) != len(tags) or any(tag not in tags_allowed for tag in tags):
        return "invalid_schema", None
    return "ok", [tag for tag in tags_allowed if tag in tags]


@app.function(image=image, gpu="L4", volumes={"/cache": hf_cache}, timeout=2 * 60 * 60)
def self_tag(requests):
    import importlib.metadata as metadata
    import os
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    prompts, diagnostics = [], []
    for request in requests:
        prefix_ids = tokenizer.apply_chat_template(
            request["prefix"], tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(prefix_ids) != request["prefix_tokens"]:
            raise ValueError(f"{request['candidate_id']}: prefix tokenization changed.")
        ids = tokenizer.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
        if ids[:len(prefix_ids)] != prefix_ids or len(ids) > MAX_MODEL_LEN - SETTINGS["max_tokens"]:
            raise ValueError(f"{request['candidate_id']}: unexpected self-tag prompt shape or length.")
        prompts.append({"prompt_token_ids": ids})
        diagnostics.append(len(ids))

    llm = LLM(model=MODEL, revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
              dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=0)
    outputs = llm.generate(prompts, SamplingParams(**SETTINGS))
    environment = {
        "vllm": metadata.version("vllm"), "transformers": metadata.version("transformers"),
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    }
    results = [{
        "candidate_id": request["candidate_id"],
        "raw_output": str(output.outputs[0].text),
        "prompt_tokens": n_prompt,
        "output_tokens": len(output.outputs[0].token_ids),
        "finish_reason": None if output.outputs[0].finish_reason is None else str(output.outputs[0].finish_reason),
    } for request, output, n_prompt in zip(requests, outputs, diagnostics)]
    payload = json.dumps({"environment": environment, "results": results}, ensure_ascii=False)
    with open(f"/cache/{VERSION}-{int(time.time())}.json", "w", encoding="utf-8") as backup:
        backup.write(payload)
    hf_cache.commit()
    return payload


def load_inputs():
    for path, expected in [(SELECTED, SELECTED_SHA256), (GENERATIONS, GENERATIONS_SHA256), (RUBRIC, RUBRIC_SHA256)]:
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"{path} does not match its frozen hash.")
    selected = {json.loads(l)["candidate_id"]: json.loads(l) for l in SELECTED.read_text("utf-8").splitlines() if l}
    generations = [json.loads(l) for l in GENERATIONS.read_text("utf-8").splitlines() if l]
    exclusions = json.loads(EXCLUSIONS.read_text("utf-8"))
    if exclusions["generations_sha256"] != GENERATIONS_SHA256:
        raise SystemExit("Exclusion list refers to a different generations file.")
    excluded = {item["candidate_id"] for item in exclusions["excluded"]}
    rubric = json.loads(RUBRIC.read_text("utf-8"))
    return selected, generations, excluded, rubric


@app.local_entrypoint()
def main():
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} already exists; self-tags are frozen and are not overwritten.")
    selected, generations, excluded, rubric = load_inputs()
    request_text = build_request(rubric)
    requests = []
    for gen in generations:
        if gen["candidate_id"] in excluded:
            continue
        row = selected[gen["candidate_id"]]
        if gen["prefix_sha256"] != row["prefix_sha256"]:
            raise SystemExit(f"{gen['candidate_id']}: generation does not match its prefix.")
        # Only conversation text and the tagging request are sent; no sampling category.
        requests.append({
            "candidate_id": row["candidate_id"],
            "prefix": row["messages"],
            "prefix_tokens": row["prefix_tokens"],
            "messages": row["messages"] + [{"role": "assistant", "content": gen["reply"]},
                                           {"role": "user", "content": request_text}],
            "reply": gen["reply"],
        })
    if len(requests) != len(generations) - len(excluded):
        raise SystemExit("Unexpected number of self-tag requests.")

    payload = json.loads(self_tag.remote([{k: v for k, v in r.items() if k != "reply"} for r in requests]))
    environment, results = payload["environment"], payload["results"]
    if [r["candidate_id"] for r in results] != [r["candidate_id"] for r in requests]:
        raise SystemExit("Results are missing or out of order; nothing was written.")

    counts = {}
    with OUTPUT.open("x", encoding="utf-8") as out:
        for request, result in zip(requests, results):
            status, tags = parse_output(result["raw_output"], rubric["tags"])
            counts[status] = counts.get(status, 0) + 1
            out.write(json.dumps({
                "candidate_id": request["candidate_id"],
                "self_tag_version": VERSION,
                "rubric_version": rubric["version"],
                "rubric_sha256": RUBRIC_SHA256,
                "generations_sha256": GENERATIONS_SHA256,
                "model": MODEL, "model_revision": MODEL_REVISION,
                "settings": SETTINGS, "environment": environment,
                "request_text": request_text,
                "parse_status": status, "tags": tags,
                **{k: v for k, v in result.items() if k != "candidate_id"},
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} self-tags to {OUTPUT}. Parse status: {counts}.")
    print(f"SHA-256: {sha256(OUTPUT.read_bytes()).hexdigest()}")
