"""Stage 1: sample English multi-turn WildChat prefixes before behavior screening.

Install: python -m pip install datasets==5.2.0 transformers==5.24.1 jinja2==3.1.6
Run:     python scripts/sample_wildchat.py --count 9000 > data/candidates-9000.jsonl
The first run downloads and caches the public dataset and the tokenizer.
Full commit SHAs can be used with HF_HUB_OFFLINE=1 once both are cached.
To extend the pool, rerun with a larger --count and the same seed and revisions.
Each JSON line records the resolved revisions. Only messages go to the screener.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

DATASET = "allenai/WildChat-1M"
DATASET_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def choose_prefix(turns, rng, tokenizer, budget):
    """Choose uniformly among eligible endpoints and return only the visible history."""
    languages = Counter()
    messages, endpoints = [], []
    for i, turn in enumerate(turns):
        role = "assistant" if i % 2 else "user"
        content = turn.get("content")
        if turn.get("role") != role or not isinstance(content, str) or not content.strip():
            break
        messages.append({"role": role, "content": content})
        if turn.get("language"):
            languages[turn["language"]] += 1
        # Estimate the dialogue language through this endpoint only. Human review
        # confirms English and can retain code and occasional foreign terms.
        if (role == "user" and i >= 2
                and i + 1 < len(turns) and turns[i + 1].get("role") == "assistant"
                and languages["English"] > sum(languages.values()) / 2):
            endpoints.append(i)

    rng.shuffle(endpoints)
    for i in endpoints:
        prefix = messages[:i + 1]
        tokens = tokenizer.apply_chat_template(
            prefix, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(tokens) <= budget:
            return i, prefix, len(tokens)
    return None


def sample(dataset, tokenizer, args, provenance):
    rng = random.Random(args.seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    seen_roots, seen_hashes, seen_prefixes = set(), set(), set()
    for row_index in order:
        row = dataset[row_index]
        turns = row["conversation"]
        if not turns or len(turns) < 3:
            continue
        root = turns[0].get("turn_identifier")
        digest = row["conversation_hash"]
        if digest in seen_hashes or root in seen_roots:
            continue
        selection = choose_prefix(turns, rng, tokenizer, args.max_prefix_tokens)
        if selection is None:
            continue
        i, messages, n_tokens = selection
        prefix = tuple((message["role"], message["content"]) for message in messages)
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        seen_hashes.add(digest)
        if root is not None:
            seen_roots.add(root)
        yield {**provenance, "seed": args.seed, "max_prefix_tokens": args.max_prefix_tokens,
               "row_index": row_index, "conversation_hash": digest,
               "user_turn_index": i, "prefix_tokens": n_tokens, "messages": messages}
        if len(seen_hashes) == args.count:
            return
    raise ValueError(f"Only {len(seen_hashes)} eligible conversations were available")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prefix-tokens", type=int, default=4096)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()
    if min(args.count, args.max_prefix_tokens) < 1:
        parser.error("count and max-prefix-tokens must be positive")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    api = HfApi()
    data_revision = args.dataset_revision
    if not re.fullmatch(r"[0-9a-f]{40}", data_revision):
        data_revision = api.dataset_info(DATASET, revision=data_revision).sha
    model_revision = args.model_revision
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        model_revision = api.model_info(MODEL, revision=model_revision).sha
    provenance = {"dataset": DATASET, "dataset_revision": data_revision,
                  "tokenizer": MODEL, "tokenizer_revision": model_revision}
    # Select columns after loading so the standard WildChat cache can be reused.
    dataset = load_dataset(DATASET, split="train", revision=data_revision, keep_in_memory=False)
    # Datasets can fall back to its newest cached copy even when given a revision.
    if not dataset.cache_files or any(
        Path(entry["filename"]).parent.name != data_revision for entry in dataset.cache_files
    ):
        raise ValueError(f"WildChat cache does not match the requested revision {data_revision}")
    dataset = dataset.select_columns(["conversation_hash", "conversation"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=model_revision)
    for record in sample(dataset, tokenizer, args, provenance):
        print(json.dumps(record, ensure_ascii=False))
    print(f"Wrote {args.count} candidates.", file=sys.stderr)


if __name__ == "__main__":
    main()
