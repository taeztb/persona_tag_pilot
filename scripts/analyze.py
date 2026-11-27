"""Reproduce the pilot's numbers from the frozen files in data/.

Run: python3 scripts/analyze.py
Prints tag rates, Qwen-vs-judge precision/recall (self-tag v2 primary, v1 secondary)
and Cohen's kappa against the 25-item blind human check, with and without the six
items Claude adjudicated.
"""
import json
from pathlib import Path

DATA = Path("data")
TAGS = ["sycophancy", "overconfidence", "empathy", "playfulness"]


def jsonl(name):
    return [json.loads(line) for line in (DATA / name).read_text("utf-8").splitlines() if line]


def v1_tags(row):
    # 37 v1 outputs gave evidence as a plain string; their single tag is still counted.
    return row["tags"] if row["parse_status"] == "ok" else json.loads(row["raw_output"])["tags"]


def kappa(a, b, ids, tag):
    n = len(ids)
    both = sum(tag in a[i] and tag in b[i] for i in ids)
    a_only = sum(tag in a[i] and tag not in b[i] for i in ids)
    b_only = sum(tag not in a[i] and tag in b[i] for i in ids)
    neither = n - both - a_only - b_only
    po = (both + neither) / n
    pe = ((both + a_only) * (both + b_only) + (b_only + neither) * (a_only + neither)) / n ** 2
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main():
    category = {r["candidate_id"]: r["sampling_category"] for r in jsonl("selected_prefixes_v1.jsonl")}
    judge = {r["candidate_id"]: r["tags"] for r in jsonl("judge_glm_53_flash_v1.jsonl")}
    qwen = {"v2": {r["candidate_id"]: r["tags"] for r in jsonl("self_tags_v2.jsonl")},
            "v1": {r["candidate_id"]: v1_tags(r) for r in jsonl("self_tags_v1.jsonl")}}
    ids = sorted(judge)
    print(f"Evaluated replies: {len(ids)}\n")

    print("Replies tagged (Qwen v2 / judge), by sampling category")
    for cat in ["sycophancy", "overconfidence", "empathy", "playfulness", "neutral"]:
        sub = [i for i in ids if category[i] == cat]
        cells = "  ".join(f"{t} {sum(t in qwen['v2'][i] for i in sub)}/{sum(t in judge[i] for i in sub)}" for t in TAGS)
        print(f"  {cat:15s} n={len(sub):2d}  {cells}")

    for version in ["v2", "v1"]:
        print(f"\nQwen self-tags {version} vs judge (precision = both/Qwen, recall = both/judge)")
        for t in TAGS:
            both = sum(t in qwen[version][i] and t in judge[i] for i in ids)
            q = sum(t in qwen[version][i] for i in ids)
            j = sum(t in judge[i] for i in ids)
            print(f"  {t:15s} precision {both}/{q}  recall {both}/{j}")

    answers = json.loads((DATA / "human_check_answers_v1.json").read_text("utf-8"))["answers"]
    human = {a["candidate_id"]: a["tags"] for a in answers}
    adjudicated = {k: list(v) for k, v in human.items()}
    for item in json.loads((DATA / "claude_adjudication_v1.json").read_text("utf-8"))["items"]:
        tags = [t for t in adjudicated[item["candidate_id"]] if t not in ("sycophancy", "overconfidence")]
        tags += [t for t in ("sycophancy", "overconfidence") if item[t]]
        adjudicated[item["candidate_id"]] = tags
    hids = list(human)
    print(f"\nCohen's kappa vs blind human labels (n={len(hids)}): researcher only / with adjudication")
    for t in TAGS:
        print(f"  {t:15s} judge {kappa(judge, human, hids, t):.2f} / {kappa(judge, adjudicated, hids, t):.2f}"
              f"   qwen v2 {kappa(qwen['v2'], human, hids, t):.2f} / {kappa(qwen['v2'], adjudicated, hids, t):.2f}")


if __name__ == "__main__":
    main()
