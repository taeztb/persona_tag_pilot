# Persona self-tagging pilot

Can a small model recognize behaviors in a reply it just wrote? We took 100 real WildChat conversations, had `Qwen3-4B-Instruct-2507` write the next reply, and asked Qwen to tag its own reply for four behaviors: sycophancy, overconfidence, empathy and playfulness. A blind judge (GLM-5.3-Flash) tagged the same replies, and a human checked the judge on 25 of them.

**Result:** Qwen's self-tags barely track behavior. It tags empathy on 94 of 98 replies and sycophancy on 2; the judge tags them on 16 and 40. Playfulness is the only tag where Qwen and the judge agree, and the human reviewer disagreed with both there. Run `python3 scripts/analyze.py` for every number.

## Files

| Path | What it is |
| --- | --- |
| `scripts/sample_wildchat.py` | Samples conversation prefixes from WildChat-1M (seed 42, pinned revisions) |
| `scripts/screen_wildchat.py` | GLM-5.3 proposes each prefix's behavior opportunity (frozen prompt `opportunity-v3`) |
| `scripts/filter_empathy_topup.py` | Keyword filter used to find more empathy prefixes |
| `scripts/generate_qwen.py` | Generates Qwen's replies on a Modal L4 |
| `scripts/self_tag_qwen.py` | Qwen tags its own replies |
| `scripts/judge_replies.py` | GLM-5.3-Flash tags the same replies, blind |
| `scripts/analyze.py` | Prints all reported results |
| `data/screening-v3.sqlite3` | Screening proposals for 2,600 candidates |
| `data/review_decisions.jsonl` | Accept/skip decision and reason for each proposal (`reviewer`: human or Claude) |
| `data/selected_prefixes_v1.jsonl` | The 100 selected prefixes and their sampling category |
| `data/generations_v1.jsonl` | Qwen's 100 replies |
| `data/generation_exclusions_v1.json` | 2 replies excluded for repetition loops (98 evaluated) |
| `data/response_rubric_v1.json` | The four behavior definitions (Qwen's and judge's wording) |
| `data/self_tags_v2.jsonl` | Qwen's self-tags (primary) |
| `data/self_tags_v1.jsonl` | Earlier self-tags that also asked for evidence quotes (secondary) |
| `data/judge_glm_53_flash_v1.jsonl` | Judge's tags |
| `data/human_check_answers_v1.json` | Blind human tags for 25 replies |
| `data/claude_adjudication_v1.json` | Claude's calls on 6 human items marked "can't tell" |

Sample composition: 25 each for sycophancy, overconfidence and playfulness; 15 empathy (10 natural, 5 via the keyword filter); and 10 neutral controls. Each later script checks the SHA-256 of its inputs and never overwrites its output.

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python3 scripts/analyze.py
```

`analyze.py` needs only the files in `data/`. Re-running the other stages needs a Modal account (`modal setup`). Screening and judging also need `MODAL_PROXY_TOKEN` set in your shell. The candidate stream isn't stored; `sample_wildchat.py --count 9000` recreates it exactly.

## Caveats

**Small samples.** There are only 25 human labels (κ ±0.3) and 15 empathy prefixes.

**Judge and reviewer.** There is one judge, from the same model family as the screener. After the first six decisions, prefix review was done by Claude. Six human labels were adjudicated by Claude, not blind.

**Choices made after seeing results:**

- empathy target cut from 25 to 15, with neutral controls and the keyword filter added
- 2 looping replies excluded
- self-tag format v2 adopted after v1
- relaxed parsing of 37 v1 outputs

## Data notice

`data/` contains WildChat-1M conversation text, so check the dataset card's license before sharing outside the team.
