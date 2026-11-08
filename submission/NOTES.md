# Notes: a knowledge system that knows what's true *now*

> **For grading, set `ANTHROPIC_API_KEY`.** On the public questions it raises the final score from 0.726 to about 0.90 (4 runs), at about $0.012 per 40 questions. Without a key the system still runs and answers extractively.

`solution.py` → `build()` returns an object with exactly `ingest`, `answer` and `conflicts`, using the shapes in `harness/contract.py`. Retrieval and conflict detection are stdlib Python. One Claude call per question writes the answer from evidence that has already been filtered, and deterministic guards check it.

## Run it (from the kit root)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # recommended, paid
.venv/bin/python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
.venv/bin/python -m unittest discover -s submission/tests       # no key, no snapshots needed
```

## API keys

- **`ANTHROPIC_API_KEY`** (recommended, paid Anthropic API): one call per question to `claude-haiku-5-5` with thinking off (`KH_MODEL` overrides it). The fixed instructions use prompt caching, and repeated questions are answered from a local cache.
- **Without a key,** or with `KH_NO_LLM=1`, each answer is the best current table row or sentence, verbatim.
- **No other keys, and no network calls besides that API.** The snapshots are the only data source.
- **Python:** graded on 3.11. On 3.9 the SDK (which needs 3.10+) is skipped and the no-key path runs.

## How "current" is decided

1. **Answer from the site as it stood on the date asked.** Each snapshot becomes a view: the newest copy of every URL seen up to that date. A page that left a later crawl is kept and marked removed. A date before the first snapshot gets "I don't know".
2. **Only compare values that describe the same fact:** same product, term, tier (balance, loan amount, model year) and kind of number (rate vs APY vs APR). Standard vs jumbo and personal vs business are different facts. A fee's condition ("within six months of closing") is not a term.
3. **When sources disagree, the winner is, in order:**
   1. the newer *effective* date on a rate table;
   2. the newer *publish* date for articles;
   3. a page re-fetched in that crawl over a copy carried forward from an earlier one;
   4. rate and product pages and fee schedules over articles, promotions and calculators;
   5. the value most pages agree on.

   An undated passing mention never beats a dated statement.
4. **Offers whose stated end date has passed are never used.**
5. **Losing values are removed from the evidence before the answer is written** (stale rows, older counts), so they can't be repeated.
6. **If the site doesn't state what was asked,** the answer is "That isn't listed on our website… please contact us", with no digits. For a fee or a count, every specific thing the question names (a place, a service) must appear in the evidence.

## How it works

**Ingest** splits pages into table rows (with header, section and effective date), fee lines and paragraphs. PDF conversions are cleaned first: HTML entities are decoded and placeholder `ColN` headers dropped. Each page's type, publish date and offer end dates are recorded. Per-snapshot lookups (repeated-statement facts, page text, terms the site defines as a list, dates the site states) are built during loading, which is untimed, so answers don't pay for them. Ingest takes about 10s for all five snapshots.

**Answer** first drops from the search words any date that only names the snapshot being answered ("as of May 2026", "today (October 2026)"); dates the site states, like deadlines, stay, and the model still sees the whole question. It runs BM25 over the view, applying the question's qualifiers (term, model year, amount, jumbo or business, branch, kind of value). Wrong-tier rows are dropped, and the system declines before any LLM call if nothing supports the question. The model sees the evidence with each page's type and dates, plus the neighbouring paragraphs and the best paragraphs a per-page cap left out. For "what … do you offer" questions it also sees business and other product lines. Two small fact links help: a question about something a repeated statement counts ("the Nth time …") gets the newest winning sentence first, with pages stating the beaten value dropped; and "what link should I use" questions are answered without the model, from the link the site's own text gives most often for that subject, only if that page is live. The model returns structured JSON (no sampling parameters, thinking off, 8s timeout, ≤ 1 retry) with a ≤ 15-word reason written first.

Guards run on the reply: every number must appear in the evidence; no meta-talk; no links to deep pages missing from the snapshot; and any failure falls back to the extractive answer. If the answer uses a term the site defines as a list on at least two pages (a service area, say) without saying what it covers, the site's definition is added verbatim and its page cited first.

**Conflicts** are deterministic, from the newest view: the same product, term and tier priced differently on two pages; counts in boilerplate repeated on 5 or more pages (grouped by sentence pattern, with the short form "N branches" alongside the site's wording); "the Nth time" ordinals; and rates quoted in prose. Each comes with its winner (for a tiered product, every tier's winner) and reason. An LLM reviewer was tried and removed: it discarded stale statements still on the site.

## Results (measured on this code)

| Run | Result |
|---|---|
| Kit selftest (`--snapshots harness/selftest/snapshots --eval harness/selftest/eval.jsonl --conflicts-gold harness/selftest/conflicts_gold.json`) | No key and with key: accuracy 1.0, conflicts F1 1.0, final 1.0 |
| Public eval, no key (`KH_NO_LLM=1`) | Accuracy 0.726 (starter: 0.093), p95 0.03s, 0 stale values, 0 errors |
| Public eval, with key, 4 runs | Accuracy 0.899 / 0.899 / 0.915 / 0.899; final 0.899 / 0.875 / 0.915 / 0.899; p95 1.55–2.07s, except 4.81s in one run (two slow API calls); 0 stale values; 0 errors; 35 model calls and $0.0115 per run (about 35k input tokens read from the prompt cache) |
| Same questions over the 4 runs | 37 of 40 all right or giving the same numbers each time |
| Public questions with an as-of date added ("as of September 2026", "(October 2026)", …), no key | Same score as without the date (0.717 on the 39 undated questions); before this rule, 0.684–0.693 |
| Fee schedule check (local questions written from the fee PDF: 15 fees at 3 dates, 4 unlisted) | No key 49/49; with key 48/49 |
| Conflicts vs the practice set's stale values (7 same-fact contradictions its wrong answers point to) | All 7 listed; 7 of 7 winners match; pillar 0.726 by the harness formula |
| Fresh clone, no snapshots | 0 errors |

**Per pillar.** Accuracy 0.90–0.915: the rubric caps it at 0.969 (one practice question has no required facts; only `--judge` can award it), and the misses below cap it near 0.915 without hard-coding. Speed 1.0 in 3 of 4 runs (0.84 in the run with two slow API calls). Consistency 1.0 on the public set (rewordings can only be checked with repeat runs). Conflicts 1.0 on the selftest; the real answer key is hidden, so the practice set's stale values are the only local check.

**Model choice.** On identical code (an earlier revision), Haiku 5.5 with thinking off scored final 0.806 against Opus 5.5's 0.773, at 1/40 of the cost. Opus can't turn thinking off, so its answers take more than 2s. The default was picked by a rule fixed before the runs.

**Still missed, and why it is left:** one expected answer using an older figure than newest-wins gives (only listing several years' numbers would match it); one yes/no answer that leaves out the named product's own top rate (only hiding true rows would fix it); one question the scorer can't award; and one list answer where the model sometimes leaves out one of two schedules (1.0 in half the runs; its request is identical each time). **Tried and removed** (no gain in 2 of 3 runs): prompt rules for yes/no answers, links and "same or different", and sending yes/no rate questions to Opus (no gain, nearly double the cost). Matching these by hand would hard-code the practice set.

## Limitations

- LLM answers vary between runs; a reworded question gets a fresh call.
- p95 is 1.5–2.1s with a key, so a slow network or an occasional slow API call (3 of 140 calls took about 4.8s) costs some of the speed score.
- Without a key, list and multi-part questions score partially.
- Prose conflicts are limited to counts, ordinals and quoted rates; contradictions only *across* old snapshots are not listed.
- The public runs need the kit's `snapshots_zip/`, which by request is not in this repo.
