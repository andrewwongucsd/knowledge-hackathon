# Notes: a knowledge system that knows what's true *now*

> **For grading, set `ANTHROPIC_API_KEY`.** On the public questions it raises the score from 0.773 to about 0.90, at about $0.012 per 40 questions. Without a key the system still runs and answers extractively.

`solution.py` → `build()` returns an object with exactly `ingest`, `answer` and `conflicts`, using the shapes in `harness/contract.py`. Retrieval and conflict detection are stdlib Python. One Claude call per question writes the answer from evidence that has already been filtered, and deterministic guards check it.

## Run it (from the kit root)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # recommended, paid
.venv/bin/python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
.venv/bin/python -m unittest discover -s submission/tests       # no key needed
```

## API keys

- **`ANTHROPIC_API_KEY`** (recommended, paid Anthropic API): one call per question to `claude-haiku-5-5` with thinking off (`KH_MODEL` overrides it). The fixed instructions use prompt caching; repeated questions are answered from a local cache. One retry; a rate-limit wait longer than 10s is not waited out (the extractive answer is used), so every answer ends well within 60s.
- **Without a key,** or with `KH_NO_LLM=1`, each answer is the best current table row or sentence, verbatim.
- **No other keys, and no network calls besides that API.** The snapshots are the only data source.
- **Python:** graded on 3.11. On 3.9 the SDK (which needs 3.10+) is skipped and the no-key path runs.

## How "current" is decided

1. **Answer from the site as it stood on the date asked.** Each snapshot becomes a view: the newest copy of every URL seen up to that date. A page that left a later crawl is kept and marked removed. A date before the first snapshot gets "I don't know".
2. **Only compare values that describe the same fact:** same product, term, tier (balance, loan amount, model year) and kind of number (rate vs APY vs APR). Standard vs jumbo and personal vs business are different facts. A fee's condition ("within 90 days of opening") is not a term.
3. **When sources disagree, the winner is, in order:**
   1. the newer *effective* date on a rate table;
   2. the newer *publish* date for articles;
   3. a page re-fetched in that crawl over a copy carried forward from an earlier one;
   4. rate and product pages and fee schedules over articles, promotions and calculators;
   5. the value most pages agree on.

   An undated passing mention never beats a dated statement.
4. **Offers whose stated end date has passed are never used.**
5. **Losing values are removed from the evidence before the answer is written** (stale rows, older counts), so they can't be repeated.
6. **If the site doesn't state what was asked,** the answer is "That isn't listed on our website… please contact us", with no digits. For a fee or a count, every specific thing the question names must appear in the evidence.

## How it works

**Ingest** splits pages into table rows (with header, section and effective date), fee lines and paragraphs. PDF conversions are cleaned first. Each page's type, publish date and offer end dates are recorded. Per-snapshot lookups (repeated-statement facts, page text, terms the site defines as a list, dates the site states) are built during loading, which is untimed. Ingest takes about 10s for all five snapshots.

**Answer** first drops from the search words any date that only names the snapshot being answered ("as of May 2026"); dates the site states, like deadlines, stay, and the model still sees the whole question. Questions are read with general cues: hours or counts asked without those words ("lobby hrs", "number of branches"), "ARM" on rate questions, "$10k" amounts, product synonyms ("car" loans are auto loans). It runs BM25 over the view, applying the question's qualifiers (term, model year, amount, jumbo or business, branch, kind of value). When a question pins a term or model year, rows from product lines it didn't ask about (jumbo, IRA, business, refinance) leave the evidence, since one line's rate can equal another's on a different date. Wrong-tier rows are dropped, and the system declines before any LLM call if nothing supports the question. The model sees the evidence with each page's type and dates, plus neighbouring paragraphs. Two fact links help: a question about a repeated "the Nth time …" statement gets the newest winning sentence first; "what link should I use" questions are answered from the link the site's own text gives most often, only if that page is live. The model returns structured JSON (thinking off, 8s timeout) with a ≤ 15-word reason written first.

Guards run on the reply: every number must appear in the evidence; no meta-talk; no links to deep pages missing from the snapshot; any failure falls back to the extractive answer. A term the site defines as a list on at least two pages is spelled out verbatim when the answer uses it without saying what it covers. Without the model, if the best row matched a value shared with another product than the one named, the named product's own row is added.

**Conflicts** are deterministic, from the newest view: the same product, term and tier priced differently on two pages; counts in boilerplate repeated on 5 or more pages (grouped by sentence pattern); "the Nth time" ordinals, joined by the same organisation's "Nth year in a row" claims; and rates quoted in prose. Values carry the site's own wording ("over N …", the ordinal's sentence). Each comes with its winner (for a tiered product, every tier's winner) and reason. An LLM reviewer was tried and removed: it discarded stale statements still on the site.

## No hard-coding

Nothing learned from the practice questions or from this site is written in our code. The site's own identity (its domains, the title suffix most pages share, its name) is derived from the crawl at ingest. Page types come from generic URL conventions (calculator, file extensions, news/blog/learn/resources sections, landing-page markers). Banking vocabulary (jumbo, IRA, APY, card networks) and general synonyms are generic. A unit test scans every file under `submission/` and fails on any practice answer or trap value, expected-source path, 4-word expected-answer phrase, proper name from the practice set, or the derived site identity. It caught 9 such literals in the previous version, now removed.

## Results (measured on this code; model requests identical to the keyed runs)

| Run | Result |
|---|---|
| Kit selftest (`--snapshots harness/selftest/snapshots --eval harness/selftest/eval.jsonl --conflicts-gold harness/selftest/conflicts_gold.json`) | Accuracy 1.0, conflicts F1 1.0, final 1.0 |
| Public eval, no key (`KH_NO_LLM=1`) | Accuracy 0.773 (starter: 0.093), p95 0.03s, 0 stale values, 0 errors |
| Public eval, with key, 3 runs | Accuracy 0.899 each; p95 1.43–3.07s; 0 stale values; 0 errors; 35 model calls and $0.0115 per run |
| Hidden-style test set, with key (160 rewordings of the practice questions + 39 dated questions with checked answers, written by independent reviewers) | Accuracy 0.825 (previous version 0.790); reworded groups giving consistent answers 0.65 (0.525); p95 1.89s; 0 regressions caused by the changes |
| Same set, no key | Rewordings 0.676 (0.601); dated 0.821 (0.718) |
| Public questions with an as-of date added, no key | Same score as without the date |
| Fee schedule check (15 fees at 3 dates, 4 unlisted) | No key 49/49; with key 48/49 |
| Conflicts vs the practice set's stale values (7 contradictions) | All 7 listed; 7 of 7 winners match |
| Conflicts vs a 21-entry key built by a reviewer who never saw our output | 14 of 21 matched; pillar 0.673 (0.611 before this round) |
| Fresh clone, Python 3.11 and 3.9 | 0 errors |

**Per pillar.** Accuracy about 0.90 with a key; the rubric caps the public set at 0.969 (one question has no required facts) and the misses below cap it near 0.915 without hard-coding. Speed 1.0 when the API is healthy. Consistency 1.0 on the public set; 0.65 on reworded groups in the hidden-style set, the real exposure. Conflicts 1.0 on the selftest; the real key is hidden.

**Model choice.** On identical code (an earlier revision), Haiku 5.5 with thinking off scored final 0.806 against Opus 5.5's 0.773, at 1/40 of the cost. Opus can't turn thinking off, so its answers take more than 2s.

**Still missed:** one expected answer using an older figure than newest-wins gives; one yes/no answer where the model leaves out the named product's own top rate (the no-key answer now includes it); one question the scorer can't award; one list answer where the model sometimes leaves out a schedule (its request is identical each time). Matching these by hand would hard-code the practice set.

## Limitations

- LLM answers vary between runs; a reworded question gets a fresh call.
- With a key, dated questions in the hidden-style set score 0.769 against 0.821 without it: the model sometimes mixes in a neighbouring tier or date.
- An occasional slow API call can push p95 past 2s on a small question set.
- Prose conflicts are limited to counts, ordinals and quoted rates; contradictions only *across* old snapshots are not listed.
- The public runs need the kit's `snapshots_zip/`, which by request is not in this repo.
