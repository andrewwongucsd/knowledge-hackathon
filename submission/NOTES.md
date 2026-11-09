# Notes: a knowledge system that knows what's true *now*

**API key.** Set `ANTHROPIC_API_KEY` for grading. This uses the **paid Anthropic API**: one `claude-haiku-5-5` call (thinking off) per question that needs one, about $0.01 per 40 questions. With no key, or with `KH_NO_LLM=1`, answers are extractive and everything still runs. No other keys, and no network calls besides that API. On Python 3.9 the SDK is skipped.

**Run** from the kit root:
- `pip install -r requirements.txt`
- `python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`
- Unit tests, which need no key: `python -m unittest discover -s submission/tests`

## How it works

**`ingest`** turns each snapshot into a view: the newest copy of every page seen up to that date. Pages that left the crawl are kept and marked removed. Each page is split into table rows (header, section, effective date), fee lines and paragraphs, with its type, publish date and offer end dates.

**`answer`** reads the question's qualifiers:
- term: a "5-year" loan is the 60-month row;
- model year and amount;
- product line: jumbo, business or IRA, unless ruled out ("not the jumbo");
- the kind of value asked for.

It then runs BM25 over the right view, drops rows for other tiers or lines, and searches the top 400 hits before declining.

A rate question whose pinned term or model year picks out one row of the product named gets that row (and the matching rows of its table) with no model call. Otherwise Claude answers from the filtered evidence, and guards check the result:
- every number must be in the evidence;
- no links to pages missing from the crawl;
- a repeated statement's newest value must appear;
- a member quoting a value for a named product also gets that product's own row.

After an API timeout or connection error, the model is skipped for the next few questions, then probed once on a short timeout.

**`conflicts`** is deterministic, from the newest view. It finds:
- one product, term and tier priced differently on two pages (all model years in one entry);
- boilerplate counts that disagree;
- "Nth time" and "Nth year in a row" claims, including ones contradicting each other on one page.

Each conflict names its winner and why.

## How "current" is decided

1. **Answer from the site as it stood on the date asked:** `as_of`, or one moment the question names inside the crawl window ("as of May 2026", "back in June 2026"; "before June 2026" means the day before). A date that dates a change ("changed to on…", "effective…") or an article does not move the snapshot. Nor does a side remark in a present-tense question, or a date when the best evidence is an article the older snapshot doesn't have yet. The model is told that date.
2. **Only the same fact competes:** same product, term, tier and kind of number. Tiers and rate-vs-APY are not conflicts.
3. **Winner:**
   1. newer effective date;
   2. newer publish date;
   3. re-fetched page over a stale copy;
   4. rate and product pages over articles, promotions and calculators;
   5. most pages.
4. **Expired offers are never used.** Losing values leave the evidence before answering.
5. **If the site doesn't say,** the answer is "That isn't listed on our website… please contact us", with no digits.

## No hard-coding

Nothing from the practice questions or this site is in the code. The site's name, domains and title suffix are derived from the crawl; page types come from generic URL conventions. A unit test scans every file under `submission/` and fails on any of:
- a practice value, tier amount, expected-source path, expected-answer phrase or practice proper name;
- the derived site identity;
- the site's own headings or the counts its boilerplate repeats.

## Measured results

| Run | Result |
|---|---|
| Kit selftest with `--conflicts-gold` | accuracy 1.0, conflicts F1 1.0 |
| Public 40, with key | accuracy 0.946, final 0.946 (p95 1.9s), 0 stale values, 0 errors, 30 model calls, $0.010 |
| Public 40, no key | accuracy 0.796, p95 0.04s |
| 239 reworded and dated questions written by independent reviewers, with key | accuracy 0.889 (previous version 0.825), reworded groups consistent 0.70 (0.65), p95 1.8s, 0 stale values |
| 32 held-out dated questions checked against the pages | with key 0.922; no key 0.797 (previous version 0.516) |
| Conflicts vs a 21-entry key built independently | pillar 0.687 (previous version 0.673) |

The last adjustment changed 9 of the 239 model requests. Each was re-checked: deterministic and correct, or 3 of 3 correct with the key.

## Limitations

- Model answers vary between runs, so reworded questions that reach the model can disagree.
- With 40 questions, two slow API replies push p95 past 2s (measured speed 0.94–1.0).
- Contradictions that exist only across old snapshots are not listed.
