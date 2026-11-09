# Notes: a knowledge system that knows what's true *now*

**API key.** Set `ANTHROPIC_API_KEY` for grading. This uses the **paid Anthropic API**: one `claude-haiku-5-5` call (thinking off) per question that needs one, about $0.01 per 40 questions. With no key, or with `KH_NO_LLM=1`, answers are extractive and everything still runs. No other keys or network calls.

**Run** from the kit root: `pip install -r requirements.txt`, then `python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`. Unit tests need no key: `python -m unittest discover -s submission/tests`.

## How it works

**`ingest`** turns each snapshot into a view: the newest copy of every page seen up to that date (pages that left the crawl marked removed), split into table rows, fee lines and paragraphs with type, publish, effective and offer-end dates.

**`answer`** reads the qualifiers (term: a "5-year" loan is the 60-month row; model year; amount; product line unless ruled out; kind of value) and members' wording with general English rules (number words, abbreviations, greetings and asides ignored, judging words like "best" never a product, "the credit union" as the institution, branch names in any case from the site's own proper names, a loan type such as "7/1" required on the row). It runs BM25 over the right view, drops other tiers' and lines' rows, and searches the top 400 hits before declining. A rate question always sees its product's own page, and rows of a line it didn't name (business, jumbo) come last.

A question about how a value changed ("between June and September 2026", "since May 2026") gets the value at each date, from the site as it stood then; it needs both change wording and a date. A rate question whose term or model year picks out one row of the product named gets that row, with no model call. Otherwise Claude answers from the filtered evidence, and guards check that:
- every number is in the evidence;
- no link goes to a page missing from the crawl;
- a repeated statement's newest value appears;
- a value quoted for a named product comes with that product's own row, looked up without the quoted number.

If the model finds no answer to an open question (how / who / what-is), it picks up to 3 of the site's own pages by title (a list built from the crawl) and answers once more from those: only after such a decline (about 2% of questions), adding about 2–3 seconds. A "which link" question gets the live, non-article page the site's text links for it. After an API timeout or connection error, the model is skipped for a few questions, then probed once.

**`conflicts`** is deterministic, from the newest view, and names each winner and why. It finds:
- one product, term and tier priced differently on two pages (all model years in one entry);
- boilerplate counts of the same thing that disagree;
- "Nth time" and "Nth year in a row" claims, including ones contradicting each other on one page.

## How "current" is decided

1. **Answer from the site as it stood on the date asked:** `as_of`, or one moment the question names inside the crawl window ("as of May 2026"; "before June 2026" is the day before). An article's date or a side remark does not move the snapshot. The model is told the date.
2. **Only the same fact competes** (product, term, tier, kind of number): tiers and rate-vs-APY are not conflicts.
3. **Winner:** newer effective date, then newer publish date, then a re-fetched page over a stale copy, then rate and product pages over articles, promotions and calculators, then most pages.
4. **Expired offers are never used.** Losing values leave the evidence before answering.
5. **If the site doesn't say,** the answer is "That isn't listed on our website… please contact us", with no digits.

## No hard-coding

Nothing from the practice questions or this site is in the code or tests: the site's name, domains, proper names and page list are derived from the crawl; rules use general English and banking vocabulary. A unit test fails if `submission/` contains a practice value, source, answer phrase or name, or the site's identity, headings or boilerplate counts.

## Measured results

"With key" figures come from one full keyed run of this version, plus three reruns of each question whose model request changed.

| Run | Result |
|---|---|
| Kit selftest with `--conflicts-gold` | accuracy 1.0, conflicts F1 1.0 |
| Public 40 | with key 0.946 (p95 1.8–2.3s across runs, 0 stale values, 0 errors, about $0.01); no key 0.811 (previous version 0.796) |
| Battery: the 40 practice questions plus 160 rewordings and 39 dated questions by independent reviewers | with key 0.925 (previous version 0.889), reworded groups consistent 0.875 (0.70); no key 0.772 (0.748); 0 stale values |
| 32 held-out dated questions checked against the pages | with key 0.984 (previous version 0.922); no key 0.859 (0.797) |
| 12 "how did it change" questions written from the snapshots | with key 0.90 (previous version 0.08); no key 0.875 (0.29) |
| Conflicts vs a 21-entry key built independently | pillar 0.716 (previous version 0.687) |

## Limitations

- Model answers vary between runs, so reworded questions can disagree.
- With 40 questions, two slow API replies push p95 past 2s.
- Contradictions that exist only across old snapshots are not listed.
- Wording far from the site's can still miss the right page when the model answers a nearby question instead of declining ("Who can join?" answered with how to join).
