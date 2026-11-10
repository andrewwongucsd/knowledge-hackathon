# Notes: a knowledge system that knows what's true *now*

**API key.** Set `ANTHROPIC_API_KEY` for grading. This uses the **paid Anthropic API**: one `claude-haiku-5-5` call (thinking off) per question that needs one, about $0.01 per 40 questions. With no key, or with `KH_NO_LLM=1`, answers are extractive and everything still runs. No other keys or network calls.

**Run** from the kit root: `pip install -r requirements.txt`, then `python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`. Unit tests need no key: `python -m unittest discover -s submission/tests`.

## How it works

**`ingest`** turns each snapshot into a view: the newest copy of every page seen up to that date (pages that left the crawl marked removed), split into table rows, fee lines and paragraphs with type, publish, effective and offer-end dates.

**`answer`** reads the question with general English and banking rules: term ("5-year" = the 60-month row), model year, amount ("under $X" is just below X), product line unless ruled out, kind of value, loan type ("7/1"), number words, abbreviations, greetings and asides, judging words ("best" is never a product), branch names from the site's own proper names. It runs BM25 over the right view, drops other tiers' and lines' rows, and searches the top 400 hits before declining. A rate question always sees its product's own page; rows of a line it didn't name (business, jumbo) come last. An introductory rate is never answered with the standard one ("intro" = "introductory"; a rate "for the first N months" is one), unless the member names the special they mean.

A rate question whose term or model year picks out one row gets that row with no model call. A "how did it change" question gets the value at each date named. Otherwise Claude answers from the filtered evidence, and guards check that every number is in the evidence, no link goes to a page missing from the crawl, a repeated statement's newest value appears, and a value quoted for a named product comes with that product's own row. If the model can't answer an open question (how / who / what-is), it picks up to 3 of the site's pages by title (a list built from the crawl) and answers once more from those (about 2% of questions, 2–3 seconds more).

**Link questions** ("link", "URL", "web address") get the live, non-article page the site's own text links for what was asked, never a calculator unless asked:
- an application portal on one of the site's uncrawled hosts counts when a live page gives it and its own words name what was asked; an "apply for X" question gets X's online application, from the evidence or X's own page (words no application of the site uses, "my husband", are not the product; a minivan or truck is a vehicle), unless it asks what applying needs;
- a link worded for another line ("a business auto loan") gives way to the site's page titled with what was asked;
- a question about joining ("apply for membership", "become / sign up as a member", opening someone's first account, or a bare "join") also reads the site's pages about joining and gets the application link they give, or the one for continuing an application. Joining something else ("the newsletter", "a board member") is not joining, and a second request in the same sentence keeps the usual pick.

**`conflicts`** is deterministic, from the newest view, and names each winner and why: one product, term and tier priced differently on two pages; boilerplate counts that disagree; "Nth time" / "Nth year in a row" claims, including ones contradicting each other on one page.

## How "current" is decided

1. **Answer from the site as it stood on the date asked:** `as_of`, or one moment the question names inside the crawl window ("as of May 2026"; "before June 2026" is the day before). An article's date doesn't move the snapshot. The model is told the date.
2. **Only the same fact competes** (product, term, tier, kind of number): tiers, rate vs APY and intro vs standard are not conflicts.
3. **Winner:** newer effective date, then newer publish date, then a re-fetched page over a stale copy, then rate and product pages over articles, promotions and calculators, then most pages.
4. **Expired offers are never used.** Losing values leave the evidence before answering.
5. **If the site doesn't say,** the answer is "That isn't listed on our website… please contact us", with no digits.

## No hard-coding

Nothing from the practice questions or this site is in the code or tests. The site's name, domains, proper names and page list come from the crawl, and the rules use general English and banking words. A unit test fails if `submission/` contains a practice value, source, answer phrase or name, or the site's identity, headings or boilerplate counts.

## Measured results

"With key" figures keep the last full keyed run for questions whose model request is unchanged and average three reruns of those whose request changed, so run-to-run luck isn't counted.

| Run | Result |
|---|---|
| Kit selftest with `--conflicts-gold` | accuracy 1.0, conflicts F1 1.0 |
| Public 40 | with key 0.946 (p95 1.8–2.3s across runs, 0 stale values, 0 errors, about $0.01); no key 0.811 |
| Battery: the 40 practice questions, 160 rewordings and 39 dated questions by independent reviewers | with key 0.938, reworded groups consistent 0.925; no key 0.780; 0 stale values, every should-refuse item declined |
| 32 held-out dated questions checked against the pages | with key 0.984; no key 0.859 |
| 12 "how did it change" questions | with key 0.90; no key 0.875 |
| 100 link, eligibility and intro-rate questions in 6 independent sets, each written against the previous version | every set improved on the version it was written against, with no question worse beyond run-to-run variation (latest set: 0.230 → 0.815) |
| Conflicts vs a 21-entry key built independently | pillar 0.716 |

## Limitations

- Model answers vary between runs, so reworded questions can disagree, and two slow API replies push p95 past 2s.
- Contradictions that exist only across old snapshots are not listed.
- Wording far from the site's can still miss the right page when the model answers a nearby question instead of declining.
- A joining question whose extra words name something other links do, when they sit in the same message as the request ("my cousin in Dubuque", "my first job"), keeps the usual pick.
- A link question that rules a line out ("for personal use, not for a business") still counts that line as asked.
