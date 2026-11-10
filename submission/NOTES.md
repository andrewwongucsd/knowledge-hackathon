# Notes: a knowledge system that knows what's true *now*

**API key.** Set `ANTHROPIC_API_KEY` for grading. This uses the **paid Anthropic API**: one `claude-haiku-5-5` call (thinking off) per question that needs one, about $0.01 per 40 questions. With no key, or with `KH_NO_LLM=1`, answers are extractive and everything still runs. No other keys or network calls.

**Run** from the kit root: `pip install -r requirements.txt`, then `python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`. Unit tests need no key: `python -m unittest discover -s submission/tests`.

## How it works

**`ingest`** turns each snapshot into a view: the newest copy of every page seen up to that date (pages that left the crawl marked removed), split into table rows, fee lines and paragraphs with type, publish, effective and offer-end dates. A table under an in-page tab takes the tab's name when its own heading says less (a plain "Savings Rates" heading under a "Health Savings Rates" tab), and a heading that names no kind of product gets the tab's product (a brand name under a money-market tab, a one-word "Regular" under a certificates tab).

**`answer`** reads the question with general English and banking rules: term ("5-year" = the 60-month row), model year, amount ("under $X" is just below X, "over $X" just above; a monthly amount, a payment, an income or a down payment is not a balance; an amount below a row's minimum doesn't fit it; nested tiers give the tightest), product line unless ruled out ("not the jumbo", "not for my business"), kind of value, loan type ("7/1"), number words, abbreviations ("HSA" rows are written "Health Savings Account"), greetings and asides, judging words ("best" is never a product), branch names from the site's own proper names. An aside after the question that only tells about the member ("…? I've been a member for 10 years", "…? I keep my savings with you") adds no product, term, model year or line; one that says what is meant ("…? I'm looking at the 24-month term", "Five years", "Also the 36-month") still counts, and a term in the question itself is never overridden. It runs BM25 over the right view, drops other tiers' and lines' rows, and searches the top 400 hits before declining. A rate question always sees its product's own page; rows of a line it didn't name (business, jumbo) come last. An introductory rate is never answered with the standard one ("intro" = "introductory"; a rate "for the first N months" is one), unless the member names the special they mean.

A rate question whose term, model year or amount picks out one row gets that row with no model call. Sending a wire is never the incoming-wire fee. A "how did it change" question gets the value at each date named. Otherwise Claude answers from the filtered evidence, and guards check that every number is in the evidence, no link goes to a page missing from the crawl, a repeated statement's newest value appears, and a value quoted for a named product comes with that product's own row. If the model can't answer an open question (how / who / what-is), it picks up to 3 of the site's pages by title (a list built from the crawl) and answers once more from those (about 2% of questions, 2–3 seconds more).

**Link questions** ("link", "URL", "web address") get the live, non-article page the site's own text links for what was asked, never a calculator unless asked:
- an application portal on one of the site's uncrawled hosts counts when a live page gives it and its own words name what was asked; an "apply for X" question gets X's online application, from the evidence or X's own page (words no application of the site uses, "my husband", are not the product; a minivan or truck is a vehicle; "personal" may sit anywhere in the address once another word picks the product), unless it asks what applying needs;
- a link worded for another line ("a business auto loan") gives way to the site's page titled with what was asked;
- a question about joining ("apply for membership", "become / sign up as a member", opening someone's first account, or a bare "join") also reads the site's pages about joining and gets the application link they give, or the one for continuing an application. Joining something else ("the newsletter", "a board member") is not joining, and a second request in the same sentence keeps the usual pick. In a message of several sentences, the others are context when the request is plainly about joining ("My cousin just moved here. Can you send me the link so she can join?"), unless they name an event, a product line or one of the site's own pages, or the request also asks to apply for something. How the link is passed on ("email it to her") is never what it is for.

**`conflicts`** is deterministic, from the newest view, and names each winner and why: one product, term and tier priced differently on two pages; boilerplate counts that disagree; "Nth time" / "Nth year in a row" claims, including ones contradicting each other on one page.

## How "current" is decided

1. **Answer from the site as it stood on the date asked:** `as_of`, or one moment the question names ("as of May 2026", "By June 2026, …"; "before June 2026" is the day before; holidays, seasons, quarters, halves and "2026-07" are read as dates). An article's date doesn't move the snapshot. Two dates compared get the value at each. A date before the first crawl is answered from it only if what the answer rests on took effect by then; otherwise it declines. The model is told the date.
2. **Only the same fact competes** (product, term, tier, kind of number): tiers, rate vs APY and intro vs standard are not conflicts.
3. **Winner:** newer effective date, then newer publish date, then a re-fetched page over a stale copy, then rate and product pages over articles, promotions and calculators, then most pages.
4. **Expired offers are never used.** Losing values leave the evidence before answering.
5. **If the site doesn't say,** the answer is "That isn't listed on our website… please contact us", with no digits.

## No hard-coding

Nothing from the practice questions or this site is in the code or tests. The site's name, domains, proper names and page list come from the crawl, and the rules use general English and banking words. A unit test fails if `submission/` contains a practice value, source, answer phrase or name, or the site's identity, headings or boilerplate counts.

## Measured results

The evaluation sets were rebuilt from the pages by independent agents after the earlier ones were lost; "with key" figures are full keyed runs.

| Run | Result |
|---|---|
| Kit selftest with `--conflicts-gold` | accuracy 1.0, conflicts F1 1.0 |
| Public 40 | with key 0.946 (p95 1.7–2.3s across runs, 0 stale values, 0 errors, about $0.01); no key 0.811 |
| Battery: the 40 practice questions, 160 rewordings and 39 dated questions | with key 0.904, reworded groups consistent 0.825; no key 0.747; 0 stale values |
| 32 held-out dated questions | with key 0.987; no key 0.960; 0 stale values |
| Fee probes at three dates | 48 of 50 |
| 82 questions with a date phrase, labelled with the snapshot that should answer | 72 read that snapshot |
| 28 regression questions from an adversarial review | no key 23 |
| 136 link, joining, eligibility, intro-rate, HSA, aside, tier and date questions in 9 independent sets, each written against the previous version | no question worse than on the version it was written against beyond run-to-run variation; latest set: 0.325 → 0.917 with and without the key |
| Conflicts vs a 20-entry key built independently | pillar 0.605 |
| Earlier sets (history) | battery with key 0.937 / no key 0.780; held-out 0.984 / 0.859; 12 "how did it change" questions 0.90 / 0.875; conflicts vs a 21-entry key 0.716 |

## Limitations

- Model answers vary between runs, so reworded questions can disagree, and two slow API replies push p95 past 2s.
- Contradictions that exist only across old snapshots are not listed.
- Wording far from the site's can still miss the right page when the model answers a nearby question instead of declining.
- A product named in words the site doesn't use gets the nearest match: "credit builder loan" gets the credit-card application, though the site's credit-building loan goes by another name on its personal-loans page.
- A program the site offers, named in words its page title doesn't use ("my wife wants to join that challenge you run"), can get the joining link.
- With no key, a generic "savings account rate" question from June on answers with certificate rates. Asked whether any savings tier pays a money market's rate, the answer, with or without the key, points to the money market tier rather than saying no savings tier does.
- Year-only dates ("as of 2025"), relative dates ("a year ago") and "right now, as of June" aren't read as moments; a loan amount isn't worked out from a price and a down payment; a fee question naming a place ("a wire to a bank in another city") can decline.
- An aside in a sentence of its own after a request that isn't a question ("Tell me the 17-month special rate. This is my savings.") is read as part of the request.
