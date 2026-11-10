# Notes: a knowledge system that knows what's true *now*

**API key.** Set `ANTHROPIC_API_KEY` for grading. This uses the **paid Anthropic API**: one `claude-haiku-5-5` call (thinking off) per question that needs one, about $0.01 per 40 questions. With no key, or with `KH_NO_LLM=1`, answers are extractive and everything still runs. No other keys or network calls.

**Run** from the kit root: `pip install -r requirements.txt`, then `python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`. Unit tests need no key: `python -m unittest discover -s submission/tests`.

## How it works

**`ingest`** turns each snapshot into a view: the newest copy of every page seen up to that date (pages that left the crawl marked removed), split into table rows, fee lines and paragraphs with type, publish, effective and offer-end dates. A table under an in-page tab takes the tab's name when its own heading says less (a plain "Savings Rates" heading under a "Health Savings Rates" tab), and a heading that names no kind of product gets the tab's product (a brand name under a money-market tab, a one-word "Regular" under a certificates tab). A table continued after a page break, whose first row holds a dollar amount, has no header row.

**`answer`** reads the question with general English and banking rules: term ("5-year" = the 60-month row), model year, amount ("under $X" is just below X, "over $X" just above; a monthly amount, a payment or an income is not a balance; a price with a down payment gives the loan amount, "a $90,000 canoe with $10,000 down" is $80,000, while a stated loan amount is kept; an amount below a row's minimum doesn't fit it; nested tiers give the tightest), product line unless ruled out ("not the jumbo", "not for my business"), kind of value, loan type ("7/1"), number words, abbreviations ("HSA" rows are written "Health Savings Account"), greetings and asides, judging words ("best" is never a product), branch names from the site's own proper names (a branch the site lists with an address or hours doesn't narrow a rate or fee; one it doesn't list, "your <city> location" included, still declines), and a percent someone "pays" or "earns" makes a rate question. An aside after the question that only tells about the member ("…? I've been a member for 10 years", "…? I keep my savings with you") adds no product, term, model year or line; one that says what is meant ("…? I'm looking at the 24-month term", "Five years", "Also the 36-month") still counts, and a term in the question itself is never overridden. It runs BM25 over the right view, drops other tiers' and lines' rows, and searches the top 400 hits before declining. A rate question always sees its product's own page; rows of a line it didn't name (business, jumbo) come last. An introductory rate is never answered with the standard one ("intro" = "introductory"; a rate "for the first N months" is one), unless the member names the special they mean. A rate question about a product line whose best row is another kind of product under the same heading (a certificate table under a savings heading) is answered from that line's plainest product; "what does a savings account pay?" is a rate question; APY and APR never name a product, and a plain word ("regular", "basic", "share") counts only if a product of that kind uses it.

A rate question whose term, model year or amount picks out one row gets that row with no model call. A wire question is read for which wire it means: received ("incoming", "wire me", "into my checking account", "from overseas") gets the incoming fee from anywhere; sent abroad ("internationally", "to my sister in London", "a Canadian bank", "in euros", from a general list of countries, capitals and big cities, never a US state, territory or big US city, and "Paris, Texas" stays domestic) gets the international fee; several kinds asked together ("domestic and international", "your wire fees") get each one's fee line; anything else is a domestic wire sent. Who, where, when and why ("to my brother, for his wedding", "someone who isn't a member", "fast", "yesterday") are not what is charged, nor is what was done with the thing in a clause after it ("a check I wrote"), unless the site states a fee next to that word ("lost"); a promotion's offer ("we'll give you $X") is never a fee. Other payments to a named service still need the site to name it. Words in parentheses restate a name ("an official check (bank draft)"). A "how did it change" question gets the value at each date named. Otherwise Claude answers from the filtered evidence, and guards check that every number is in the evidence, no link goes to a page missing from the crawl, a repeated statement's newest value appears, and a value quoted for a product line that only another kind of product pays ("do any savings tiers pay X%?" when only a money market does) comes with the plain product's own top tier, with or without the key; the model then reads that product's table first, so it can say no. If the model can't answer an open question (how / who / what-is), it picks up to 3 of the site's pages by title (a list built from the crawl) and answers once more from those (about 2% of questions, 2–3 seconds more).

**Link questions** ("link", "URL", "web address") get the live, non-article page the site's own text links for what was asked, never a calculator unless asked:
- an application portal on one of the site's uncrawled hosts counts when a live page gives it and its own words name what was asked; an "apply for X" question gets X's online application, from the evidence or X's own page (words no application of the site uses, "my husband", are not the product; a minivan or truck is a vehicle; "personal" may sit anywhere in the address once another word picks the product), unless it asks what applying needs;
- a link worded for another line ("a business auto loan") gives way to the site's page titled with what was asked;
- a question about joining ("apply for membership", "become / sign up as a member", opening someone's first account, or a bare "join") also reads the site's pages about joining and gets the application link they give, or the one for continuing an application. Joining something else ("the newsletter", "a board member") is not joining, and a second request in the same sentence keeps the usual pick. In a message of several sentences, the others are context when the request is plainly about joining ("My cousin just moved here. Can you send me the link so she can join?"), unless they name an event, a product line or one of the site's own pages, or the request also asks to apply for something. How the link is passed on ("email it to her") is never what it is for.

**`conflicts`** is deterministic, from the newest view, and names each winner and why: one product, term and tier priced differently on two pages; boilerplate counts that disagree; "Nth time" / "Nth year in a row" claims, including ones contradicting each other on one page.

## How "current" is decided

1. **Answer from the site as it stood on the date asked:** `as_of`, or one moment the question names ("as of May 2026", "By June 2026, …"; "before June 2026" is the day before; holidays, seasons, quarters, halves and "2026-07" are read as dates; a year after "as of / back in / before" is that year's last day or the day before it starts, never a model year; "right now, as of June 2026" is June, but "is it now higher than back in May?" is now). Relative dates count back from `as_of`, or else from a month after the newest crawl (questions are asked after it): "last May", "a year ago", and in a question about a rate or a fee also "in 2025" / "last year" (that year's end), "last month" / "two months ago" (that month's end) and a week phrase whose crawl doesn't depend on when exactly "now" is. In a count, an event or a span ("how many … last year", "over the last year") they name a period and aren't read. An article's date doesn't move the snapshot. Two dates compared get the value at each. A date before the first crawl is answered from it only if what the answer rests on was in effect then: not published or dated later (the earliest date any crawl gives its page), and, if undated, no more than a year before the first crawl. The model is told the date asked, and sees the question with it written out ("two months ago" as "in August 2026").
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
| Public 40 | with key 0.93–0.95 across runs (p95 1.9–2.1s, 0 stale values, 0 errors, about $0.01); no key 0.811 |
| Battery: the 40 practice questions, 160 rewordings and 39 dated questions | with key 0.90–0.91 across runs, reworded groups consistent 0.80–0.85; no key 0.768; 0 stale values |
| 32 held-out dated questions | with key 0.987; no key 0.960; 0 stale values |
| Fee probes at three dates | 48 of 50 |
| 82 questions with a date phrase, labelled with the snapshot that should answer | 79 read that snapshot |
| 63 regression questions from four adversarial reviews | no key 60 |
| 10 reworded question pairs that must give the same numbers | 6 agree (1 before this version) |
| Crash sweep: every kit question plus branch, wire, date and down-payment templates (844) | 0 errors |
| 200 link, joining, eligibility, intro-rate, HSA, aside, tier, date, wire, down-payment, branch, fee-wording and plain-product questions in 13 independent sets, each written against the previous version | no question worse than on the version it was written against beyond run-to-run variation; latest set (16): with key 13 → 14, no key 10 → 12 |
| Conflicts vs a 20-entry key built independently | pillar 0.605 |
| Earlier sets (history) | battery with key 0.937 / no key 0.780; held-out 0.984 / 0.859; 12 "how did it change" questions 0.90 / 0.875; conflicts vs a 21-entry key 0.716 |

## Limitations

- Model answers vary between runs, so reworded questions can disagree, and two slow API replies push p95 past 2s.
- Contradictions that exist only across old snapshots are not listed.
- Wording far from the site's can still miss the right page when the model answers a nearby question instead of declining.
- A product named in words the site doesn't use gets the nearest match: "credit builder loan" gets the credit-card application, though the site's credit-building loan goes by another name on its personal-loans page.
- A program the site offers, named in words its page title doesn't use ("my wife wants to join that challenge you run"), can get the joining link.
- Asked whether any savings tier pays another account's rate, the model sometimes still answers "yes, the money market does"; the plain account's top tier is always given too. With no key, "what does a savings account pay?" can name a savings product other than the plainest one when that product leads the evidence.
- "Now" for relative dates is an assumption (a month after the newest crawl); a question written much later reads "two months ago" a crawl too early. "Last year" in a total ("how many … last year") is a period, not a moment.
- A foreign place the geography list lacks gets the domestic wire fee with no key (the model, with the key, still knows it); a foreign-named US town without its state ("my cousin in Paris") reads as abroad.
- A fee question can still decline on a word no fee line uses outside a relative clause ("a check written by me", "a check that bounced"), and "a lost X" with no fee for it can get another fee that mentions "lost".
- An aside in a sentence of its own after a request that isn't a question ("Tell me the 17-month special rate. This is my savings.") is read as part of the request.
