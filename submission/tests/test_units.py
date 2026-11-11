"""Unit + selftest checks. Stdlib only, no snapshots, no API key:
`python -m unittest discover -s submission/tests` (run from the kit root)."""
import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # kit root: submission/tests/ -> submission/ -> root
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "harness")]
os.environ["KH_NO_LLM"] = "1"

import run_eval  # noqa: E402
import scoring  # noqa: E402
from contract import Answer, Conflict  # noqa: E402
from ingest import Parser, View, strip_footnotes, url_key  # noqa: E402
from retrieve import _stem, amount_ranges, model_year_ranges, parse_query, tokenize  # noqa: E402
from solution import build  # noqa: E402
from synth import REFUSAL, _drop_banned, _drop_dead_links, _sentences, numbers, render_row  # noqa: E402

SELFTEST = ROOT / "harness" / "selftest"


class Parsing(unittest.TestCase):
    def test_footnote_markers_are_stripped_but_values_kept(self):
        self.assertEqual(strip_footnotes("4.44%1"), "4.44%")
        self.assertEqual(strip_footnotes("9.87% APR2"), "9.87% APR")
        self.assertEqual(strip_footnotes("Annual Percentage Rate (As Low As) 1"), "Annual Percentage Rate (As Low As)")
        self.assertEqual(strip_footnotes("11.00% - 18.00% APR"), "11.00% - 18.00% APR")

    def test_url_key_matches_the_scorer(self):
        for u in ("https://www.example.org/borrow/boat-loans#rates", "https://www.example.org/",
                  "https://www.example.org/privacy?campaignId=x"):
            self.assertEqual(url_key(u), scoring._url_key(u))

    def test_model_year_ranges_handle_both_orders_and_open_ends(self):
        self.assertEqual(model_year_ranges("2019-21 Model Year"), [(2019, 2021)])
        self.assertEqual(model_year_ranges("Model Year 2022-18"), [(2018, 2022)])
        self.assertEqual(model_year_ranges("2018–22 Model Year"), [(2018, 2022)])
        self.assertEqual(model_year_ranges("2009 Model Year (and older)"), [(0, 2009)])
        self.assertEqual(model_year_ranges("2009 Model Year or earlier"), [(0, 2009)])

    def test_amount_tiers(self):
        self.assertEqual(amount_ranges("$5,000 – $9,999.99"), [(5000.0, 9999.99)])
        self.assertEqual(amount_ranges("$50,000+")[0][0], 50000.0)
        self.assertEqual(amount_ranges("$500,000 or less"), [(0.0, 500000.0)])

    def test_question_qualifiers(self):
        q = parse_query("What's the APR for a 48-month loan on a 2022 model-year boat?")
        self.assertEqual(q.terms, {(48, "month")})
        self.assertEqual(q.years, {2022})
        self.assertEqual(q.attribute, "rate")
        q = parse_query("What was the rate on the 14-month jumbo special as of March 2027?")
        self.assertEqual(q.years, set())                     # a date, not a model year
        self.assertIn("jumbo", q.qualifiers)
        q = parse_query("Current APY on a six month share certificate?")
        self.assertEqual(q.terms, {(6, "month")})
        q = parse_query("How many offices does the bank run?")
        self.assertEqual((q.attribute, q.count_noun), ("count", _stem("offices")))
        q = parse_query("What are the lobby hours at the Elm Avenue branch?")
        self.assertEqual((q.attribute, q.entity), ("hours", "elm"))

    def test_a_fee_condition_is_not_a_product_term(self):
        q = parse_query("What is the fee to close an account within three months of opening?")
        self.assertEqual((q.attribute, q.terms), ("fee", set()))

    def test_fee_schedule_vocabulary(self):
        self.assertIn("official", parse_query("How much do you charge for a cashier's check?").tokens)
        self.assertIn("international", parse_query("Do you charge for overseas transfers?").tokens)

    def test_list_intent_only_for_catalogue_questions(self):
        self.assertTrue(parse_query("What savings accounts do you offer?").list_intent)
        self.assertFalse(parse_query("What is the APR on the Gold card?").list_intent)

    def test_pdf_conversion_artifacts_are_cleaned(self):
        md = ("# Fees\n\n|Services|Col2|\n|---|---|\n|Copy &amp;#45; per page|$1.00 each|\n\n"
              "Wire &amp; transfer help is available.\n")
        page = Parser().parse("example.com/fees.pdf", "https://example.com/fees.pdf", "Fees", md)
        text = " ".join(c.text for c in page.chunks)
        self.assertNotIn("&amp;", text)
        self.assertIn("Copy - per page", text)
        row = next(c for c in page.chunks if c.is_row)
        self.assertNotIn("Col2", render_row(row))
        self.assertIn("$1.00", render_row(row))


class Guardrails(unittest.TestCase):
    """No hard-coding in our code (assignment §7, and the owner's stricter rule). Nothing learned from the
    practice set or from this site may appear in any file under submission/ (code, tests, notes):
      * practice answer and trap values; expected-source paths; 4-word expected-answer phrases;
      * proper names from the practice questions and answers;
      * the site's own identity (domain, brand, title suffix): the system derives it from the crawl.
    Exempt: generic vocabulary any credit-union site uses (ordinals, "noon", street words, "share", the card
    network the code names as a product head, calendar names) and bare years."""
    EXEMPT = set("noon first second third fourth fifth sixth seventh eighth ninth tenth avenue lane street "
                 "share visa".split())
    CALENDAR = set("monday tuesday wednesday thursday friday saturday sunday january february march april may "
                   "june july august september october november december mon tue wed thu fri sat sun".split())

    def _banned(self) -> dict:
        import re
        from types import SimpleNamespace
        from ingest import site_identity
        from retrieve import _BASE_STOP
        items = [json.loads(l) for l in (ROOT / "eval" / "public_eval.jsonl").read_text().splitlines() if l.strip()]
        texts = [t for it in items for t in (it["question"], it.get("expected_answer", ""))]
        lower_words = set(re.findall(r"\b[a-z][a-z'’$-]+\b", " ".join(texts)))
        banned = {}
        for it in items:
            for t in it.get("must_include", []) + it.get("must_not_include", []):
                if len(t) >= 3 and t.lower() not in self.EXEMPT and not re.fullmatch(r"(?:19|20)\d\d", t):
                    banned[t.lower()] = f"practice value ({it['id']})"
            for u in it.get("expected_sources", []):
                path = url_key(u).partition("/")[2]
                if len(path) >= 8:
                    banned[path.lower()] = f"expected source ({it['id']})"
            words = re.findall(r"[A-Za-z0-9$%.'’-]+", it.get("expected_answer", ""))
            for i in range(len(words) - 3):
                gram = [w.lower().strip(".") for w in words[i:i + 4]]
                if sum(w not in _BASE_STOP for w in gram) >= 3:
                    banned[" ".join(gram)] = f"expected-answer phrase ({it['id']})"
        for t in texts:                                  # proper names: capitalised mid-sentence, never lower-case
            for sentence in re.split(r"(?<=[.?!:;])\s+|\s+-\s+|\(", t):
                for w in re.findall(r"[A-Za-z][A-Za-z'’$-]*", sentence)[1:]:
                    w = re.sub(r"['’]s$", "", w)
                    if (w[0].isupper() and len(w) >= 4 and w.lower() not in lower_words | self.EXEMPT
                            and not any(part in self.CALENDAR for part in re.split(r"[-–]", w.lower()))):
                        banned.setdefault(w.lower(), "practice proper name")
        for t in texts:                                  # dollar amounts the practice set states (tier bounds)
            for amt in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d\d)?|\d+\.\d\d(?=\b)(?!%)", t):
                if "," in amt and not amt.endswith(",000"):     # tier bounds like 9,999.99; round sums are generic
                    banned.setdefault(amt, "practice amount")
        snaps = sorted((ROOT / "snapshots").glob("0*/manifest.json"))
        if snaps:                                        # the site's identity, derived the same way ingest does
            pages = json.loads(snaps[-1].read_text())["pages"]
            domains, suffix, words = site_identity([SimpleNamespace(url=p.get("url"), title=p.get("title"))
                                                    for p in pages])
            for x in domains | words | ({suffix} if suffix else set()):
                banned[x.lower()] = "site identity"
            # the site's own wording: multi-word headings, and counted things its boilerplate repeats
            from collections import Counter
            heads, counts = Counter(), Counter()
            units = {"month", "months", "year", "years", "day", "days", "hour", "hours", "min", "minute", "minutes",
                     "percent", "mo", "yr", "yrs", "am", "pm", "points", "point", "times", "time"}
            for p in pages:
                f = snaps[-1].parent / p.get("file", "")
                md = f.read_text(errors="ignore") if p.get("file") and f.exists() else ""
                for h in re.findall(r"^#{1,4}\s+(.+?)\s*$", md, re.M):
                    w = re.findall(r"[a-z][a-z'’-]+", h.lower())
                    if len(w) >= 3 and sum(x not in _BASE_STOP for x in w) >= 3:
                        heads[" ".join(w)] += 1
                for n, noun in set(re.findall(r"\b(\d[\d,]*)\+?\s+([a-z]+)\b", md.lower())):
                    if noun not in units and noun not in _BASE_STOP and int(n.replace(",", "") or 0) > 1:
                        counts[f"{n} {noun}"] += 1
            for h in heads:
                if h not in ("annual percentage rate", "annual percentage yield"):    # regulatory terms, every bank's
                    banned.setdefault(h, "site heading")
            for c, k in counts.items():
                if k >= 5:
                    banned.setdefault(c, "site boilerplate count")
        return banned

    def test_no_practice_or_site_knowledge_in_our_files(self):
        import re
        banned = self._banned()
        self.assertGreater(len(banned), 100)             # the guard really has something to look for
        files = [p for p in (ROOT / "submission").rglob("*")
                 if p.is_file() and p.suffix in (".py", ".md", ".txt", ".json") and "__pycache__" not in p.parts]
        hits = []
        for p in files:
            text = " ".join(p.read_text(errors="ignore").lower().split())
            hits += [(str(p.relative_to(ROOT)), t, why) for t, why in banned.items()
                     if re.search(rf"(?<![\w.]){re.escape(t)}(?![\w])", text)]
        self.assertEqual(hits, [])


class Edges(unittest.TestCase):
    """Link picker and fact edge, on synthetic pages (no site data)."""

    def test_link_picker_takes_the_live_link_the_text_gives(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        from solution import pick_link
        live = lambda kind: NS(status="fresh", page=NS(kind=kind))
        view = NS(pages={"example.com/rates": live("product"), "example.com/blog/rates-explained": live("article"),
                         "example.com/old-rates": NS(status="missing", page=NS(kind="product"))})
        text = ("Compare certificate rates (example.com/rates). Certificate rates explained (example.com/blog/rates-explained). "
                "Certificate rates, explained again (example.com/blog/rates-explained). "
                "Old certificate rates (example.com/old-rates). Certificate archive (example.com/gone).")
        ctx = [Chunk(1, "example.com/a", "product", text, text)]
        self.assertEqual(pick_link(ctx, view, parse_query("What link shows certificate rates?")), "example.com/rates")
        self.assertIsNone(pick_link(ctx, view, parse_query("What link shows mortgage payoffs?")))

    def test_fact_edge_puts_the_winner_first_and_drops_beaten_values(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        from solution import apply_fact, match_fact
        win = Chunk(7, "example.com/news-3", "article", "Acme ranked the lender among its best, its seventh time doing so.", "")
        fact = {"topic": "How many times Acme recognized: the Nth time", "_losers": ["sixth"],
                "_win_chunk": win, "_win_sentence": win.text}
        index = NS(chunks=[None] * 1000, postings={"acm": [(0, 1)]})
        self.assertIs(match_fact([fact], parse_query("Has Acme ever ranked this lender?"), index), fact)
        self.assertIsNone(match_fact([fact], parse_query("What are the lobby hours?"), index))
        old = Chunk(2, "example.com/news-2", "article", "Acme ranked the lender among its best, its sixth time doing so.", "")
        other = Chunk(3, "example.com/about", "product", "We serve members in three states.", "")
        out = apply_fact([old, other], fact)
        self.assertEqual([c.text for c in out], [win.text, other.text])


class Rev20(unittest.TestCase):
    """Glossary, conflict output, branch possessives and as-of dates (synthetic data only)."""

    def test_gloss_adds_the_sites_definition_only_when_missing(self):
        from solution import gloss
        gl = {"service area": ((("Region A", "Any town"), ("Region B", "North, South")), "example.com/grants")}
        text, page = gloss("You can join if you live within our service area.", gl)
        self.assertIn("Region A: Any town; Region B: North, South.", text)
        self.assertEqual(page, "example.com/grants")
        same = "Our service area covers Region A and Region B."
        self.assertEqual(gloss(same, gl), (same, None))

    def test_count_values_also_carry_the_short_noun_and_families_ignore_quantity_words(self):
        from conflicts import _family, _short_count
        self.assertEqual(_short_count("8 office locations"), "8 offices (8 office locations)")
        self.assertEqual(_short_count("400 members"), "400 members")
        self.assertEqual(_family("It has over 400 members."), _family("It has nearly 500 members."))

    def test_branch_possessive_and_dated_model_years(self):
        self.assertEqual(parse_query("When does the Elm's branch open?").entity, "elm")
        from retrieve import use_site_words
        use_site_words({"example"})                     # a site named "Example" (derived from the crawl at ingest)
        try:
            self.assertEqual(parse_query("What are the lobby hours at Example's Elm Avenue branch?").entity, "elm")
        finally:
            use_site_words(set())
        q = parse_query("What is the auto loan APR for a 2031 vehicle as of March 3, 2032?")
        self.assertEqual(q.years, {2031})

    def test_as_of_dates_leave_the_search_but_content_dates_stay(self):
        from solution import strip_temporal
        same = lambda d: d >= "2031-09-01"                     # stand-in for "maps to the asked snapshot"
        self.assertEqual(strip_temporal("What is the APY as of September 2031?", same), "What is the APY ?")
        for q in ("Did it close on January 15, 2031?", "What happens in July 2031?"):
            self.assertEqual(strip_temporal(q, same), q)
        # a deadline the site states stays a search word even when it falls in the answered window
        stated = {("2031-11-12", True), ("2032-04-30", False)}
        for q in ("Is it open until Nov. 12, 2031?", "What is due in April 2032?"):
            self.assertEqual(strip_temporal(q, same, stated), q)
        self.assertEqual(strip_temporal("What is the APY (October 2031)?", same, stated), "What is the APY ?")

    def test_review_fixes_synonyms_model_years_and_branch_words(self):
        from retrieve import _stem
        change = _stem("change")
        self.assertNotIn(change, parse_query("Is there a charge to edit my profile?").tokens)
        self.assertIn(change, parse_query("How do I edit my profile?").tokens)
        self.assertIn(_stem("update"), parse_query("How do I amend my profile?").tokens)
        self.assertEqual(parse_query("Which APR may 2029 model year cars get for 66 months?").years, {2029})
        from retrieve import use_site_words
        use_site_words({"example"})
        try:
            self.assertIsNone(parse_query("What are Example's Saturday branch hours?").entity)
        finally:
            use_site_words(set())
        self.assertEqual(parse_query("Saturday lobby hours at the Elm branch?").entity, "elm")


class Rev21(unittest.TestCase):
    """Rewording robustness and conflict wording (synthetic data only)."""

    def test_hours_and_count_cues(self):
        self.assertEqual(parse_query("office hrs pls?").attribute, "hours")
        self.assertEqual(parse_query("When is the main office open?").attribute, "hours")
        self.assertIsNone(parse_query("What time period does the special cover?").attribute)
        self.assertNotEqual(parse_query("What time does the elm branch close?").attribute, "hours")
        q = parse_query("Total number of offices?")
        self.assertEqual((q.attribute, q.count_noun), ("count", _stem("offices")))

    def test_arm_amounts_frame_words_and_product_synonyms(self):
        from ingest import Chunk
        from retrieve import supports
        q = parse_query("What is the 10/6 ARM rate?")
        self.assertIn("adjustabl", q.tokens)
        self.assertEqual(q.raw, "What is the 10/6 ARM rate?")                 # the model sees the member's words
        self.assertNotIn("adjustabl", parse_query("What is a 5/1 ARM?").tokens)
        self.assertEqual(parse_query("Rate for a $10k line of credit?").amounts, [10000.0])
        self.assertNotIn("find", parse_query("Where can I find checking rates?").descriptors)
        text = "Auto Loans — 60 Months: 5.00% APR"
        self.assertTrue(supports(Chunk(1, "example.com/rates", "product", text, text), parse_query("What is the car loan rate?")))

    def test_an_ordinal_is_part_of_a_caption(self):
        md = "1st Choice Savings\n\n| Balance | APY |\n|---|---|\n| $0 - $999 | 0.07% |\n"
        page = Parser().parse("example.com/rates", "https://example.com/rates", "Rates", md)
        self.assertIn("1st Choice Savings", next(c for c in page.chunks if c.is_row).index_text)

    def test_other_product_lines_leave_term_questions(self):
        from types import SimpleNamespace as Row
        from retrieve import on_other_line, other_lines
        top = Row(is_row=True, index_text="Share Specials 14 Month Share Special 2.22% APY")
        lines = other_lines(parse_query("What is the 14-month share special APY?"), top)
        self.assertIn("jumbo", lines)
        self.assertTrue(on_other_line(Row(is_row=True, index_text="14 Month Jumbo Share Special 2.32%"), lines))
        self.assertEqual(other_lines(parse_query("What does the 14-month special pay on $120,000?"), top), set())
        self.assertNotIn("refinance", other_lines(parse_query("What is the 20-year rate for refinancing?"), top))

    def test_conflict_values_carry_the_sites_wording(self):
        from conflicts import STREAK_RE, with_phrases
        self.assertEqual(with_phrases("400 members", ["over 400 members"]), "400 members (over 400 members)")
        self.assertEqual(with_phrases("seventh", None), "seventh")
        m = STREAK_RE.search("For the fourth consecutive year, Acme ranked the lender among the best.")
        self.assertEqual((m.group(1), m.group(2)), ("fourth", "Acme"))


class RetryWait(unittest.TestCase):
    """A rate-limited reply asking for a long wait must not hold an answer past the 60s limit."""

    def _client(self, retry_after: str):
        try:
            import httpx2
            from synth import _new_client
        except ImportError:                       # no SDK (e.g. Python 3.9): nothing to test
            self.skipTest("anthropic SDK not installed")
        calls = []

        def handler(request):
            calls.append(request)
            return httpx2.Response(429, headers={"retry-after": retry_after},
                                   json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
        client = _new_client(api_key="test-key", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        return client, calls

    def _ask(self, client):
        import time
        start = time.monotonic()
        with self.assertRaises(Exception):
            client.messages.create(model="claude-haiku-5-5", max_tokens=8,
                                   messages=[{"role": "user", "content": "hi"}])
        return time.monotonic() - start

    def test_a_long_retry_after_is_not_waited_out(self):
        client, calls = self._client("75")
        self.assertLess(self._ask(client), 3.0)
        self.assertEqual(len(calls), 2)                 # the one retry still happens, without the wait

    def test_a_short_retry_after_is_still_honoured(self):
        client, calls = self._client("1")
        self.assertGreaterEqual(self._ask(client), 0.9)
        self.assertEqual(len(calls), 2)


class Rev23(unittest.TestCase):
    """Dates a question names, and the date the model is told (synthetic snapshots and values)."""

    @staticmethod
    def _store(*days):
        from types import SimpleNamespace as NS
        return NS(views=[NS(taken_on=d) for d in days])

    def test_a_moment_the_question_names_selects_the_site_as_it_stood_then(self):
        from solution import date_asked
        st = self._store(date(2030, 1, 10), date(2030, 3, 1), date(2030, 6, 15))
        for q, want in [("What was the savings APY in February 2030?", "2030-02-28"),
                        ("As of 1/20/2030, what is the savings APY?", "2030-01-20"),
                        ("Back in March 2030, what did checking pay?", "2030-03-31"),
                        ("As of August 2030, what was the APY?", None),          # after the newest crawl
                        ("What was the APY on 2030-01-10?", "2030-01-10"),
                        ("What was the APY in December 2029?", "2029-12-31"),    # before the first crawl: kept
                        ("What is the APY in February 2030?", None),             # present tense, soft frame
                        ("Is the APY still the same as of February 2030?", "2030-02-28"),
                        ("What was the current APY as of February 2030?", "2030-02-28"),
                        ("What was the savings APY today (February 2030)?", None),
                        ("How has the APY moved since February 2030?", None),
                        ("What was the APY between January 2030 and March 2030?", None),
                        # the frame is read from the words before the date, with qualifiers
                        ("What was the APY before March 2030?", "2030-02-28"),
                        ("What was the APY prior to March 2030?", "2030-02-28"),
                        ("What was the APY up until March 2030?", "2030-02-28"),
                        ("What was the APY before the March 3, 2030 change?", "2030-03-02"),
                        ("What was the APY in mid-February 2030?", "2030-02-15"),
                        ("What was the APY in early March 2030?", "2030-03-01"),
                        ("What is the APY as of the end of February 2030?", "2030-02-28"),
                        ("What is the APY as of last month (February 2030)?", "2030-02-28"),
                        ("What was the APY on 15 February 2030?", "2030-02-15"),
                        ("Back on Feb. 20, 2030, what did savings pay?", "2030-02-20"),
                        ("As at 2/20/2030, what APY did balances of $1,000 to $9,999.99 earn?", "2030-02-20"),
                        # a change, an article or event: the newest crawl records it
                        ("What did the APY change to on February 20, 2030?", None),
                        ("What was the APY effective February 20, 2030?", None),
                        ("What new APY took effect on February 20, 2030?", None),
                        ("What did the February 20, 2030 blog post say about savings?", None),
                        ("What was the news story on February 20, 2030 about?", None),
                        # a date in a side remark of a present-tense question
                        ("I opened an account in February 2030. What APY does it pay?", None),
                        ("My account was opened on Feb. 20, 2030; what APY do new accounts earn?", None),
                        ("What is the APY? I saw a lower rate in February 2030.", None),
                        ("I opened an account on February 20, 2030. What APY did I get?", "2030-02-20")]:
            self.assertEqual(date_asked(q, st), want, q)
        self.assertIsNone(date_asked("What was the APY in February 2030?", self._store(date(2030, 1, 10))))
        late = self._store(date(2030, 1, 15), date(2030, 3, 1), date(2030, 6, 15))
        self.assertEqual(date_asked("What was the APY in early January 2030?", late), "2030-01-15")   # nearest crawl
        self.assertEqual(date_asked("What was the APY on January 2, 2030?", late), "2030-01-02")    # before it: kept

    def test_an_article_published_after_the_pinned_crawl_is_read_from_the_newest(self):
        from contract import Page, Snapshot
        rates = "# Savings\n\n| Account | APY |\n|---|---|\n| Basic Savings | {a} |\n"
        tips = "March 5, 2030\n\n# How to spot a fake caller\n\nHang up and call the number on your card instead.\n"
        ks = build()
        ks.ingest(Snapshot("s1", "2030-01-10", [Page("https://example.org/savings", "Savings | Example",
                                                     rates.format(a="1.10%"), "2030-01-10T09:00:00Z")]))
        ks.ingest(Snapshot("s2", "2030-06-15", [
            Page("https://example.org/savings", "Savings | Example", rates.format(a="1.40%"), "2030-06-15T09:00:00Z"),
            Page("https://example.org/blog/fake-caller", "How to spot a fake caller | Example", tips,
                 "2030-06-15T09:00:00Z")]))
        from solution import moment_asked
        tip = "What tips did you share in March 2030 about a fake caller?"
        self.assertIsNone(moment_asked(tip, ks._store, ks._indexes))          # the article is only in the newest
        self.assertEqual(ks.answer(tip).sources[0].snapshot_id, "s2")
        self.assertEqual(moment_asked("What was the Basic Savings APY in March 2030?", ks._store, ks._indexes),
                         "2030-03-31")
        self.assertIn("1.10%", ks.answer("What was the Basic Savings APY in March 2030?").text)

    def test_the_model_is_told_the_day_asked_never_an_earlier_one(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        from synth import Synthesizer
        c = Chunk(1, "example.org/rates", "product", "Basic Savings APY 1.10%.", "")
        view = NS(taken_on=date(2030, 3, 1), pages={"example.org/rates": NS(
            page=NS(url="https://example.org/rates", kind="product", publish=None), status="fresh",
            fetched=date(2030, 3, 1))})
        sent = []

        class Fake:
            class messages:
                @staticmethod
                def create(**k):
                    sent.append(k["messages"][0]["content"])
                    raise RuntimeError("offline")
        s = Synthesizer()
        s._client = Fake()
        for asked, shown in [(date(2030, 3, 20), "2030-03-20\n"), (None, "2030-03-01\n"),
                             (date(2030, 2, 1), "2030-03-01. The member asks about 2030-02-01, before this snapshot")]:
            self.assertIsNone(s._llm("What is the APY?", NS(context=[c], chunks=[c]), view, asked))
            self.assertTrue(sent[-1].startswith(f"Website snapshot date: {shown}"), sent[-1][:60])

    def test_text_dates_and_as_of_answer_from_the_same_snapshot(self):
        from contract import Page, Snapshot
        md = "# Savings\n\n| Account | Dividend Rate | APY |\n|---|---|---|\n| Basic Savings | {r} | {a} |\n"
        ks = build()
        for sid, day, r, a in [("s1", "2030-01-10", "1.05%", "1.10%"), ("s2", "2030-06-15", "1.35%", "1.40%")]:
            ks.ingest(Snapshot(sid, day, [Page("https://example.org/savings", "Savings | Example",
                                               md.format(r=r, a=a), f"{day}T09:00:00Z")]))
        self.assertIn("1.40%", ks.answer("What is the Basic Savings APY?").text)
        for q, as_of in [("What was the Basic Savings APY in March 2030?", None),
                         ("What was the Basic Savings APY?", "2030-03-01")]:
            text = ks.answer(q, as_of).text
            self.assertIn("1.10%", text, q)
            self.assertNotIn("1.40%", text, q)

    def test_the_answer_cache_is_per_day_shown_to_the_model(self):
        from contract import Page, Snapshot
        ks = build()
        for sid, day in [("s1", "2030-01-10"), ("s2", "2030-06-15")]:
            ks.ingest(Snapshot(sid, day, [Page("https://example.org/savings", "Savings | Example",
                                               "# Savings\n\nBasic Savings pays 1.10% APY.\n", f"{day}T09:00:00Z")]))
        seen = []
        ks._synth.compose = lambda question, q, ev, view, asked=None, **kw: (seen.append(asked), ("ok", [], True, ""))[1]
        ks._synth.enabled = True
        for as_of in ("2030-02-01", "2030-03-01", "2030-02-01"):
            ks.answer("What was the Basic Savings APY?", as_of)
        self.assertEqual(seen, [date(2030, 2, 1), date(2030, 3, 1)])     # one call per day, repeats cached


class Rev23b(unittest.TestCase):
    """Recall, answer guards, the clear-cut path and the breaker (synthetic data only)."""

    def test_year_terms_are_month_terms_and_negated_lines_are_not_asked(self):
        from retrieve import LIST_Q, terms_in
        self.assertEqual(terms_in("a 5-year loan"), terms_in("60 Months"))
        self.assertEqual(parse_query("What is the 3 year loan rate?").terms, {(36, "month")})
        q = parse_query("What is the 12-month certificate APY (the regular one, not the jumbo)?")
        self.assertEqual(q.qualifiers, set())
        self.assertNotIn("jumbo", q.tokens)
        self.assertEqual(parse_query("What is the non-jumbo 12-month APY?").qualifiers, set())
        self.assertEqual(parse_query("What is the jumbo 12-month APY?").qualifiers, {"jumbo"})
        self.assertTrue(LIST_Q.search("List your credit card rates"))
        self.assertTrue(parse_query("What are my options for auto loans?").list_intent)
        self.assertFalse(parse_query("Weighing my options: what is the 12-month certificate APY?").list_intent)

    def test_fact_guard_adds_the_winner_and_never_keeps_a_beaten_value(self):
        from ingest import Chunk
        from solution import fact_guard
        win = Chunk(7, "example.org/news-3", "article", "This was the seventh time Acme recognized the bank.", "")
        fact = {"resolution": "seventh", "_losers": ["sixth"], "_win_sentence": win.text, "_win_chunk": win}
        self.assertEqual(fact_guard("Acme named it a top lender.", fact),
                         ("Acme named it a top lender. " + win.text, win))
        self.assertEqual(fact_guard("It was the sixth time Acme recognized it.", fact), (win.text, win))
        self.assertEqual(fact_guard("The seventh time, says Acme.", fact), ("The seventh time, says Acme.", None))

    def test_older_editions_of_the_winning_article_leave_the_evidence(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        from solution import apply_fact
        def page(title, day):
            return NS(page=NS(title=title, kind="article", publish=date(*day)))
        view = NS(pages={"example.org/award-3": page("Acme Names the Bank a Top Lender", (2030, 6, 1)),
                         "example.org/award": page("Acme names the bank a top lender", (2028, 6, 1)),
                         "example.org/other": page("Branch news", (2027, 1, 1))})
        win = Chunk(1, "example.org/award-3", "article", "This was the seventh time Acme named the bank.", "")
        old = Chunk(2, "example.org/award", "article", "Acme ranked the bank in its 2028 list.", "")
        other = Chunk(3, "example.org/other", "article", "A new branch opened.", "")
        fact = {"_losers": ["sixth"], "_win_chunk": win, "_win_sentence": win.text}
        self.assertEqual([c.key for c in apply_fact([old, other], fact, view)], ["example.org/award-3", "example.org/other"])
        self.assertEqual(len(apply_fact([old, other], fact)), 3)          # no view: only beaten values leave

    def test_a_quoted_value_gets_the_named_products_own_row(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        from synth import named_row_guard
        def row(i, name, apy, section):
            return Chunk(i, "example.org/rates", "product", f"{name} {apy}", f"{name} {apy}", section=section,
                         is_row=True, header=["Account", "APY"], cells=[name, apy], value_cols=[1])
        other = row(1, "Premier Money Market Savings", "0.47%", "Rates · Premier Money Market Savings")
        plain = row(2, "Basic Savings", "0.13%", "Rates · Basic Savings")
        cert = row(3, "9 Month Certificate", "2.07%", "Deposit Rates · Certificates")
        ev = NS(chunks=[other, cert, plain])
        q = parse_query("Is 0.47% APY the best that any savings tier earns?")
        text, own = named_row_guard("Is 0.47% APY the best that any savings tier earns?", q, ev, "Yes, the money market earns 0.47% APY.")
        self.assertIs(own, plain)
        self.assertIn("0.13%", text)
        self.assertEqual(named_row_guard("Does savings pay 0.47%?", q, ev, "Basic Savings pays 0.13%.")[1], None)
        self.assertEqual(named_row_guard("What does savings pay?", q, ev, "It pays 0.47%.")[1], None)   # no value quoted

    def _system(self, client):
        from contract import Page, Snapshot
        md = ("# Certificates\n\n| Term | Dividend Rate | APY |\n|---|---|---|\n| 6 Month | 1.87% | 1.91% |\n"
              "| 12 Month | 2.37% | 2.41% |\n\nWe offer certificates for every goal.\n")
        ks = build()
        for sid, day in [("s1", "2030-01-10"), ("s2", "2030-06-15")]:
            ks.ingest(Snapshot(sid, day, [Page("https://example.org/certificates", "Certificates | Example", md,
                                               f"{day}T09:00:00Z")]))
        ks._synth.enabled, ks._synth._client = True, client
        return ks

    def test_a_clear_cut_rate_question_needs_no_model_call(self):
        calls = []

        class Fake:
            class messages:
                @staticmethod
                def create(**k):
                    calls.append(k)
                    raise RuntimeError("offline")
        ks = self._system(Fake())
        a = ks.answer("What is the 12-month certificate APY?")
        self.assertIn("2.41%", a.text)
        self.assertNotIn("1.91%", a.text)
        self.assertEqual(calls, [])
        ks.answer("Which is better, the 6-month or the 12-month certificate?")     # a comparison goes to the model
        self.assertEqual(len(calls), 1)

    def test_a_network_failure_pauses_the_model_then_probes_again(self):
        import time

        class APIConnectionError(Exception):
            pass
        calls = []

        class Down:
            class messages:
                @staticmethod
                def create(**k):
                    calls.append(k)
                    raise APIConnectionError("no route")
        ks = self._system(Down())
        start = time.monotonic()
        for n in range(12):
            ks._cache.clear()
            ks.answer(f"What certificates do you offer for goal {n}?")
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertEqual(len(calls), 2)                  # the first call, then one probe after 10 skipped


class Rev24(unittest.TestCase):
    """Members' wording and no-key paragraph answers (synthetic data only)."""

    def test_wording_reads_the_way_members_write(self):
        from retrieve import _stem, tokenize
        self.assertEqual(tokenize("the Easy$pend card"), tokenize("the Easyspend card"))
        q = parse_query("Where can I edit my acct info?")
        self.assertTrue({_stem("information"), _stem("account")} <= set(q.tokens))
        for own in ("After a move, how do I change my mailing address?", "How do I correct my address?",
                    "Where do I file a change of address?"):
            self.assertIsNone(parse_query(own).attribute, own)          # the member editing their own address
        for place in ("What is the correct address for the Elm branch?", "What is my Elm branch address?",
                      "Can you tell me my Pine office address?", "What is the address of the Pine office?",
                      "Where is the main office?"):
            self.assertEqual(parse_query(place).attribute, "address", place)   # where a place is

    def test_product_words_skip_judgements_and_stop_at_a_determiner(self):
        self.assertEqual(parse_query("The memo says its gold certificate APY is what?").descriptors, {"gold"})
        self.assertEqual(parse_query("What is the best 7-month certificate rate?").descriptors, set())
        self.assertEqual(parse_query("Cheapest certificate APY for 9 years?").descriptors, set())
        self.assertEqual(parse_query("What is the best widget loan APR?").descriptors, {"widget"})   # still asked for

    def test_a_how_to_answer_without_the_model_is_the_whole_paragraph(self):
        from ingest import Chunk
        from synth import PARAGRAPH_CHARS, best_sentence
        q = parse_query("How do I update my profile?")
        self.assertIsNone(q.attribute)
        steps = "Log in to online banking. Open Settings and choose Profile. Call 555-0100 for help."
        self.assertEqual(best_sentence(Chunk(1, "example.org/help", "product", steps, steps), q), steps)
        long = "Open the app and follow the prompts. " * 40
        self.assertLessEqual(len(best_sentence(Chunk(2, "example.org/help", "product", long, long), q)), PARAGRAPH_CHARS)


class Rev26(unittest.TestCase):
    """Openers, lower-case branch names, a product's own page, excerpt order and change questions
    (synthetic data only)."""

    def test_an_opener_is_not_part_of_the_question(self):
        from retrieve import asking_clause
        self.assertEqual(asking_clause("Hi there! Quick one: how many offices are open?"), "how many offices are open?")
        self.assertEqual(asking_clause("Hey - what does a bank check cost?"), "what does a bank check cost?")
        self.assertEqual(asking_clause("What is the 9-month APY?"), "What is the 9-month APY?")
        self.assertEqual(asking_clause("How many offices are open? Asking for my aunt."), "How many offices are open?")
        self.assertNotIn("aunt", parse_query("How many offices are open? Asking for my aunt.").tokens)
        from retrieve import CHAT_FILLERS, _stem
        self.assertTrue({_stem("pls"), _stem("thanks")} <= CHAT_FILLERS)

    def test_a_lower_case_branch_name_is_read_from_the_sites_own_proper_names(self):
        from ingest import Chunk
        from retrieve import proper_names, use_proper_names
        text = ("We opened our Pinecrest Branch in spring. Members love Pinecrest staff. "
                "Find your nearest branch online. Visit a nearest office.")
        names = proper_names([Chunk(1, "example.org/news", "product", text, text)])
        self.assertIn("pinecrest", names)
        self.assertNotIn("nearest", names)
        use_proper_names(names)
        try:
            self.assertEqual(parse_query("lobby hours at the pinecrest branch?").entity, "pinecrest")
            self.assertIsNone(parse_query("where is your nearest branch?").entity)
        finally:
            use_proper_names(set())

    def test_a_products_own_page_states_its_rates(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        from retrieve import product_page_rates
        own = Chunk(1, "example.org/personal/widget-cards", "product", "Gold Widget: 9.75% APR.", "")
        blog = Chunk(2, "example.org/blog/widget-tips", "article", "A widget card at 8.50% APR.", "")
        page = lambda url, kind, chunks: NS(status="fresh", page=NS(url=url, kind=kind, chunks=chunks))
        index = NS(view=NS(pages={"example.org/personal/widget-cards": page("https://example.org/personal/widget-cards", "product", [own]),
                                  "example.org/blog/widget-tips": page("https://example.org/blog/widget-tips", "article", [blog])}))
        q = parse_query("How much interest do your widget cards charge?")
        self.assertEqual([c.id for _, c in product_page_rates(index, q, [(5.0, blog)])], [1])

    def test_loan_specs_branch_nouns_and_the_institution(self):
        from ingest import Chunk
        from retrieve import supports, use_proper_names
        q = parse_query("What is the rate on a 9/2 adjustable home loan?")
        self.assertEqual(q.specs, {"9/2"})
        row = lambda t: Chunk(1, "example.org/rates", "product", t, t)
        self.assertFalse(supports(row("Home Loans 4/1 adjustable 6.10% APR"), q))
        self.assertTrue(supports(row("Home Loans 9/2 adjustable 6.40% APR"), q))
        self.assertEqual(parse_query("What was the APR on 4/30/2031?").specs, set())    # a date, not a spec
        use_proper_names({"pinecrest"})
        try:
            self.assertEqual(parse_query("Hours at the Pinecrest location?").entity, "pinecrest")
            self.assertEqual(parse_query("Address of the pinecrest office?").entity, "pinecrest")
            self.assertIsNone(parse_query("What are the Main Office hours?").entity)
        finally:
            use_proper_names(set())
        self.assertNotIn("union", parse_query("What is the credit union's routing number?").tokens)
        self.assertEqual(parse_query("Do any of the credit union savings tiers pay 0.47% APY?").descriptors, set())
        self.assertIn("credit", parse_query("What is the credit card APR?").descriptors)

    def _two_crawls(self, client=None):
        from contract import Page, Snapshot
        ks = build()
        for sid, day, rate, apy in [("s1", "2030-01-10", "2.37%", "2.41%"), ("s2", "2030-06-15", "2.57%", "2.61%")]:
            md = f"# Certificates\n\n| Term | Dividend Rate | APY |\n|---|---|---|\n| 12 Month | {rate} | {apy} |\n"
            ks.ingest(Snapshot(sid, day, [Page("https://example.org/certificates", "Certificates | Example", md,
                                               f"{day}T09:00:00Z")]))
        return ks

    def test_a_change_question_gives_each_date_its_value(self):
        from solution import change_span
        ks = self._two_crawls()
        a = ks.answer("How did the 12-month certificate APY change between January 2030 and June 2030?")
        self.assertIn("2.41%", a.text)
        self.assertIn("2.61%", a.text)
        self.assertEqual(change_span("Has the 12-month APY gone up since January 2030?", ks._store)[0], date(2030, 1, 31))
        for plain in ("What is the 12-month certificate APY?", "How do I change my address?",
                      "Can I change my certificate term after opening it in January 2030?"):
            self.assertIsNone(change_span(plain, ks._store), plain)
        self.assertNotIn("2.41%", ks.answer("What is the 12-month certificate APY?").text)


class Rev30(unittest.TestCase):
    """The second look after a decline (synthetic data, fake model client)."""

    def _site(self, replies):
        import json as _json
        from types import SimpleNamespace as NS
        from contract import Page, Snapshot
        calls = []

        class Fake:
            class messages:
                @staticmethod
                def create(**k):
                    calls.append(k)
                    reply = replies[min(len(calls), len(replies)) - 1]
                    return NS(stop_reason="end_turn", content=[NS(type="text", text=_json.dumps(reply))])
        ks = build()
        pages = [("https://example.org/about/sign-up", "Sign Up | Example",
                  "# Sign Up\n\nYou can sign up if you live or work in the valley service area.\n"),
                 ("https://example.org/certificates", "Certificates | Example",
                  "# Certificates\n\n| Term | APY |\n|---|---|\n| 6 Month | 1.91% |\n| 12 Month | 2.41% |\n"),
                 ("https://example.org/membership-perks", "Perks | Example", "# Perks\n\nMembership perks include free coffee.\n")]
        ks.ingest(Snapshot("s1", "2030-01-10", [Page(u, t, m, "2030-01-10T09:00:00Z") for u, t, m in pages]))
        ks._synth.enabled, ks._synth._client = True, Fake()
        return ks, calls

    def test_an_open_question_gets_a_second_look_at_the_pages_its_title_names(self):
        declined = {"note": "", "answer": "", "answerable": False, "evidence_ids": []}
        picked = {"paths": ["/about/sign-up"]}
        answered = {"note": "", "answer": "You can sign up if you live or work in the valley service area.",
                    "answerable": True, "evidence_ids": [1]}
        ks, calls = self._site([declined, picked, answered])
        a = ks.answer("Who qualifies for membership?")
        self.assertEqual(len(calls), 3)
        self.assertIn("valley service area", a.text)
        self.assertTrue(any("sign-up" in s.url for s in a.sources))
        self.assertIn("/about/sign-up | Sign Up", calls[1]["system"][0]["text"])

    def test_a_declined_rate_question_gets_no_second_look(self):
        declined = {"note": "", "answer": "", "answerable": False, "evidence_ids": []}
        ks, calls = self._site([declined])
        a = ks.answer("Compare the APY on the 6-month and 12-month certificates.")
        self.assertEqual(len(calls), 1)
        self.assertFalse(a.confident)


class Rev31(unittest.TestCase):
    """Eligibility wording, links to the site's own portals, introductory rates (synthetic data only)."""

    def test_asking_who_can_join_also_searches_for_eligible(self):
        for q in ("Who can join?", "Can anyone become a member?", "Do I qualify for membership?",
                  "What are the requirements to join?", "Is membership open to everyone?"):
            self.assertIn(_stem("eligible"), parse_query(q).expansion, q)
            self.assertTrue({_stem("eligible"), _stem("join")} <= set(parse_query(q).tokens), q)
        self.assertEqual(parse_query("Who can join?").expansion, [_stem("eligible")])   # "join" was asked already
        for q in ("Who can I call about my card?", "Who can help me join?", "Can non-members cash a check?",
                  "What is the minimum balance requirement for checking?", "How do I join?",
                  "What is the rate for members who join?"):
            self.assertEqual(parse_query(q).expansion, [], q)

    def _portal_view(self, kind="product"):
        from types import SimpleNamespace as NS
        page = lambda key, kind: NS(status="fresh", snapshot_id="s1", page=NS(kind=kind, key=key, url="https://" + key, title=""))
        return NS(pages={"example.org/members": page("example.org/members", kind),
                         "example.org/about": page("example.org/about", "product")})

    def test_a_portal_link_needs_its_anchor_to_name_what_was_asked(self):
        from ingest import Chunk
        from solution import link_source, pick_link
        text = "Apply now (apply.example.org). Apply today (go.example.org/track). Elsewhere (apply.other.net)."
        ctx = [Chunk(1, "example.org/members", "product", text, text)]
        view = self._portal_view()
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link to apply?")), "apply.example.org")
        self.assertEqual(link_source("apply.example.org", ctx, view).page.key, "example.org/members")
        self.assertIsNone(pick_link(ctx, view, parse_query("What's the link?")))             # nothing asked
        self.assertIsNone(pick_link(ctx, view, parse_query("What's the link to apply for a loan?")))  # "loan" not named
        self.assertIsNone(pick_link(ctx, self._portal_view("article"), parse_query("What's the link to apply?")))
        join = [Chunk(2, "example.org/members", "product", "Join now (join.example.org).", "Join now (join.example.org).")]
        self.assertEqual(pick_link(join, view, parse_query("What's the link for who can join?")), "join.example.org")
        self.assertEqual(pick_link(join, view, parse_query("link to join online pls")), "join.example.org")

    def test_dead_link_filter_keeps_the_sites_own_subdomains(self):
        from types import SimpleNamespace as NS
        view = NS(pages={"example.org/rates": NS(status="fresh")})
        out = _drop_dead_links("Open it at join.example.org/products/savings today. See example.org/old/page now.", view)
        self.assertEqual(out, "Open it at join.example.org/products/savings today.")

    def test_an_intro_rate_question_needs_an_intro_rate(self):
        from contract import Page, Snapshot
        self.assertEqual(parse_query("What is the intro APR on the harbor line?").rate_kind, "intro")
        self.assertEqual(parse_query("What introductory dividend rate does the harbor account pay?").rate_kind, "intro")
        self.assertIsNone(parse_query("What APR is offered on the harbor line, and is there an introductory rate?").rate_kind)
        table = "| Product | APR |\n|---|---|\n| Harbor Line Standard | 6.10% |\n"
        old = "# Harbor Line\n\nHarbor Line intro rate: 1.49% intro APR for 3 months.\n\n" + table
        ks = build()
        for sid, day, md in (("s1", "2030-01-10", old), ("s2", "2030-03-10", "# Harbor Line\n\n" + table)):
            ks.ingest(Snapshot(sid, day, [Page("https://example.org/harbor-line", "Harbor Line | Example", md,
                                               f"{day}T09:00:00Z")]))
        self.assertIn("1.49%", ks.answer("What is the intro APR on the harbor line?", "2030-01-31").text)
        later = ks.answer("What is the intro APR on the harbor line?")
        self.assertFalse(later.confident)
        self.assertNotIn("6.10", later.text)
        self.assertIn("6.10%", ks.answer("What is the APR on the harbor line?").text)

    def test_one_cards_intro_offer_is_not_anothers(self):
        from contract import Page, Snapshot
        md = ("# Cards\n\n## Harbor Card\n\nNew cardholder offer: 2.49% intro APR on balance transfers for 6 months.\n\n"
              "| Card | APR |\n|---|---|\n| Harbor Card | 14.00% |\n| Coral Card | 16.00% |\n")
        ks = build()
        ks.ingest(Snapshot("s1", "2030-01-10", [Page("https://example.org/cards", "Cards | Example", md,
                                                     "2030-01-10T09:00:00Z")]))
        self.assertIn("2.49%", ks.answer("What is the intro APR on the Harbor card?").text)
        coral = ks.answer("What is the intro APR on the Coral card?")
        self.assertFalse(coral.confident)
        self.assertNotIn("2.49", coral.text)


class Rev32(unittest.TestCase):
    """Portal links given by the page about joining; specials the member names (synthetic data only)."""

    def _view(self):
        from types import SimpleNamespace as NS
        page = lambda key, kind, title: NS(status="fresh", snapshot_id="s1",
                                           page=NS(kind=kind, key=key, url="https://" + key, title=title))
        return NS(pages={"example.org/join": page("example.org/join", "product", "Join"),
                         "example.org/apply-info": page("example.org/apply-info", "product", "Apply"),
                         "example.org/grow-lp": page("example.org/grow-lp", "promo", "Grow your savings"),
                         "example.org/branches": page("example.org/branches", "product", "Branches")})

    def test_the_join_pages_portal_wins_a_tie_for_a_membership_question(self):
        from ingest import Chunk
        from solution import pick_link
        ch = lambda i, key, text: Chunk(i, key, "product", text, text)
        ctx = [ch(1, "example.org/join", "Join today (join.example.org)."),
               ch(2, "example.org/grow-lp", "Join in person (example.org/branches).")]
        widely = lambda u: 50 if "branches" in u else 3          # the branch page is linked from every page
        view = self._view()
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link to join?"), widely), "join.example.org")
        become = [ch(3, "example.org/join", "Become a member (join.example.org).")]
        self.assertEqual(pick_link(become, view, parse_query("URL to become a member?"), widely), "join.example.org")
        # not a membership question: ties are settled as before (the more widely linked page)
        apply = [ch(4, "example.org/apply-info", "Apply now (apply.example.org)."),
                 ch(5, "example.org/grow-lp", "Apply in person (example.org/branches).")]
        self.assertEqual(pick_link(apply, view, parse_query("What's the link to apply?"), widely), "example.org/branches")

    def test_a_named_special_answers_an_intro_rate_question(self):
        from contract import Page, Snapshot
        md = ("# Rates\n\n## Certificates\n\n| Certificate | APY |\n|---|---|\n"
              "| 9 Month Coral Certificate Special | 3.15% |\n| 12 Month Coral Certificate | 1.40% |\n")
        ks = build()
        ks.ingest(Snapshot("s1", "2030-01-10", [Page("https://example.org/rates", "Rates | Example", md,
                                                     "2030-01-10T09:00:00Z")]))
        self.assertEqual(parse_query("What is the intro rate on the certificate special?").rate_kind, "offer")
        self.assertEqual(parse_query("Is there a special intro rate on certificates?").rate_kind, "intro")
        self.assertIn("3.15%", ks.answer("What is the intro rate on the certificate special?").text)
        for q in ("What is the intro rate on certificates?", "Is there a special intro rate on certificates?",
                  "What is the intro rate on the savings special?"):
            a = ks.answer(q)
            self.assertFalse(a.confident, q)
            self.assertNotIn("3.15", a.text, q)


class Rev33(unittest.TestCase):
    """Joining link questions, calculators, "web address", money market (synthetic data only)."""

    def test_what_counts_as_joining(self):
        from solution import joining_info
        for q in ("What's the link to join?", "link to apply for membership", "URL to become a member?",
                  "Give me the link to sign up as a new member", "membership application link?",
                  "I'd like to register as a new member online. What link do I use?"):
            joining, _, wanted = joining_info(parse_query(q))
            self.assertTrue(joining, q)
            self.assertEqual(wanted, set(), q)
        for q in ("What is the link to become a board member?", "What's the link for the annual membership meeting?",
                  "link to join the savings challenge", "membership login link", "member services link"):
            self.assertFalse(joining_info(parse_query(q))[0], q)
        joining, _, wanted = joining_info(parse_query(
            "I started my membership application but didn't finish; what link do I use to pick it back up?"))
        self.assertTrue(joining)
        self.assertEqual(wanted, {_stem("continue")})

    def _view(self):
        from types import SimpleNamespace as NS
        page = lambda key, kind, title: NS(status="fresh", snapshot_id="s1",
                                           page=NS(kind=kind, key=key, url="https://" + key, title=title))
        return NS(pages={"example.org/join": page("example.org/join", "product", "Join"),
                         "example.org/grow-lp": page("example.org/grow-lp", "promo", "Grow your savings"),
                         "example.org/branches": page("example.org/branches", "product", "Branches"),
                         "example.org/loans": page("example.org/loans", "product", "Loans"),
                         "example.org/calculators/loan-payment": page("example.org/calculators/loan-payment",
                                                                      "calculator", "Payment calculator")})

    def test_a_joining_question_reads_the_join_page(self):
        from ingest import Chunk
        from solution import link_source, pick_link
        ch = lambda i, key, text: Chunk(i, key, "product", text, text)
        joins = [ch(1, "example.org/join", "Become a member (join.example.org). Continue my application "
                                            "(join.example.org/login).")]
        promo = [ch(2, "example.org/grow-lp", "Join online (join.example.org/grow). Join in person (example.org/branches).")]
        widely = lambda u: 50 if "branches" in u else 3
        view = self._view()
        q = parse_query("What's the link to join online?")
        self.assertEqual(pick_link(promo, view, q, widely, joins), "join.example.org")
        self.assertEqual(link_source("join.example.org", promo + joins, view).page.key, "example.org/join")
        resume = parse_query("I started a membership application but didn't finish it; what link picks it back up?")
        self.assertEqual(pick_link(promo, view, resume, widely, joins), "join.example.org/login")

    def test_a_calculator_is_not_the_link_unless_asked_for(self):
        from ingest import Chunk
        from solution import LINK_Q, pick_link
        text = "Loan payment calculator (example.org/calculators/loan-payment). Make a loan payment (example.org/loans)."
        ctx = [Chunk(1, "example.org/loans", "product", text, text)]
        view = self._view()
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link to make a loan payment?")), "example.org/loans")
        self.assertEqual(pick_link(ctx, view, parse_query("link to the loan payment calculator")),
                         "example.org/calculators/loan-payment")
        self.assertIsNotNone(LINK_Q.search("What's the web address for online banking?"))

    def test_money_market_names_the_product_of_an_intro_question(self):
        q = parse_query("What's the intro rate on the money market special?")
        self.assertEqual(q.category, {"market"})
        self.assertIn("money", q.descriptors)
        self.assertEqual(parse_query("What are the money market rates?").category, set())


class Rev34(unittest.TestCase):
    """Chatty joining questions, links for another line, intro wording, "under $X" (synthetic data only)."""

    def test_chatty_words_and_first_accounts(self):
        from solution import joining_info
        q = parse_query("What URL could I text my sister so she can sign up for membership online?")
        self.assertTrue({"text", "sister"} <= joining_info(q)[2])                  # no link vocabulary given
        joining, _, wanted = joining_info(q, {"text": 1, "board": 54})              # words no link names drop out
        self.assertTrue(joining)
        self.assertEqual(wanted, set())
        self.assertTrue(joining_info(parse_query("What's the link to open my first account?"), {})[0])
        self.assertFalse(joining_info(parse_query("What is the link to rename my first account?"), {})[0])
        self.assertFalse(joining_info(parse_query("What is the link to join the newsletter?"), {})[0])

    def _view(self):
        from types import SimpleNamespace as NS
        page = lambda key, kind, title: NS(status="fresh", snapshot_id="s1",
                                           page=NS(kind=kind, key=key, url="https://" + key, title=title))
        return NS(pages={k: page(k, kind, t) for k, kind, t in (
            ("example.org/savings", "product", "Savings"), ("example.org/blog/tips", "article", "Tips"),
            ("example.org/business/loans/vehicle", "product", "Business Vehicle Loans"),
            ("example.org/personal/auto-loans", "product", "Auto Loans"))})

    def test_a_joining_portal_beats_a_product_one(self):
        from ingest import Chunk
        from solution import pick_link
        text = ("Open an account (example.org/savings). Open my account (join.example.org/products). "
                "Become a member (join.example.org).")
        ctx = [Chunk(1, "example.org/savings", "product", text, text)]
        self.assertEqual(pick_link(ctx, self._view(), parse_query("What's the link to open my first account?")),
                         "join.example.org")

    def test_a_link_for_another_line_gives_way_to_the_page_for_what_was_asked(self):
        from ingest import Chunk
        from solution import _last_clause, pick_link
        self.assertIn("business", _last_clause("Not sure? What can I use a business auto loan for?"))
        text = "What can I use a business auto loan for? (example.org/business/loans/vehicle)"
        ctx = [Chunk(1, "example.org/blog/tips", "article", text, text)]
        view = self._view()
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link to apply for a car loan?")),
                         "example.org/personal/auto-loans")
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link for business car loans?")),
                         "example.org/business/loans/vehicle")

    def test_intro_wording_and_amounts_below(self):
        from retrieve import KIND_EV
        self.assertIn(_stem("introductory"), parse_query("What is the intro rate on the harbor account?").tokens)
        self.assertNotIn(_stem("introductory"), parse_query("What is the intro balance transfer fee?").tokens)
        self.assertIsNotNone(KIND_EV.search("apy for the first 3 months"))
        self.assertIsNone(KIND_EV.search("no payments for the first 3 months"))
        self.assertEqual(parse_query("What is the rate for balances under $7,300?").amounts, [7299.99])
        self.assertEqual(parse_query("What is the rate for a $7,300 balance?").amounts, [7300.0])


class Rev35(unittest.TestCase):
    """First accounts, the clause that asks for the link, application links (synthetic data only)."""

    def test_opening_a_first_account_is_joining_and_managing_one_is_not(self):
        from solution import joining_info
        for q in ("link to open my son's very first account", "What's the link to open my first account and become a member?",
                  "URL to set up her first account online"):
            self.assertTrue(joining_info(parse_query(q), {})[0], q)
        for q in ("What's the link to see my son's first account balance?", "What's the link to get my first account statement?",
                  "link to close my daughter's first account?"):
            self.assertFalse(joining_info(parse_query(q), {})[0], q)

    def test_only_the_clause_asking_for_the_link_counts(self):
        from solution import _link_clause_words
        self.assertNotIn("work", _link_clause_words("My friend from work wants an account. What's the URL he can use to join?"))
        self.assertIn("harbor", _link_clause_words("I'd like to become a member; what's the link to the harbor schedule?"))

    def _view(self):
        from types import SimpleNamespace as NS
        page = lambda key, kind, title: NS(status="fresh", snapshot_id="s1",
                                           page=NS(kind=kind, key=key, url="https://" + key, title=title))
        return NS(pages={k: page(k, kind, t) for k, kind, t in (
            ("example.org/join", "product", "Join"), ("example.org/canoes", "product", "Canoe Loans"))})

    def test_a_joining_question_with_context_elsewhere_gets_the_join_link(self):
        from ingest import Chunk
        from solution import pick_link
        joins = [Chunk(1, "example.org/join", "product", "Become a member (join.example.org).", "Become a member (join.example.org).")]
        q = parse_query("My friend from work wants to open his very first account. What's the URL he can use to join online?")
        self.assertEqual(pick_link([], self._view(), q, None, joins, {"work": 40}), "join.example.org")
        other = parse_query("I'd like to become a member; what's the link to the harbor schedule?")
        self.assertNotEqual(pick_link([], self._view(), other, None, joins, {"harbor": 9, "schedul": 9}), "join.example.org")

    def test_an_apply_question_gets_the_application_naming_the_product(self):
        from ingest import Chunk
        from solution import pick_link
        text = ("Canoe loans (example.org/canoes). Apply today (apply.example.org/loan-application/canoe-loan). "
                "Apply today (apply.example.org/loan-application/kayak-loan).")
        ctx = [Chunk(1, "example.org/canoes", "product", text, text)]
        view = self._view()
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link to apply for a canoe loan?")),
                         "apply.example.org/loan-application/canoe-loan")
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link to apply for a loan?")), "example.org/canoes")
        self.assertEqual(pick_link(ctx, view, parse_query("What do I need to apply for a canoe loan? Link please.")),
                         "example.org/canoes")


class Rev36(unittest.TestCase):
    """Requirement questions vs link requests; application words; vehicle types (synthetic data only)."""

    def test_only_a_requirements_question_skips_the_application(self):
        from solution import APPLY_NEEDS
        for q in ("I need the link to apply for a canoe loan online. What is it?",
                  "Which link do I need to apply for a canoe loan?", "What URL do I need to apply for a canoe loan?",
                  "Link to apply for a canoe loan? I have all my documents ready."):
            self.assertIsNone(APPLY_NEEDS.search(q), q)
        for q in ("What do I need to apply for a canoe loan?", "What will I need when applying for a canoe loan? Link please.",
                  "Which documents do I need to apply? Link?"):
            self.assertIsNotNone(APPLY_NEEDS.search(q), q)

    def _view(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        text = ("Auto loans (example.org/autos). Apply today (apply.example.org/loan-application/auto-loan). "
                "Apply today (apply.example.org/loan-application/canoe-loan).")
        chunk = Chunk(1, "example.org/autos", "product", text, text)
        page = lambda key, title, chunks=(): NS(status="fresh", snapshot_id="s1",
                                                page=NS(kind="product", key=key, url="https://" + key, title=title,
                                                        chunks=list(chunks)))
        return NS(pages={"example.org/autos": page("example.org/autos", "Auto Loans", [chunk]),
                         "example.org/canoe-cover": page("example.org/canoe-cover", "Canoe Insurance")}), [chunk]

    def test_vehicle_types_and_words_no_application_uses(self):
        from solution import pick_link
        view, ctx = self._view()
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link to apply for a truck loan?")),
                         "apply.example.org/loan-application/auto-loan")
        chatty = parse_query("My husband and I want to apply online for a canoe loan for the family. What link starts it?")
        self.assertEqual(pick_link(ctx, view, chatty), "apply.example.org/loan-application/canoe-loan")
        # a product the site has a page for ("insurance") is not another product's application
        self.assertNotEqual(pick_link(ctx, view, parse_query("What's the link to apply for canoe insurance?")),
                            "apply.example.org/loan-application/canoe-loan")


class Rev37(unittest.TestCase):
    """Context in another sentence, how a link is passed on, lines ruled out, line words in addresses
    (synthetic data only)."""

    def test_context_in_another_sentence_is_not_what_is_asked(self):
        from solution import _request_words
        def words(q, named=None):
            return _request_words(q, set(tokenize(q)), named)
        moved = "My cousin just moved to Harborview. Can you send me the link so she can join?"
        self.assertFalse(set(tokenize("harborview moved")) & words(moved))
        teen = ("My teen just started her first job and wants a canoe account. "
                "What's the web address where she can apply for membership?")
        self.assertFalse(set(tokenize("job canoe")) & words(teen))
        # kept: no one named as joining, an event, a second request, a line, or a page the site names
        for q, kept in (("The harbor picnic is on Saturday. What's the link to join?", "picnic"),
                        ("My son has a webinar tonight. Can you send me the link so he can join?", "webinar"),
                        ("My son wants a canoe loan. What's the link so he can join and apply?", "canoe"),
                        ("I need a business canoe account. What's the link to apply for membership?", "business")):
            self.assertTrue(set(tokenize(kept)) <= words(q), q)
        regatta = "My wife heard about the harbor regatta. Can you send me the link so she can join?"
        self.assertTrue(set(tokenize("regatta")) <= words(regatta, set(tokenize("regatta"))))
        self.assertFalse(set(tokenize("regatta")) & words(regatta))
        one = "What's the link on example.org where she can join?"        # a domain doesn't end a sentence
        self.assertEqual(words(one), set(tokenize(one)))

    def test_how_a_link_is_passed_on_is_not_what_it_is_for(self):
        from solution import joining_info
        for q in ("Can you share the link so my cousin can join?", "Email me the web address so my dad can become a member."):
            joining, _, wanted = joining_info(parse_query(q), {"share": 50, "email": 9, "web": 9})
            self.assertTrue(joining, q)
            self.assertFalse(wanted, q)

    def test_a_line_ruled_out_and_one_asked_about(self):
        self.assertEqual(parse_query("What's the canoe loan rate, not for my business?").excluded, {"business"})
        self.assertEqual(parse_query("Cashier's check fee? Except for business.").excluded, {"business"})
        self.assertNotIn("business", parse_query("Canoe loan rate, not for a business?").tokens)
        asked = parse_query("Not sure if I qualify or not for a jumbo special. What's the jumbo special rate?")
        self.assertEqual(asked.excluded, set())
        self.assertIn("jumbo", asked.qualifiers)

    def _view(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        text = ("Canoe loans (example.org/canoes). Apply today (apply.example.org/loan-application-personal/canoe-loan). "
                "Personal loans (example.org/personal-loans). "
                "Apply today (apply.example.org/loan-application-personal/personal-loan).")
        chunk = Chunk(1, "example.org/canoes", "product", text, text)
        page = lambda key, title, chunks=(): NS(status="fresh", snapshot_id="s1",
                                                page=NS(kind="product", key=key, url="https://" + key, title=title,
                                                        chunks=list(chunks)))
        return NS(pages={"example.org/canoes": page("example.org/canoes", "Canoe Loans", [chunk]),
                         "example.org/personal-loans": page("example.org/personal-loans", "Personal Loans")}), [chunk]

    def test_a_line_word_may_sit_anywhere_in_an_application_address(self):
        from solution import pick_link
        view, ctx = self._view()
        for q in ("What's the link to apply for a personal canoe loan?",
                  "I need the link to apply for a loan on a canoe. It's for personal use, not for a business."):
            self.assertEqual(pick_link(ctx, view, parse_query(q)), "apply.example.org/loan-application-personal/canoe-loan", q)
        self.assertEqual(pick_link(ctx, view, parse_query("What's the link to apply for a personal loan?")),
                         "apply.example.org/loan-application-personal/personal-loan")


class Rev38(unittest.TestCase):
    """Tabs that name a table under a plainer heading; one line written two ways; asides after a question
    (synthetic data only)."""

    def test_a_tab_names_the_table_under_a_plainer_heading(self):
        md = ("## Rates\n\n- [Kayak Savings Rates](https://example.org/personal/kayak-savings-account-options#)\n\n"
              "Saving Rates\n\n| Balance | Dividend Rate | APY |\n| --- | --- | --- |\n| $40,000+ | 0.45% | 0.45% |\n\n"
              "- [Canoe Loan Rates](https://example.org/personal/kayak-savings-account-options#)\n\n"
              "Canoe Loans\n\n| Term | Rate | APR |\n| --- | --- | --- |\n| 36 Months | 6.00% | 6.10% |\n")
        p = Parser()
        p.domains = {"example.org"}
        rows = [c for c in p.parse("example.org/personal/kayak-savings-account-options",
                                   "https://example.org/personal/kayak-savings-account-options", "Rates", md).chunks if c.is_row]
        self.assertIn("Kayak Savings Rates", rows[0].text)             # a long tab line still counts
        self.assertIn("Canoe Loans", rows[1].text)                     # the tab adds only "Rates": the heading stays

    def test_one_line_written_two_ways(self):
        from ingest import Chunk
        from retrieve import qualify
        q = parse_query("What's the HSA rate on $45,000?")
        self.assertTrue({"hsa", "health"} <= q.qualifiers)
        row = Chunk(1, "example.org/rates", "product", "Health Savings Account — $40,000+: 0.45%",
                    "Rates Health Savings Account — $40,000+: 0.45%", is_row=True, cells=["$40,000+", "0.45%"], value_cols=[1])
        self.assertNotEqual(qualify(row, q)[0], "mismatch")
        self.assertEqual(parse_query("How do I open an HSA?").qualifiers, {"hsa"})   # prose questions unchanged

    def test_an_aside_about_the_member_is_context(self):
        p = parse_query("What's the 17-month certificate special APY? I keep my personal savings with you.")
        self.assertNotIn("savings", p.category)
        self.assertNotIn("personal", p.descriptors)
        self.assertEqual(p.terms, {(17, "month")})
        self.assertEqual(parse_query("What are the canoe loan rates? I've been a member for 10 years.").terms, set())
        self.assertNotIn("business", parse_query("What's the canoe loan rate? My business banks elsewhere.").qualifiers)
        self.assertEqual(parse_query("What's the 60-month canoe loan APR? I have a 36-month car loan with you.").terms,
                         {(60, "month")})                               # a term in the question is never overridden

    def test_an_aside_that_says_what_is_meant_still_counts(self):
        self.assertEqual(parse_query("What's the certificate rate? I'm looking at the 24-month term.").terms, {(24, "month")})
        self.assertEqual(parse_query("Can you tell me the rate on a certificate? Five years.").terms, {(60, "month")})
        self.assertEqual(parse_query("What's the 12-month certificate rate? Also the 24-month.").terms,
                         {(12, "month"), (24, "month")})
        self.assertEqual(parse_query("What's the auto loan APR? My car is a 2025.").years, {2025})
        bare = parse_query("What's the rate? It's for a 12-month certificate.")
        self.assertIn("certificate", bare.category)
        self.assertEqual(bare.terms, {(12, "month")})


class Rev39(unittest.TestCase):
    """Amounts and tiers, calendar phrases, "by" and comparisons, sending a wire, one-word headings and product
    tabs, dates before the first crawl (synthetic data only)."""

    def test_amounts_that_are_not_balances_and_amounts_over(self):
        self.assertEqual(parse_query("I can put $400 a month into a kayak certificate. What APY would I get?").amounts, [])
        self.assertEqual(parse_query("I earn $70,000 a year. What's the 30-year canoe loan rate?").amounts, [])
        self.assertEqual(parse_query("What's the canoe loan rate on a loan over $40,000?").amounts, [40000.01])
        self.assertEqual(parse_query("I'm buying a $90,000 canoe with $10,000 down. What's the rate?").amounts, [80000.0])

    def _system(self):
        from contract import Page, Snapshot
        md = ("# Canoe Loan Rates\n\n| Term | Rate | APR |\n|---|---|---|\n"
              "| 30-Year Canoe Loan $500,000 or less | 6.50% | 6.60% |\n| 30-Year Canoe Loan $200,000 or less | 6.30% | 6.40% |\n"
              "| 30-Year Canoe Loan $80,000 or less | 6.10% | 6.20% |\n\n"
              "# Kayak Certificate Specials\n\n| Term | Dividend Rate | APY | Min. to Earn |\n|---|---|---|---|\n"
              "| 9 Month Kayak Certificate Special | 3.10% | 3.15% | $1,000 |\n"
              "| 9 Month Kayak Jumbo Certificate Special | 3.40% | 3.45% | $90,000 |\n\n"
              "Rates are effective as of 02/01/2030.\n")
        ks = build()
        ks.ingest(Snapshot("s1", "2030-03-01", [Page("https://example.org/rates", "Rates | Example", md, "2030-03-01T09:00:00Z")]))
        ks.ingest(Snapshot("s2", "2030-06-15", [Page("https://example.org/rates", "Rates | Example", md, "2030-06-15T09:00:00Z")]))
        return ks

    def test_the_tightest_tier_and_a_rows_minimum(self):
        ks = self._system()
        loan = ks.answer("What's the 30-year canoe loan rate for $70,000?").text
        self.assertIn("6.10%", loan)
        self.assertNotIn("6.50%", loan)
        small = ks.answer("What's the 9-month kayak certificate special APY for $5,000?").text
        self.assertIn("3.15%", small)
        self.assertNotIn("3.45%", small)
        big = ks.answer("What's the 9-month kayak certificate special APY for $95,000?").text
        self.assertIn("3.45%", big)
        self.assertNotIn("3.15%", big)
        self.assertNotIn("3.45%", ks.answer("What's the 9-month kayak certificate special APY, not the jumbo?").text)

    def test_a_date_before_the_first_crawl_reads_it_only_if_the_rate_was_in_effect(self):
        ks = self._system()
        self.assertIn("3.15%", ks.answer("What was the 9-month kayak certificate special APY for $5,000 on February 10, 2030?").text)
        early = ks.answer("What was the 9-month kayak certificate special APY for $5,000 in January 2030?")
        self.assertFalse(early.confident)                      # those rates took effect on February 1
        self.assertFalse(ks.answer("What was the 9-month kayak certificate special APY for $5,000?", "2030-01-15").confident)

    def test_calendar_phrases(self):
        from solution import calendar_phrases
        for text, want in (("On Labor Day 2030, what was it?", "On September 2, 2030, what was it?"),
                           ("In Q2 2030, what?", "In June 2030, what?"),
                           ("During the second half of 2030, what?", "During December 2030, what?"),
                           ("In the fall of 2030, what?", "In November 2030, what?"),
                           ("What was it in 2030-04?", "What was it in April 2030?")):
            self.assertEqual(calendar_phrases(text), want)
        for text in ("My fall 2030 transcript", "the 48-month APR for a 2011-12 model year canoe"):
            self.assertEqual(calendar_phrases(text), text)       # not a date

    def test_by_opens_a_moment_and_two_dated_comparisons(self):
        from types import SimpleNamespace as NS
        from solution import change_span, date_asked
        st = NS(views=[NS(taken_on=date(2030, 1, 10)), NS(taken_on=date(2030, 3, 1)), NS(taken_on=date(2030, 6, 15))])
        self.assertEqual(date_asked("By April 2030, what was the kayak APY?", st), "2030-04-30")
        self.assertIsNone(date_asked("The special had to be opened by April 30, 2030 — what's its APY?", st))
        self.assertEqual(change_span("Was the kayak APY the same on February 1, 2030 and on April 1, 2030?", st),
                         (date(2030, 2, 1), date(2030, 4, 1)))
        self.assertIsNone(change_span("Is the 9-month kayak certificate different from the 18-month one? I opened mine "
                                      "on February 1, 2030 and it matures April 1, 2031.", st))

    def test_sending_a_wire_is_never_the_incoming_fee(self):
        from ingest import Chunk
        from retrieve import supports
        q = parse_query("How much does it cost to send a canoe wire?")
        incoming = Chunk(1, "example.org/fees", "document", "Incoming canoe wire $10.00 each", "Fees Incoming canoe wire $10.00 each")
        outgoing = Chunk(2, "example.org/fees", "document", "Domestic canoe wire $20.00 each", "Fees Domestic canoe wire $20.00 each")
        self.assertFalse(supports(incoming, q))
        self.assertTrue(supports(outgoing, q))

    def test_one_word_headings_and_product_tabs(self):
        md = ("- [Rowboat Certificate Rates](https://example.org/rates#)\n\nSpecial\n\n| Term | APY |\n|---|---|\n| 9 Month | 3.15% |\n\n"
              "Standard\n\n| Term | APY |\n|---|---|\n| 12 Month | 2.95% |\n\n"
              "- [Canoe Account Rates](https://example.org/rates#)\n\nHarbor Choice\n\n| Balance | APY |\n|---|---|\n| $5,000+ | 0.55% |\n")
        p = Parser()
        p.domains = {"example.org"}
        rows = [c for c in p.parse("example.org/rates", "https://example.org/rates", "Rates", md).chunks if c.is_row]
        self.assertTrue(rows[0].text.startswith("Special"))                       # names a kind of product already
        self.assertTrue(rows[1].text.startswith("Standard Certificate"))          # a one-word heading, under the tab
        self.assertTrue(rows[2].text.startswith("Canoe Account Rates · Harbor Choice"))


class Rev40(unittest.TestCase):
    """Branch names never crash a question; the asked product's own row; wires and their recipients; year-only and
    relative dates; down payments; branches the site has (synthetic data only)."""

    def test_no_question_shape_raises(self):
        heads = ["What's the canoe loan rate at the Harbor branch", "What's the official check fee at the Elm Street branch",
                 "What are the hours of the Harbor branch", "What's the link to the kayak page for the Harbor branch",
                 "How much is a wire at the harbor branch", "What's the 12-month certificate rate"]
        tails = ["?", "? I'm a member.", "? It's for a 12-month certificate.", " as of March 2030?", "? Asking for a friend.",
                 ", not the jumbo?", " back in 2029?"]
        for h in heads:
            for t in tails:
                parse_query(h + t)                          # must not raise

    def _rows(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        def row(i, cell, apy, section, span):
            return Chunk(i, "example.org/rates", "product", f"{cell} {apy}", f"{section} Balance APY {cell} {apy}",
                         section=section, is_row=True, header=["Balance", "APY"], cells=[cell, apy], value_cols=[1], table_span=span)
        rows = [row(1, "$0.00 - $999.99", "0.05%", "Rates · Basic Savings", (1, 4)),
                row(2, "$1,000+", "0.17%", "Rates · Basic Savings", (1, 4)),
                row(3, "$0.01+", "1.92%", "Rates · Kayak Savings", (6, 8)),
                row(4, "$10,000+", "0.61%", "Rates · Canoe Money Market", (10, 12)),
                row(5, "$10,000+", "0.61%", "Rates · Health Savings Account", (14, 16)),
                row(6, "—", "0.00%", "Rates · Basic Checking", (18, 20)),
                row(7, "$5,000 - $19,999.99", "2.65%", "Rates · Kayak Checking", (22, 24))]
        view = NS(pages={"example.org/rates": NS(page=NS(chunks=rows))}, taken_on=date(2030, 6, 1))
        return rows, view, NS

    def test_the_asked_products_own_row_is_the_plainest_at_its_top_tier(self):
        from synth import named_row, named_row_guard
        rows, view, NS = self._rows()
        q = parse_query("Do any savings tiers pay 0.61% APY?")
        ev = NS(chunks=[rows[3], rows[4], rows[2]], dropped=[])
        self.assertIs(named_row(q, ev, view=view), rows[1])            # not the named or health account, nor the $0 tier
        text, own = named_row_guard("Do any savings tiers pay 0.61% APY?", q, ev, "Yes, the money market pays 0.61%.", view=view)
        self.assertIs(own, rows[1])
        self.assertIn("0.17%", text)
        checking = parse_query("What APY does checking pay on a $10,000 balance?")
        self.assertIs(named_row(checking, NS(chunks=[rows[6], rows[5]], dropped=[]), view=view), rows[6])   # the tier that fits

    def test_a_value_of_the_same_kind_of_product_appends_nothing(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        from synth import named_row_guard
        def row(i, cell, apy, section):
            return Chunk(i, "example.org/rates", "product", f"{cell} {apy}", f"{section} {cell} {apy}", section=section,
                         is_row=True, header=["Term", "APY"], cells=[cell, apy], value_cols=[1], table_span=(i, i))
        regular = row(1, "12 Month Certificate", "3.05%", "Rates · Certificates")
        jumbo = row(2, "12 Month Jumbo Certificate", "3.45%", "Rates · Certificates")
        q = parse_query("Do any certificates pay 3.45% APY?")
        self.assertIsNone(named_row_guard("Do any certificates pay 3.45% APY?", q, NS(chunks=[jumbo, regular], dropped=[]),
                                          "Yes, the 12-month jumbo pays 3.45%.")[1])

    def _fees(self):
        from contract import Page, Snapshot
        fees = ("# Fee Schedule\n\n| Service | Fee |\n|---|---|\n| Domestic wire | $20.00 each |\n| Incoming wire | $10.00 |\n"
                "| International wire | $45.00 each |\n| Official check | $5.00 each |\n| Stop payment | $30.00 per item |\n")
        contact = "# Contact Us\n\n## Harbor Branch\n\n12 Main St., Rivertown, ST 00000\n\nMon - Fri 9 a.m. - 4 p.m.\n"
        ks = build()
        for sid, day in (("s1", "2030-01-10"), ("s2", "2030-06-15")):
            ks.ingest(Snapshot(sid, day, [Page("https://example.org/fees", "Fees | Example", fees, f"{day}T09:00:00Z"),
                                          Page("https://example.org/contact", "Contact | Example", contact, f"{day}T09:00:00Z")]))
        return ks

    def test_a_wires_recipient_is_not_what_is_charged(self):
        from ingest import Chunk
        from retrieve import supports
        ks = self._fees()
        self.assertIn("$20.00", ks.answer("How much does it cost to wire money to my nephew's bank in Oakvale?").text)
        self.assertFalse(ks.answer("What does it cost to send money to Zapcash?").confident)     # not a wire: still declines
        self.assertIn("$5.00", ks.answer("What's the fee for an official check (bank draft)?").text)
        incoming = Chunk(1, "example.org/fees", "document", "Incoming wire $10.00", "Fees Incoming wire $10.00")
        self.assertTrue(supports(incoming, parse_query("What's the fee for a wire to my checking account?")))   # received
        self.assertFalse(supports(incoming, parse_query("What's the fee for a wire to my nephew's bank?")))     # sent

    def test_a_branch_the_site_has_does_not_narrow_a_fee(self):
        ks = self._fees()
        self.assertIn("$5.00", ks.answer("What's the official check fee at the Harbor branch?").text)
        self.assertFalse(ks.answer("What's the official check fee at the Oakdale branch?").confident)
        self.assertIn("$5.00", ks.answer("What's the official check fee at your Harbor location?").text)
        self.assertFalse(ks.answer("What's the official check fee at your Oakdale location?").confident)
        self.assertEqual(parse_query("What's the canoe loan rate at your Harbor, Ohio branch?").entity, "harbor")

    def test_only_what_a_wire_fee_line_can_say_narrows_a_wire_question(self):
        ks = self._fees()
        self.assertIn("$20.00", ks.answer("My cousin banks in Oakvale and I want to wire her some money for her "
                                          "graduation. What do you charge to send it?").text)
        self.assertIn("$10.00", ks.answer("Someone is wiring money to me from another bank. Is there a fee for that "
                                          "incoming wire into my account?").text)
        self.assertIn("$45.00", ks.answer("What's the fee for an international wire to my cousin?").text)

    def test_a_percent_paid_is_a_rate_question(self):
        self.assertEqual(parse_query("Is there a kayak savings tier that earns 0.45%?").attribute, "rate")
        self.assertNotEqual(parse_query("Do I earn 2% cash back on the canoe card?").attribute, "rate")

    def test_down_payments(self):
        self.assertEqual(parse_query("I'm buying a $250,000 canoe with $50,000 down. What's the rate?").amounts, [200000.0])
        self.assertEqual(parse_query("I'm buying a $250,000 canoe and putting 20% down. What's the rate?").amounts, [200000.0])
        self.assertEqual(parse_query("What's the rate with a down payment of $40,000 on a $200,000 canoe?").amounts, [160000.0])
        self.assertEqual(parse_query("What's the rate on a $240,000 loan if I put $50,000 down?").amounts, [240000.0])
        self.assertEqual(parse_query("What's the rate on a canoe loan of $250,000 with $60,000 down?").amounts, [250000.0])

    def test_year_only_and_relative_dates(self):
        from types import SimpleNamespace as NS
        from solution import calendar_phrases, date_asked
        st = NS(views=[NS(taken_on=date(2030, 1, 10)), NS(taken_on=date(2030, 3, 1)), NS(taken_on=date(2030, 6, 15))])
        self.assertEqual(calendar_phrases("As of 2029, what was it?"), "As of December 2029, what was it?")
        self.assertEqual(calendar_phrases("the 2029 canoe grant"), "the 2029 canoe grant")
        self.assertEqual(date_asked("As of 2029, what was the canoe fee?", st), "2029-12-31")
        self.assertEqual(date_asked("Before 2030, what was the canoe fee?", st), "2029-12-31")
        self.assertEqual(date_asked("What was the canoe fee in 2029?", st), "2029-12-31")      # a value: a moment
        self.assertIsNone(date_asked("How many canoes did you sell in 2029?", st))             # a count: a period
        self.assertIsNone(date_asked("What was the APR on a canoe trailer built prior to 2029?", st))   # a model year
        self.assertEqual(date_asked("What was the kayak APY last March?", st), "2030-03-31")
        self.assertEqual(date_asked("What was the kayak APY a year ago?", st), "2029-07-15")   # now: a month after the newest crawl
        self.assertIsNone(date_asked("What was the kayak APY last month?", st))                 # June 2030: the newest crawl
        self.assertEqual(date_asked("What was the canoe fee last year?", st), "2029-12-31")
        self.assertIsNone(date_asked("How many canoe members joined last year?", st))
        self.assertIsNone(date_asked("Have canoe loan rates gone up over the last year?", st))  # a span
        self.assertEqual(date_asked("Right now, as of March 2030, what's the kayak APY?", st), "2030-03-31")
        self.assertIsNone(date_asked("Is the kayak APY now higher than back in March 2030?", st))

    def test_an_amount_over_a_certificate_minimum_still_asks_about_savings(self):
        from contract import Page, Snapshot
        md = ("# Rates\n\n## Basic Savings\n\n| Balance | APY |\n|---|---|\n| $0.00 - $19,999.99 | 0.10% |\n| $20,000+ | 0.23% |\n\n"
              "## Certificates\n\n| Term | APY | Min. to Earn |\n|---|---|---|\n| 12 Month | 3.05% | $1,000 |\n")
        ks = build()
        ks.ingest(Snapshot("s1", "2030-01-10", [Page("https://example.org/rates", "Rates | Example", md, "2030-01-10T09:00:00Z")]))
        text = ks.answer("What APY would $30,000 in savings earn?").text
        self.assertIn("0.23%", text)
        self.assertNotIn("3.05%", text)


class Rev41(unittest.TestCase):
    """Dates before the first crawl and their lower bound; which way a wire goes; value questions about a year or a
    recent month (synthetic data only)."""

    def test_a_day_before_the_first_crawl_needs_what_was_in_effect_then(self):
        from contract import Page, Snapshot
        undated = "# Fee Schedule\n\n| Service | Fee |\n|---|---|\n| Canoe wire | $20.00 each |\n"
        dated = undated + "\nFees effective as of 02/01/2030.\n"
        kayak = "# Kayak Rentals\n\nThe kayak launch fee is $7.00 per day.\n"
        ks = build()
        ks.ingest(Snapshot("s1", "2030-03-01", [Page("https://example.org/fees", "Fees | Example", undated, "2030-03-01T09:00:00Z"),
                                                Page("https://example.org/kayak", "Kayak | Example", kayak, "2030-03-01T09:00:00Z")]))
        ks.ingest(Snapshot("s2", "2030-06-15", [Page("https://example.org/fees", "Fees | Example", dated, "2030-06-15T09:00:00Z"),
                                                Page("https://example.org/kayak", "Kayak | Example", kayak, "2030-06-15T09:00:00Z")]))
        self.assertIn("$20.00", ks.answer("What was the canoe wire fee as of February 15, 2030?").text)
        early = ks.answer("What was the canoe wire fee in 2028?")      # a later crawl dates the schedule February 2030
        self.assertFalse(early.confident)
        self.assertEqual(numbers(early.text), set())
        self.assertIn("$7.00", ks.answer("What was the kayak launch fee in 2029?").text)     # undated, the year before
        self.assertFalse(ks.answer("What was the kayak launch fee in 2027?").confident)      # undated, long before

    def test_which_way_a_wire_goes(self):
        from retrieve import wire_direction
        for text, way in [("My cousin wants to wire me money from Germany. Is there a fee?", "in"),
                          ("What's the fee for wires from overseas?", "in"),
                          ("Is there a fee for wire transfers coming in from abroad?", "in"),
                          ("What's the fee to receive an international wire?", "in"),
                          ("What's the fee for a wire to my checking account?", "in"),
                          ("What's the fee to send a wire to my own account at another bank?", "out"),
                          ("How much is a wire?", "out"),
                          ("How much to wire money to my brother in Georgia?", "out"),
                          ("What's the fee to wire money to Puerto Rico?", "out"),
                          ("What's the fee to wire money to my sister in Birmingham?", "out"),
                          ("How much to wire money to my cousin in Paris, Texas?", "out"),
                          ("How much to wire $500 to Chad?", "out"),
                          ("What's the fee to wire money to my sons account in Toronto?", "abroad"),
                          ("How much to wire money to a Canadian bank account?", "abroad"),
                          ("What's the fee to wire a payment in euros?", "abroad"),
                          ("What does it cost to wire money to Europe?", "abroad"),
                          ("How much to wire money to the Philippines?", "abroad"),
                          ("What's the fee to wire money internationally?", "abroad"),
                          ("How much to wire money to my friend in Lyon, France?", "abroad"),
                          ("Do you charge for domestic and international wires?", "all"),
                          ("What are your wire transfer fees?", "all")]:
            self.assertEqual(wire_direction(text), way, text)

    def test_wire_answers_and_how_much_to(self):
        ks = Rev40()._fees()
        self.assertIn("$45.00", ks.answer("What's the fee to wire money to my friend in Toronto?").text)
        self.assertIn("$10.00", ks.answer("My cousin wants to wire me money from Germany. Is there a fee?").text)
        self.assertIn("$10.00", ks.answer("My employer is wiring my bonus into my checking account. Is there a fee "
                                          "for receiving it?").text)          # "wiring" is searched as "wire"
        both = ks.answer("What are your wire transfer fees?").text
        self.assertTrue(all(v in both for v in ("$10.00", "$20.00", "$45.00")), both)
        self.assertEqual(parse_query("How much to wire money overseas?").attribute, "fee")
        self.assertNotEqual(parse_query("How much to open a canoe account?").attribute, "fee")

    def test_a_table_continued_after_a_page_break_has_no_header(self):
        md = "# Fee Schedule\n\n|Official kayak check|$5.00* each|\n|---|---|\n|Incoming canoe wire|$10.00|\n|Domestic canoe wire|$20.00 each|\n"
        p = Parser()
        p.domains = {"example.org"}
        rows = [c for c in p.parse("example.org/fees", "https://example.org/fees", "Fees", md).chunks if c.is_row]
        self.assertEqual([r.cells[0] for r in rows], ["Official kayak check", "Incoming canoe wire", "Domestic canoe wire"])
        self.assertNotIn("$5.00", rows[1].text)          # the first row's fee is not the second row's column

    def test_a_year_or_a_recent_month_in_a_value_question(self):
        from types import SimpleNamespace as NS
        from solution import date_asked
        st = NS(views=[NS(taken_on=date(2030, 1, 10)), NS(taken_on=date(2030, 3, 1)), NS(taken_on=date(2030, 6, 15))])
        self.assertEqual(date_asked("What was the kayak APY two months ago?", st), "2030-05-31")     # now: mid-July
        self.assertEqual(date_asked("How much was a canoe wire last year?", st), "2029-12-31")
        self.assertEqual(date_asked("What was the canoe wire fee last year?", st), "2029-12-31")     # its twin, the same
        self.assertIsNone(date_asked("Is the canoe fee charged in the last week of the month?", st))
        self.assertIsNone(date_asked("What did the canoe club announce last month?", st))           # an event


class Rev42(unittest.TestCase):
    """Relative clauses in fee questions; the model's question with its date written out; the asked kind of product
    before another kind under the same heading; a wire someone else sends (synthetic data only)."""

    def _fees(self):
        from contract import Page, Snapshot
        md = ("# Fee Schedule\n\n| Service | Fee |\n|---|---|\n| Canoe check stop payment | $30.00 per item |\n"
              "| Re-key canoe locker due to lost key | $100.00 |\n| Kayak money order | $5.00 each |\n")
        ks = build()
        ks.ingest(Snapshot("s1", "2030-06-15", [Page("https://example.org/fees", "Fees | Example", md, "2030-06-15T09:00:00Z")]))
        return ks

    def test_a_relative_clause_is_not_what_is_charged_unless_a_fee_names_it(self):
        import retrieve
        ks = self._fees()
        idx = ks._indexes[-1]
        old, retrieve.RARE_SHARE = retrieve.RARE_SHARE, 0.5         # in a tiny index, make one-line words rare
        try:
            def subject(text):
                q = parse_query(text)
                return {w for alts in retrieve.rare_subject(idx, q) for w in alts}
            self.assertNotIn(_stem("wrote"), subject("What would it cost to stop a canoe check I wrote?"))
            self.assertNotIn(_stem("wife"), subject("What would it cost to stop a canoe check my wife wrote?"))
            self.assertIn(_stem("lost"), subject("What's the fee for a canoe card I lost?"))        # a fee names "lost"
            self.assertIn(_stem("overdraw"), subject("What's the canoe fee each time I overdraw?"))  # not a relative clause
            self.assertNotIn(_stem("yesterday"), subject("What would it cost to stop a canoe check from yesterday?"))
        finally:
            retrieve.RARE_SHARE = old
        self.assertIn("$30.00", ks.answer("What would it cost to stop payment on a canoe check I wrote?").text)

    def test_the_model_reads_the_date_the_question_named(self):
        from contract import Page, Snapshot
        md = "# Kayak Rates\n\n| Account | APY |\n|---|---|\n| Kayak Savings | {} |\n"
        ks = build()
        for sid, day, apy in (("s1", "2030-01-10", "1.10%"), ("s2", "2030-03-01", "1.20%"), ("s3", "2030-06-15", "1.30%")):
            ks.ingest(Snapshot(sid, day, [Page("https://example.org/rates", "Rates | Example", md.format(apy), f"{day}T09:00:00Z")]))
        sent = []
        class Fake:
            class messages:
                @staticmethod
                def create(**k):
                    sent.append(k["messages"][0]["content"])
                    raise RuntimeError("offline")
        ks._synth.enabled, ks._synth._client = True, Fake()
        ks.answer("What was the Kayak Savings APY last March?")
        self.assertIn("Question: What was the Kayak Savings APY in March 2030?", sent[-1])
        ks.answer("What was the Kayak Savings APY last March?", "2030-06-20")             # as_of: as written
        self.assertIn("Question: What was the Kayak Savings APY last March?", sent[-1])

    def test_the_asked_kind_of_product_before_another_under_the_same_heading(self):
        from contract import Page, Snapshot
        md = ("# Kayak Savings Rates\n\n| Term | APY | Min. to Earn |\n|---|---|---|\n| 9 Month Certificate Special | 3.15% | $1,000 |\n\n"
              "## Basic Savings\n\n| Balance | APY |\n|---|---|\n| $0.00 - $999.99 | 0.05% |\n| $1,000+ | 0.17% |\n")
        ks = build()
        ks.ingest(Snapshot("s1", "2030-06-15", [Page("https://example.org/rates", "Rates | Example", md, "2030-06-15T09:00:00Z")]))
        text = ks.answer("What is the savings rate?").text
        self.assertIn("0.17%", text)
        self.assertNotIn("3.15%", text)
        self.assertIn("3.15%", ks.answer("What's the 9-month certificate special APY?").text)
        sent = []
        class Fake:
            class messages:
                @staticmethod
                def create(**k):
                    sent.append(k["messages"][0]["content"])
                    raise RuntimeError("offline")
        ks._synth.enabled, ks._synth._client = True, Fake()
        ks._cache.clear()
        ks.answer("What is the savings rate?")
        self.assertIn("0.17%", sent[-1])               # the model reads the savings table ...
        self.assertNotIn("3.15%", sent[-1])            # ... and not the certificate under the same heading

    def test_kinds_rate_cues_and_descriptors(self):
        from retrieve import kinds_in, wire_direction
        self.assertEqual(kinds_in("Canoe CD Rates"), {"certificate"})
        self.assertEqual(kinds_in("Premier Money Market Savings"), {"money market"})
        self.assertEqual(parse_query("What does a canoe savings account pay?").attribute, "rate")
        self.assertEqual(parse_query("What will my canoe checking account pay in fees each month?").attribute, "fee")
        self.assertEqual(parse_query("What was the 9-month kayak certificate special paying two months ago?").attribute, "rate")
        self.assertNotIn("apy", parse_query("What APY does canoe savings earn?").descriptors)
        self.assertEqual(wire_direction("What's the fee for a wire my employer sends to my account?"), "in")
        self.assertEqual(wire_direction("What's the fee when I send a wire to my account at another bank?"), "out")


class Rev43(unittest.TestCase):
    """What a fee is for (fee items, state words, clauses, who and why); rate cues, amounts in words, ranges,
    vehicles; the asked kind of product missing; the site's own name before "branch" (synthetic data only)."""

    FEES = ("# Fee Schedule\n\n| Service | Fee |\n|---|---|\n| Canoe check stop payment | $30.00 per item |\n"
            "| Canoe check cashing | $8.00 each |\n| Official canoe check | $5.00 each |\n"
            "| Re-key canoe locker due to lost key | $100.00 |\n| Kayak money order | $5.00 each |\n"
            "| Paddle rush fee for harbor card | $50.00* |\n| Harbor transaction fee | 1% of Amount |\n"
            "| Canoe locker 3 X 5 | $25.00 |\n| Canoe locker 5 X 10 | $60.00 |\n| Canoe locker past due late fee | $10.00 |\n"
            "| Abandoned canoe account fee - unclaimed share fee before escheating to the harbor state | $50.00 |\n"
            "| Domestic wire | $25.00 each |\n| International wire transfer | $55.00 each |\n| Incoming wire transfer | $10.00 |\n\n"
            "Harbor Service Charges Harbor card transactions at Canoe ATMs Free Paddle ATM usage (after 8 free per month) "
            "$0.50* per transaction Harbor exchange fee 2% of Amount\n\n"
            "Other fees include: Canoe advance fee—$10.00 or 3.00% of the amount of each canoe advance, whichever is "
            "greater; late payment fee—up to $15.00; returned payment fee—up to $25.00.\n\n"
            "We won't charge canoe overdraft fees for transactions less than $5.\n\n"
            "The Canoe Overdraft Privilege service fee is $30 per overdraft.\n\n"
            "To waive the $2.00 monthly service fee, keep a balance of $2,500 in your kayak HSA.\n")

    def _ks(self, md=None, url="https://example.org/canoe-credit-union-fees"):
        from contract import Page, Snapshot
        ks = build()
        ks.ingest(Snapshot("s1", "2030-06-15", [Page(url, "Fees | Example", md or self.FEES, "2030-06-15T09:00:00Z")]))
        return ks

    def _subject(self, ks, text):
        import retrieve
        old, retrieve.RARE_SHARE = retrieve.RARE_SHARE, 0.5         # in a tiny index, make few-chunk words rare
        try:
            return {w for alts in retrieve.rare_subject(ks._indexes[-1], parse_query(text)) for w in alts}
        finally:
            retrieve.RARE_SHARE = old

    def test_fee_items_labels_thresholds_and_tails(self):
        from retrieve import fee_items
        items = fee_items("Harbor card transactions at Canoe ATMs Free Paddle rush fee for kayak card $45.00* "
                          "Harbor exchange fee 2% of Amount")
        self.assertEqual(len(items), 3)
        self.assertEqual(items[1][1], "Paddle rush fee for kayak card $45.00*")
        self.assertEqual(items[2][1], "Harbor exchange fee 2% of Amount")
        self.assertTrue(fee_items("We won't charge overdraft fees for transactions less than $5.")[0][2])
        self.assertEqual(fee_items("Notary services $10.00 for non-members")[0][1], "Notary services $10.00 for non-members")
        self.assertIn("whichever is greater", fee_items("Canoe advance fee—$10.00 or 3.00% of the amount, whichever is "
                                                        "greater; late fee—up to $15.00")[0][1])

    def test_a_lost_thing_needs_a_fee_for_that_lost_thing(self):
        ks = self._ks()
        declined = lambda q: not ks.answer(q).confident
        self.assertTrue(declined("What's the fee for a lost kayak money order?"))       # only "lost key" is priced
        self.assertTrue(declined("Is there a fee if I lose my harbor card?"))
        self.assertTrue(declined("How much is a lost credit card?"))                   # "credit" only in the URL
        self.assertIn("$100.00", ks.answer("How much is a lost canoe locker key?").text)
        self.assertIn("$100.00", ks.answer("What's the fee if I lose my canoe locker key?").text)
        self.assertIn("$30.00", ks.answer("What's the fee to stop payment on a lost canoe check?").text)
        self.assertIn("$50.00", ks.answer("My harbor card was stolen. How much is a paddle rush fee for a harbor card?").text)
        self.assertIn("$25.00", ks.answer("What's the fee if my canoe payment bounces?").text)    # bounced = returned
        self.assertTrue(declined("A canoe check my tenant gave me bounced after I deposited it — what's your returned "
                                 "deposited item fee?"))                                 # "after" names no fee

    def test_what_was_done_with_it_and_who_and_why(self):
        ks = self._ks()
        self.assertNotIn(_stem("written"), self._subject(ks, "How much is it to stop payment on a canoe check written by me?"))
        self.assertNotIn(_stem("bounced"), self._subject(ks, "What's the stop payment fee for a canoe check that bounced?"))
        self.assertIn(_stem("wrote"), self._subject(ks, "What's the fee for a canoe check I wrote?"))   # nothing named
        self.assertIn(_stem("lose"), self._subject(ks, "Is there a fee if I lose my harbor card?"))      # "if" is no noun
        s = self._subject(ks, "My son needs an official canoe check for his apartment deposit — how much do you charge for one?")
        self.assertNotIn(_stem("son"), s)
        self.assertNotIn(_stem("apartment"), s)
        s = self._subject(ks, "I'm sending a wire to my cousin, who lives in Germany — what will that cost me?")
        self.assertNotIn(_stem("lives"), s)
        self.assertIn(_stem("overdraft"), self._subject(ks, "What's the canoe fee each time I overdraw?"))
        self.assertNotIn("\x00unpriced", self._subject(ks, "What's the monthly fee for a kayak HSA?"))   # a product word
        self.assertIn("$5.00", ks.answer("My son needs an official canoe check for his apartment deposit — how much do you charge for one?").text)
        self.assertIn("$55.00", ks.answer("I'm sending a wire to my cousin, who lives in Germany — what will that cost me?").text)
        self.assertIn("$5.00", ks.answer("For a kayak money order, what's the fee?").text)
        self.assertFalse(ks.answer("What does it cost to send money to Venmo?").confident)
        self.assertIn("$2.00", ks.answer("What's the monthly fee for a kayak HSA?").text)

    def test_fee_answers_name_one_fee(self):
        ks = self._ks()
        text = ks.answer("How much is a canoe locker?").text
        self.assertIn("$25.00", text)
        self.assertIn("$60.00", text)
        self.assertNotIn("$10.00", text)                    # the late fee is another fee, not a size
        self.assertIn("$50.00", ks.answer("What is the abandoned canoe account fee?").text)
        self.assertIn("1% of Amount", ks.answer("What's the harbor transaction fee?").text)
        self.assertIn("3.00%", ks.answer("What's the canoe advance fee?").text)
        self.assertIn("$30", ks.answer("What's the canoe overdraft fee?").text)
        self.assertIn("$5", ks.answer("Will I be charged a canoe overdraft fee for a transaction under $5?").text)
        text = ks.answer("Is the international wire fee higher than domestic?").text
        self.assertIn("$55.00", text)
        self.assertIn("$25.00", text)
        self.assertNotIn("$10.00", text)                    # only the two kinds named
        text = ks.answer("What's the fee on an incoming wire versus an outgoing domestic one?").text
        self.assertIn("$10.00", text)
        self.assertIn("$25.00", text)
        self.assertNotIn("$55.00", text)

    def test_rate_cues_amounts_ranges_and_vehicles(self):
        from retrieve import wire_direction
        self.assertEqual(parse_query("What does the canoe HSA pay?").attribute, "rate")
        self.assertEqual(parse_query("If I lock my money in a 2-year kayak CD, what does it earn?").attribute, "rate")
        self.assertEqual(parse_query("How much does a one-year kayak certificate pay?").attribute, "rate")
        self.assertNotEqual(parse_query("What does a canoe HSA pay for?").attribute, "rate")
        self.assertNotIn(_stem("pay"), parse_query("What does a canoe savings account pay?").tokens)
        self.assertEqual(parse_query("What does a canoe HSA pay on 15k?").amounts, [15000])
        self.assertEqual(parse_query("What does a canoe HSA pay on fifteen thousand dollars?").amounts, [15000])
        self.assertEqual(parse_query("What does a canoe HSA pay with five grand?").amounts, [5000])
        self.assertEqual(parse_query("What does a canoe HSA pay on 7,500 dollars?").amounts, [7500])
        self.assertEqual(parse_query("Can I roll my 401k into a canoe IRA? What's the rate?").amounts, [])
        for text in ("canoe savings rate for a balance between $20,000 and $50,000?", "canoe savings rate for 20k to 50k?",
                     "canoe savings rate for a balance between 20 and 50 thousand dollars?"):
            self.assertEqual(parse_query(text).amounts, [20000, 49999.99], text)
        self.assertEqual(parse_query("canoe savings rate for a $20k-$48,500 balance?").amounts, [20000, 48500])
        self.assertEqual(parse_query("What's the canoe certificate rate for 12 to 24 months?").amounts, [])
        self.assertIn("loan", parse_query("If I finance a 2030 car for 7 years, what APR is that?").category)
        self.assertNotIn("loan", parse_query("What's the APY if I'm saving for a car?").category)
        self.assertNotIn("loan", parse_query("What's the kayak rate? I have a 60-month car loan with you.").category)
        self.assertIn("rv", parse_query("What's the rate on a camper?").descriptors)
        self.assertEqual(parse_query("I'm financing a 2028 pickup over 84 months, what rate would I get?").years, {2028})
        self.assertEqual(wire_direction("Is the international wire fee higher than domestic?"), "all")
        self.assertEqual(wire_direction("Compare the fees for sending a wire domestically versus internationally"), "all")

    def test_a_missing_kind_of_product_declines(self):
        md = ("# Kayak Rates\n\n## Kayak Certificates\n\n| Term | APY |\n|---|---|\n| 12 Month | 3.10% |\n| 24 Month | 3.20% |\n\n"
              "## Kayak Auto Loans\n\n| Term | APR |\n|---|---|\n| 84 Months | 6.15% |\n")
        ks = self._ks(md, "https://example.org/rates")
        self.assertFalse(ks.answer("What's the rate on an 84-month kayak certificate?").confident)
        self.assertIn("3.20%", ks.answer("What's the 24-month kayak certificate rate?").text)

    def test_the_sites_own_name_is_never_a_branch(self):
        import retrieve
        saved = (set(retrieve.SITE_WORDS), set(retrieve.PROPER_NAMES))
        try:
            retrieve.use_site_words({"canoeco"})
            retrieve.use_proper_names({"canoeco's", "elmwood"})
            q = parse_query("What are Canoeco's general branch lobby hours of operation?")
            self.assertIsNone(q.entity)
            self.assertEqual(q.attribute, "hours")
            self.assertEqual(parse_query("When are Canoeco's branch lobbies usually open?").attribute, "hours")
            self.assertEqual(parse_query("What are the hours of the elmwood branch?").entity, "elmwood")
        finally:
            retrieve.use_site_words(saved[0])
            retrieve.use_proper_names(saved[1])


class Contract(unittest.TestCase):
    def test_exactly_the_three_contract_methods(self):
        own = {n for n, v in vars(type(build())).items() if callable(v) and not n.startswith("__")}
        self.assertEqual(own, {"ingest", "answer", "conflicts"})

    def test_refusal_is_digit_free_and_reads_as_a_refusal(self):
        self.assertFalse(numbers(REFUSAL))
        self.assertTrue(scoring.REFUSAL_RE.search(REFUSAL))

    def test_sentence_split_never_cuts_inside_times(self):
        self.assertEqual(_sentences("Open 9 a.m. Monday to 4 p.m. at Main Street. Closed Sunday."),
                         ["Open 9 a.m. Monday to 4 p.m. at Main Street.", "Closed Sunday."])
        self.assertEqual(_drop_banned("Hours are 9 a.m. to noon. These excerpts disagree."), "Hours are 9 a.m. to noon.")

    def test_answers_never_link_to_a_deep_page_missing_from_the_snapshot(self):
        view = View("s2", date(2030, 6, 15), {"example.org/borrow/sample-loan": None})
        self.assertEqual(_drop_dead_links("Use example.org/old/retired-page. Or call us.", view), "Or call us.")
        kept = "See example.org/borrow/sample-loan. Apply at example.org/join."   # crawled; vanity
        self.assertEqual(_drop_dead_links(kept, view), kept)

    def test_empty_system_declines_instead_of_raising(self):
        a = build().answer("What is the routing number?")
        self.assertIsInstance(a, Answer)                    # the harness's own class (repeat check)
        self.assertFalse(a.confident)
        self.assertEqual(build().conflicts(), [])


class Selftest(unittest.TestCase):
    """The kit's synthetic snapshots: as_of, an expired promo, a page that left the crawl, a refusal."""

    @classmethod
    def setUpClass(cls):
        cls.ks = build()
        for snap in run_eval.load_snapshots(SELFTEST / "snapshots"):
            cls.ks.ingest(snap)

    def test_current_value_ignores_the_expired_promo(self):
        self.assertIn("3.50%", self.ks.answer("What is the 6-month certificate APY?").text)
        self.assertNotIn("2.95%", self.ks.answer("Current APY on a six month share certificate?").text)

    def test_as_of_answers_from_the_site_as_it_stood(self):
        self.assertIn("2.95%", self.ks.answer("What was the 6-month certificate APY?", as_of="2026-01-31").text)
        self.assertFalse(self.ks.answer("What was it?", as_of="2025-12-31").confident)   # before any snapshot

    def test_history_survives_a_page_leaving_the_crawl(self):
        self.assertIn("$2.50", self.ks.answer("How much is a lost debit card replacement?").text)

    def test_declines_when_the_site_does_not_say(self):
        a = self.ks.answer("What is the 60-month jumbo certificate rate?")
        self.assertFalse(a.confident)
        self.assertFalse(numbers(a.text))

    def test_cites_every_live_page_stating_the_value_but_never_a_removed_one(self):
        from solution import _pages_stating
        ix = self.ks._indexes[-1]
        self.assertTrue(any("certificates" in k for k in _pages_stating(ix, {"3.50%"})))
        self.assertEqual(_pages_stating(ix, {"$2.50"}), [])     # only on a page the latest crawl dropped

    def test_fee_question_about_something_never_listed_declines(self):
        a = self.ks.answer("How much is a lost debit card replacement in Springfield?")   # rare word absent
        self.assertFalse(a.confident)
        self.assertFalse(numbers(a.text))

    def test_llm_context_adds_only_live_prose(self):
        from retrieve import parse_query, select
        for question in ("What is the 6-month certificate APY?", "How much is a lost debit card replacement?"):
            ev = select(self.ks._indexes[-1], parse_query(question))
            added = [c for c in ev.context if c not in ev.chunks]
            self.assertTrue(all(c in ev.context for c in ev.chunks))          # nothing retrieved is lost
            self.assertFalse(any(c.is_row for c in added))                     # no table rows sneak back in
            self.assertFalse({c.id for c in ev.context} & {c.id for c in ev.dropped})   # nor stale values
            self.assertEqual(len({c.id for c in ev.context}), len(ev.context))   # each chunk once

    def test_conflicts_are_typed_strings_and_pick_the_current_value(self):
        found = self.ks.conflicts()
        self.assertTrue(all(isinstance(c, Conflict) for c in found))
        self.assertTrue(all(isinstance(v, str) for c in found for v in c.values))
        report = scoring.score_conflicts(
            [{"id": "c1", "topic_keywords": ["6-month", "certificate"], "values": ["2.95%", "3.50%"],
              "correct_value": "3.50%"}], found)
        self.assertEqual((report["f1"], report["resolution_accuracy"]), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
