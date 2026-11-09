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
from retrieve import _stem, amount_ranges, model_year_ranges, parse_query  # noqa: E402
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
                        ("What was the APY in December 2029?", None),            # before the first crawl
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
        self.assertIsNone(date_asked("What was the APY on January 2, 2030?", late))                  # a day before it

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
        for asked, shown in [(date(2030, 3, 20), "2030-03-20"), (None, "2030-03-01"), (date(2030, 2, 1), "2030-03-01")]:
            self.assertIsNone(s._llm("What is the APY?", NS(context=[c], chunks=[c]), view, asked))
            self.assertTrue(sent[-1].startswith(f"Website snapshot date: {shown}\n"), sent[-1][:40])

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
