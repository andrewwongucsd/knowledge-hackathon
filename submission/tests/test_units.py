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
from retrieve import amount_ranges, model_year_ranges, parse_query  # noqa: E402
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
        q = parse_query("How many branch locations are there?")
        self.assertEqual((q.attribute, q.count_noun), ("count", "branch"))
        q = parse_query("What are the lobby hours at the Elm Avenue branch?")
        self.assertEqual((q.attribute, q.entity), ("hours", "elm"))

    def test_a_fee_condition_is_not_a_product_term(self):
        q = parse_query("What is the fee to close an account within three months of opening?")
        self.assertEqual((q.attribute, q.terms), ("fee", set()))

    def test_fee_schedule_vocabulary(self):
        self.assertIn("official", parse_query("What is the cashier's check fee?").tokens)
        self.assertIn("international", parse_query("How much is a foreign wire?").tokens)

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
    Exempt: generic vocabulary any credit-union site uses (ordinals, "noon", street words, card networks,
    wallet brands, calendar names) and bare years."""
    EXEMPT = set("noon first second third fourth fifth sixth seventh eighth ninth tenth aprs avenue lane street "
                 "west share visa rewards platinum google samsung volunteer".split())
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
        snaps = sorted((ROOT / "snapshots").glob("0*/manifest.json"))
        if snaps:                                        # the site's identity, derived the same way ingest does
            pages = json.loads(snaps[-1].read_text())["pages"]
            domains, suffix, words = site_identity([SimpleNamespace(url=p.get("url"), title=p.get("title"))
                                                    for p in pages])
            for x in domains | words | ({suffix} if suffix else set()):
                banned[x.lower()] = "site identity"
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
        view = NS(pages={"example.com/rates": NS(status="fresh"), "example.com/old-rates": NS(status="missing")})
        text = ("Compare certificate rates (example.com/rates). See certificate rates (example.com/rates). "
                "Old certificate rates (example.com/old-rates). Certificate archive (example.com/gone).")
        ctx = [Chunk(1, "example.com/a", "product", text, text)]
        self.assertEqual(pick_link(ctx, view, parse_query("What link shows certificate rates?")), "example.com/rates")
        self.assertIsNone(pick_link(ctx, view, parse_query("What link shows mortgage payoffs?")))

    def test_fact_edge_puts_the_winner_first_and_drops_beaten_values(self):
        from types import SimpleNamespace as NS
        from ingest import Chunk
        from solution import apply_fact, match_fact
        win = Chunk(7, "example.com/news-3", "article", "This was the seventh time Acme recognized the bank.", "")
        fact = {"topic": "How many times Acme recognized: the Nth time", "_losers": ["sixth"],
                "_win_chunk": win, "_win_sentence": win.text}
        index = NS(chunks=[None] * 1000, postings={"acm": [(0, 1)]})
        self.assertIs(match_fact([fact], parse_query("What recognition has the bank received from Acme?"), index), fact)
        self.assertIsNone(match_fact([fact], parse_query("What are the lobby hours?"), index))
        old = Chunk(2, "example.com/news-2", "article", "It was the sixth time Acme recognized the bank.", "")
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
        q = parse_query("What is the auto loan APR for a 2025 vehicle as of May 15, 2026?")
        self.assertEqual(q.years, {2025})

    def test_as_of_dates_leave_the_search_but_content_dates_stay(self):
        from solution import strip_temporal
        same = lambda d: d >= "2026-09-01"                     # stand-in for "maps to the asked snapshot"
        self.assertEqual(strip_temporal("What is the APY as of September 2026?", same), "What is the APY ?")
        for q in ("Did it close on January 15, 2026?", "What happens in July 2026?"):
            self.assertEqual(strip_temporal(q, same), q)
        # a deadline the site states stays a search word even when it falls in the answered window
        stated = {("2026-11-12", True), ("2027-04-30", False)}
        for q in ("Is it open until Nov. 12, 2026?", "What is due in April 2027?"):
            self.assertEqual(strip_temporal(q, same, stated), q)
        self.assertEqual(strip_temporal("What is the APY (October 2026)?", same, stated), "What is the APY ?")

    def test_review_fixes_synonyms_model_years_and_branch_words(self):
        from retrieve import _stem
        change = _stem("change")
        self.assertNotIn(change, parse_query("Is there a fee to update my account information?").tokens)
        self.assertIn(change, parse_query("How do I update my personal information?").tokens)
        self.assertEqual(parse_query("Which APR may 2022 model year cars get for 72 months?").years, {2022})
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
        self.assertEqual(parse_query("lobby hrs?").attribute, "hours")
        self.assertEqual(parse_query("When is the drive-up open?").attribute, "hours")
        self.assertIsNone(parse_query("What time period does the special cover?").attribute)
        self.assertNotEqual(parse_query("What time does the elm branch close?").attribute, "hours")
        q = parse_query("Number of branch locations?")
        self.assertEqual((q.attribute, q.count_noun), ("count", "branch"))

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
        md = "1st Choice Savings\n\n| Balance | APY |\n|---|---|\n| $0 - $999 | 0.10% |\n"
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
        m = STREAK_RE.search("For a third year in a row, Acme named the bank a top employer.")
        self.assertEqual((m.group(1), m.group(2)), ("third", "Acme"))


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
        view = View("05", date(2026, 9, 28), {"example.org/borrow/sample-loan": None})
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
