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
        self.assertEqual(strip_footnotes("3.25%1"), "3.25%")
        self.assertEqual(strip_footnotes("6.10% APR1"), "6.10% APR")
        self.assertEqual(strip_footnotes("Annual Percentage Rate (As Low As) 1"), "Annual Percentage Rate (As Low As)")
        self.assertEqual(strip_footnotes("11.00% - 18.00% APR"), "11.00% - 18.00% APR")

    def test_url_key_matches_the_scorer(self):
        for u in ("https://www.dupaco.com/personal/loans/home-loans#rates", "https://www.dupaco.com/",
                  "https://www.dupaco.com/privacy-policy?campaignId=x"):
            self.assertEqual(url_key(u), scoring._url_key(u))

    def test_model_year_ranges_handle_both_orders_and_open_ends(self):
        self.assertEqual(model_year_ranges("2024-27 Model Year"), [(2024, 2027)])
        self.assertEqual(model_year_ranges("Model Year 2026-23"), [(2023, 2026)])
        self.assertEqual(model_year_ranges("2023–26 Model Year"), [(2023, 2026)])
        self.assertEqual(model_year_ranges("2016 Model Year (and older)"), [(0, 2016)])

    def test_amount_tiers(self):
        self.assertEqual(amount_ranges("$5,000 – $9,999.99"), [(5000.0, 9999.99)])
        self.assertEqual(amount_ranges("$50,000+")[0][0], 50000.0)
        self.assertEqual(amount_ranges("$500,000 or less"), [(0.0, 500000.0)])

    def test_question_qualifiers(self):
        q = parse_query("What's the APR for a 60-month auto loan on a 2025 model-year vehicle?")
        self.assertEqual(q.terms, {(60, "month")})
        self.assertEqual(q.years, {2025})
        self.assertEqual(q.attribute, "rate")
        q = parse_query("What was the rate on the 10-month jumbo special as of May 2026?")
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
    """Assignment §7: hard-coding answers to the practice questions is out of bounds. No expected or
    wrong-answer string from eval/public_eval.jsonl may appear in the submission's code, apart from
    generic vocabulary that any implementation needs (ordinal words, "noon")."""
    GENERIC = {"noon", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"}

    def test_no_practice_answers_in_submission_code(self):
        items = [json.loads(l) for l in (ROOT / "eval" / "public_eval.jsonl").read_text().splitlines() if l.strip()]
        tokens = {t for it in items for k in ("must_include", "must_not_include") for t in it.get(k, [])
                  if len(t) >= 4 and t.lower() not in self.GENERIC}
        hits = sorted((p.name, t) for p in (ROOT / "submission").glob("*.py")
                      for t in tokens if t.lower() in p.read_text().lower())
        self.assertEqual(hits, [])


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
        view = View("05", date(2026, 9, 28), {"dupaco.com/personal/loans/example-loan": None})
        self.assertEqual(_drop_dead_links("Use dupaco.com/old/retired-page. Or call us.", view), "Or call us.")
        kept = "See dupaco.com/personal/loans/example-loan. Apply at dupaco.com/offer."   # crawled; vanity
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
