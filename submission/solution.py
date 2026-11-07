"""Knowledge system for the hackathon harness: build() -> object with ingest / answer / conflicts.

How "current" is decided (also in NOTES.md):
  1. Answer from the website as it stood on the requested date (default: newest snapshot), keeping
     every page ever seen up to then, so facts from pages that dropped out of a crawl survive.
  2. When sources disagree on the same fact: newest effective date (rate tables), then newest
     publish date (articles), then a page re-fetched in that crawl over a stale copy, then rate and
     product pages over documents, articles, promotions and calculators.
  3. Expired offers, and rows for a different term / model year / balance tier than asked, are
     never used. If nothing states what was asked, answer "isn't listed; please contact us".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                                   # under run_eval, harness/ is already importable
    from contract import Answer, Conflict, Snapshot, Source
except ImportError:                    # standalone (tests): find the kit's harness/ folder
    sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents
                                if (d / "harness" / "contract.py").exists())))
    from contract import Answer, Conflict, Snapshot, Source

from conflicts import find_conflicts   # noqa: E402
from ingest import Store               # noqa: E402
from retrieve import Index, parse_query, select   # noqa: E402
from synth import REFUSAL, Synthesizer  # noqa: E402


class KnowledgeSystem:
    """Exactly the three contract methods; the work lives in module-level functions."""

    def __init__(self):
        self._store = Store()
        self._indexes: list[Index] = []
        self._synth = Synthesizer()
        self._cache: dict = {}

    def ingest(self, snapshot: Snapshot) -> None:
        view = self._store.add_snapshot(snapshot)
        self._indexes.append(Index(view))
        self._cache.clear()

    def answer(self, question: str, as_of: str | None = None) -> Answer:
        vi = self._store.view_for(as_of)
        key = (re.sub(r"[^a-z0-9$%.]+", " ", question.lower()).strip(), vi)
        if key not in self._cache:
            # vi is None when nothing is ingested or the date is before the first snapshot
            index = self._indexes[vi] if vi is not None else None
            self._cache[key] = _respond(index, self._synth, question)
        return self._cache[key]

    def conflicts(self) -> list[Conflict]:
        if not self._indexes:
            return []
        found = find_conflicts(self._indexes[-1].view)
        return [Conflict(topic=str(f["topic"]), values=[str(v) for v in f["values"]],
                         sources=[Source(url=str(u), snapshot_id=str(s)) for u, s in f["sources"]],
                         resolution=str(f["resolution"]), reason=str(f["reason"])) for f in found]


MAX_SOURCES = 6     # pages that state the values given; the same press release often has 3 URL variants


def _respond(index: Index | None, synth: Synthesizer, question: str) -> Answer:
    if index is None:
        return Answer(text=REFUSAL, confident=False)
    view = index.view
    q = parse_query(question)
    ev = select(index, q)
    if not ev.chunks or (q.attribute and not ev.supported):
        return Answer(text=REFUSAL, confident=False)
    text, cited, confident, note = synth.compose(question, q, ev, view)
    if not confident:
        return Answer(text=REFUSAL, confident=False)
    # Cite the pages actually used, then other evidence pages stating the same values
    # (extra citations cost nothing and the "official" page is often the second copy).
    vals = set(re.findall(r"\d+(?:\.\d+)?\s*%|\$\s?[\d,]+(?:\.\d+)?|\b\d{1,2}(?::\d\d)?\s?[ap]\.m\.|"
                          r"\b\d{3}-\d{3}-\d{4}\b|\b\d{6,}\b", text))   # rates, amounts, times, phones, IDs
    also = [c for c in ev.chunks if c not in cited and vals and any(v in c.text for v in vals)]
    sources, seen = [], set()
    for c in cited + also:
        for key in [c.key] + ev.copies.get(c.id, []):     # identical rows on other pages count too
            vp = view.pages[key]
            if key in seen or len(sources) >= MAX_SOURCES:
                continue
            seen.add(key)
            sources.append(Source(url=vp.page.url, snapshot_id=vp.snapshot_id, quote=c.text[:240]))
    return Answer(text=text, sources=sources, confident=True, conflict_note=note or None)


def build() -> KnowledgeSystem:
    return KnowledgeSystem()
