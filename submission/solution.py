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

from collections import Counter        # noqa: E402

from conflicts import find_conflicts, statement_conflicts   # noqa: E402
from ingest import KIND_RANK, STATUS_RANK, Chunk, Store, url_key   # noqa: E402
from retrieve import Index, parse_query, select, tokenize   # noqa: E402
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


MAX_SOURCES = 8     # pages that state the values given; the same press release often has 3 URL variants


def _respond(index: Index | None, synth: Synthesizer, question: str) -> Answer:
    if index is None:
        return Answer(text=REFUSAL, confident=False)
    view = index.view
    q = parse_query(question)
    ev = select(index, q)
    if LINK_Q.search(question):                       # "what link should I use…": the site's own link
        link = pick_link(ev.context or ev.chunks, view, q)
        if link:
            vp = view.pages[url_key(link)]
            return Answer(text=f"You can find that at {link}.", confident=True,
                          sources=[Source(url=vp.page.url, snapshot_id=vp.snapshot_id)])
    if not ev.chunks or (q.attribute and not ev.supported):
        return Answer(text=REFUSAL, confident=False)
    fact = match_fact(_facts(index), q, index)
    if fact:
        ev.context = apply_fact(ev.context or ev.chunks, fact)
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
    # Then any other live page on the site that states one of those values (the routing number, a
    # phone number, a time), official product pages first.
    for key in _pages_stating(index, vals):
        if key in seen or len(sources) >= MAX_SOURCES:
            continue
        vp = view.pages[key]
        quote = next((c.text[:240] for c in vp.page.chunks if any(v in c.text for v in vals)), None)
        seen.add(key)
        sources.append(Source(url=vp.page.url, snapshot_id=vp.snapshot_id, quote=quote))
    return Answer(text=text, sources=sources, confident=True, conflict_note=note or None)


LINK_Q = re.compile(r"\b(link|url|web ?page|website address)\b", re.I)
LINK_RE = re.compile(r"([^()\n]{0,80})\(([a-z0-9.-]+\.[a-z]{2,}/[^)\s]*)\)")
LINK_WORDS = set(tokenize("link url use see find page web webpage website address where visit"))
FACT_RARE = 0.01        # a question word on under 1% of chunks names the fact asked about


def pick_link(context: list[Chunk], view, q) -> str | None:
    """The link the site's own text gives for what was asked: the most-cited link in the evidence
    whose anchor text shares a word with the question and whose page is live in this snapshot.
    Dead or retired links (pages the crawl never found, or that left it) can't be picked."""
    subject = set(q.tokens) - LINK_WORDS
    votes: Counter = Counter()
    for c in context:
        for m in LINK_RE.finditer(c.text):
            key = url_key(m.group(2))
            vp = view.pages.get(key)
            if vp is not None and vp.status != "missing" and set(tokenize(m.group(1))) & subject:
                votes[m.group(2).rstrip("/")] += 1
    return votes.most_common(1)[0][0] if votes else None


def _facts(index: Index) -> list[dict]:
    facts = getattr(index, "_facts", None)
    if facts is None:                        # built once per view
        facts = [f for f in statement_conflicts(index.view) if f.get("_win_sentence")]
        index._facts = facts
    return facts


def match_fact(facts: list[dict], q, index: Index) -> dict | None:
    """A repeated statement the question asks about: it shares a rare word with the fact's topic."""
    limit = FACT_RARE * max(1, len(index.chunks))
    best = None
    for f in facts:
        rare = [t for t in set(q.tokens) & set(tokenize(f["topic"])) if len(index.postings.get(t, ())) < limit]
        if rare and (best is None or len(rare) > best[0]):
            best = (len(rare), f)
    return best[1] if best else None


def apply_fact(context: list[Chunk], fact: dict) -> list[Chunk]:
    """The winning statement goes first; evidence repeating a value it beat is removed (rule 5)."""
    beaten = [re.compile(rf"\b{re.escape(v)}\b", re.I) for v in fact["_losers"]]
    kept = [c for c in context if not any(p.search(c.text) for p in beaten)]
    win = fact["_win_chunk"]
    stated = Chunk(-1, win.key, win.kind, fact["_win_sentence"], fact["_win_sentence"], section=win.section)
    return [stated] + kept


def _pages_stating(index: Index, vals: set) -> list[str]:
    if not vals:
        return []
    texts = getattr(index, "_page_texts", None)
    if texts is None:                        # built once per view, then reused for every answer
        texts = {k: " ".join(c.text for c in vp.page.chunks)
                 for k, vp in index.view.pages.items() if vp.status != "missing"}
        index._page_texts = texts
    pages = index.view.pages
    return sorted((k for k, t in texts.items() if any(v in t for v in vals)),
                  key=lambda k: (KIND_RANK[pages[k].page.kind], STATUS_RANK[pages[k].status], k))


def build() -> KnowledgeSystem:
    return KnowledgeSystem()
