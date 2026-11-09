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
import calendar                        # noqa: E402
from datetime import date, timedelta   # noqa: E402

from ingest import KIND_RANK, MONTHS, STATUS_RANK, Chunk, Store, _on_site, url_key   # noqa: E402
import retrieve                        # noqa: E402
from retrieve import (CHAT_FILLERS, MEMBERSHIP_Q, STOP, Evidence, Index, _stem, parse_query, proper_names,  # noqa: E402
                      select, tokenize, use_proper_names, use_site_words)
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
        use_site_words(self._store.site_words)      # derived from the crawl; must precede tokenizing
        index = Index(view)
        self._store.proper_names = proper_names(index.chunks)   # branch names written in lower case
        use_proper_names(self._store.proper_names)
        _facts(index)                  # build per-view lookups now: loading is untimed, answers are
        _page_list(index)
        _page_texts(index)
        _glossary(index)
        _dates_stated(index)
        self._indexes.append(index)
        self._cache.clear()

    def answer(self, question: str, as_of: str | None = None) -> Answer:
        _own_site(self._store)
        span = change_span(question, self._store) if as_of is None else None
        if span:                                 # "how did it change between June and September?"
            v1, v2 = (self._store.view_for(d.isoformat() if d else None) for d in span)
            if v1 is not None and v2 is not None and v1 != v2:
                key = ("change", re.sub(r"[^a-z0-9$%.]+", " ", question.lower()).strip(), v1, v2)
                if key not in self._cache:
                    self._cache[key] = _respond_change(self._indexes[v1], self._indexes[v2], self._synth, question)
                return self._cache[key]
        asked = as_of or moment_asked(question, self._store, self._indexes)   # or the moment it names
        vi = self._store.view_for(asked)
        day = _iso(asked)                       # the day the model is told, when later than the crawl
        shown = day if vi is not None and day and day > self._store.views[vi].taken_on else None
        key = (re.sub(r"[^a-z0-9$%.]+", " ", question.lower()).strip(), vi, shown)
        if key not in self._cache:
            # vi is None when nothing is ingested or the date is before the first snapshot
            index = self._indexes[vi] if vi is not None else None
            # A date that only names the snapshot being answered from is not a search word
            search = strip_temporal(question, lambda d: self._store.view_for(d) == vi,
                                    _dates_stated(index)) if index else question
            self._cache[key] = _respond(index, self._synth, question, search, _iso(asked))
        return self._cache[key]

    def conflicts(self) -> list[Conflict]:
        _own_site(self._store)
        if not self._indexes:
            return []
        found = find_conflicts(self._indexes[-1].view)
        return [Conflict(topic=str(f["topic"]), values=[str(v) for v in f["values"]],
                         sources=[Source(url=str(u), snapshot_id=str(s)) for u, s in f["sources"]],
                         resolution=str(f["resolution"]), reason=str(f["reason"])) for f in found]


def _own_site(store: Store) -> None:
    """Read questions with this site's name (derived from its crawl), even if another system in the same
    process was loaded since."""
    if retrieve.SITE_WORDS != store.site_words:
        use_site_words(store.site_words)
    if retrieve.PROPER_NAMES != getattr(store, "proper_names", set()):
        use_proper_names(getattr(store, "proper_names", set()))


MAX_SOURCES = 8     # pages that state the values given; the same page is often served under several URL variants


def _respond(index: Index | None, synth: Synthesizer, question: str, search: str | None = None,
             asked: date | None = None) -> Answer:
    if index is None:
        return Answer(text=REFUSAL, confident=False)
    view = index.view
    q = parse_query(search or question)       # the model still sees the full question
    ev = select(index, q)
    if LINK_Q.search(question):                       # "what link should I use…": the site's own link
        carried = lambda u: sum(u in t for t in _page_texts(index).values())
        near, wide = ev.context or ev.chunks, [c for _, c in index.search(q.tokens, top=30)]
        link = pick_link(near, view, q, carried) or pick_link(wide, view, q, carried)
        if link:
            vp = link_source(link, near + wide, view)
            return Answer(text=f"You can find that at {link}.", confident=True,
                          sources=[Source(url=vp.page.url, snapshot_id=vp.snapshot_id)] if vp else [])
    if not ev.chunks or (q.attribute and not ev.supported):
        return Answer(text=REFUSAL, confident=False)
    fact = match_fact(_facts(index), q, index)
    if fact:
        ev.context = apply_fact(ev.context or ev.chunks, fact, view)
    def second_look(text: str):
        pages, paths = _page_list(index)
        keys = [paths[p] for p in map(_path, synth.pick_pages(text, pages)) if p in paths]
        return _pages_evidence(index, q, keys) if keys else None
    text, cited, confident, note = synth.compose(question, q, ev, view, asked,
                                                 lookup=lambda text: select(index, parse_query(text)),
                                                 second_look=second_look)
    if not confident:
        return Answer(text=REFUSAL, confident=False)
    if fact:
        text, stated = fact_guard(text, fact)
        if stated is not None:
            cited = [stated] + [c for c in cited if c.id != stated.id]
    text, defined_on = gloss(text, _glossary(index))     # a site-defined term -> what the site says it covers
    # Cite the pages actually used, then other evidence pages stating the same values
    # (extra citations cost nothing and the "official" page is often the second copy).
    vals = set(re.findall(r"\d+(?:\.\d+)?\s*%|\$\s?[\d,]+(?:\.\d+)?|\b\d{1,2}(?::\d\d)?\s?[ap]\.m\.|"
                          r"\b\d{3}-\d{3}-\d{4}\b|\b\d{6,}\b", text))   # rates, amounts, times, phones, IDs
    also = [c for c in ev.chunks if c not in cited and vals and any(v in c.text for v in vals)]
    sources, seen = [], set()
    if defined_on:
        vp = view.pages[defined_on]
        sources.append(Source(url=vp.page.url, snapshot_id=vp.snapshot_id))
        seen.add(defined_on)
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


DATE_PHRASE = re.compile(
    r"\(?\b(?:(?:as of|as at|in|on|during|back in|by|since|before|after)\s+)?"
    r"((?:%s|sept?|jan|feb|mar|apr|jun|jul|aug|oct|nov|dec)\.?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?20\d\d"
    r"|20\d\d-\d{2}-\d{2}|\d{1,2}/\d{1,2}/20\d\d)\)?" % "|".join(MONTHS), re.I)


def _phrase_date(text: str) -> date | None:
    """'May 2026' (last day of the month), 'May 15, 2026', '2026-05-15' or '5/15/2026'."""
    t = text.strip().lower()
    try:
        if m := re.fullmatch(r"(20\d\d)-(\d{2})-(\d{2})", t):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(20\d\d)", t):
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        m = re.fullmatch(r"([a-z]+)\.?\s+(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(20\d\d)", t)
        month = next((n for name, n in MONTHS.items() if m and name.startswith(m.group(1)[:3])), None)
        if month:
            year = int(m.group(3))
            return date(year, month, int(m.group(2)) if m.group(2) else calendar.monthrange(year, month)[1])
    except ValueError:
        return None
    return None


_DAY_PRECISE = re.compile(r"\d{1,2}(?:st|nd|rd|th)?,?\s+20|-\d\d-|/")


def _dates_stated(index: Index) -> set:
    """Every date the view's own pages state, as (ISO date, day-precise?). Built once per view at ingest."""
    st = getattr(index, "_dates", None)
    if st is None:
        st = {(d.isoformat(), bool(_DAY_PRECISE.search(m.group(1))))
              for t in _page_texts(index).values() for m in DATE_PHRASE.finditer(t)
              if (d := _phrase_date(m.group(1)))}
        index._dates = st
    return st


# A date written in the question picks the snapshot only when it names a moment to read the site at.
# The frame is read from the words just before the date, the tense from the clause that asks.
FRAME_RE = re.compile(r"\b(as of|as at|back in|back on|prior to|up until|until|till|before|since|after|by|"
                      r"from|between|and|to|in|on|during|for|at)\b", re.I)
PIN_FRAMES = {"as of", "as at", "back in", "back on"}           # the site as it stood then
BEFORE_FRAMES = {"prior to", "up until", "until", "till", "before"}   # the day before
BLOCK_FRAMES = {"since", "after", "by", "from", "between", "and", "to"}   # a span or a change, not a moment
QUALIFIER_RE = re.compile(r"\b(mid|middle|early|start|beginning|late|end)\b", re.I)
AUX_RE = re.compile(r"\b(used to|is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\b|"
                    r"\b(?:what|who|where|when|how|which|that|it|there|here)['’]s\b", re.I)
PAST_AUX = {"was", "were", "did", "had", "used to"}
NOW_Q = re.compile(r"\b(now|today|currently|presently|nowadays|anymore|any longer|these days)\b", re.I)
# A question about a change, or about an article or event, is answered from the newest crawl, which
# records the change or the article; the date then says when it happened, not when to read the site.
CHANGE_Q = re.compile(r"\b(chang\w*|rise[sn]?|rising|rose|raise[ds]?|increas\w*|decreas\w*|drop\w*|fell|fall\w*|"
                      r"lowered|cut|jump\w*|climb\w*|(?:went|go|goes|gone) (?:up|down)|effective|"
                      r"(?:took|take[ns]?|went into) effect|set (?:on|at|to)|updated?|adjust\w*)\b", re.I)
EVENT_Q = re.compile(r"\b(posts?|posted|articles?|blogs?|story|stories|news\w*|press|releases?|released|"
                     r"announc\w*|reports?|reported|publish\w*|episodes?|podcasts?|videos?|webinars?|"
                     r"seminars?|events?|meetings?)\b", re.I)
CLAUSE_END = re.compile(r"(?<=[;!?])\s+|(?<=[a-z0-9%)\]][.])\s+(?=[A-Z])")   # never inside 9,999.99 or Sept. 1


def date_asked(question: str, store: Store) -> str | None:
    """When no date is passed, a single moment the question itself names inside the crawl window
    ("as of March 2031", "what was it on 3/14/2031", "before April 2031") selects the site as it stood then.
    "As of / back in" always pin; "in / on / during" pin only in a past-tense question that is not about a
    change or an article or event. Not for questions about now, spans ("since", "between"), dates outside
    the window, or a date in a side remark of a present-tense question."""
    q = question.strip()
    if len(store.views) < 2 or NOW_Q.search(q):
        return None
    cuts = [0] + [m.end() for m in CLAUSE_END.finditer(q)] + [len(q)]
    spans = list(zip(cuts, cuts[1:]))
    ask = next((s for s in reversed(spans) if "?" in q[s[0]:s[1]]), spans[-1])
    aux = AUX_RE.search(q[ask[0]:ask[1]])
    past = bool(aux and aux.group(1) and aux.group(1).lower() in PAST_AUX)
    found = set()
    for m in DATE_PHRASE.finditer(q):
        d = _phrase_date(m.group(1))
        if d is None:
            continue
        before = q[:m.start(1)]
        tail = " ".join(before.split()[-5:])
        frames = list(FRAME_RE.finditer(tail))
        frame = " ".join(frames[-1].group(1).lower().split()) if frames else ""
        if frame in BLOCK_FRAMES or not (ask[0] <= m.start(1) < ask[1] or past):
            return None
        precise = bool(_DAY_PRECISE.search(m.group(1)))
        if frame not in PIN_FRAMES | BEFORE_FRAMES and (not past or CHANGE_Q.search(q) or EVENT_Q.search(q)):
            return None
        if not precise:                          # "mid-March", "early April", "14 March 2031"
            day = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*$", before)
            qual = {w.lower() for w in QUALIFIER_RE.findall(tail[frames[-1].end():] if frames else tail)}
            if day and 1 <= int(day.group(1)) <= d.day:
                d = d.replace(day=int(day.group(1)))
            elif qual & {"mid", "middle"}:
                d = d.replace(day=15)
            elif qual & {"early", "start", "beginning"}:
                d = d.replace(day=1)
            elif frame in BEFORE_FRAMES:
                d = d.replace(day=1)
        if frame in BEFORE_FRAMES:
            d = d - timedelta(days=1)
        found.add((d, precise))
    if len({d for d, _ in found}) != 1:
        return None
    d, precise = found.pop()
    first, last = store.views[0].taken_on, store.views[-1].taken_on
    if not precise and d < first and (d.year, d.month) == (first.year, first.month):
        d = first                       # "early March" when the first crawl is mid-March: the nearest reading
    return d.isoformat() if first <= d < last else None


def moment_asked(question: str, store: Store, indexes: list) -> str | None:
    """date_asked, unless what the question is about is an article the older snapshot doesn't have yet:
    the newest crawl's top evidence is a page published after that snapshot and by the date named."""
    d = date_asked(question, store)
    vi = store.view_for(d) if d else None
    if vi is None or vi == len(store.views) - 1 or not indexes:
        return d
    newest = indexes[-1]
    ev = select(newest, parse_query(question))
    if not ev.chunks:
        return d
    asked = date.fromisoformat(d)
    pages = [newest.view.pages[k].page for k in dict.fromkeys(c.key for c in ev.chunks)][:5]
    # the best evidence (the top three pages when it is prose, not a rate or fee row)
    best = pages[:1] if ev.chunks[0].is_row else pages[:3]
    since = store.views[vi].taken_on
    if any(p.publish and since < p.publish <= asked for p in best) or any(p.publish == asked for p in pages):
        return None                     # ... or the question names the day one of those articles came out
    return d


# A question about how a value changed over time ("How did X change between June and September 2026?",
# "Has X gone up since May 2026?") is answered from the site as it stood at each date, both values given
# with their dates. Both the change wording and a date in the question are needed; otherwise nothing changes.
CHANGE_ASK = re.compile(
    r"\bhow (?:did|has|have|had)\b[^?]{0,120}?\bchange[ds]?\b|"
    r"\b(?:did|has|have|had)\b[^?]{0,120}?\b(?:go(?:ne)? (?:up|down)|went (?:up|down)|increase[ds]?|"
    r"decrease[ds]?|rise[ns]?|rose|fall(?:en)?|fell|drop(?:ped)?|change[ds]?)\b|"
    r"\bwhat (?:was|is|were) the change\b", re.I)
_MONTH = "|".join(MONTHS) + "|sept?|jan|feb|mar|apr|jun|jul|aug|oct|nov|dec"
MONTH_PAIR = re.compile(rf"\b(?:between|from)\s+({_MONTH})\.?\s+(?:and|to)\s+(?:{_MONTH})\.?\s+"
                        rf"(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?(20\d\d)", re.I)          # "between May and June 2026"
SPAN_FROM = re.compile(r"\b(?:since|after|from)\s+(?:\S+\s+){0,2}$", re.I)


def change_span(question: str, store: Store) -> tuple | None:
    """(start, end) dates of a change the question asks about; end None means "now". A month alone means
    the site as it stood by the end of that month."""
    if len(store.views) < 2 or not CHANGE_ASK.search(question):
        return None
    found = [(m.start(1), d) for m in DATE_PHRASE.finditer(question) if (d := _phrase_date(m.group(1)))]
    if len(found) < 2 and (mp := MONTH_PAIR.search(question)) and (d := _phrase_date(f"{mp.group(1)} {mp.group(2)}")):
        found.append((mp.start(1), d))
    days = sorted({d for _, d in found})
    if len(days) >= 2:
        return days[0], days[-1]
    if len(days) == 1 and SPAN_FROM.search(question[:found[0][0]]):
        return days[0], None
    return None


def _respond_change(first: Index, last: Index, synth: Synthesizer, question: str) -> Answer:
    search = strip_temporal(question, lambda d: True)       # the dates name the two moments, not search words
    parts = []
    for index in (first, last):
        q = parse_query(search)
        parts.append((index.view, q, select(index, q)))
    if not any(ev.chunks and (not q.attribute or ev.supported) for _, q, ev in parts):
        return Answer(text=REFUSAL, confident=False)
    text, cited = synth.compose_change(question, parts)
    if text == REFUSAL:
        return Answer(text=REFUSAL, confident=False)
    sources, seen = [], set()
    for view, c in cited:
        vp = view.pages[c.key]
        if (c.key, vp.snapshot_id) not in seen and len(sources) < MAX_SOURCES:
            seen.add((c.key, vp.snapshot_id))
            sources.append(Source(url=vp.page.url, snapshot_id=vp.snapshot_id, quote=c.text[:240]))
    return Answer(text=text, sources=sources, confident=True)


def _iso(value: str | None) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def strip_temporal(question: str, same_view, stated=frozenset()) -> str:
    """Drop date phrases that only point at the snapshot being answered from ('as of September 2026').
    Dates that are facts in their own right stay as search words: any date the site itself states (a
    closing date, a deadline, an event), and any date in another snapshot's window."""
    def drop(m):
        d = _phrase_date(m.group(1))
        if not d or not same_view(d.isoformat()):
            return m.group(0)
        if (d.isoformat(), bool(_DAY_PRECISE.search(m.group(1)))) in stated:
            return m.group(0)
        return " "
    return re.sub(r"\s{2,}", " ", DATE_PHRASE.sub(drop, question)).strip(" ,")


LINK_Q = re.compile(r"\b(link|url|web ?page|website address)\b", re.I)
LINK_RE = re.compile(r"([^()\n]{0,80})\(([a-z0-9.-]+\.[a-z]{2,}(?:/[^)\s]*)?)\)")
LINK_WORDS = set(tokenize("link url use see find page web webpage website address where visit"))
MEDIUM_WORDS = set(tokenize("online"))      # how, not what: "join online" asks for the link to join
FACT_RARE = 0.01        # a question word on under 1% of chunks names the fact asked about


def pick_link(context: list[Chunk], view, q, carried=None) -> str | None:
    """The link the site's own text gives for what was asked: the link whose anchor text shares a word with
    the question, given by the most pages of the evidence (then carried by the most pages of the site), whose
    page is live in this snapshot and is not an article (a "which page" question wants the product or tool
    page, not a blog post). Dead or retired links (pages the crawl never found, or that left it) can't be
    picked.

    A link to one of the site's own hosts that the crawl never fetched (an application portal such as
    join.<site>) can't be checked against a page, so it needs more: a live, non-article page gives it, its
    anchor text names everything asked, and its own host or path shares a word asked (so a click tracker
    behind "Apply today" is never the answer).

    For a question about joining or membership (becoming a member is joining), "join" also counts for that
    host or path word, and a portal given by the site's page about joining (its title or path says so) wins a
    tie: the page that gives a portal is the only evidence for it."""
    subject = set(q.tokens) - LINK_WORDS - CHAT_FILLERS - set(q.expansion)
    cover = subject - MEDIUM_WORDS
    joining = MEMBERSHIP_Q.search(q.raw) is not None
    addr = cover | ({_stem("join")} if joining else set())
    hosts = {k.split("/", 1)[0] for k in view.pages}
    pages: dict = {}
    portals, given_by_own_page = set(), set()
    for c in context:
        for m in LINK_RE.finditer(c.text):
            link, anchor = m.group(2).rstrip("/"), set(tokenize(m.group(1)))
            vp = view.pages.get(url_key(link))
            if vp is None:
                carrier = view.pages.get(c.key)
                if (link.split("/", 1)[0] in hosts or not _on_site(link, hosts) or carrier is None
                        or carrier.status == "missing" or carrier.page.kind == "article"
                        or not cover or not cover <= anchor or not set(tokenize(link)) & addr):
                    continue
                portals.add(link)
                if joining and addr & set(tokenize(f"{carrier.page.title or ''} {c.key.split('/', 1)[-1]}")):
                    given_by_own_page.add(link)
            elif vp.status == "missing" or vp.page.kind == "article":
                continue
            if anchor & subject:
                pages.setdefault(link, set()).add(c.key)
    if not pages:
        return None
    def weight(u):                 # a portal counts where it is written as a link, not inside longer links
        return 0 if not carried else carried(f"({u})") if u in portals else carried(u)
    return max(pages, key=lambda u: (len(pages[u]), u in given_by_own_page, weight(u)))


def link_source(link: str, context: list[Chunk], view):
    """The page to cite for a picked link: the page itself, or for a portal the page that gives the link."""
    vp = view.pages.get(url_key(link))
    if vp is not None:
        return vp
    givers = sorted({c.key for c in context if f"({link})" in c.text or f"({link}/)" in c.text})
    live = [view.pages[k] for k in givers if k in view.pages and view.pages[k].status != "missing"]
    return min(live, key=lambda p: (KIND_RANK[p.page.kind], p.page.key), default=None)


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
        # a count is the fact asked about only when the question counts that same thing
        if re.match(r"\s*\d", f.get("resolution", "")) and not (
                q.attribute == "count" and q.count_noun and q.count_noun in set(tokenize(f["topic"]))):
            continue
        rare = [t for t in set(q.tokens) & set(tokenize(f["topic"])) if len(index.postings.get(t, ())) < limit]
        if rare and (best is None or len(rare) > best[0]):
            best = (len(rare), f)
    return best[1] if best else None


def _fact_token(value: str) -> str:
    m = re.match(r"\s*([\d,]+|[a-z]+)", value.lower())          # "12 branches" -> 12, "third" -> third
    return m.group(1) if m else ""


def _beaten(fact: dict) -> list:
    """Patterns for the values a repeated statement beat: a count with its noun ("9 offices", "9 office
    locations", never "9-month"), an ordinal, and the "Nth year in a row" claims it outranks."""
    pats = []
    for v in fact["_losers"]:
        m = re.match(r"\s*([\d,]+)\+?\s+([a-z]+)", v.lower())
        pats.append(rf"\b{re.escape(m.group(1))}\+?\s+(?:[\w’'-]+\s+){{0,2}}{re.escape(m.group(2)[:5])}" if m
                    else rf"\b{re.escape(v)}\b")
    for s in fact.get("_streaks", ()):
        if m := re.search(r"\b\w+\s+(?:years?\s+in\s+a\s+row|consecutive\s+years?|straight\s+years?)", s, re.I):
            pats.append(rf"\b{re.escape(m.group(0))}\b")
    return [re.compile(p, re.I) for p in pats]


def fact_guard(text: str, fact: dict) -> tuple[str, Chunk | None]:
    """A repeated statement the question asks about: its newest value must reach the answer, and a value
    it beat must not (rule 5). Returns the text and the chunk stating the winner, if it was added."""
    sentence = fact["_win_sentence"]
    if any(p.search(text) for p in _beaten(fact)):
        return sentence, fact["_win_chunk"]
    win = _fact_token(fact["resolution"])
    if not win or re.search(rf"\b{re.escape(win)}\b", text, re.I):
        return text, None
    return f"{text.rstrip()} {sentence}", fact["_win_chunk"]


def apply_fact(context: list[Chunk], fact: dict, view=None) -> list[Chunk]:
    """The winning statement goes first; evidence repeating a value it beat is removed (rule 5), and so is
    every older edition of the article that states the winner (a yearly press release re-posted under the
    same title): its figures are the ones the newer edition replaced."""
    beaten = _beaten(fact)
    kept = [c for c in context if not any(p.search(c.text) for p in beaten)]
    win = fact["_win_chunk"]
    if view is not None and win.key in view.pages:
        wp = view.pages[win.key].page
        title = " ".join(re.findall(r"\w+", (wp.title or "").lower()))

        def older_edition(c):
            p = view.pages[c.key].page if c.key in view.pages else None
            return (p is not None and c.key != win.key and p.kind == "article" and title
                    and " ".join(re.findall(r"\w+", (p.title or "").lower())) == title
                    and p.publish and wp.publish and p.publish < wp.publish)
        kept = [c for c in kept if not older_edition(c)]
    stated = Chunk(-1, win.key, win.kind, fact["_win_sentence"], fact["_win_sentence"], section=win.section)
    return [stated] + kept


TERM_DEF = re.compile(r"\b([a-z][a-z’'-]+(?: [a-z][a-z’'-]+){1,2})\s*:\s*-\s", re.I)   # a 2-3 word term
ITEM = re.compile(r"(?:^|\s)-\s*([A-Z][A-Za-z .'’]{1,30}?):\s*(.+?)(?=\s+-\s|$)")


def _glossary(index: Index) -> dict:
    """Terms the site defines as a list ("service area: - Region A: … - Region B: …"), kept only when the
    identical list appears on at least two live pages and the term has a rare word. term -> (items, page)."""
    gl = getattr(index, "_gloss", None)
    if gl is not None:
        return gl
    found: dict = {}
    for key, vp in index.view.pages.items():
        if vp.status == "missing":
            continue
        chunks = vp.page.chunks
        for j, c in enumerate(chunks):
            if c.is_row or not (m := TERM_DEF.search(c.text)):
                continue
            words = m.group(1).lower().split()
            items = ITEM.findall(c.text[m.end() - 2:])
            for nxt in chunks[j + 1:j + 3]:                    # the list often runs into the next chunks
                more = ITEM.findall(nxt.text)
                if not nxt.text.lstrip().startswith("-") or not more:
                    break
                items += more
            items = tuple((n.strip(), v.strip()) for n, v in items)
            if len(items) >= 2:                  # "the bank's service area": the term is a shared ending
                for n in range(2, len(words) + 1):                 # of at least two words
                    if words[-n] not in STOP:
                        found.setdefault((" ".join(words[-n:]), items), []).append(key)
    limit = 0.01 * max(1, len(index.chunks))
    gl = {}
    defined = set()
    for (term, items), keys in sorted(found.items(), key=lambda kv: len(kv[0][0])):   # shortest shared term
        rare = any(len(index.postings.get(t, ())) < limit for t in tokenize(term))
        if len(set(keys)) >= 2 and rare and term not in gl and items not in defined:
            defined.add(items)
            best = min(keys, key=lambda k: (KIND_RANK[index.view.pages[k].page.kind], k))
            gl[term] = (items, best)
    index._gloss = gl
    return gl


def gloss(text: str, glossary: dict) -> tuple[str, str | None]:
    """If the answer relies on a site-defined term without saying what it covers, add the site's own
    definition, verbatim. Returns (text, page key of the definition or None)."""
    low = text.lower()
    for term, (items, page) in glossary.items():
        if term in low and not all(n.lower() in low for n, _ in items):
            return f"{text} The {term} is: " + "; ".join(f"{n}: {v}" for n, v in items) + ".", page
    return text, None


def _page_list(index: Index) -> tuple[str, dict]:
    """The site's own non-article pages as "path | title" lines, and path -> page key. Built once per view."""
    pl = getattr(index, "_page_list", None)
    if pl is None:
        rows, paths = [], {}
        for key, vp in sorted(index.view.pages.items()):
            if vp.status == "missing" or vp.page.kind == "article":
                continue
            path = _path(key)
            paths[path] = key
            rows.append(f"{path} | {vp.page.title or ''}")
        pl = index._page_list = ("\n".join(rows), paths)
    return pl


def _path(ref: str) -> str:
    """'https://example.org/a/b/' or 'example.org/a/b' -> '/a/b'."""
    ref = re.sub(r"^[a-z]+://", "", str(ref).strip().lower())
    return "/" + ref.split("/", 1)[1].strip("/") if "/" in ref else "/"


def _pages_evidence(index: Index, q, keys: list) -> Evidence | None:
    """Evidence from the pages picked: each page's opening paragraphs, then its best matches."""
    view, wanted = index.view, set(keys)
    opening = [c for k in keys for c in view.pages[k].page.chunks[:8]]
    best = [c for _, c in index.search(q.tokens, top=3000) if c.key in wanted][:8]
    chunks = [c for c in {c.id: c for c in opening + best}.values()
              if not (c.expired_on and c.expired_on < view.taken_on)][:16]
    return Evidence(chunks, [], True, {}, chunks) if chunks else None


def _page_texts(index: Index) -> dict:
    texts = getattr(index, "_page_texts", None)
    if texts is None:                        # built once per view (at ingest), then reused
        texts = {k: " ".join(c.text for c in vp.page.chunks)
                 for k, vp in index.view.pages.items() if vp.status != "missing"}
        index._page_texts = texts
    return texts


def _pages_stating(index: Index, vals: set) -> list[str]:
    if not vals:
        return []
    texts = _page_texts(index)
    pages = index.view.pages
    return sorted((k for k, t in texts.items() if any(v in t for v in vals)),
                  key=lambda k: (KIND_RANK[pages[k].page.kind], STATUS_RANK[pages[k].status], k))


def build() -> KnowledgeSystem:
    return KnowledgeSystem()
