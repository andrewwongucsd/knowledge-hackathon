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

import copy
import math
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

from ingest import KIND_RANK, MONTH_RE, MONTHS, STATUS_RANK, Chunk, Store, _on_site, url_key   # noqa: E402
import ingest                          # noqa: E402
import retrieve                        # noqa: E402
from retrieve import (ASKS_ELSE, BACK_REFERENCE, CHAT_FILLERS, DIMENSIONS, FRAME_WORDS, GENERIC, GENERIC_HEADS, HEDGES, MEMBERSHIP_Q,  # noqa: E402
                      PLACE_DEICTIC, QUIET, RESTRICTION, STOP, Evidence, Index, _stem, carried_in_context, count_clause, fee_clause,
                      ADDRESS_EV, FACILITY_ASK, facility_places, lead_names, parse_query, place_phrases, proper_names, qualify, select, terms_in,
                      tokenize, use_proper_names, use_site_words)
from synth import ABSENT, REFUSAL, Synthesizer, _says_absent, spelled_as_digits  # noqa: E402


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
        # the earliest effective date any crawl gives each page: a fee schedule dated only in later crawls was
        # still not in effect before that date (see _too_new)
        dated = self._store.__dict__.setdefault("first_effective", {})
        for key, vp in view.pages.items():
            effs = [c.effective for c in vp.page.chunks if c.effective]
            if effs:
                dated[key] = min(effs + ([dated[key]] if key in dated else []))
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
        before = None                           # a date before the first crawl: read it, guarded (see _respond)
        if vi is None and day and self._store.views and day < self._store.views[0].taken_on:
            vi, before = 0, day
        shown = day if vi is not None and day and day > self._store.views[vi].taken_on else None
        key = (re.sub(r"[^a-z0-9$%.]+", " ", question.lower()).strip(), vi, shown, before)
        if key not in self._cache:
            # vi is None when nothing is ingested or the date is before the first snapshot
            index = self._indexes[vi] if vi is not None else None
            # A date that only names the snapshot being answered from is not a search word
            now, crawls = (_iso(as_of), None) if as_of else (_now(self._store), _crawls(self._store))
            search = strip_temporal(calendar_phrases(question, now, crawls), lambda d: self._store.view_for(d) == vi or
                                    (before is not None and _iso(d) is not None and _iso(d) < self._store.views[0].taken_on),
                                    _dates_stated(index)) if index else question
            told = question
            if asked and not as_of:
                # the date the question named is the one read: the model sees it written out ("Two months ago, …" ->
                # "In August 2026, …"), or it counts back from the snapshot date itself
                told = calendar_phrases(question, now, crawls)
                told = told[:1].upper() + told[1:]
            self._cache[key] = _respond(index, self._synth, told, search, _iso(asked), before,
                                        getattr(self._store, "first_effective", {}))
        return self._cache[key]

    def conflicts(self) -> list[Conflict]:
        _own_site(self._store)
        if not self._indexes:
            return []
        found = find_conflicts(self._indexes[-1].view, [ix.view for ix in self._indexes[:-1]])
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


UNDATED_REACH_DAYS = 365     # how far before the first crawl an undated page is taken to have read the same


def _too_new(c: Chunk, view, before: date, dated: dict) -> bool:
    """Before the first crawl, a chunk can't answer for a day before it took effect (its own effective date, or
    the earliest any crawl gives its page) or before its page was published; an undated one answers only for
    the year before the first crawl."""
    vp = view.pages[c.key]
    first = dated.get(c.key)
    if (c.effective and c.effective > before) or (first and first > before) or (vp.page.publish and vp.page.publish > before):
        return True
    return not (c.effective or first) and (view.taken_on - before).days > UNDATED_REACH_DAYS


NO_SUCH = re.compile(r"\bno\b[^.?!]{0,60}\b(?:appears?|listed|found|mentioned)\b|"
                     r"\b(?:doesn['’]t|does not|don['’]t|do not)\s+(?:list|have|show|publish|operate|appear)\b|"
                     r"\bnot\s+(?:listed|found|mentioned)\b", re.I)
FILLERS = {_stem(w) for w in "total altogether all together actual actually even know day days now today".split()}


RECOVER_BUDGET = 12000
# A recovered answer that denies, limits or excludes anything is no recovery: a refusal must be a refusal, and a missed
# recovery costs nothing (the question stays declined).
RECOVER_ABSENT = re.compile(r"\b(?:no|none|not|never|nothing|nobody|without|only|cannot|can['’]t|isn['’]t|aren['’]t|doesn['’]t|"
                            r"don['’]t|didn['’]t|won['’]t|hasn['’]t|haven['’]t)\b", re.I)


def _respond(index: Index | None, synth: Synthesizer, question: str, search: str | None = None,
             asked: date | None = None, before: date | None = None, dated: dict | None = None) -> Answer:
    calls = synth.calls
    ans = _respond_core(index, synth, question, search, asked, before, dated)
    if ans.confident or index is None or not synth.enabled or before is not None or synth.calls != calls:
        return ans                              # only a question the rules declined before any model call can be recovered
    return _recover(index, synth, question, search, asked) or ans


def _recover(index: Index, synth: Synthesizer, question: str, search: str | None, asked: date | None) -> Answer | None:
    """A decline the rules made before any model call, of a question that may be answerable: the model picks pages by title
    from the site's own list and answers once from them, under the guards of every model answer (numbers from the evidence,
    nothing that says the site doesn't have it). Never for a rate or hours question (a term, tier or place the site lacks is
    the usual reason: "not listed" is the truth), nor a branch or ATM asked of a place no page writes."""
    q = parse_query(search or question)
    if q.attribute in ("rate", "hours"):
        return None
    named = _places_asked(index, q.raw)
    if named and (not named[0] or not _place_has_page(index, named[1])):
        return None                              # a place no page is about has nothing to recover
    pages, paths = _page_list(index)
    keys = [paths[p] for p in map(_path, synth.pick_pages(question, pages)) if p in paths]
    keys = _pages_about(index, q, [set(p) for p in place_phrases(q.raw)] + (named[1] if named else [])) + keys
    ev = _pages_evidence(index, q, keys, RECOVER_BUDGET) if keys else None
    if ev is None or not ev.chunks:
        return None
    out = synth._llm(question, ev, index.view, asked, q.list_intent)
    if not out or not out.get("answerable", False):
        return None
    accepted = synth._accept(question, q, ev, index.view, out, asked=asked)
    if accepted is None or _says_absent(accepted[0], question) or RECOVER_ABSENT.search(accepted[0]) or ABSENT.search(accepted[0]):
        return None                              # an answer that says something is not there is no recovery
    text, cited, note = accepted
    if q.attribute == "fee" and not retrieve.WIRE.search(q.raw):
        focus = retrieve.fee_focus(index, q)
        if focus[0] and not any(retrieve.chunk_cover(c, q, focus)[1] for c in cited):
            return None                          # a fee line that names none of what was asked is a near miss, not its fee
    seen, sources = set(), []
    for c in cited:
        vp = index.view.pages[c.key]
        if (c.key, vp.snapshot_id) not in seen and len(sources) < MAX_SOURCES:
            seen.add((c.key, vp.snapshot_id))
            sources.append(Source(url=vp.page.url, snapshot_id=vp.snapshot_id, quote=c.text[:240]))
    return Answer(text=text, sources=sources, confident=True, conflict_note=note or None)


# "Mini-Max Loans: Add 1% to conventional rates": a program the site prices as another line's rate plus a stated amount
MODIFIER_RE = re.compile(r"([A-Z][\w-]*(?:[ -][A-Z][\w-]*)?)\s+Loans?:\s*Add\s+(\d+(?:\.\d+)?)\s*%\s+to\s+([a-z]+(?:\s+[a-z]+)?)\s+rates?")
HOW_MUCH_MORE = re.compile(r"\b(?:how much|by how much)\b[^?]{0,60}\b(?:higher|more|extra|above|over)\b|"
                           r"\b(?:higher|more|extra)\b[^?]{0,40}\bthan\b|\b(?:difference|markup|surcharge|premium)\b", re.I)


def _rate_modifiers(index: Index) -> list:
    """The programs a live product page prices as another line's rate plus an amount: [(name, tokens, percent, base, chunk)]."""
    found = getattr(index, "_modifiers", None)
    if found is None:
        found, seen = [], set()
        for c in index.chunks:
            vp = index.view.pages[c.key]
            if c.is_row or vp.page.kind != "product" or vp.status != "fresh":
                continue
            for m in MODIFIER_RE.finditer(c.text):
                toks = frozenset(tokenize(m.group(1)))
                if toks and (toks, m.group(2)) not in seen:
                    seen.add((toks, m.group(2)))
                    found.append((m.group(1), toks, float(m.group(2)), m.group(3), c))
        index._modifiers = found
    return found


# a person's biography sits under a heading that gives the name and the title under it: the title is in the section, not the text
TITLE_ASK = re.compile(r"\b(?:title|role|position|job)\b|\bwho\s+(?:is|was)\b|\bwhat\s+does\b[^?]*\bdo\b", re.I)
TITLE_ONLY = re.compile(r"\b(?:title|role|position|job)\b", re.I)


def _with_heading_title(cited: list, question: str, text: str) -> str:
    """"Name · Title" over the chunk the answer rests on, the name asked about: say the title ("Name is Title.") for a title
    question, and put it before the biography for a who-is question, unless the answer already says it."""
    low = set(tokenize(question))
    for c in cited:
        name, _, title = (c.section or "").partition(" · ")
        if not title or not ingest.NAME_HEADING.fullmatch(name) or not (ingest.ROLE_WORD.search(title) or not ingest.NAME_HEADING.fullmatch(title)) \
                or title.lower() in text.lower():
            continue
        if set(tokenize(name)) <= low:
            lead = f"{name} is {title}."
            return lead if TITLE_ONLY.search(question) else f"{lead} {text}"
    return text


def _staff_titles(index: Index) -> dict:
    """{frozenset of name tokens: (name, title, chunk)} for every "Name · Title" section of a live product page (the staff
    pages: a name heading and the title heading under it). A name given two different titles is left out. Built once per view."""
    found = getattr(index, "_staff", None)
    if found is None:
        found, clash = {}, set()
        for c in index.chunks:
            name, _, title = (c.section or "").partition(" · ")
            vp = index.view.pages[c.key]
            if (not title or vp.status != "fresh" or vp.page.kind != "product" or not ingest.NAME_HEADING.fullmatch(name)
                    or not (ingest.ROLE_WORD.search(title) or not ingest.NAME_HEADING.fullmatch(title)) or len(title) > 70):
                continue
            key = frozenset(tokenize(name))
            if key in found and found[key][1].lower() != title.lower():
                clash.add(key)
            found.setdefault(key, (name, title, c))
        for key in clash:
            found.pop(key, None)
        index._staff = found
    return found


def _respond_title(index: Index, question: str) -> Answer | None:
    """"What is <Name>'s title?": the title the site's staff page gives under the name, whatever a blog post from years ago
    says (a quote in an old article names the title the person held then)."""
    if not TITLE_ONLY.search(question):
        return None
    low = set(tokenize(question))
    hits = [v for k, v in _staff_titles(index).items() if k <= low]
    if len(hits) != 1:
        return None
    name, title, c = hits[0]
    vp = index.view.pages[c.key]
    return Answer(text=f"{name} is {title}.", sources=[Source(url=vp.page.url, snapshot_id=vp.snapshot_id, quote=c.text[:240])],
                  confident=True)


# a question that sets the program aside ("not Mini-Max", "conventional vs Mini-Max", "the base rate") asks for the plain line
SETS_ASIDE = re.compile(r"\b(?:not|non|without|excluding|except|instead\s+of|rather\s+than|versus|vs|other\s+than|besides|don['’]t|do\s+not|"
                        r"base\s+rates?)\b", re.I)
# a row of the line the footnote prices from: "Auto & Motorcycle Loans — 2024-27 Model Year, 60 Months: 6.39% APR"
ROW_ANSWER = re.compile(r"^[^.:]+ — [^:]+: .*?\d%")


def _on_footnoted_line(chunk, name: str, answer: str) -> bool:
    """Is the base answer a row of the line the footnote sits under ("Auto Loan Rates: ... <Name> Loans: Add 1% ...")? The
    footnote's own label ("Auto Loan Rates") names the line; the answer must name it too, or the amount is not for it."""
    before = chunk.text[:max(chunk.text.find(name), 0)]
    labels = re.findall(r"((?:[A-Z][\w&]*\s+){1,3})Rates?:", before)
    words = {_stem(w.lower()) for w in re.findall(r"[A-Za-z]+", labels[-1])} - {_stem(w) for w in ("loan", "loans", "rate", "rates", "apr")} if labels else set()
    return bool(words) and bool(words & set(tokenize(answer)))


def _respond_modifier(index: Index, synth: Synthesizer, question: str, search: str | None, asked: date | None,
                      before: date | None, dated: dict | None) -> Answer | None:
    """A rate question that names a program priced as another line plus an amount: that line's rate for the rest of the
    question, each APR raised by the amount ("Mini-Max loans add 1% to the conventional rate: 6.39% APR ..."); the rule alone
    when the question asks how much more, or when no one row of the base line answers it."""
    words = {t for t in tokenize(search or question)}
    hit = next((m for m in _rate_modifiers(index) if m[1] <= words or _stem(re.sub(r"[\s-]+", "", m[0].lower())) in words), None)
    if hit is None or SETS_ASIDE.search(question):
        return None
    name, toks, pct, base, chunk = hit
    vp = index.view.pages[chunk.key]
    rule = f"{name} loans add {pct:g}% to {base} rates."
    sources = [Source(url=vp.page.url, snapshot_id=vp.snapshot_id, quote=chunk.text[:240])]
    if HOW_MUCH_MORE.search(question):
        return Answer(text=rule, sources=sources, confident=True)
    stripped = re.sub(r"\b" + r"[\s-]*".join(re.escape(w) for w in re.split(r"[\s-]+", name.lower())) + r"\b(?:\s+(?:loans?|program|plan))?",
                      " ", search or question, flags=re.I)
    plain = copy.copy(synth)
    plain.enabled = False                                # the base line's own row: no model, so the figures are the site's
    ans = _respond_core(index, plain, stripped, None, asked, before, dated, modifiers=False)
    figures = re.findall(r"(\d+(?:\.\d+)?)%", ans.text) if ans.confident else []
    if not figures or len(set(figures)) > 6 or not ROW_ANSWER.search(ans.text) or not _on_footnoted_line(chunk, name, ans.text):
        return Answer(text=rule, sources=sources, confident=True)
    raised = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: f"{float(m.group(1)) + pct:.2f}%", ans.text)
    return Answer(text=f"{name} loans add {pct:g}% to {base} rates: {raised}", sources=ans.sources[:MAX_SOURCES - 1] + sources,
                  confident=True)


def _respond_core(index: Index | None, synth: Synthesizer, question: str, search: str | None = None,
                  asked: date | None = None, before: date | None = None, dated: dict | None = None,
                  modifiers: bool = True) -> Answer:
    """`before`: the question asks about a day before the first crawl, answered from the first crawl. The answer
    stands only if nothing it rests on took effect after that day (see _too_new)."""
    if index is None:
        return Answer(text=REFUSAL, confident=False)
    view = index.view
    q = parse_query(search or question)       # the model still sees the full question
    q.spelled_counts = synth.enabled          # a count a page spells in words is for the model to read
    if modifiers and q.attribute == "rate" and (modified := _respond_modifier(index, synth, question, search, asked, before, dated)):
        return modified
    if (titled := _respond_title(index, question)) is not None:
        return titled
    linking = LINK_Q.search(question) is not None
    vocab = _anchor_vocab(index) if linking else None
    named = _title_names(index) if linking else None
    joining, _, wanted = joining_info(q, vocab, named) if linking else (False, set(), set())
    if joining and not wanted and _stem("join") not in q.tokens:
        q.expansion = q.expansion + [_stem("join")]        # "sign up as a member": the link to join
        q.tokens = q.tokens + [_stem("join")]
    ev = select(index, q)
    named = _places_asked(index, q.raw)
    if named and not named[0]:
        return Answer(text=REFUSAL, confident=False)       # a branch or ATM asked of a place that no page of the site writes
    lead_split = count_clause(q.raw) if q.attribute == "count" else fee_clause(q.raw) if q.attribute == "fee" else None
    antecedent = []
    if lead_split and not ev.supported:
        antecedent = _antecedent(*lead_split)
        q, ev = _retry_clause(index, synth, q, ev, *lead_split, antecedent)
    if linking:                                       # "what link should I use…": the site's own link
        carried = lambda u: sum(u in t for t in _page_texts(index).values())
        near, wide = ev.context or ev.chunks, [c for _, c in index.search(q.tokens, top=30)]
        joins = _join_chunks(index) if joining else []
        link = (pick_link(near, view, q, carried, joins, vocab, named)
                or pick_link(wide, view, q, carried, joins, vocab, named))
        if not link and any(t in LINK_WORDS for t in q.tokens):
            # "which page should I visit to view <X>": the frame words searched too; look again without them
            q2 = parse_query(search or question)
            q2.tokens = [t for t in q2.tokens if t not in LINK_WORDS] or q2.tokens
            ev2 = select(index, q2)
            link = pick_link(ev2.context or ev2.chunks, view, q, carried, joins, vocab, named)
            if link:
                near = (ev2.context or ev2.chunks) + near
        if link:
            vp = link_source(link, near + wide + joins, view)
            return Answer(text=f"You can find that at {link}.", confident=True,
                          sources=[Source(url=vp.page.url, snapshot_id=vp.snapshot_id)] if vp else [])
    if not ev.chunks or (q.attribute and not ev.supported and not (synth.enabled and carried_in_context(ev, q))):
        return Answer(text=REFUSAL, confident=False)
    if q.attribute == "count" and synth.enabled and lead_split:
        ev.context = _with_count_chunks(index, q, ev, list(q.tokens) + antecedent)
    if q.list_intent and synth.enabled:
        ev.context = _with_listed_pages(index, q, ev)
    if synth.enabled and named and q.attribute in (None, "address") and _place_has_page(index, named[1]):
        ev.context = _with_place_chunks(index, q, ev, named[1])
    if q.year_kind and not _model_year_rows(index, q):
        return Answer(text=REFUSAL, confident=False)       # the vehicle's year and term fit no row of its kind
    if q.year_guess:
        rows = _model_year_rows(index, q)
        if not rows or not synth.enabled:
            # no model-year row fits the year and term, or the word beside the year is not one the vehicle list knows and
            # only the model can say what it is and which product line it is financed under
            return Answer(text=REFUSAL, confident=False)
        have = {c.id for c in ev.context or ev.chunks}
        ev.context = (ev.context or ev.chunks) + [c for c in rows if c.id not in have]
    fact = match_fact(_facts(index), q, index)
    if fact:
        ev.context = apply_fact(ev.context or ev.chunks, fact, view)
    def second_look(text: str):
        pages, paths = _page_list(index)
        keys = [paths[p] for p in map(_path, synth.pick_pages(text, pages)) if p in paths]
        return _pages_evidence(index, q, keys) if keys else None
    told = question
    if q.place:                                       # the model isn't told a branch it might think matters
        told = re.sub(rf"[,\s]*\b(?:at|in|from|near|for|of)?\s*(?:the|your|our|a)?\s*{re.escape(q.place)}\b[\w\s,'’.-]*?"
                      r"\b(?:branch|location|office)\b", " ", question, count=1, flags=re.I)
        told = re.sub(r"\s+([?.!,;:])", r"\1", re.sub(r"\s{2,}", " ", told)).strip()
    text, cited, confident, note = synth.compose(told, q, ev, view, asked,
                                                 lookup=lambda text: select(index, parse_query(text)),
                                                 second_look=second_look)
    if not confident and q.list_intent and q.attribute is None and not before:
        listed = [c for c in _outline_chunks(index, q) if " lists: " in c.text]
        if listed:                                   # the model declined a list the best page spells out heading by heading
            vp = view.pages[listed[0].key]
            return Answer(text=listed[0].text.split(". ")[0].rstrip(".") + ".", confident=True,
                          sources=[Source(url=vp.page.url, snapshot_id=vp.snapshot_id, quote=listed[0].text[:240])])
    if not confident or (before and any(_too_new(c, view, before, dated or {}) for c in cited)):
        return Answer(text=REFUSAL, confident=False)
    if named and (NO_SUCH.search(text) or ABSENT.search(text)) and not _is_a_branch(index, named[1]) \
            and not any(_page_is_about(index, c, named[1]) for c in cited):
        return Answer(text=REFUSAL, confident=False)       # "no branch in X is listed" and nothing cited is about X: no other figure answers it
    if q.attribute == "count" and synth.enabled:
        text = spelled_as_digits(text, small=True)        # "Thirty-two" on the page: the member and the scorer read "32"
    if TITLE_ASK.search(question):
        text = _with_heading_title(cited, question, text)
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


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th weekday of a month (n = -1: the last one)."""
    if n > 0:
        d = date(year, month, 1)
        return d + timedelta(days=(weekday - d.weekday()) % 7, weeks=n - 1)
    d = date(year, month, calendar.monthrange(year, month)[1])
    return d - timedelta(days=(d.weekday() - weekday) % 7)


# US holidays with a year ("Memorial Day 2026"), quarters, halves, seasons after a frame word, and "2026-07"
HOLIDAYS = {r"new year['’]?s(?: day)?": lambda y: date(y, 1, 1),
            r"(?:martin luther king|mlk)(?: jr\.?)? day": lambda y: _nth_weekday(y, 1, 0, 3),
            r"presidents['’]? day": lambda y: _nth_weekday(y, 2, 0, 3),
            r"memorial day": lambda y: _nth_weekday(y, 5, 0, -1),
            r"juneteenth": lambda y: date(y, 6, 19),
            r"(?:independence day|the fourth of july|fourth of july)": lambda y: date(y, 7, 4),
            r"labor day": lambda y: _nth_weekday(y, 9, 0, 1),
            r"(?:columbus|indigenous peoples['’]?) day": lambda y: _nth_weekday(y, 10, 0, 2),
            r"halloween": lambda y: date(y, 10, 31),
            r"veterans['’]? day": lambda y: date(y, 11, 11),
            r"thanksgiving(?: day)?": lambda y: _nth_weekday(y, 11, 3, 4),
            r"christmas eve": lambda y: date(y, 12, 24),
            r"christmas(?: day)?": lambda y: date(y, 12, 25),
            r"new year['’]?s eve": lambda y: date(y, 12, 31)}
SEASON_END = {"spring": 5, "summer": 8, "fall": 11, "autumn": 11}     # meteorological seasons, by their last month
_SEASON = re.compile(r"\b(in|during|as of|as at|back in|by|since|before|after)\s+(?:the\s+)?(spring|summer|fall|autumn)"
                     r"\s+(?:of\s+)?(20\d\d)\b(?!\s+(?:semester|transcript|term|edition|cycle|session|class|catalog))", re.I)
_QUARTER = re.compile(r"\bQ([1-4])\s+(20\d\d)\b|\b(?:the\s+)?(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(20\d\d)\b", re.I)
_HALF = re.compile(r"\b(?:the\s+)?(first|second)\s+half\s+(?:of\s+)?(20\d\d)\b", re.I)
_YEAR_MONTH = re.compile(r"\b(20\d\d)-(0[1-9]|1[0-2])\b(?!-\d)(?![\s-]*(?:model|vehicle|car|truck|suv)s?\b)", re.I)
_ORDINAL = ["first", "second", "third", "fourth"]
# A year alone names a moment only after "as of / back in / by" (the year's end) or "before / until" (the day
# before it starts), and only where it ends the phrase. "In 2025" names a period, and a year next to a vehicle
# word is a model year ("a car built prior to 2020").
_YEAR_ONLY = re.compile(r"\b(as of|as at|back in|by|before|prior to|until|till)\s+(20\d\d)\b(?=\s*(?:[,?.!;:]|$))", re.I)
_MODEL_CUE = re.compile(r"\b(?:model|vehicle|car|truck|suv|built|made|manufactured)s?\b", re.I)
# Relative dates that name the same month whether "now" is the newest crawl or a little after it: "last May",
# "this past May", "a year ago". "Last month", "two weeks ago" and "last year" (a period) are not read.
_LAST_MONTH = re.compile(r"\b(?:last|this past)\s+(%s)\b(?!\s+\d)" % "|".join(calendar.month_name[1:]), re.I)
_YEARS_AGO = re.compile(r"\b(a|an|one|two|three|four|five|[1-9])\s+years?\s+ago\b", re.I)
_YEARS_N = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_FRAMED = re.compile(r"\b(?:as of|as at|back in|back on|prior to|up until|until|till|before|since|after|by|from|"
                     r"between|and|to|in|on|during)\s*$", re.I)


# In a question about a value (a rate or a fee, "how much was …"), a year or a recent stretch of time names the
# moment to read the site at: "in 2025", "last year", "last month", "two months ago". In a count, an event or an
# article it names a period ("how many employees volunteered last year?") and is not read. "Over the last year"
# and "the last week of the month" are spans, never moments.
_IN_YEAR = re.compile(r"\b(in|during)\s+(20\d\d)\b(?=\s*(?:[,?.!;:]|$))", re.I)
_LAST_UNIT = re.compile(r"(?<!the\s)\blast\s+(year|month)\b(?!['’]s)(?!\s+of\b)", re.I)
_MONTHS_AGO = re.compile(r"\b(a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|[1-9]|1[01])\s+months?\s+ago\b", re.I)
_WEEKS_AGO = re.compile(r"(?<!the\s)\blast\s+week\b(?!['’]s)(?!\s+of\b)|\b(a|one|two|three|four|five|six|[1-6])\s+weeks?\s+ago\b", re.I)
_UNITS_N = {**_YEARS_N, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11}
# a week phrase reads a crawl only if the crawl is the same for any "now" from two to six weeks after the newest crawl
WEEK_NOW_RANGE = (14, 42)


def _value_question(text: str) -> bool:
    low = text.lower()
    return bool((retrieve.RATE_Q.search(low) or retrieve.FEE_Q.search(low) or retrieve.PCT_PAID.search(low)
                 or retrieve.DEPOSIT_PAYS.search(low)) and not retrieve.COUNT_Q.search(low) and not EVENT_Q.search(low))


def _crawl_at(day: date, crawls) -> int:
    return max((i for i, c in enumerate(crawls) if c <= day), default=-1)


def _relative(text: str, now: date, crawls=None) -> str:
    """"last May" -> "in May 2030" (the latest May before `now`), "a year ago" -> "on September 28, 2029"; in a
    value question also "in 2029" / "last year" -> "in December 2029", "last month" / "two months ago" -> that
    month, and a week phrase whose crawl doesn't depend on `now` (see WEEK_NOW_RANGE) -> its day. "In" / "on" only
    when no frame word comes before. `crawls`: the crawl dates when `now` is only assumed (no as-of date)."""
    put = lambda m, prep, d: d if _FRAMED.search(m.string[:m.start()]) else f"{prep} {d}"
    day = lambda d: f"{calendar.month_name[d.month]} {d.day}, {d.year}"
    def last(m):
        month = [n.lower() for n in calendar.month_name].index(m.group(1).lower())
        return put(m, "in", f"{calendar.month_name[month]} {now.year if month < now.month else now.year - 1}")
    def ago(m):
        y = now.year - (_YEARS_N.get(m.group(1).lower()) or int(m.group(1)))
        return put(m, "on", f"{calendar.month_name[now.month]} {min(now.day, calendar.monthrange(y, now.month)[1])}, {y}")
    text = _YEARS_AGO.sub(ago, _LAST_MONTH.sub(last, text))
    if not _value_question(text) or DATE_PHRASE.search(text):     # a date written out says it ("last month (May 2030)")
        return text
    def month_back(n):
        y, m = divmod(now.year * 12 + now.month - 1 - n, 12)
        return f"{calendar.month_name[m + 1]} {y}"
    if not _MODEL_CUE.search(text):                      # "a car bought in 2025" is a model year
        text = _IN_YEAR.sub(lambda m: f"{m.group(1)} December {m.group(2)}", text)
    text = _LAST_UNIT.sub(lambda m: put(m, "in", f"December {now.year - 1}" if m.group(1).lower() == "year" else month_back(1)), text)
    text = _MONTHS_AGO.sub(lambda m: put(m, "in", month_back(_UNITS_N.get(m.group(1).lower()) or int(m.group(1)))), text)
    def weeks(m):
        n = 1 if m.group(1) is None else (_UNITS_N.get(m.group(1).lower()) or int(m.group(1)))
        if crawls:
            lo, hi = (crawls[-1] + timedelta(days=d) - timedelta(weeks=n) for d in WEEK_NOW_RANGE)
            if _crawl_at(lo, crawls) != _crawl_at(hi, crawls):
                return m.group(0)                        # which crawl depends on when "now" is: not read
        return put(m, "on", day(now - timedelta(weeks=n)))
    return _WEEKS_AGO.sub(weeks, text)


def calendar_phrases(text: str, now: date | None = None, crawls=None) -> str:
    """Calendar phrases a date reader doesn't know, as dates it does: "Memorial Day 2026" -> "May 25, 2026",
    "Q3 2026" / "the third quarter of 2026" -> "September 2026", "the first half of 2026" -> "June 2026",
    "in the summer of 2026" -> "in August 2026", "2026-07" -> "July 2026", "as of 2025" -> "as of December 2025",
    "before 2026" -> "before January 2026"; with `now` (the newest crawl), "last May" and "a year ago"."""
    day = lambda d: f"{calendar.month_name[d.month]} {d.day}, {d.year}"
    month = lambda y, m: f"{calendar.month_name[m]} {y}"
    for name, when in HOLIDAYS.items():
        text = re.sub(rf"\b{name},?\s+(20\d\d)\b", lambda m: day(when(int(m.group(1)))), text, flags=re.I)
    text = _SEASON.sub(lambda m: f"{m.group(1)} {month(int(m.group(3)), SEASON_END[m.group(2).lower()])}", text)
    text = _QUARTER.sub(lambda m: month(int(m.group(2) or m.group(4)),
                                        3 * (int(m.group(1)) if m.group(1) else _ORDINAL.index(m.group(3).lower()) + 1)), text)
    text = _HALF.sub(lambda m: month(int(m.group(2)), 6 if m.group(1).lower() == "first" else 12), text)
    if not _MODEL_CUE.search(text):
        text = _YEAR_ONLY.sub(lambda m: f"{m.group(1)} {'January' if m.group(1).lower() in ('before', 'prior to', 'until', 'till') else 'December'}"
                                        f" {m.group(2)}", text)
    if now is not None:
        text = _relative(text, now, crawls)
    return _YEAR_MONTH.sub(lambda m: month(int(m.group(1)), int(m.group(2))), text)


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
BLOCK_FRAMES = {"since", "after", "from", "between", "and", "to"}   # a span or a change, not a moment
BY_OPENS = re.compile(r"(?:^|[.!?;:,—–]\s*|\s-\s)by\b[^.!?;:,—–]*$", re.I)   # "By (the end of) June 2026, ..." opens it
QUALIFIER_RE = re.compile(r"\b(mid|middle|early|start|beginning|late|end)\b", re.I)
AUX_RE = re.compile(r"\b(used to|is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\b|"
                    r"\b(?:what|who|where|when|how|which|that|it|there|here)['’]s\b", re.I)
PAST_AUX = {"was", "were", "did", "had", "used to"}
NOW_Q = re.compile(r"\b(now|today|currently|presently|nowadays|anymore|any longer|these days)\b", re.I)
# "now" set against a past date compares the two ("is it now higher than back in May?"): the newest crawl answers
COMPARE_NOW = re.compile(r"\b(?:than|compar\w*|versus|vs\.?|same|differ\w*|still|chang\w*|since)\b", re.I)
# A question about a change, or about an article or event, is answered from the newest crawl, which
# records the change or the article; the date then says when it happened, not when to read the site.
CHANGE_Q = re.compile(r"\b(chang\w*|rise[sn]?|rising|rose|raise[ds]?|increas\w*|decreas\w*|drop\w*|fell|fall\w*|"
                      r"lowered|cut|jump\w*|climb\w*|(?:went|go|goes|gone) (?:up|down)|effective|"
                      r"(?:took|take[ns]?|went into) effect|set (?:on|at|to)|updated?|adjust\w*)\b", re.I)
EVENT_Q = re.compile(r"\b(posts?|posted|articles?|blogs?|story|stories|news\w*|press|releases?|released|"
                     r"announc\w*|reports?|reported|publish\w*|episodes?|podcasts?|videos?|webinars?|"
                     r"seminars?|events?|meetings?)\b", re.I)
CLAUSE_END = re.compile(r"(?<=[;!?])\s+|(?<=[a-z0-9%)\]][.])\s+(?=[A-Z])")   # never inside 9,999.99 or Sept. 1


# "Now", for relative dates when no as-of date is given: questions are asked after the newest crawl, about a month
# after it ("today" is the month after the last crawl)
NOW_AFTER_NEWEST_DAYS = 30


def _now(store: Store) -> date | None:
    return store.views[-1].taken_on + timedelta(days=NOW_AFTER_NEWEST_DAYS) if store.views else None


def _crawls(store: Store) -> list:
    return [v.taken_on for v in store.views]


def date_asked(question: str, store: Store) -> str | None:
    """When no date is passed, a single moment the question itself names inside the crawl window
    ("as of March 2031", "what was it on 3/14/2031", "before April 2031") selects the site as it stood then.
    "As of / back in" always pin; "in / on / during" pin only in a past-tense question that is not about a
    change or an article or event. Not for questions about now (unless every date named is pinned: "Right now, as
    of March 2031, ..."; never when "now" is compared with it), spans ("since", "between"), dates outside the window,
    or a date in a side remark of a present-tense question."""
    q = calendar_phrases(question, _now(store), _crawls(store)).strip()
    now = NOW_Q.search(q)
    if len(store.views) < 2 or (now and COMPARE_NOW.search(q)):
        return None
    pins = set()                            # "Right now, as of June 2026, ...": the date it pins is meant
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
        if frame == "by" and not BY_OPENS.search(before):
            return None                         # "...had to be opened by August 31": a deadline, not a moment
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
        pins.add(frame in PIN_FRAMES)
    if (now and pins != {True}) or len({d for d, _ in found}) != 1:
        return None
    d, precise = found.pop()
    first, last = store.views[0].taken_on, store.views[-1].taken_on
    if not precise and d < first and (d.year, d.month) == (first.year, first.month):
        d = first                       # "early March" when the first crawl is mid-March: the nearest reading
    return d.isoformat() if d < last else None      # before the first crawl: see KnowledgeSystem.answer


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
    if any(p.publish and since < p.publish <= asked for p in best) or any(p.publish == asked for p in best):
        return None                     # ... or the question names the day one of those articles came out
    return d


# A question about how a value changed over time ("How did X change between June and September 2026?",
# "Has X gone up since May 2026?") is answered from the site as it stood at each date, both values given
# with their dates. Both the change wording and a date in the question are needed; otherwise nothing changes.
CHANGE_ASK = re.compile(
    r"\bhow (?:did|has|have|had)\b[^?]{0,120}?\bchange[ds]?\b|"
    r"\b(?:did|has|have|had)\b[^?]{0,120}?\b(?:go(?:ne)? (?:up|down)|went (?:up|down)|increase[ds]?|"
    r"decrease[ds]?|rise[ns]?|rose|fall(?:en)?|fell|drop(?:ped)?|change[ds]?)\b|"
    r"\bwhat (?:was|is|were) the change\b|\bwhat happened to\b|"
    r"\bwhat(?:['’]s|\s+has|\s+have)?\s+(?:been\s+)?(?:changed|changes?|different|updated)\b|"
    r"\bwhat\s+(?:are|were)\s+the\s+(?:changes|differences)\b|\b(?:anything|something)\s+(?:changed|different)\b", re.I)
# "Was it the same on May 12 and June 9, 2026?", "compare … on A with … on B": a comparison across two dates
COMPARE_ASK = re.compile(r"\b(?:was|were|is|are)\b[^?]{0,120}?\bthe same\b|\bcompare[ds]?\b|\b(?:differ|different)\b", re.I)
_MONTH = "|".join(MONTHS) + "|sept?|jan|feb|mar|apr|jun|jul|aug|oct|nov|dec"
MONTH_PAIR = re.compile(rf"\b(?:between|from)\s+({_MONTH})\.?\s+(?:and|to)\s+(?:{_MONTH})\.?\s+"
                        rf"(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?(20\d\d)", re.I)          # "between May and June 2026"
SPAN_FROM = re.compile(r"\b(?:since|after|from)\s+(?:\S+\s+){0,2}$", re.I)


# "What was it in May and what is it now?", "Has it changed since the middle of May?", "…when I first checked in May":
# one past moment, and "now" (or "since" it), asks for the site at both. A moment is a date, a day without a year
# ("May 12"), or a month alone; the newest crawl is "now".
# "the most recent round" / "the latest cohort" names a thing, not the moment "now": the adjective counts only when no such noun follows
NOW_REF = re.compile(r"\b(?:now|currently|today|at present|these days|most\s+recently|latest(?:\s+update)?|most\s+recent|newest)\b"
                     r"(?!\s+(?:round|cohort|class|batch|group|recipients?|winners?|awardees?|edition|announcement|news|press|article|"
                     r"story|post|issue|grant|award|program|list)\b)", re.I)
FIRST_CHECKED = re.compile(r"\b(?:first\s+checked|when\s+i\s+first|at\s+first|originally)\b", re.I)
AS_OF_PIN = re.compile(r"\bas\s+(?:of|at)\b", re.I)
SINCE_RE = re.compile(r"\bsince\b", re.I)
# "is it still the same as it was in August?": the past moment set against the newest crawl, with no "now" said
SAME_AS = re.compile(r"\b(?:still\s+(?:the\s+)?same|(?:the\s+)?same\s+as|unchanged)\b", re.I)
# a season is a stretch of months: "when I first looked in the spring" is the first crawl in March to May
SEASON = re.compile(r"\b(?:(?P<q>early|late|mid)[\s-]+|(?:in|since|from|during|last|this|over)\s+(?:the\s+)?|the\s+)"
                    r"(?P<s>spring|summer|fall|autumn)\b", re.I)
SEASON_MONTHS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11)}
# a value that can change, said without a rate, fee or count: "Did the maximum grant change since May?"
CHANGED_WORDS = re.compile(r"\b(?:chang\w*|gone\s+up|go(?:es)?\s+up|went\s+up|go(?:es)?\s+down|went\s+down|gone\s+down|increas\w*|"
                           r"decreas\w*|rise[ns]?|rose|fell|fall(?:en)?|drop(?:ped|s)?|lower(?:ed)?|higher|rais(?:e[ds]?|ing)|differ\w*|"
                           r"same|unchanged)\b", re.I)
_MONTH_FULL = "|".join(MONTHS)                      # in full: "APR" is a loan term, "may" a verb
MONTH_ONLY = re.compile(rf"\b(?:(?:the\s+)?(?P<q1>start|beginning|middle|end)\s+of\s+|(?P<q2>early|mid|late)[\s-]+)?"
                        rf"(?P<m>{_MONTH_FULL})\b(?!\.?\s*\d)", re.I)
MONTH_DAY = re.compile(rf"\b(?P<m>{_MONTH_FULL})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*,?\s*\d)", re.I)
MAY_AFTER = re.compile(r"\b(?:in|since|from|between|of|by|back|and|to)\s+$", re.I)     # "May I ask…" is no month
MONTH_QUALIFIER = re.compile(r"(?:(?:the\s+)?(?P<q1>start|beginning|middle|end)\s+of|(?P<q2>early|mid|late)[\s-]*)\s*$", re.I)
NEAR_FIRST_CRAWL_DAYS = 45      # a dated day this close before the first crawl is read as the first crawl


def _month_moment(store: Store, year: int, month: int, qualifier: str, first_checked: bool) -> date | None:
    """The day a month (with a qualifier) names, as a moment to read the site at; None outside the crawls' months.
    The start of a month, "early", or "when I first checked" is the first crawl in it (or the latest before it);
    the middle is the 15th; the end, "late" or a month alone is its last day."""
    views = store.views
    start, end = date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    if end < views[0].taken_on.replace(day=1) or start > views[-1].taken_on:
        return None
    if qualifier in ("start", "beginning", "early") or first_checked:
        inside = [v.taken_on for v in views if start <= v.taken_on <= end]
        before = [v.taken_on for v in views if v.taken_on <= start]
        return inside[0] if inside else (before[-1] if before else views[0].taken_on)
    return date(year, month, 15) if qualifier in ("middle", "mid") else end


def _season_moment(store: Store, season: str, qualifier: str, first_checked: bool) -> date | None:
    """The day a season names: the first crawl in it (early, or "when I first looked"), else its last day."""
    views = store.views
    year = views[-1].taken_on.year
    first_month, last_month = SEASON_MONTHS[season]
    start, end = date(year, first_month, 1), date(year, last_month, calendar.monthrange(year, last_month)[1])
    if end < views[0].taken_on.replace(day=1) or start > views[-1].taken_on:
        return None
    if qualifier == "early" or first_checked:
        inside = [v.taken_on for v in views if start <= v.taken_on <= end]
        before = [v.taken_on for v in views if v.taken_on <= start]
        return inside[0] if inside else (before[-1] if before else views[0].taken_on)
    return start + (end - start) // 2 if qualifier == "mid" else end


def _then_now_span(question: str, original: str, store: Store) -> tuple | None:
    """(then, None): one past moment against the newest crawl. Fires on an explicit "now" (not pinned by "as of"),
    on "since <moment>", or on "when I first checked"; and only for a value that can change (a rate, fee or count,
    or a change verb). A month before the first crawl's month or after the newest crawl is no moment."""
    first_checked = bool(FIRST_CHECKED.search(question))
    pinned = AS_OF_PIN.search(question)
    if not (((NOW_REF.search(question) or SAME_AS.search(question)) and not pinned) or SINCE_RE.search(question) or first_checked):
        return None
    pq = parse_query(original)
    if not (pq.attribute in ("rate", "fee", "count") or (pq.attribute is None and CHANGED_WORDS.search(question))):
        return None
    views = store.views
    first, newest = views[0].taken_on, views[-1].taken_on
    days = []
    for m in DATE_PHRASE.finditer(question):
        d = _phrase_date(m.group(1))
        if d is None:
            continue
        if _DAY_PRECISE.search(m.group(1)):
            if d >= first:
                days.append(d)
            elif (first - d).days <= NEAR_FIRST_CRAWL_DAYS:
                days.append(first)
        else:                                       # "the middle of May 2026"
            q = MONTH_QUALIFIER.search(question[:m.start(1)])
            moment = _month_moment(store, d.year, d.month, ((q.group("q1") or q.group("q2")) if q else "").lower(), first_checked)
            if moment:
                days.append(moment)
    if not days:                                    # "May 12": the crawl year
        for m in MONTH_DAY.finditer(question):
            try:
                d = date(newest.year, MONTHS[m.group("m").lower()], int(m.group("d")))
            except (ValueError, KeyError):
                continue
            if first <= d <= newest:
                days.append(d)
            elif 0 < (first - d).days <= NEAR_FIRST_CRAWL_DAYS:
                days.append(first)                  # "around May 11", the day before the first crawl
    if not days:
        for m in SEASON.finditer(question):
            moment = _season_moment(store, m.group("s").lower(), (m.group("q") or "").lower(), first_checked)
            if moment:
                days.append(moment)
    if not days:
        for m in MONTH_ONLY.finditer(question):
            q = (m.group("q1") or m.group("q2") or "").lower()
            if m.group("m").lower() == "may" and not (q or MAY_AFTER.search(question[:m.start()])):
                continue
            moment = _month_moment(store, newest.year, MONTHS[m.group("m").lower()], q, first_checked)
            if moment:
                days.append(moment)
    days = sorted(set(days))
    if not days:
        return None
    return (days[0], days[-1]) if len(days) >= 2 else (days[0], None)


# "in May and in June 2026", "in May vs June", "in May compared to June"
IN_AND_FRAME = re.compile(r"\b(?:in|on|during|for)\s+(?P<a>.+?)\s+(?:and|vs\.?|versus|compared\s+(?:to|with))\s+(?:in\s+|on\s+)?"
                          r"(?P<b>(?:[^?,]|,(?=\s*\d))+)", re.I)
BETWEEN_FRAME = re.compile(r"\b(?:between|from)\s+(?P<a>.+?)\s+(?:and|to|through|until|till)\s+(?P<b>(?:[^?,]|,(?=\s*\d))+)", re.I)


def _half_moment(text: str, year: int, store: Store) -> date | None:
    """One side of "between A and B": a date, a day without a year, or a month with an optional qualifier."""
    first, newest = store.views[0].taken_on, store.views[-1].taken_on
    if (m := DATE_PHRASE.search(text)) and (d := _phrase_date(m.group(1))):
        if _DAY_PRECISE.search(m.group(1)):
            return d if d >= first else (first if (first - d).days <= NEAR_FIRST_CRAWL_DAYS else None)
        q = MONTH_QUALIFIER.search(text[:m.start(1)])
        return _month_moment(store, d.year, d.month, ((q.group("q1") or q.group("q2")) if q else "").lower(), False)
    if m := MONTH_DAY.search(text):
        try:
            d = date(year, MONTHS[m.group("m").lower()], int(m.group("d")))
        except (ValueError, KeyError):
            return None
        return d if first <= d <= newest else (first if 0 < (first - d).days <= NEAR_FIRST_CRAWL_DAYS else None)
    if m := MONTH_ONLY.search(text):
        return _month_moment(store, year, MONTHS[m.group("m").lower()], (m.group("q1") or m.group("q2") or "").lower(), False)
    return None


def _between_span(question: str, store: Store) -> tuple | None:
    """(earlier, later) for "between A and B" where each side is a date, a day or a month, and only one of them says the
    year ("between May 19 and June 9, 2026", "between mid-May and late September 2026")."""
    if not (m := BETWEEN_FRAME.search(question) or IN_AND_FRAME.search(question)):
        return None
    years = [d.year for x in DATE_PHRASE.finditer(question) if (d := _phrase_date(x.group(1)))]
    year = years[-1] if years else store.views[-1].taken_on.year
    a, b = _half_moment(m.group("a"), year, store), _half_moment(m.group("b"), year, store)
    return (min(a, b), max(a, b)) if a and b and a != b else None


def change_span(question: str, store: Store) -> tuple | None:
    """(start, end) dates of a change the question asks about; end None means "now". A month alone means
    the site as it stood by the end of that month. Two dated months ("between May and June 2026"), or one
    moment against now ("since the middle of May", "what was it in June and what is it now?")."""
    original = question
    question = calendar_phrases(question, _now(store), _crawls(store))
    if len(store.views) < 2:
        return None
    asks_change = CHANGE_ASK.search(question) or (COMPARE_ASK.search(question) and len(terms_in(question.lower())) <= 1)
    if asks_change:
        found = [(m.start(1), d) for m in DATE_PHRASE.finditer(question) if (d := _phrase_date(m.group(1)))]
        if len(found) < 2 and (mp := MONTH_PAIR.search(question)) and (d := _phrase_date(f"{mp.group(1)} {mp.group(2)}")):
            found.append((mp.start(1), d))
        days = sorted({d for _, d in found})
        if len(days) >= 2:
            return days[0], days[-1]
        if len(days) == 1 and SPAN_FROM.search(question[:found[0][0]]):
            return days[0], None
    if asks_change and not AS_OF_PIN.search(question) and (span := _between_span(question, store)):
        return span
    return _then_now_span(question, original, store)


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


LINK_Q = re.compile(r"\b(link|url|web ?page|website address|web ?address)\b|"
                    r"\b(?:what|which)(?:['’]s|\s+is|\s+are)?\s+(?:the\s+|your\s+)?(?:web)?(?:site|page)\b", re.I)
LINK_RE = re.compile(r"([^()\n]{0,80})\(([a-z0-9.-]+\.[a-z]{2,}(?:/[^)\s]*)?)\)")
LINK_WORDS = set(tokenize("link url use see find page web webpage website address where visit view check look browse"))
MEDIUM_WORDS = set(tokenize("online"))      # how, not what: "join online" asks for the link to join
# Joining the credit union, in general English. "membership" followed by another noun ("membership meeting",
# "membership grant") is something else; a member of a kind ("a board member") is not joining either.
_AFTER_MEMBERSHIP = r"(?=\s*(?:$|[?.!,]|\s+(?:application|form|online|today|now|please|pls|page|site|portal|link|url)\b))"
JOIN_ACT = re.compile(
    r"\b(?:(?:apply(?:ing)?|sign(?:ing)?\s+up|register(?:ing)?|enrol+(?:ing)?)\s+(?:for|to)\s+"
    r"(?:(?:a|an|the|my|\w+['’]s|\w+)\s+)?membership" + _AFTER_MEMBERSHIP +
    r"|(?:be|become|becoming|sign(?:ing)?\s+up\s+as|register(?:ing)?\s+as|enrol+(?:ing)?\s+as)\s+an?\s+(?:(\w+(?:\s+\w+)?)\s+)?member\b"
    r"|membership\s+application"
    r"|(?:open(?:ing)?|get(?:ting)?|start(?:ing)?|set(?:ting)?\s+up)\s+(?:(?:my|your|our|their|his|her|a|an)\s+)?"
    r"(?:\w+['’]s\s+)?(?:very\s+)?first\s+account(?=\s*(?:$|[?.!,;:]|\s+(?:online|with|today|now|here|for|at|and|or|so)\b)))",
    re.I)    # opening someone's (very) first account is joining; "my first account balance" is not
# "join the newsletter / webinar / committee": joining something else, unless it is the institution itself
JOIN_OBJECT = re.compile(r"\bjoin(?:ing)?\s+(?:the|our|your|a|an)\s+(?!credit\s+unions?\b)(\w+)", re.I)
JOIN_BARE = re.compile(r"\bjoin(?:ing)?\b|\bmembership" + _AFTER_MEMBERSHIP, re.I)
# picking up an application where one left off is continuing it ("complete a new application" is not)
RESUME = re.compile(r"\b(?:continu\w*|resum\w*|finish\w*|complet\w*\s+(?:my|the|our)\b|pick(?:ing)?\s+"
                    r"(?:it\s+|this\s+|that\s+)?(?:back\s+)?up|get\s+back\s+to|return\s+to)", re.I)
JOIN_WORDS = {_stem(w) for w in ("join", "member", "membership", "become")}
JOIN_ACT_WORDS = JOIN_WORDS | set(tokenize("apply sign up register enrol enroll application open first account"))
FIRST_ACCOUNT = re.compile(r"\bfirst\s+account\b", re.I)
# the clause that asks for the link: a word asked elsewhere in the message ("my buddy from work …") is context
CLAUSE_BREAK = re.compile(r"[,;:.?!—–]|\s-\s|\b(?:if|because|since|so|when|after|before)\b", re.I)
# words that frame a request rather than say what it is about
FRAME_TALK = set(tokenize("if am so also but didn doesn don isn wasn just then than very really click go send let type enter first "
                          "lets help used using before after since early late last week end beginning start started "
                          "middle mid she he her him his hers someone somebody anybody people"))
FRAME_TALK |= {_stem(w) for w in FRAME_WORDS} | {_stem(w) for w in GENERIC}
MONTH_WORD = re.compile(rf"^(?:{MONTH_RE})$", re.I)
SENTENCE_END = re.compile(r"(?<=[.?!;])\s+")         # "…join. My cousin…" ends a sentence; "site.com" does not
# the request names who joins ("so she can join", "for my cousin to join")
JOINER = re.compile(r"\b(?:she|he|they|(?:my|our|his|her|their)\s+\w+)\b[^.?!;]*\bjoin", re.I)
# how the link is passed on, never what it is for ("email it to her", "share the link", "text me the web address")
SEND_IT = re.compile(r"\b(?:e-?mail|text|message|send|forward|share)\s+(?:(?:me|her|him|them|us)\s+)?"
                     r"(?:it|this|that|them|the\s+(?:link|url|(?:web\s?)?address|web\s?page|website))\b", re.I)
# things people join besides the credit union
EVENT_WORDS = re.compile(r"\b(?:webinar|meeting|class|seminar|workshop|session|call|event|zoom|team|club|group|board|"
                         r"committee|program|challenge|contest)s?\b", re.I)
FACT_RARE = 0.01        # a question word on under 1% of chunks names the fact asked about


def joining_info(q, vocab=None, named=None) -> tuple[bool, set, set]:
    """(is the question about joining, the words asked, what else is asked beyond joining). Joining: an action
    ("apply for membership", "become / sign up as a member", "membership application"), or a bare "join" or
    "membership" when nothing else is asked. A member of a kind ("a board member") is not joining. "Picking an
    application back up" asks to continue it. With the site's link vocabulary, a word that at most one link
    on the site names ("text it to my sister") can't be what a link is asked for. How the link is passed on
    ("email it to her") is never what it is for. In a message of several sentences, the ones that don't ask
    for the link are context (see _request_words)."""
    site = {w.lower() for w in retrieve.SITE_WORDS}
    phrase, other_member = set(), False
    for m in JOIN_ACT.finditer(q.raw):
        phrase |= set(tokenize(m.group(0))) if FIRST_ACCOUNT.search(m.group(0)) else set(tokenize(m.group(0))) & JOIN_ACT_WORDS
        if m.group(1) and set(tokenize(m.group(1))) - FRAME_TALK - site:
            other_member = True
    subject = set(q.tokens) - LINK_WORDS - CHAT_FILLERS - set(q.expansion)
    sent = set(tokenize(" ".join(m.group(0) for m in SEND_IT.finditer(q.raw))))
    wanted = {w for w in _request_words(q.raw, subject, named) - sent - JOIN_WORDS - phrase - MEDIUM_WORDS - FRAME_TALK - site
              if not w[0].isdigit() and not MONTH_WORD.match(w)}
    if vocab is not None:
        wanted = {w for w in wanted if vocab.get(w, 0) >= 2}
    if RESUME.search(q.raw):
        wanted = (wanted - set(tokenize(" ".join(m.group(0) for m in RESUME.finditer(q.raw))))) | {_stem("continue")}
    joins_other = any(m.group(1).lower() not in site for m in JOIN_OBJECT.finditer(q.raw))
    joining = not other_member and (bool(phrase) or (JOIN_BARE.search(q.raw) is not None and not wanted
                                                     and not joins_other))
    return joining, subject, wanted


def _request_words(raw: str, subject: set, named=None) -> set:
    """The words of a joining request, without its context. In a message of several sentences, a sentence
    with neither the link word nor a joining word is context ("My cousin just moved here. Can you send me
    the link so she can join?") when the request is plainly about joining: a joining phrase ("apply for
    membership"), or the link sentence names who joins. The whole message counts when the request also
    asks to apply for something ("so he can join and apply"), the context names a line of products, or,
    for "so she can join", the context names something else one joins: an event, or a word that names
    one of the site's own pages and little else (`named`, see _title_names)."""
    sentences = [s for s in SENTENCE_END.split(raw.strip()) if s.strip()]
    asking = [s for s in sentences if LINK_Q.search(s) or JOIN_BARE.search(s) or JOIN_ACT.search(s)]
    context = " ".join(s for s in sentences if s not in asking)
    if not asking or not context.strip():
        return subject
    acted = any(JOIN_ACT.search(s) for s in asking)
    if not acted and not any(JOINER.search(s) for s in asking if LINK_Q.search(s)):
        return subject
    if (any(APPLY_Q.search(JOIN_ACT.sub(" ", s)) for s in asking)
            or re.search(rf"\b(?:{'|'.join(OTHER_LINES)})", context, re.I)):
        return subject
    if not acted and (EVENT_WORDS.search(context) or set(tokenize(context)) & (named or set())):
        return subject
    return subject & set(tokenize(" ".join(asking)))


def _title_names(index: Index) -> set:
    """Words that name one of the site's own pages and little else: in a live, non-article page's title, and
    on under 1% of the site's chunks (a program's own name). Built once per view."""
    names = getattr(index, "_title_names", None)
    if names is None:
        site = {w.lower() for w in retrieve.SITE_WORDS}
        titled = set()
        for vp in index.view.pages.values():
            if vp.status != "missing" and vp.page.kind != "article":
                titled |= set(tokenize(vp.page.title or "")) - site
        limit = FACT_RARE * max(1, len(index.chunks))
        names = {w for w in titled if len(index.postings.get(w, ())) < limit}
        index._title_names = names
    return names


def _anchor_vocab(index: Index) -> Counter:
    """How many link texts on the site (this view) contain each word. Built once per view."""
    vocab = getattr(index, "_anchor_vocab", None)
    if vocab is None:
        vocab = Counter()
        for vp in index.view.pages.values():
            if vp.status != "missing":
                for c in vp.page.chunks:
                    for m in LINK_RE.finditer(c.text):
                        vocab.update(set(tokenize(_clause(m.group(1)))))
        index._anchor_vocab = vocab
    return vocab


def _join_chunks(index: Index) -> list:
    """The chunks of the site's own pages about joining: live, non-article pages whose title (the site's name
    aside) is only joining words, such as "Join" or "Membership". Built once per view."""
    chunks = getattr(index, "_join_chunks", None)
    if chunks is None:
        site = {w.lower() for w in retrieve.SITE_WORDS}
        chunks = []
        for key, vp in sorted(index.view.pages.items()):
            title = set(tokenize(vp.page.title or "")) - site
            if vp.status != "missing" and vp.page.kind != "article" and title and title <= JOIN_WORDS:
                chunks += vp.page.chunks
        index._join_chunks = chunks
    return chunks


def _clause(text: str) -> str:
    """A link's own words: the last clause of the text before it."""
    return re.split(r"[.!?;:—–|]|\s{2,}|\\\\", text)[-1]


def pick_link(context: list[Chunk], view, q, carried=None, joins=(), vocab=None, named=None) -> str | None:
    """The site's own link for what was asked (see _pick_any). A question about joining the credit union also
    reads the site's pages about joining, and prefers, in order: a link whose own words name more of what else
    was asked ("Continue my application" for picking one back up), an application portal given by a page about
    joining, then the usual order. When something else was asked and no link names it, the usual pick stands."""
    joining, subject, wanted = joining_info(q, vocab, named)
    if not joining:
        return _pick_any(context, view, q, carried)
    join_keys = {c.key for c in joins}
    hosts = {k.split("/", 1)[0] for k in view.pages}
    site = {w.lower() for w in retrieve.SITE_WORDS}
    asked = set(q.tokens) | JOIN_WORDS | wanted
    pages: dict = {}
    portals, given_by_join_page, exact, named, joining_portals = set(), set(), set(), {}, set()
    for c in list(context) + list(joins):
        for m in LINK_RE.finditer(c.text):
            link, anchor = m.group(2).rstrip("/"), set(tokenize(m.group(1)))
            own = set(tokenize(_clause(m.group(1))))
            vp = view.pages.get(url_key(link))
            if vp is None:
                carrier = view.pages.get(c.key)
                if (link.split("/", 1)[0] in hosts or not _on_site(link, hosts) or carrier is None
                        or carrier.status == "missing" or carrier.page.kind == "article"
                        or _stem("join") not in set(tokenize(link)) or not own & (JOIN_WORDS | wanted)):
                    continue
                portals.add(link)
                if c.key in join_keys:
                    given_by_join_page.add(link)
                if own & JOIN_WORDS:
                    joining_portals.add(link)          # an application link that says it is for joining
            elif vp.status == "missing" or vp.page.kind == "article" or (vp.page.kind == "calculator"
                                                                         and not _asks_calculator(q)):
                continue
            if anchor & (subject | JOIN_WORDS | wanted):
                pages.setdefault(link, set()).add(c.key)
                if own - site and own - site <= asked:
                    exact.add(link)
                named[link] = max(named.get(link, 0), len(own & wanted))
    if not pages:
        return _pick_any(context, view, q, carried)
    def weight(u):
        return 0 if not carried else carried(f"({u})") if u in portals else carried(u)
    best = max(pages, key=lambda u: (named[u], u in given_by_join_page, u in joining_portals, len(pages[u]),
                                     u in exact, weight(u)))
    if wanted and named[best] == 0 and wanted & _link_clause_words(q.raw):
        return _pick_any(context, view, q, carried)        # something else asked, in the request itself
    if wanted and named[best] and best not in given_by_join_page and named[best] < len(wanted):
        return _pick_any(context, view, q, carried)
    return best


def _link_clause_words(text: str) -> set:
    """The words of the clause(s) that ask for the link ("what's the URL he can use to join online?")."""
    clauses = [c for c in CLAUSE_BREAK.split(text) if c and LINK_Q.search(c)]
    return set(tokenize(" ".join(clauses))) if clauses else set(tokenize(text))


def _asks_calculator(q) -> bool:
    return any(t.startswith("calcul") for t in q.tokens)


def _pick_any(context: list[Chunk], view, q, carried=None) -> str | None:
    """The link the site's own text gives for what was asked: the link whose anchor text shares a word with
    the question, given by the most pages of the evidence (then carried by the most pages of the site), whose
    page is live in this snapshot and is not an article (a "which page" question wants the product or tool
    page, not a blog post). Dead or retired links (pages the crawl never found, or that left it) can't be
    picked, and neither can a calculator unless one was asked for.

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
    words: dict = {}                              # link -> its own words, where the text gives it
    for c in context:
        for m in LINK_RE.finditer(c.text):
            link, anchor = m.group(2).rstrip("/"), set(tokenize(m.group(1)))
            words.setdefault(link.rstrip("/"), []).append(_last_clause(m.group(1)).lower())
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
            elif vp.status == "missing" or vp.page.kind == "article" or (vp.page.kind == "calculator"
                                                                         and not _asks_calculator(q)):
                continue
            if anchor & subject:
                pages.setdefault(link, set()).add(c.key)
    apps = _applications(context, view, q, subject, hosts)
    for link, keys in apps.items():
        pages.setdefault(link, set()).update(keys)
        portals.add(link)
    if not pages:
        return None
    def weight(u):                 # a portal counts where it is written as a link, not inside longer links
        return 0 if not carried else carried(f"({u})") if u in portals else carried(u)
    best = max(pages, key=lambda u: (u in apps, len(pages[u]), u in given_by_own_page, weight(u)))
    return _own_line_page(best, words.get(best, []), view, q) or best


APPLY_Q = re.compile(r"\b(?:apply(?:ing)?|applications?)\b", re.I)
_LINKISH = r"(?:link|url|web\s?page|website|address)"
# "what do I need to apply?" asks for the requirements; "I need the link to apply" / "which link do I need" asks for the link
APPLY_NEEDS = re.compile(
    rf"\b(?:what|which)\b(?:(?!\b{_LINKISH}\b)[^?.])*?\b(?:need(?:ed|s)?|require[sd]?)\b(?:(?!\b{_LINKISH}\b)[^?.])*?\bappl"
    r"|\b(?:documents?|paperwork|requirements?)\b[^?.]*?\bappl|\bappl\w*\b[^?.]*?\b(?:documents?|paperwork|requirements?)\b",
    re.I)
# vehicle types are vehicles (an auto loan covers them); used only to match application links
VEHICLE_KINDS = {_stem(w): {_stem("auto"), _stem("vehicle")} for w in ("minivan", "van", "suv", "truck", "pickup")}
LINE_WORDS = {_stem(w) for w in ("personal", "business")}     # lines of products an address may name anywhere
APPLY_LINK = re.compile(r"\b(?:apply|application|request|start)\w*", re.I)
NOT_APPLY_PRODUCT = set(tokenize("apply applying application applications submit check complete"))


def _applications(context: list[Chunk], view, q, subject: set, hosts: set) -> dict:
    """For "where do I apply for X": the site's own application link for X (link -> pages giving it). The
    link's words say apply / application / request / start, it is on one of the site's hosts the crawl never
    fetched, a live non-article page gives it, and the product asked is named by the link's own words or the
    last part of its address (a word every product shares, such as "loan", may be named anywhere in it).
    A question about what applying needs ("what documents do I need to apply…") wants the page, not this."""
    if not APPLY_Q.search(q.raw) or APPLY_NEEDS.search(q.raw):
        return {}
    site = {w.lower() for w in retrieve.SITE_WORDS}
    heads = {_stem(w) for w in GENERIC_HEADS}
    def syn(w):
        return {w} | set(retrieve._SYN.get(w, ())) | VEHICLE_KINDS.get(w, set())
    used, titled = _application_words(view, hosts)
    # a word no application link of the site uses ("my husband", "for the family") is not the product, unless a
    # product page is named with it (then the site has that product and no application of another one is it)
    product = {w for w in subject - FRAME_TALK - MEDIUM_WORDS - site - NOT_APPLY_PRODUCT
               if not w[0].isdigit() and (syn(w) & used or w in titled)}
    specific = product - heads
    if not specific:
        return {}
    apps: dict = {}
    # a line ("personal", "business") may be named anywhere in an address once another word picks the
    # product ("a personal van loan"); "personal loan", "personal line" name a product of their own
    lines = {w for w in LINE_WORDS if not re.search(rf"\b{w}\w*\s+(?:{'|'.join(GENERIC_HEADS)})\b", q.raw, re.I)}
    pinned = specific - lines
    def names_product(title):                 # the product's own page: its title names what was asked
        words = set(tokenize(title or ""))
        return all(syn(w) & words for w in (pinned or specific))
    own_pages = [c for _, vp in sorted(view.pages.items())
                 if vp.status != "missing" and vp.page.kind not in ("article", "calculator")
                 and names_product(vp.page.title) for c in getattr(vp.page, "chunks", ())]
    for c in list(context) + own_pages:
        carrier = view.pages.get(c.key)
        if carrier is None or carrier.status == "missing" or carrier.page.kind == "article":
            continue
        for m in LINK_RE.finditer(c.text):
            link, own = m.group(2).rstrip("/"), _last_clause(m.group(1))
            if (view.pages.get(url_key(link)) is not None or link.split("/", 1)[0] in hosts
                    or not _on_site(link, hosts) or not APPLY_LINK.search(own)):
                continue
            path = link.split("/", 1)[1] if "/" in link else ""
            last = set(tokenize(path.rsplit("/", 1)[-1])) | set(tokenize(own))
            whole = set(tokenize(path)) | set(tokenize(own))
            if (all((syn(w) - heads | {w}) & (whole if w in lines and pinned else last) for w in specific)
                    and all(syn(w) & whole for w in product - specific)):
                apps.setdefault(link, set()).add(c.key)
    return apps


def _application_words(view, hosts: set) -> tuple[set, set]:
    """The words the site's application links use (their addresses and own words), and the words of its
    product pages' titles. Built once per view."""
    cached = getattr(view, "_application_words", None)
    if cached is None:
        used, titled = set(), set()
        for key, vp in view.pages.items():
            if vp.status == "missing":
                continue
            if vp.page.kind == "product":
                titled |= set(tokenize(getattr(vp.page, "title", "") or ""))
            for c in getattr(vp.page, "chunks", ()):
                for m in LINK_RE.finditer(c.text):
                    link, own = m.group(2).rstrip("/"), _last_clause(m.group(1))
                    if (view.pages.get(url_key(link)) is None and link.split("/", 1)[0] not in hosts
                            and _on_site(link, hosts) and APPLY_LINK.search(own)):
                        used |= set(tokenize(link.split("/", 1)[1] if "/" in link else "")) | set(tokenize(own))
        cached = (used, titled)
        try:
            view._application_words = cached
        except AttributeError:
            pass
    return cached


OTHER_LINES = ("business", "jumbo")      # a line of products the question has to name to be asked about


def _last_clause(text: str) -> str:
    """A link's own words, ignoring the punctuation that ends them ("…a business auto loan for?")."""
    return _clause(text.rstrip(" .!?;:"))


def _own_line_page(link: str, own_words: list, view, q) -> str | None:
    """A link the site words for another line ("a business auto loan") when the question names no such line:
    the site's own live page whose title names everything asked, and not that line, is the answer instead
    (rows of a line the question didn't name come last, so do its links). None keeps the link."""
    asked = set(q.tokens)
    lines = [w for w in OTHER_LINES if _stem(w) not in asked
             and any(re.search(rf"\b{w}(?:es|s)?\b", t) for t in own_words)]
    if not lines:
        return None
    site = {w.lower() for w in retrieve.SITE_WORDS}
    product = set(q.tokens) - LINK_WORDS - CHAT_FILLERS - FRAME_TALK - site - set(q.expansion)
    groups = [{w} | set(retrieve._SYN.get(w, ())) for w in product if not w[0].isdigit()]
    if not groups:
        return None
    for key, vp in sorted(view.pages.items()):
        title = (vp.page.title or "").lower()
        named = set(tokenize(title)) - site
        if (vp.status != "missing" and vp.page.kind not in ("article", "calculator") and named
                and all(g & named for g in groups)
                and not any(re.search(rf"\b{w}", f"{title} {key}") for w in lines)):
            return key
    return None


def link_source(link: str, context: list[Chunk], view):
    """The page to cite for a picked link: the page itself, or for a portal the page that gives the link."""
    vp = view.pages.get(url_key(link))
    if vp is not None:
        return vp
    givers = sorted({c.key for c in context if f"({link})" in c.text or f"({link}/)" in c.text})
    if not givers:                            # a link found on a page outside the evidence: any live page giving it
        givers = sorted(k for k, vp in view.pages.items() if vp.status != "missing"
                        and any(f"({link})" in c.text for c in getattr(vp.page, "chunks", ())))
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
        if f["topic"].startswith("Boilerplate figure ("):
            if _asks_the_figure(q, f["topic"]):
                return f                               # "how much in assets does <the credit union> have?"
            continue
        # a count is the fact asked about only when the question counts that same thing
        if re.match(r"\s*\d", f.get("resolution", "")) and not (
                q.attribute == "count" and q.count_noun and q.count_noun in set(tokenize(f["topic"]))):
            continue
        rare = [t for t in set(q.tokens) & set(tokenize(f["topic"])) if len(index.postings.get(t, ())) < limit]
        if rare and (best is None or len(rare) > best[0]):
            best = (len(rare), f)
    return best[1] if best else None


def _asks_the_figure(q, topic: str) -> bool:
    """Is the question only about the thing a boilerplate figure counts ("assets", "membership"), plus the site's own name:
    no place, and no other word that would make it a different question ("members who got a payout")?"""
    label = topic.split("(", 1)[1].split(")", 1)[0]
    labels = {label} | ({_stem("member"), _stem("membership")} if label in (_stem("member"), _stem("membership")) else set())
    site = {_stem(w) for w in retrieve.SITE_WORDS}
    other = {t for t in q.tokens if t not in GENERIC and t not in STOP and t not in site and not t[0].isdigit()} - labels
    return bool(labels & set(q.tokens)) and not other and not FACILITY_ASK.search(q.raw)


def _fact_token(value: str) -> str:
    m = re.match(r"\s*(\$[\d,.]+(?:\s+(?:billion|million))?|[\d,]+|[a-z]+)", value.lower())   # "12 branches" -> 12, "$3.6 billion"
    return m.group(1) if m else ""


def _beaten(fact: dict) -> list:
    """Patterns for the values a repeated statement beat: a count with its noun ("9 offices", "9 office
    locations", never "9-month"), an ordinal, and the "Nth year in a row" claims it outranks."""
    pats = []
    for v in fact["_losers"]:
        m = re.match(r"\s*([\d,]+)\+?\s+([a-z]+)", v.lower())
        pats.append(rf"\b{re.escape(m.group(1))}\+?\s+(?:[\w’'-]+\s+){{0,2}}{re.escape(m.group(2)[:5])}" if m
                    else rf"(?<![\w$]){re.escape(v)}(?!\w)")
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
    if not win or re.search(rf"(?<![\w$]){re.escape(win)}(?!\w)", text, re.I):
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


PAGES_BUDGET = 16000      # characters of the picked pages the model reads on a second look


def _pages_evidence(index: Index, q, keys: list, budget: int | None = None) -> Evidence | None:
    """Evidence from the pages picked: the paragraphs that best match the question, the rest of the sections they
    sit in, each page's opening paragraphs, then whatever else the page says, in page order, to a size budget."""
    view, wanted = index.view, set(keys)
    live = lambda c: not (c.expired_on and c.expired_on < view.taken_on)
    best = [c for _, c in index.search(q.tokens, top=3000) if c.key in wanted and live(c)][:8]
    pages = [[c for c in view.pages[k].page.chunks if live(c)] for k in keys]
    sections = {(c.key, c.section) for c in best if c.section}
    near = [c for page in pages for c in page if (c.key, c.section) in sections]
    opening = [c for page in pages for c in page[:5]]
    rest = [c for page in pages for c in page]
    chunks, used = [], 0
    for c in {c.id: c for c in best + near + opening + rest}.values():
        if used + len(c.text) > (budget or PAGES_BUDGET) or len(chunks) >= 60:
            continue
        chunks.append(c)
        used += len(c.text)
    return Evidence(chunks, [], True, {}, chunks) if chunks else None


SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


SIZE_COMPARED = re.compile(r"\b(?:too\s+(?:big|small|large|little|tight)|bigger|larger|smaller|(?:next|another|different|other)\s+size|"
                          r"instead|rather\s+than|upgrade|downsize)\b", re.I)


def _resolve_size(clause: str, lead: str) -> str:
    """"What does that size cost?" after "a 5 by 10 safe deposit box": the size, and the thing it is the size of. Only when
    the lead-in gives one size and does not compare it ("too big", "the next size up"), and says what the thing is (a
    box); with none or two, a comparison or no box named, the clause is left as it is."""
    m = re.search(r"\b(?:that|this|the same)\s+(?:size|one)\b", clause, re.I)
    sizes = [(d.group(1).strip(), d.group(2).strip()) for d in DIMENSIONS.finditer(lead)]
    if not m or len({tuple(re.findall(r"\d+", x)) for x, _ in sizes}) != 1 or SIZE_COMPARED.search(lead):
        return clause
    size, noun = sizes[-1]
    noun = noun or ("box" if re.search(r"\bbox", lead, re.I) else "")
    return clause[:m.start()] + f"a {size} {noun}".strip() + clause[m.end():] if noun else clause


def _antecedent(clause: str, lead: str) -> list:
    """What "that grant" or "this program" in a clause refers to: the content words of the lead-in's sentence that names the
    same thing, to search with (never to require)."""
    out = []
    for m in re.finditer(r"\b(?:that|this|those|these|the same)\s+([a-z]+)", clause, re.I):
        noun = _stem(m.group(1).lower())
        for sentence in SENTENCE_END_RE.split(lead):
            toks = tokenize(sentence)
            if noun in toks:
                out += [t for t in toks if t not in STOP and t not in GENERIC and not t[0].isdigit()][:8]
    return list(dict.fromkeys(out))


def _retry_clause(index: Index, synth: Synthesizer, q, ev: Evidence, clause: str, lead: str, antecedent: list) -> tuple:
    """A lead-in before the question ("I'm writing a paper … Roughly how many …?"): its words may be what nothing answers.
    Returns (q, ev) of the clause asked alone when it answers, else the ones given. Used only when the whole question found
    nothing, and only if the lead-in restricts, places or asks nothing else ("not counting …", "I live in …" with "close to
    me"); a clause that refers back ("that grant", "it") needs a lead-in that names what it refers to and an answer on a
    page that carries the names; and what answers must share a word of the clause beyond the thing counted. A fee question
    reads its lead-in only to say what "that size" is the size of."""
    if q.attribute == "fee":
        resolved = _resolve_size(clause, lead)
        if resolved == clause:
            return q, ev
        clause = resolved
    names = lead_names(lead)
    back = BACK_REFERENCE.search(clause)
    if PLACE_DEICTIC.search(clause) or RESTRICTION.search(lead) or ASKS_ELSE.search(lead) or (back and not names):
        return q, ev
    q2 = parse_query(clause)
    q2.spelled_counts = synth.enabled
    if back:
        q2.tokens = list(dict.fromkeys(q2.tokens + sorted(n for n in names if not n[0].isdigit()) + antecedent))
    ev2 = select(index, q2)
    words = {n for n in names if not n[0].isdigit()}
    rest = set(q2.tokens) - {q2.count_noun} - QUIET - HEDGES - FILLERS if q2.attribute == "count" else set()
    ok = lambda c: retrieve.supports(c, q2) and (not rest or rest & set(c.tf or {}))
    carries = not back or (bool(words) and any(ok(c) and words <= set(tokenize(_page_texts(index).get(c.key, "")))
                                                 for c in ev2.chunks))
    if (q2.attribute == q.attribute and (q.attribute != "count" or q2.count_noun == q.count_noun) and carries
            and ((ev2.supported and any(ok(c) for c in ev2.chunks)) or (synth.enabled and carried_in_context(ev2, q2)))):
        return q2, ev2
    return q, ev


def _with_count_chunks(index: Index, q, ev: Evidence, tokens: list) -> list:
    """With a key, a count question that has a lead-in also shows the model the paragraphs that state a count of the thing
    asked about (a number in digits or in words), the five that share most with the question first."""
    pat = retrieve.count_re(q.count_noun, spelled=True)
    have = ev.context or ev.chunks
    seen = {c.id for c in have}
    wanted = set(tokens)
    found = []
    for pos, _ in index.postings.get(q.count_noun, []):
        c = index.chunks[pos]
        if c.id in seen or c.kind == "calculator" or (c.expired_on and c.expired_on < index.view.taken_on):
            continue
        if pat.search(c.text.lower()):
            found.append((-len(wanted & set(c.tf)), c.id, c))
    out, starts = [], set()
    for _, _, c in sorted(found, key=lambda x: x[:2]):
        if c.text[:80] not in starts:                  # the same paragraph on several pages once
            starts.add(c.text[:80])
            out.append(c)
        if len(out) == 5:
            break
    return have + out


def _places_asked(index: Index, raw: str) -> tuple | None:
    """(any named place is on a page, the places as sets of stems) for a branch, ATM or lobby asked of a town written next
    to the facility ("a branch in davenport"). A lower-case word that is on the site ("the same lobby hours") is no town;
    None when the question names none."""
    places = []
    for word, capitalised in facility_places(raw):
        stems = set(tokenize(word))                           # tokens are the index's own stems
        if capitalised or not any(index.postings.get(t) for t in stems):
            places.append(stems)
    if not places:
        return None
    return any(index.postings.get(t) for p in places for t in p), places


DATELINE = re.compile(r"(?:^|\n)\s*([A-Z][A-Z .'’-]{2,},\s*[A-Za-z.]+)\s*[–—-]")


DATELINE_MID = re.compile(r"(?:^|\n|\)\s)([A-Z][A-Z .'’-]{2,},\s*[A-Za-z.]+)\s*[–—-]")
FACILITY_WORD = re.compile(r"\b(?:branch|office|learning lab|lab|location|ATM|lobby|drive-?up)\b", re.I)
FACILITY_FACT = re.compile(r"\b(?:located|address|phone number|hours)\b", re.I)


def _facility_dateline(text: str):
    """A dateline that follows a byline on the same line ("... (link) MAPLE GROVE, Ohio–Finishing touches ..."), accepted only for a
    page about a facility: a facility word within 500 characters after it and an address, phone number or hours on the page. A
    member's story datelined from their hometown is not a page about a place the credit union is in."""
    m = DATELINE_MID.search(text[:1500])
    if m and FACILITY_WORD.search(text[m.end():m.end() + 500]) and FACILITY_FACT.search(text[:3000]):
        return m
    return None


def _page_subjects(index: Index) -> dict:
    """Each page's subject words: its title and its dateline ("MAPLE GROVE, Ohio-"). A page that only mentions a place in
    its text is not about it. Built once per view."""
    subjects = getattr(index, "_subjects", None)
    if subjects is None:
        texts = _page_texts(index)
        subjects = index._subjects = {}
        for key, vp in index.view.pages.items():
            if vp.status != "missing":
                text = texts.get(key, "")
                dateline = DATELINE.search(text[:1500]) or _facility_dateline(text)
                subjects[key] = set(tokenize(f"{vp.page.title} {dateline.group(1) if dateline else ''}"))
    return subjects


def _pages_about(index: Index, q, places: list, n: int = 2) -> list:
    """The pages named for a place in their title or opening (a press release about one branch: its dateline), those that
    share most with the question first. Press releases are not in the site's page list, so a question about one place's
    staff or phone number would never reach them."""
    if not places:
        return []
    want, subjects = set(q.tokens), _page_subjects(index)
    keys = [k for k, words in subjects.items() if index.view.pages[k].status == "fresh" and any(p <= words for p in places)]
    return sorted(keys, key=lambda k: (-len(want & subjects[k]), k))[:n]


PHONE_ADDRESS_ASK = {_stem(w) for w in ("phone", "number", "address", "call", "street", "located", "where")}
PHONE_ADDRESS_Q = re.compile(r"\b(?:phone|number|address|call|street|located|where)\b", re.I)


def _with_place_chunks(index: Index, q, ev: Evidence, places: list) -> list:
    """A question about one branch ("the phone number and address of the Grimes branch") shows the model the paragraphs of the
    pages about that branch that share words with it, besides what the search ranked first: the phone number and the address
    are often two paragraphs of one page."""
    have = ev.context or ev.chunks
    if q.attribute != "address" and not PHONE_ADDRESS_Q.search(q.raw):
        return have                                 # whether a branch is still open or what became of it: the page's own answer
    seen = {c.id for c in have}
    want = set(q.tokens) | PHONE_ADDRESS_ASK
    extra = []
    for key in _pages_about(index, q, places, 2):
        cands = [(len(want & set(c.tf or ())), c) for c in index.view.pages[key].page.chunks if c.id not in seen and not c.is_row]
        extra += [c for n, c in sorted(cands, key=lambda x: (-x[0], x[1].id)) if n][:3]
    return have + extra


def _is_a_branch(index: Index, places: list) -> bool:
    """Does a page that is not an article give one of the places an address? Then it is a branch or an ATM the site lists;
    a town that only news and blog posts mention is not."""
    for p in places:
        first = next(iter(p))
        for pos, _ in index.postings.get(first, ()):
            c = index.chunks[pos]
            if c.kind != "article" and p <= set(c.tf or ()) and ADDRESS_EV.search(c.text):
                return True
    return False


def _place_has_page(index: Index, places: list) -> bool:
    """Is some page of the site about one of the places (named in its title or its opening)?"""
    return any(p <= words for words in _page_subjects(index).values() for p in places)


def _page_is_about(index: Index, c: Chunk, places: list) -> bool:
    """Does the page a chunk is on name one of the places in its title or its opening (a heading, a dateline)? A page that
    only mentions a place in passing ("shared branches in Arizona") is not about it."""
    opening = _page_subjects(index).get(c.key, set())
    return any(p <= opening for p in places)


def _model_year_rows(index: Index, q) -> list:
    """A year written beside a make or model ("a 2022 <make> <model>") can be the model year of any line that prices by
    model year. Every such row of a live product page that the year and the asked term fit, the most authoritative of
    the rows that name one tier of one product, so the model can pick the line the vehicle belongs to. Never a
    calculator, promotion or article row, nor a page that was not re-fetched."""
    view, best = index.view, {}
    for c in index.chunks:
        if not c.is_row or c.expired_on and c.expired_on < view.taken_on or not retrieve._pct_values(c):
            continue
        vp = view.pages[c.key]
        if vp.page.kind != "product" or vp.status != "fresh":
            continue
        if not any(lo <= y <= hi for y in q.years for lo, hi in retrieve.model_year_ranges(c.index_text)):
            continue
        if qualify(c, q)[0] == "mismatch":
            continue
        k = (retrieve.product_name(c).lower()[:14], re.sub(r"\W+", " ", c.label().lower()))
        if k not in best or retrieve.authority(c, view) < retrieve.authority(best[k], view):
            best[k] = c
    return list(best.values())[:12]


LIST_PAGES, LIST_BUDGET = 2, 8000


OUTLINE_SKIP = re.compile(r"\b(?:frequently|faq|handpicked|fine print|contact|questions|team|carriers?|disclosures?|related|resources?|about|"
                          r"footer|menu|calculators?|support|help|rates?)\b", re.I)
OUTLINE_GENERIC = {_stem(w) for w in ("kind kinds type types list every all what which can offer offers have does do you your u guy guys "
                                      "will get need maybe later run provide through for apply open individuals individual personal "
                                      "rate rates apr would like know please sponsor give out sell are the").split()}


def _page_outlines(index: Index) -> dict:
    """{page key: (title, [section headings that name something], lead text)} for live product pages: a landing page's tiles
    ("Kayak Loan", "Canoe Loan", ...) are its sections. Built once per view."""
    found = getattr(index, "_outlines", None)
    if found is None:
        found = index._outlines = {}
        for key, vp in index.view.pages.items():
            if vp.status != "fresh" or vp.page.kind != "product":
                continue
            secs = list(dict.fromkeys((c.section or "").split(" · ")[0] for c in vp.page.chunks if c.section))
            secs = [x for x in secs if x and len(x) < 45 and not OUTLINE_SKIP.search(x)]
            lead = next((c.text for c in vp.page.chunks if not c.is_row and len(c.text) > 60), "")
            found[key] = (vp.page.title.split("|")[0].strip(), secs, lead)
    return found


def _outline_chunks(index: Index, q) -> list:
    """For a list question, one chunk per best page that lists what the page offers: its section headings (a landing page's
    product tiles), and for a page named for the thing asked ("Recreational Loans") its opening sentence. Pages rank by the
    question's salient words in the title and headings; business pages only when the question says business, personal ones
    when it says personal or individuals."""
    raw = q.raw.lower()
    business = re.search(r"\bbusiness", raw) is not None
    personal = re.search(r"\b(?:personal|individuals?|my|me)\b", raw) is not None
    want = {t for t in q.tokens if t not in OUTLINE_GENERIC}
    if re.search(r"\bindividuals?\b", raw):
        want.add(_stem("personal"))                  # "loans for individuals": the pages named personal
    scored, outlines = [], _page_outlines(index)
    df = Counter(t for title, secs, _ in outlines.values() for t in want & set(tokenize(" ".join([title] + secs))))
    weight = {t: math.log(1 + len(outlines) / df[t]) for t in df}        # a word few pages use says more than "auto" or "loan"
    rare = min(df, key=df.get) if df and min(df.values()) <= 5 else None
    for key, (title, secs, lead) in outlines.items():
        in_title = want & set(tokenize(title))
        sc = (3 * sum(weight.get(t, 0) for t in in_title)
              + sum(weight.get(t, 0) for t in {t for x in secs for t in want & set(tokenize(x))}))
        if rare and rare in in_title:
            sc += 10                                                      # a page named for the rarest word asked
        if sc > 0:
            sc += min(12, 3 * sum(k.startswith(key + "/") for k in outlines))   # a landing page: its child pages are the lines it offers
            sc += 0.1 * min(10, sum(len(x.split()) <= 3 for x in secs))   # tiles are short names
        if sc > 0 and personal and not business and ("/personal" in key or "personal" in title.lower()):
            sc += 4
        if "/business" in key and not business and personal:
            sc *= 0.4
        if "/personal" in key and business:
            sc *= 0.4
        if sc > 0:
            scored.append((sc, key, title, secs, lead))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for sc, key, title, secs, lead in scored[:2]:
        named = bool(want & set(tokenize(title)))
        text = (f"{title} lists: " + "; ".join(secs) + "." if len(secs) >= 3 else "") + (f" {title}: {lead[:260]}" if lead and named else "")
        if not text.strip():
            continue
        text = text.strip()
        out.append(Chunk(-1, key, "product", text, text))
    return out


def _with_listed_pages(index: Index, q, ev: Evidence) -> list:
    """A list question ("what protection products can I add?") is answered from every line the best pages give, not
    the few paragraphs that ranked highest: the two pages whose paragraphs match the question most (a business page
    first when the question says business; never a blog post or a calculator) are read to a budget, after the evidence
    already chosen; then the titles of the pages under the best page's section ("business/loans/…": the kinds of
    financing), the rest of the table the best row sits in when a rate is asked at every term, and every rate row of the
    kind of product asked about."""
    view = index.view
    business = re.search(r"\bbusiness(?:es)?\b", q.raw, re.I) is not None
    mass: dict = {}
    for score, c in index.search(q.tokens, top=400):
        vp = view.pages[c.key]
        if vp.page.kind == "product" and vp.status == "fresh":
            mass[c.key] = mass.get(c.key, 0.0) + score * (3 if business and "business" in c.key else 1)
    keys = sorted(mass, key=lambda k: -mass[k])[:LIST_PAGES]
    more = _pages_evidence(index, q, keys, LIST_BUDGET) if keys else None
    have = ev.context or ev.chunks
    seen = {c.id for c in have}
    out = have + [c for c in (more.chunks if more else []) if c.id not in seen]
    if keys:
        titles = [view.pages[k].page.title.split("|")[0].strip() for k in sorted(view.pages)
                  if k.startswith(keys[0] + "/") and view.pages[k].page.kind == "product" and view.pages[k].status == "fresh"
                  and view.pages[k].page.title][:14]
        if titles:
            text = "Pages under this one: " + "; ".join(dict.fromkeys(titles))
            out.append(Chunk(-1, keys[0], "product", text, text))
    if q.attribute is None:
        out = _outline_chunks(index, q) + out         # the outline first: it names every line the best pages offer
    top = next((c for c in ev.chunks if c.is_row and retrieve.supports(c, q)), None)
    if top is not None and retrieve.LIST_EACH.search(q.raw):
        # "what APR at each term?": every row of the table the best row sits in that the question's other words fit
        ids = {c.id for c in out}
        out += [c for c in view.pages[top.key].page.chunks if c.is_row and c.table_span == top.table_span and c.id not in ids
                and qualify(c, q)[0] != "mismatch"][:24]
    if q.attribute == "rate" and q.category:
        ids = {c.id for c in out}
        kind = {_stem(w.lower()) for w in q.category if w not in GENERIC_HEADS}
        rows, keyed = [], set()
        for c in index.chunks:
            if (c.is_row and c.id not in ids and retrieve._pct_values(c) and kind and kind <= set(c.tf)
                    and view.pages[c.key].page.kind == "product" and view.pages[c.key].status == "fresh"
                    and not (c.expired_on and c.expired_on < view.taken_on) and qualify(c, q)[0] != "mismatch"):
                if (rk := retrieve._row_key(c)) not in keyed:
                    keyed.add(rk)
                    rows.append(c)
        out += rows[:24]
    return out


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
