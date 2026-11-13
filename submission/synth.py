"""Turn selected evidence into the member-facing answer.

Two paths, same evidence:
  * LLM (Claude) when ANTHROPIC_API_KEY is set and KH_NO_LLM is not: writes the answer from the
    numbered evidence only, then deterministic guards check it.
  * Extractive: the best supporting table row or sentence, verbatim. Used without a key, and as the
    fallback whenever the LLM call fails, times out, or breaks a guard.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from datetime import date

from ingest import Chunk, View, url_key
from collections import Counter
from types import SimpleNamespace

from retrieve import (ASKS_THRESHOLD, KIND_NAME, SOFT_QUALIFIERS, SPELLED, SPELLED_SMALL, SYNONYMS, WIRE, WIRE_FEE_LINE, Evidence, Query, _asked,
                      asked_kinds, chunk_cover, count_sentence, fee_cover, fee_items, kinds_in, names_line, on_other_line, other_kind, other_lines,
                      place_phrases, product_name, qualify, spelled_numbers, supports, terms_in, tier_width, tokenize, wire_direction,
                      STATUS_DATE, STATUS_EV)

REFUSAL = "That isn't listed on our website, so I can't confirm it. Please contact us and we can help."
# The prompt calls the context "excerpts"; a sentence that talks about them instead of the site is dropped.
BANNED = re.compile(r"\bevidence\b|\bexcerpts?\b|\bdocuments?\b", re.I)
NUM_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
MODEL = os.getenv("KH_MODEL", "claude-haiku-5-5")
TIMEOUT_S = float(os.getenv("KH_LLM_TIMEOUT", "8"))
RETRY_WAIT_MAX_S = 10.0     # a longer server-requested wait is not waited out: each answer must end within 60s


def _new_client(**kwargs):
    """The SDK client, with one retry. A rate-limited reply can ask for a long wait (retry-after) and the
    SDK sleeps all of it, which would push an answer past the 60s limit. Waits over RETRY_WAIT_MAX_S are
    skipped instead; the retry then fails fast and the extractive answer is used. Worst case per answer
    is about 2 x TIMEOUT_S + RETRY_WAIT_MAX_S."""
    import anthropic

    class _Client(anthropic.Anthropic):
        def _calculate_retry_timeout(self, *args, **kw):
            wait = super()._calculate_retry_timeout(*args, **kw)
            return wait if wait <= RETRY_WAIT_MAX_S else 0.0

    return _Client(timeout=TIMEOUT_S, max_retries=1, **kwargs)

SYSTEM = """You answer a credit union member's question using ONLY the numbered website excerpts provided.

Rules:
1. Answer whenever an excerpt states the fact. A range, an "as low as" value, a value under a different label, and a general answer to a general question all count. If excerpts disagree, choose by rule 3; do not decline. Set answerable=false only if no excerpt states it for the specific branch, product, term or tier the question names. A conclusion that follows directly from a stated fact and the snapshot date is answerable.
2. Copy numbers, dates, times, names, phone numbers and URLs exactly as written in the excerpts (do not reformat times, dates or spelled-out ordinals).
3. Excerpts are ordered most-trustworthy first. If they disagree, prefer: a newer "effective" date, then a newer "published" date, then a live page over one "not re-fetched" or "removed", then rate/product pages over calculators, promotions, blogs and press releases. Ignore incidental mentions inside bios or anecdotes. Never mention a value you rejected; explain it in "note" instead.
4. Tiers: if the question names a term, balance, loan amount or model year, answer for exactly that tier. Otherwise give the standard (not jumbo), personal (not business) product's main tier, plus the range across that same product's tiers, unless the question asks what is offered: then list every product line shown, including business ones. Never mix in other products' values.
5. For deposit accounts give both the dividend rate and the APY from the same row; for loans give the rate and the APR when both are listed.
6. If the site only has an older figure, give it with its year. Answer yes/no questions with the facts (which product, what value). For list questions, include every item the excerpts list.
7. One to three sentences, in one turn: never ask the member a follow-up question. Do not mention "excerpts", "evidence" or "documents". Do not add dates, years or page details unless the question asks for them or rule 6 applies.

Return JSON, in this order: note (at most 15 words, decided first: which value wins and why, or ""), answer (text for the member, consistent with note), answerable (true when the excerpts state the answer, however you feel about it), evidence_ids (the excerpt numbers you used)."""

# Field order is generation order: the reasoning ("note") is written before the answer, so the answer
# follows the decision instead of being rationalised after it.
SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "string"},
        "answer": {"type": "string"},
        "answerable": {"type": "boolean"},
        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["note", "answer", "answerable", "evidence_ids"],
    "additionalProperties": False,
}


# The second look: when the search results hold no answer to an open question, the model picks pages from
# the site's own page list (derived from the crawl at ingest), and is asked once more from those pages.
ROUTE_SYSTEM = """You route a credit union member's question to pages of the credit union's own website. The site's pages are listed below, one per line as "path | title". Return the paths of up to 3 pages most likely to answer the question, best first, copied exactly as listed, or an empty list if none would."""
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
    "required": ["paths"],
    "additionalProperties": False,
}


def numbers(text: str) -> set[float]:
    return {float(m.group(1).replace(",", "") + ("." + m.group(2) if m.group(2) else ""))
            for m in NUM_RE.finditer(text or "")}


def describe(c: Chunk, view: View) -> str:
    vp = view.pages[c.key]
    meta = [vp.page.url, vp.page.kind]
    if vp.page.publish:
        meta.append(f"published {vp.page.publish.isoformat()}")
    if c.effective:
        meta.append(f"effective {c.effective.isoformat()}")
    if vp.status == "carried":
        meta.append(f"not re-fetched since {vp.fetched.isoformat()}")
    elif vp.status == "missing":
        meta.append("removed from the site")
    return " | ".join(meta)


def render_row(c: Chunk) -> str:
    vals = ", ".join(v if _short_header(h).lower() in v.lower() else f"{_short_header(h)} {v}".strip()
                     for h, v in c.values())
    ctx = c.section.split(" · ")[-1] if c.section else ""
    label = c.label().replace(" | ", ", ")
    lead = f"{ctx} — {label}" if ctx and label and ctx.lower() not in label.lower() else (label or ctx)
    return f"{lead}: {vals}" if vals else c.text


def _short_header(h: str) -> str:
    h = re.sub(r"\(as low as\)|\*", "", h, flags=re.I).strip()
    if re.search(r"annual percentage yield|\bapy\b", h, re.I):
        return "APY"
    if re.search(r"annual percentage rate|\bapr\b", h, re.I):
        return "APR"
    return h


PARAGRAPH_CHARS = 500


# A schedule is a block per facility ("Lobby", "Drive-up") of weekday lines ("Monday-Thursday: 6:45 a.m.-3:15 p.m."). Quoted
# whole, or the facility and the weekdays asked about.
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAYNAME = r"(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day"
_DAYSPEC = re.compile(rf"^\s*({_DAYNAME})(?:\s*[-–—]\s*({_DAYNAME}))?\s*:", re.I)
_CMP_DAYS = re.compile(r"\b(?:later|earlier|longer|shorter|than|vs\.?|versus|compared?|same|different|differ)\b", re.I)


def _schedule_blocks(text: str) -> list:
    """[(facility label or None, [weekday lines])]; the label may sit on its own line or lead the first weekday line."""
    blocks, label, lines = [], None, []
    for line in text.split("\n"):
        m = re.match(rf"\s*([A-Za-z][A-Za-z -]{{1,20}}?)\s*(?=(?:{_DAYNAME})\b|$)", line)
        head = m.group(1).strip() if m and not re.search(r"\d", m.group(1)) and len(m.group(1).split()) <= 2 else None
        if head and not re.match(_DAYNAME, head, re.I):
            if lines:
                blocks.append((label, lines))
            label, lines, line = head, [], line[m.end():].strip()
        if _DAYSPEC.match(line):
            lines.append(line.strip())
    if lines:
        blocks.append((label, lines))
    return blocks


def hours_part(text: str, question: str, by_day: bool = True) -> str:
    """The lobby or the drive-up block when one is asked about; with a place named, only the weekdays asked about
    ("weekdays", "weekends", one day); a comparison word ("later", "vs", "than") keeps the whole block."""
    blocks = _schedule_blocks(text)
    if not blocks:
        return text.replace("\n", "; ")[:400]
    low = question.lower()
    lobby, drive = bool(re.search(r"\blobb(?:y|ies)\b", low)), bool(re.search(r"\bdrive[- ]?(?:up|thru|through)", low))
    want = [b for b in blocks if b[0] and ((lobby and not drive and b[0].lower().startswith("lobby"))
                                           or (drive and not lobby and "drive" in b[0].lower()))]
    days = {i for i, d in enumerate(_DAYS) if re.search(rf"\b{d}s?\b", low)}
    days |= set(range(5)) if re.search(r"\bweekdays?\b", low) else set()
    days |= {5, 6} if re.search(r"\bweekends?\b", low) else set()
    if not by_day or _CMP_DAYS.search(low):
        days = set()
    out = []
    for lab, ls in want or blocks:
        keep = ls
        if days:
            def covers(l):
                m = _DAYSPEC.match(l)
                a = _DAYS.index(m.group(1).lower()); b = _DAYS.index(m.group(2).lower()) if m.group(2) else a
                return any(a <= d <= b for d in days)
            keep = [l for l in ls if covers(l)] or ls
        out.append(f"{lab + ' ' if lab else ''}{'; '.join(keep)}")
    return "; ".join(out)[:600]


def compare_hours(q: Query, lead: bool = True):
    """Each distinct schedule among the q.schedules, with its branch's town (at most six), compared by clock times
    ("dash and spacing styles don't matter"). With `lead`, "No, hours differ by branch." or "Yes."."""
    seen, parts = {}, []
    asked = q.raw if (not lead or re.search(r"lobb|drive", q.raw, re.I)) else ""
    for c in q.schedules:
        part = hours_part(c.text, asked)
        sig = tuple(re.sub(r"\W", "", t.lower()) for t in re.findall(r"\d{1,2}(?::\d{2})?\s*[ap]\.?\s?m\b|noon|closed", part, re.I))
        if sig and sig not in seen:
            seen[sig] = c
            parts.append((c.branch, part, c))
    if not parts:
        return None
    if len(parts) == 1:
        return (("Yes. " + parts[0][1] + ".") if lead else parts[0][1]), [parts[0][2]]
    return ("No, hours differ by branch. " if lead else "") + " ".join(f"{name + ': ' if name else ''}{part}." for name, part, _ in parts[:6]), \
        [c for _, _, c in parts[:6]]


# "Is the <place> branch still open?", "has the offer ended?": the passage that says so with a date
STATUS_Q = re.compile(r"\bstill\s+(?:open|operating|operational|in\s+operation|in\s+business|running|around|available|offered|"
                      r"active|going|valid)\b|\b(?:has|have|did|is|are|was)\b[^?]*\b(?:close[sd]?|closing|shut\s+down|shutting\s+down|"
                      r"ended|discontinued|expired)\b", re.I)
FORM_LABELS = re.compile(r"[A-Za-z?]\*(?=\s|$)")


def best_sentence(c: Chunk, q: Query) -> str:
    if q.attribute == "hours":
        return hours_part(c.text, q.raw, bool(q.places))
    if q.attribute is None:                             # how-to and "what is…": the steps or the description
        return c.text.replace("\n", " ")[:PARAGRAPH_CHARS]   # span the paragraph, not one sentence of it
    if q.attribute == "count" and q.count_noun:         # the clause that counts the asked thing
        sent = count_sentence(c, q)[0]
        if sent:
            return sent[:400]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])|\n", c.text) if s.strip()]
    if q.attribute == "fee":
        # a sentence holding several fees is quoted as one of them; a sentence with an amount is preferred, a
        # threshold ("less than $5") only when the question asks one
        allow = bool(ASKS_THRESHOLD.search(q.raw))
        items = lambda s: [it for _, it, threshold in fee_items(s) if not threshold or allow]
        split = [(p, True) for s in sents if len(fee_items(s)) > 1 for p in items(s)]
        whole = [(s, bool(items(s))) for s in sents if len(fee_items(s)) <= 1]
        sents = [p for p, ok in split + whole if ok] or sents
        if q.focus and q.focus[0] and sents:
            # the item that names what was asked ("foreign transaction fee on my debit card" is not "Rush fee for
            # debit card"); a free one only when no priced one names it as much
            cov = lambda s: fee_cover(tokenize(s), q.focus, bool(re.search(r"[$%]", s)))
            best = max(sents, key=cov)
            if cov(best)[1]:
                return best[:400]
    qt = set(q.tokens)

    def score(s):
        status, _ = qualify(_as_chunk(s, c), q)
        if status == "mismatch":
            return -1
        return (len(qt & set(tokenize(s))) + (3 if status == "match" else 0)
                + (2 if q.attribute and supports(_as_chunk(s, c), q) else 0))
    if not sents:
        return c.text[:400]
    best = max(sents, key=score)
    return best[:400] if score(best) >= 0 else REFUSAL


def _as_chunk(text: str, like: Chunk) -> Chunk:
    return Chunk(-1, like.key, like.kind, text, f"{like.index_text} {text}", section=like.section)


WIRE_KIND_NAMED = {"incoming": r"\b(?:incoming|inbound|receiv\w*)\b", "domestic": r"\bdomestic\w*",
                   "international": r"\b(?:international\w*|abroad|overseas|foreign)\b"}


def extractive(q: Query, ev: Evidence, view: View) -> tuple[str, list[Chunk]]:
    pool = [c for c in ev.chunks if supports(c, q)] if q.attribute else ev.chunks
    if _leads_with_other_kind(q, ev):
        # "what's the kayak savings rate?" led by a certificate table under a savings heading: the asked line answers
        own = asked_line_row(q, ev, view)
        rest = [c for c in pool if c is not own and not (c.is_row and other_kind(c, q))]
        pool = ([own] if own else []) + rest      # never the other kind's row (see compose)
    if q.attribute == "fee":                   # a promotion's offer ("we'll give you $X") is never a fee
        pool = [c for c in pool if c.kind != "promo"] or pool
    if not pool:
        return REFUSAL, []
    if q.attribute == "fee" and q.focus and len(q.focus[0]) >= 2 and q.focus[2] != "both" and not WIRE.search(q.raw) \
            and chunk_cover(pool[0], q, q.focus)[1] * 2 < len(q.focus[0]):
        return REFUSAL, []           # the thing asked isn't priced: "a box of checks" is not the drill-box fee
    if q.attribute == "count" and q.no_year:            # a year asked that the site counts nothing for
        return REFUSAL, []
    if q.attribute == "count" and q.series:             # one dated sentence per year (see select)
        return " ".join(s if s.endswith((".", "!", "?")) else s + "." for _, s, _ in q.series), [c for _, _, c in q.series]
    if q.attribute == "hours" and q.places and len(q.schedules) > 1:
        several = compare_hours(q, lead=False)          # the place names more than one branch: each distinct schedule
        if several:
            return several
    if q.attribute == "hours" and q.compare:
        both = compare_hours(q)
        if both:
            return both
    if q.attribute is None and STATUS_Q.search(q.raw):
        # the passage that states a closing, ending or opening with a full date and names the place asked about
        asked = place_phrases(q.raw)
        dated = [c for c in pool if STATUS_DATE.search(c.text) and STATUS_EV.search(c.text)
                 and (not asked or any(p & set(tokenize(c.index_text)) for p in asked))]
        pool = dated + [c for c in pool if c not in dated]
    if q.attribute is None:                             # a form's field labels answer nothing
        forms = [c for c in pool if not c.is_row and len(FORM_LABELS.findall(c.text)) >= 2]
        pool = [c for c in pool if c not in forms] + forms
    if q.attribute == "fee" and WIRE.search(q.raw) and wire_direction(q.raw) == "all":
        # several kinds of wire asked about ("domestic and international", "your wire fees"): one fee line of each
        # kind, from the page that lists the first one
        first = next((c for c in pool if WIRE_FEE_LINE.search(c.text)), None)
        kinds = {}
        for c in ([first] + [d for d in view.pages[first.key].page.chunks if d is not first]) if first else []:
            kind = next((k for k in ("incoming", "domestic", "international") if k in c.text.lower()), None)
            if kind and kind not in kinds and WIRE_FEE_LINE.search(c.text) and supports(c, q):
                kinds[kind] = c
        # two kinds named ("the incoming wire versus the outgoing domestic one"): just those two
        named = {k for k, pat in WIRE_KIND_NAMED.items() if re.search(pat, q.raw, re.I)}
        if len(named) > 1:
            kinds = {k: c for k, c in kinds.items() if k in named}
        if len(kinds) > 1:
            lines = list(kinds.values())
            return " ".join(render_row(c) + "." if c.is_row else best_sentence(c, q) for c in lines), lines
    top = pool[0]
    one_amount = top.is_row and len(set(q.amounts)) == 1 and tier_width(top, q) is not None
    if one_amount:
        # an amount inside several nested tiers ("$100,000 or less", "$300,000 or less") asks for the tightest
        siblings = [c for c in view.pages[top.key].page.chunks if c.is_row and c.table_span == top.table_span
                    and supports(c, q) and qualify(c, q)[0] != "mismatch" and tier_width(c, q) is not None]
        top = min([top] + siblings, key=lambda c: tier_width(c, q))
    if top.is_row:
        # Tiers the question did not pin down (e.g. the loan amount for a "25-year home loan") are shown as a
        # range: every row of the same table that matches all qualifiers the question did give.
        hidden, lines = {d.id for d in ev.dropped}, other_lines(q, top)
        same_table = [c for c in view.pages[top.key].page.chunks if c.is_row and c is not top
                      and c.table_span == top.table_span and c.id not in hidden and supports(c, q)
                      and qualify(c, q)[0] != "mismatch" and not on_other_line(c, lines)
                      and not any(re.search(rf"\b{w}\b", f"{c.label()} {c.section}".lower()) for w in q.excluded)
                      and not (one_amount and tier_width(c, q) != tier_width(top, q))]
        if q.attribute == "fee":
            # a fee table's other rows are other fees, unless they name the same service (safe deposit box sizes)
            lead = lambda c: [w for w in tokenize(c.label()) if not w[0].isdigit()]
            same_table = [c for c in same_table if lead(c) == lead(top)]
        rows = [top] + same_table[:5 if q.attribute == "fee" else 3]
        if q.category and not _names(top, q.category):
            # The top row matched a value shared with a different product than the one the question
            # names ("savings"): also give the named product's own row from the evidence, so a value
            # coincidence can't crowd out the product asked about.
            own = named_row(q, ev, rows, view)
            if own:
                rows = rows + [own]
        return " ".join(render_row(r) + "." for r in rows), rows
    return best_sentence(top, q), [top]


def _mentions(text: str, words) -> bool:
    """Any of the words, singular or plural ("loan" names "Auto Loans")."""
    return any(re.search(rf"\b{re.escape(w[:-1] if w.endswith('s') and not w.endswith('ss') else w)}s?\b", text)
               for w in words)


def _line_words(q: Query) -> set:
    """"Account" is every product's head word, so it names the line only when nothing more specific does. The
    site's word for the line counts too (a CD is a certificate)."""
    words = (q.category - {"account", "accounts"}) or q.category
    return words | {s for w in words if w.rstrip("s") == "cd" for s in SYNONYMS.get("cd", ())}


def _names(c: Chunk, words) -> bool:
    return _mentions(f"{c.label()} {c.section}".lower(), words)


def _line_rows(q: Query, ev: Evidence, exclude=()) -> list[Chunk]:
    """Evidence rows of the product line the question names that also fit what it asked (descriptor,
    term, tier): never another product's row."""
    words = _line_words(q)
    return [c for c in ev.chunks if c.is_row and c not in exclude and _names(c, words)
            and supports(c, q) and qualify(c, q)[0] != "mismatch"]


# The kind of product a row's name says it is, and the words that say nothing beyond its kind ("Basic Savings
# Account" is the plain savings account; "Kayak Savings" is a named one).
PLAIN_NAME = re.compile(r"\b(?:share|accounts?|rates?|regular|basic|standard|personal)\b")
# lines that are a different kind of account, not a tier of the same one (jumbo, refinance)
OTHER_LINE = ("health", "hsa", "ira", "traditional", "roth", "business")


_kinds, _product_name, _line_kinds = kinds_in, product_name, asked_kinds      # see retrieve


def _foreign(c: Chunk, q: Query, lines=SOFT_QUALIFIERS) -> bool:
    """A row of a line the question didn't ask about, or of another kind of product than the one it names."""
    low = f"{c.label()} {c.section}".lower()
    if any(re.search(rf"\b{w}\b", low) and not _asked(w, q) for w in lines):
        return True
    return bool(_kinds(_product_name(c)) - _line_kinds(q))


def _name_extra(c: Chunk, q: Query) -> set:
    name = KIND_NAME.sub(" ", PLAIN_NAME.sub(" ", _product_name(c).lower()))
    return set(re.findall(r"[a-z]+", name)) - {d.lower() for d in q.descriptors}


def _names_descriptors(c: Chunk, q: Query) -> bool:
    low = f"{c.label()} {c.section}".lower()
    return all(re.search(rf"\b{re.escape(d.lower())}", low) for d in q.descriptors)


def _top_pct(c: Chunk) -> float:
    return max((n for _, v in c.values() if "%" in v for n in numbers(v)), default=0.0)


def named_row(q: Query, ev: Evidence, exclude=(), view: View | None = None) -> Chunk | None:
    """The asked product's own row. Rows of another kind or of an unasked line never count. With the view, every
    row of that line on the page of the first such evidence row is a candidate (minus stale rows and expired
    offers): a row fitting the asked amount, term or model year first (the tightest tier), else the plainest
    product (fewest name words beyond its kind, then page order) at its top tier (the highest rate a deposit
    pays, the lowest a loan charges)."""
    words = _line_words(q)
    plain = [c for c in _line_rows(q, ev, exclude) if not _foreign(c, q)]
    if not plain:
        return None
    rows = plain
    if view is not None and plain[0].key in view.pages:
        gone = {d.id for d in getattr(ev, "dropped", ())}
        rows = [c for c in view.pages[plain[0].key].page.chunks
                if c.is_row and c not in exclude and c.id not in gone and not (c.expired_on and c.expired_on < view.taken_on)
                and _names(c, words) and supports(c, q) and qualify(c, q)[0] != "mismatch" and not _foreign(c, q)] or plain
    pinned = bool(q.amounts or q.terms or q.years)
    fits = lambda c: pinned and qualify(c, q)[0] == "match"
    best = min(rows, key=lambda c: (not fits(c), not _names_descriptors(c, q), len(_name_extra(c, q)), c.id))
    table = [c for c in rows if c.key == best.key and c.table_span == best.table_span]
    if fits(best):
        tiers = [c for c in table if fits(c) and tier_width(c, q) is not None]
        return min(tiers, key=lambda c: (tier_width(c, q), c.id)) if tiers else best
    if _line_kinds(q) & {"loan", "mortgage", "card", "heloc", "line"}:
        return min(table, key=lambda c: (_top_pct(c), c.id))
    return max(table, key=lambda c: (_top_pct(c), -c.id))


QUOTED_PCT = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent\b|apy\b|apr\b)", re.I)


INTRO_ROW = re.compile(r"\b(?:intro\w*|promo\w*|(?:1st|first)\s+\d+\s+months?)\b", re.I)


def _leads_with_other_kind(q: Query, ev: Evidence) -> bool:
    """A rate question naming a product line whose best evidence row is another kind of product by its own name
    ("Kayak Rates · Special Certificates" for "what's the kayak savings rate?")."""
    if q.attribute != "rate" or not q.category - {"account", "accounts"}:
        return False
    top = next((c for c in ev.chunks if supports(c, q)), None)
    return top is not None and top.is_row and other_kind(top, q)


def asked_line_row(q: Query, ev: Evidence, view: View | None) -> Chunk | None:
    """The asked line's own row for such a question: of that kind and no unasked line, naming every descriptor and
    qualifier as a word, never an intro rate unless one was asked; from the evidence, else from the crawl's
    re-fetched pages; on the page with the most such rows; then the plainest product (see named_row)."""
    def own(c):
        text = f"{c.label()} {c.section}".lower()
        word = lambda w: any(re.search(rf"\b{re.escape(x.lower())}\b", text) for x in [w] + list(SYNONYMS.get(w, ())))
        return (not other_kind(c, q) and not _foreign(c, q) and all(word(d) for d in q.descriptors)
                and (not q.qualifiers or any(word(w) for w in q.qualifiers))
                and (q.rate_kind or not INTRO_ROW.search(f"{text} {' '.join(c.header).lower()}")))
    words, gone = _line_words(q), {d.id for d in getattr(ev, "dropped", ())}
    def candidates(chunks):
        return [c for c in chunks if c.is_row and c.id not in gone and _names(c, words) and supports(c, q)
                and qualify(c, q)[0] != "mismatch" and not (c.expired_on and c.expired_on < view.taken_on) and own(c)]
    rows = [c for c in _line_rows(q, ev) if own(c)]
    if not rows and view is not None:
        rows = candidates(c for vp in view.pages.values() if vp.status == "fresh" for c in vp.page.chunks)
    if not rows:
        return None
    page = Counter(c.key for c in rows).most_common(1)[0][0]
    if view is not None and page in view.pages:           # every such row of that page, not just the evidence's
        rows = candidates(view.pages[page].page.chunks) or rows
    return named_row(q, SimpleNamespace(chunks=[c for c in rows if c.key == page], dropped=[]))


def _table_of(own: Chunk, view: View | None, ev: Evidence) -> list[Chunk]:
    pool = view.pages[own.key].page.chunks if view is not None and own.key in view.pages else ev.chunks
    name = _product_name(own) or own.section
    return [c for c in pool if c.is_row and c.key == own.key and c.table_span == own.table_span
            and (_product_name(c) or c.section) == name] or [own]


def line_table_first(q: Query, ev: Evidence, view: View) -> Evidence:
    """For such a question the model reads the asked line's own table first, and never the other kind's rows."""
    if not _leads_with_other_kind(q, ev):
        return ev
    own = asked_line_row(q, ev, view)
    if own is None:
        return ev
    table = _table_of(own, view, ev)
    rest = [c for c in (ev.context or ev.chunks) if c not in table and not (c.is_row and other_kind(c, q))]
    return replace(ev, context=table + rest)


def _quoted_elsewhere(question: str, q: Query, ev: Evidence) -> bool:
    """The member quotes a value for a product line ("do any savings tiers pay 1.23%?") that only another kind of
    product's rows carry (a money market's, a health savings account's), not a tier or special of the line itself."""
    if not (q.category and QUOTED_PCT.search(question)):
        return False
    quoted = numbers(" ".join(QUOTED_PCT.findall(question)))
    carriers = [c for c in ev.chunks if c.is_row and quoted & {n for _, v in c.values() if "%" in v for n in numbers(v)}]
    return bool(carriers) and all(_foreign(c, q, OTHER_LINE) for c in carriers)


def _own_row_and_table(question: str, q: Query, ev: Evidence, lookup=None, view: View | None = None):
    """The asked product's own row and its table. The quoted number pulls in the rows that carry it, so `lookup`
    searches again without it to find the product's own rows."""
    if lookup is not None:
        more = lookup(QUOTED_PCT.sub(" ", question))
        if more is not None and more.chunks:
            have = {c.id for c in more.chunks}
            ev = replace(ev, chunks=list(more.chunks) + [c for c in ev.chunks if c.id not in have])
    own = named_row(q, ev, view=view)
    if own is None:
        return None, []
    pool = view.pages[own.key].page.chunks if view is not None and own.key in view.pages else ev.chunks
    name = _product_name(own) or own.section
    table = [c for c in pool if c.is_row and c.key == own.key and c.table_span == own.table_span
             and (_product_name(c) or c.section) == name] or [own]
    return own, table


def named_row_guard(question: str, q: Query, ev: Evidence, text: str, lookup=None,
                    view: View | None = None) -> tuple[str, Chunk | None]:
    """A member who quotes a value for a named product that belongs to another kind of product's table gets the
    named product's own value too (its top tier, see named_row)."""
    if not _quoted_elsewhere(question, q, ev):
        return text, None
    own, table = _own_row_and_table(question, q, ev, lookup, view)
    if own is None or {n for c in table for _, v in c.values() if "%" in v for n in numbers(v)} & numbers(text):
        return text, None                      # the answer already gives that product's own value
    return f"{text.rstrip()} {render_row(own)}.", own


def own_table_first(question: str, q: Query, ev: Evidence, view: View, lookup=None) -> Evidence:
    """For such a question the model reads the named product's own table first, so it can say whether that
    product pays the value, not only which other product does."""
    if not _quoted_elsewhere(question, q, ev):
        return ev
    own, table = _own_row_and_table(question, q, ev, lookup, view)
    if own is None:
        return ev
    shown = list(ev.context or ev.chunks)
    return replace(ev, context=table + [c for c in shown if c not in table])


COMPARE_Q = re.compile(r"\b(compar\w*|vs\.?|versus|differ\w*|chang\w*|better|higher|lower|than|which is|should i|"
                       r"more|most|less|least)\b", re.I)
SECOND_ASK = re.compile(r"\b(?:and|also|plus),?\s+(?:what|when|how|is|are|does|do|can|should|will|the\s+\d)", re.I)


def clear_cut(question: str, q: Query, ev: Evidence) -> bool:
    """A rate question whose pinned term, model year or amount picks out exactly the top evidence row:
    that row is the answer, so it is given verbatim with no model call (faster, cheaper, and the same
    text for every wording). Lists, comparisons, changes and rows of an unasked product line still go
    to the model. A count stated per year is the dated sentence itself, for the same reason."""
    if q.attribute == "count" and (q.series or q.no_year):
        return True
    if q.attribute == "hours" and (q.places or q.need_schedule or (q.compare and q.schedules)):
        return any(supports(c, q) for c in ev.chunks)     # a schedule is given verbatim: the same for every wording
    if q.year_guess:
        return False                                          # which product line the vehicle belongs to is for the model
    if q.attribute != "rate" or not (q.terms or q.years or q.amounts) or q.list_intent or COMPARE_Q.search(question) \
            or SECOND_ASK.search(question) or len({w.rstrip("s") for w in _line_words(q)}) > 1:
        return False
    pool = [c for c in ev.chunks if supports(c, q)]
    if not pool or not pool[0].is_row or qualify(pool[0], q)[0] != "match":
        return False
    top, low = pool[0], pool[0].index_text.lower()
    if q.terms and not q.terms <= terms_in(low):              # every asked term is on that row
        return False
    if q.category and not (_names(top, _line_words(q)) or names_line(f"{top.label()} {top.section}".lower(), q)):
        return False                                          # the row belongs to the product asked about
    return not any(re.search(rf"\b{w}\b", low) for w in SOFT_QUALIFIERS if not _asked(w, q))


# After a transport failure (no route, timeout, rate limit, overload) the next questions skip the model
# instead of each waiting out the same failure; then one call probes again, on a short timeout and with
# no retry, so a dead network costs one slow answer, not one per probe. A rejected key stops the model.
COOLDOWN = {"APITimeoutError": 10, "APIConnectionError": 10, "RequestTimeoutError": 10, "GatewayTimeoutError": 10,
            "RateLimitError": 3, "InternalServerError": 3, "OverloadedError": 3, "ServiceUnavailableError": 3,
            "AuthenticationError": 10**9, "PermissionDeniedError": 10**9}
PROBE_TIMEOUT_S = 2.5


class Synthesizer:
    def __init__(self):
        self._client = None
        self._cooldown = 0
        self.calls = 0                  # model requests made so far (a recovery is only tried when a question made none)
        self._probe = False             # the last call failed in transport: the next one is a probe
        self.enabled = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")) \
            and not os.getenv("KH_NO_LLM")
        if self.enabled:                # one-time setup here: build() is untimed, the first answer is not
            try:
                self._client = _new_client()
            except Exception:           # no SDK (e.g. Python 3.9): _llm keeps its lazy path and falls back
                self._client = None

    def _llm(self, question: str, ev: Evidence, view: View, asked: date | None = None, listing: bool = False,
             guess: bool = False) -> dict | None:
        excerpts = "\n\n".join(f"[{i}] ({describe(c, view)})\n{c.text}" for i, c in enumerate(_llm_chunks(ev), 1))
        # A dated question is answered from the site as it stood that day, i.e. the newest crawl
        # on or before it; showing the crawl's own (earlier) day reads as "can't confirm".
        if asked and asked < view.taken_on:
            # a day before the first crawl: the model is told the day, or "a year ago" reads as a year before the crawl
            head = (f"Website snapshot date: {view.taken_on.isoformat()}. The member asks about {asked.isoformat()}, before "
                    f"this snapshot: answer as of that day from excerpts already in effect then (an effective date on or "
                    f"before it, or none).")
        else:
            head = f"Website snapshot date: {(asked if asked and asked > view.taken_on else view.taken_on).isoformat()}"
        if guess:                              # a year beside a word that may be a vehicle's make or model
            head += ("\nThe member gives a year beside a name that may be a vehicle's make or model. If it is a vehicle, answer with "
                     "the row of the product line that vehicle is financed under; if it is not a vehicle, set answerable=false.")
        if listing:                            # asked what is offered: every line, not the member's own
            head += ("\nThe member asks what is offered: list every product line the excerpts show for it, personal and "
                     "business alike, each with its own rate. An excerpt that says \"<page> lists: A; B; C\" names the page's "
                     "own product lines: name each of them that fits the question.")
        return self._ask(f"{head}\n\nExcerpts:\n{excerpts}", question)

    def _ask(self, content: str, question: str, system: str = SYSTEM, schema: dict = SCHEMA,
             max_tokens: int = 1024) -> dict | None:
        self.calls += 1
        try:
            if self._client is None:                   # lazy: a missing package must not break build()
                self._client = _new_client()
            client = self._client
            if self._probe and hasattr(client, "with_options"):
                client = client.with_options(timeout=PROBE_TIMEOUT_S, max_retries=0)
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                # the fixed instructions are identical on every call, so they are cached
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"{content}\n\nQuestion: {question}"}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                **_thinking_off(MODEL),
            )
            self._probe = False
            if resp.stop_reason == "refusal":
                return None
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return json.loads(text)
        except Exception as e:
            self._cooldown = next((COOLDOWN[k.__name__] for k in type(e).__mro__ if k.__name__ in COOLDOWN), 0)
            self._probe = self._cooldown > 0
            return None

    def compose_change(self, question: str, parts: list) -> tuple[str, list]:
        """How a value changed: the site as it stood at each date [(view, query, evidence), ...]. Returns
        (text, [(view, cited chunk), ...]). Without the model: each date's own answer, with its date."""
        said, cited = [], []
        for view, q, ev in parts:
            text, used = extractive(q, ev, view)
            if text != REFUSAL:
                d = view.taken_on
                said.append(f"As of {d:%B} {d.day}, {d.year}: {text}")
                cited += [(view, c) for c in used]
        fallback = " ".join(said) if said else REFUSAL
        cooling = self._cooldown > 0
        self._cooldown = max(0, self._cooldown - 1)
        if self.enabled and not cooling:
            out = self._llm_change(question, parts)
            if out is not None:
                if not out.get("answerable", False):
                    return REFUSAL, []
                text = _drop_banned(str(out.get("answer", "")))
                shown = [(view, c) for view, _, ev in parts for c in _llm_chunks(ev)]
                allowed = numbers(question) | numbers(" ".join(f"{c.text} {describe(c, view)} {view.taken_on}"
                                                               for view, c in shown))
                if text and numbers(text) <= allowed:
                    ids = [i for i in out.get("evidence_ids", []) if isinstance(i, int) and 1 <= i <= len(shown)]
                    return _drop_dead_links(text, parts[-1][0]), [shown[i - 1] for i in ids] or cited
        return fallback, cited

    def _llm_change(self, question: str, parts: list) -> dict | None:
        blocks, n = [], 0
        for view, _, ev in parts:
            lines = []
            for c in _llm_chunks(ev):
                n += 1
                lines.append(f"[{n}] ({describe(c, view)})\n{c.text}")
            blocks.append(f"The website as it stood on {view.taken_on.isoformat()}:\n\n" + "\n\n".join(lines))
        content = ("The member asks how something changed over time. Give the value as the website stood on each "
                   "date below, each with its date, from that date's excerpts only.\n\n" + "\n\n".join(blocks))
        return self._ask(content, question)

    def compose(self, question: str, q: Query, ev: Evidence, view: View,
                asked: date | None = None, lookup=None, second_look=None) -> tuple[str, list[Chunk], bool, str]:
        """Returns (text, cited chunks, confident, note)."""
        note = _stale_note(ev, view)
        if _leads_with_other_kind(q, ev) and asked_line_row(q, ev, view) is None and not any(
                not c.is_row and supports(c, q) and _mentions(f"{c.section} {c.text}".lower(), _line_words(q))
                for c in ev.chunks):
            return REFUSAL, [], False, note         # "an 84-month certificate" with only an 84-month loan row
        cooling = self._cooldown > 0
        self._cooldown = max(0, self._cooldown - 1)
        if self.enabled and not cooling and (not ev.supported or not clear_cut(question, q, ev)):
            ev = line_table_first(q, own_table_first(question, q, ev, view, lookup), view)
            out = self._llm(question, ev, view, asked, q.list_intent, q.year_guess)
            if out is not None:
                said_absent = bool(out.get("answerable", False)) and _says_absent(str(out.get("answer", "")), question)
                if not out.get("answerable", False) or said_absent:
                    again = self._second_look(question, q, view, asked, second_look)
                    if again is not None:
                        return again[0], again[1], True, "; ".join(x for x in (again[2], note) if x)
                    return REFUSAL, [], False, note
                accepted = self._accept(question, q, ev, view, out, lookup, asked)
                if accepted is not None:
                    text, cited, llm_note = accepted
                    if q.attribute in ("count", "when") and _says_absent(text, question):
                        return REFUSAL, [], False, note         # "the site only says …": that is not an answer
                    return text, cited, True, "; ".join(x for x in (llm_note, note) if x)
        if not ev.supported:                    # only the model may answer what the search could not support
            return REFUSAL, [], False, note
        text, cited = extractive(q, ev, view)
        if text != REFUSAL:                     # the no-key answer gets the same own row
            text, own = named_row_guard(question, q, ev, text, lookup, view)
            cited = cited + ([own] if own else [])
        return text, cited, text != REFUSAL, note

    def _accept(self, question: str, q: Query, ev: Evidence, view: View, out: dict, lookup=None,
                asked: date | None = None):
        """The model's answer if it passes the guards: (text, cited chunks, note), else None."""
        shown = _llm_chunks(ev)                # the exact excerpts the model saw, same numbering
        text = _dates_as_written(_digits_as_written(_drop_dead_links(_drop_banned(str(out.get("answer", ""))), view), shown), shown)
        small = q.attribute == "count"
        said = spelled_numbers(" ".join(c.text for c in shown), small)   # numbers the excerpts write in words
        if small:
            text = spelled_as_digits(text, said, small=True)           # "Thirty-two" on the page, asked how many: the figure
        allowed = (numbers(question) | numbers(" ".join(f"{c.text} {describe(c, view)}" for c in shown))
                   | {float(v) for v in said.values()})
        if asked:                              # the day it was told it may write back, in any form
            allowed |= numbers(f"{asked.isoformat()} {asked.day} {asked.year} {asked.month}")
        if not text or not numbers(text) <= allowed:
            return None
        ids = [i for i in out.get("evidence_ids", []) if isinstance(i, int) and 1 <= i <= len(shown)]
        cited = [shown[i - 1] for i in ids] or ev.chunks[:1]
        text, own = named_row_guard(question, q, ev, text, lookup, view)
        return text, cited + ([own] if own else []), str(out.get("note") or "").strip()

    def _second_look(self, question: str, q: Query, view: View, asked, second_look):
        """The search results held no answer to an open question ("who qualifies for membership?"): the
        model picks the site's own pages by title, then answers once more from those pages. Never for a
        rate, hours or address question, where "not listed" is usually the truth."""
        if second_look is None or not (q.attribute in (None, "when") or (q.attribute in ("count", "fee") and asked is None)):
            return None
        ev = second_look(question)
        if ev is None or not ev.chunks:
            return None
        out = self._llm(question, ev, view, asked, q.list_intent, q.year_guess)
        if not out or not out.get("answerable", False):
            return None
        accepted = self._accept(question, q, ev, view, out, asked=asked)
        return None if accepted is not None and _says_absent(accepted[0], question) else accepted

    def pick_pages(self, question: str, pages: str) -> list[str]:
        """Up to 3 page paths from the site's own page list ("path | title" lines) for a question."""
        out = self._ask("Pick the pages.", question, system=ROUTE_SYSTEM + "\n\n" + pages, schema=ROUTE_SCHEMA,
                        max_tokens=200)
        paths = out.get("paths", []) if isinstance(out, dict) else []
        return [p for p in paths if isinstance(p, str)][:3]


def _thinking_off(model: str) -> dict:
    """Haiku 5.5 thinks by default and accepts `disabled` at effort <= high; the speed score wants
    answers under 2s. Opus 5.5 rejects `disabled` (thinking is always on), so it gets nothing here."""
    return {"thinking": {"type": "disabled"}} if model.startswith("claude-haiku") else {}


ABSENT = re.compile(r"\b(?:only (?:say|says|state|states|mention|mentions)|(?:does|do|did)(?:n['’]t| not) (?:say|state|list|mention|specify|"
                    r"provide|give|show|publish|have)|no (?:information|mention|details?|record)|not (?:stated|listed|specified|mentioned|"
                    r"published)|(?:isn['’]t|is not|aren['’]t|are not) (?:stated|listed|specified|mentioned|given|available))\b", re.I)


def _says_absent(text: str, question: str) -> bool:
    """An answer that says the site doesn't state it is a decline in other words ("they only say that in …"), unless it
    also gives a figure the question didn't contain ("it doesn't give 2025; 317 used it in 2023")."""
    return ABSENT.search(text) is not None and not numbers(text) - numbers(question)


NUMBER_WORDS = set("""zero two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety""".split())
WORD_THEN_DIGITS = re.compile(r"(?<![\w-])([A-Za-z]+(?:-[A-Za-z]+)?)\s*\((\d[\d,]*)\)")


def _digits_as_written(text: str, shown: list) -> str:
    """The site writes "ten (10) per member": an answer that says "ten" says "10", the figure the site gives in digits.
    Whole words only ("one-time", "twenty-four" stay), never the pronoun "one", and a quote of the site's own
    "ten (10)" is left alone."""
    given = {}
    for c in shown:
        for m in WORD_THEN_DIGITS.finditer(c.text):
            if all(w in NUMBER_WORDS for w in m.group(1).lower().split("-")):
                given[m.group(1).lower()] = m.group(2)
    if not given:
        return text
    return re.sub(r"(?<![\w-])(" + "|".join(map(re.escape, given)) + r")(?![\w-])(?!\s*\()", lambda m: given[m.group(1).lower()],
                  text, flags=re.I)


AP_MONTHS = {"jan": "January", "feb": "February", "aug": "August", "sept": "September", "oct": "October", "nov": "November", "dec": "December"}
AP_DATE = re.compile(r"\b(Jan|Feb|Aug|Sept|Oct|Nov|Dec)\.\s*(\d{1,2})\b")
FULL_DATE = re.compile(r"\b(" + "|".join(AP_MONTHS.values()) + r")\s+(\d{1,2})\b")


def _dates_as_written(text: str, shown: list) -> str:
    """The site writes "Oct. 1" and "Nov. 12, 2026": an answer that says "October 1" says it as the site does, since the
    member (and anything that checks the answer against the site) reads the site's own words. Only dates the excerpts
    write in that short form, and only the same month and day."""
    given = {}
    for c in shown:
        for m in AP_DATE.finditer(c.text):
            given[(AP_MONTHS[m.group(1).lower()], int(m.group(2)))] = f"{m.group(1)}. {int(m.group(2))}"
    if not given:
        return text
    return FULL_DATE.sub(lambda m: given.get((m.group(1), int(m.group(2))), m.group(0)), text)


def spelled_as_digits(text: str, said: dict | None = None, small: bool = False) -> str:
    """An answer to a count says the digits for a number spelled in words ("Thirty-two" -> "32"; with `small`, "six" -> "6"),
    by default for every number the text spells, or for those of `said` (what the excerpts spell)."""
    by_word = {k.lower(): v for k, v in (spelled_numbers(text, small) if said is None else said).items()}
    return (SPELLED_SMALL if small else SPELLED).sub(lambda m: str(by_word.get(m.group(0).lower(), m.group(0))), text)


def _llm_chunks(ev: Evidence) -> list[Chunk]:
    return ev.context or ev.chunks


def _sentences(text: str) -> list[str]:
    """Split on sentence ends, but never inside "9 a.m. Monday" or "4 p.m. at another branch"."""
    return [s for s in re.split(r"(?<=[.!?])(?<![ap]\.m\.)\s+(?=[A-Z0-9\"“(])", text.strip()) if s]


def _drop_banned(text: str) -> str:
    return " ".join(s for s in _sentences(text) if not BANNED.search(s)).strip()


def _drop_dead_links(text: str, view: View) -> str:
    """Drop sentences linking to a multi-level page on the site's own host that is not in this snapshot.

    Old articles still link to pages that have since moved, and the crawl covers the site's pages, so an
    uncrawled deep link is a dead one. One-level vanity links (host/short-name) and other hosts, including
    the site's own subdomains (join.<host>/...), are redirects the crawl can't check, so they are kept. The
    hosts come from the crawled pages themselves."""
    hosts = sorted({k.split("/", 1)[0] for k in view.pages})
    if not hosts:
        return text
    site_link = re.compile(r"(?<![\w.-])(?:https?://)?(?:www\.)?(" + "|".join(map(re.escape, hosts)) +
                           r")((?:/[A-Za-z0-9_\-.]+){2,})", re.I)

    def dead(s: str) -> bool:
        return any(url_key(m.group(1) + m.group(2).rstrip(".")) not in view.pages for m in site_link.finditer(s))
    return " ".join(s for s in _sentences(text) if not dead(s)).strip()


def _stale_note(ev: Evidence, view: View) -> str:
    rows = [c for c in ev.dropped if c.is_row][:3]
    if not rows:
        return ""
    parts = [f"{'/'.join(v for _, v in c.values() if '%' in v)} on {view.pages[c.key].page.url}"
             + (f" (effective {c.effective.isoformat()})" if c.effective else "") for c in rows]
    return "Ignored older or less authoritative values: " + "; ".join(parts)
