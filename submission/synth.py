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
from retrieve import (SOFT_QUALIFIERS, WIRE, WIRE_FEE_LINE, Evidence, Query, _asked, names_line, on_other_line,
                      other_lines, qualify, supports, terms_in, tier_width, tokenize, wire_direction)

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


def best_sentence(c: Chunk, q: Query) -> str:
    if q.attribute == "hours":                          # schedules only make sense whole
        return c.text.replace("\n", "; ")[:400]
    if q.attribute is None:                             # how-to and "what is…": the steps or the description
        return c.text.replace("\n", " ")[:PARAGRAPH_CHARS]   # span the paragraph, not one sentence of it
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])|\n", c.text) if s.strip()]
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


def extractive(q: Query, ev: Evidence, view: View) -> tuple[str, list[Chunk]]:
    pool = [c for c in ev.chunks if supports(c, q)] if q.attribute else ev.chunks
    if q.attribute == "fee":                   # a promotion's offer ("we'll give you $X") is never a fee
        pool = [c for c in pool if c.kind != "promo"] or pool
    if not pool:
        return REFUSAL, []
    if q.attribute == "fee" and WIRE.search(q.raw) and wire_direction(q.raw) == "all":
        # several kinds of wire asked about ("domestic and international", "your wire fees"): one fee line of each
        # kind, from the page that lists the first one
        first = next((c for c in pool if WIRE_FEE_LINE.search(c.text)), None)
        kinds = {}
        for c in ([first] + [d for d in view.pages[first.key].page.chunks if d is not first]) if first else []:
            kind = next((k for k in ("incoming", "domestic", "international") if k in c.text.lower()), None)
            if kind and kind not in kinds and WIRE_FEE_LINE.search(c.text) and supports(c, q):
                kinds[kind] = c
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
        rows = [top] + same_table[:3]
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
    """"Account" is every product's head word, so it names the line only when nothing more specific does."""
    return (q.category - {"account", "accounts"}) or q.category


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
KIND_NAME = re.compile(r"\b(loan|mortgage|certificate|cd|card|savings|checking|heloc|money\s+market)s?\b")
PLAIN_NAME = re.compile(r"\b(?:share|accounts?|rates?|regular|basic|standard|personal)\b")
# lines that are a different kind of account, not a tier of the same one (jumbo, refinance)
OTHER_LINE = ("health", "hsa", "ira", "traditional", "roth", "business")


def _kinds(text: str) -> set:
    return {re.sub(r"\s+", " ", m.group(1)) for m in KIND_NAME.finditer(text.lower())}


def _product_name(c: Chunk) -> str:
    """The row's own product name: its first label cell, else the last part of its heading, else the heading."""
    for t in (c.label().split(" | ")[0], c.section.split(" · ")[-1], c.section):
        if _kinds(t):
            return t
    return ""


def _line_kinds(q: Query) -> set:
    return {w if w == "savings" else w.rstrip("s") for w in _line_words(q)}


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
    to the model."""
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
        self._probe = False             # the last call failed in transport: the next one is a probe
        self.enabled = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")) \
            and not os.getenv("KH_NO_LLM")
        if self.enabled:                # one-time setup here: build() is untimed, the first answer is not
            try:
                self._client = _new_client()
            except Exception:           # no SDK (e.g. Python 3.9): _llm keeps its lazy path and falls back
                self._client = None

    def _llm(self, question: str, ev: Evidence, view: View, asked: date | None = None) -> dict | None:
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
        return self._ask(f"{head}\n\nExcerpts:\n{excerpts}", question)

    def _ask(self, content: str, question: str, system: str = SYSTEM, schema: dict = SCHEMA,
             max_tokens: int = 1024) -> dict | None:
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
        cooling = self._cooldown > 0
        self._cooldown = max(0, self._cooldown - 1)
        if self.enabled and not cooling and not clear_cut(question, q, ev):
            ev = own_table_first(question, q, ev, view, lookup)
            out = self._llm(question, ev, view, asked)
            if out is not None:
                if not out.get("answerable", False):
                    again = self._second_look(question, q, view, asked, second_look)
                    if again is not None:
                        return again[0], again[1], True, "; ".join(x for x in (again[2], note) if x)
                    return REFUSAL, [], False, note
                accepted = self._accept(question, q, ev, view, out, lookup, asked)
                if accepted is not None:
                    text, cited, llm_note = accepted
                    return text, cited, True, "; ".join(x for x in (llm_note, note) if x)
        text, cited = extractive(q, ev, view)
        if text != REFUSAL:                     # the no-key answer gets the same own row
            text, own = named_row_guard(question, q, ev, text, lookup, view)
            cited = cited + ([own] if own else [])
        return text, cited, text != REFUSAL, note

    def _accept(self, question: str, q: Query, ev: Evidence, view: View, out: dict, lookup=None,
                asked: date | None = None):
        """The model's answer if it passes the guards: (text, cited chunks, note), else None."""
        text = _drop_dead_links(_drop_banned(str(out.get("answer", ""))), view)
        shown = _llm_chunks(ev)                # the exact excerpts the model saw, same numbering
        allowed = numbers(question) | numbers(" ".join(f"{c.text} {describe(c, view)}" for c in shown))
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
        rate, fee, hours, count, date or address question, where "not listed" is usually the truth."""
        if second_look is None or q.attribute is not None:
            return None
        ev = second_look(question)
        if ev is None or not ev.chunks:
            return None
        out = self._llm(question, ev, view, asked)
        if not out or not out.get("answerable", False):
            return None
        return self._accept(question, q, ev, view, out, asked=asked)

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
