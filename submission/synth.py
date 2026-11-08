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

from ingest import Chunk, View, url_key
from retrieve import Evidence, Query, on_other_line, other_lines, qualify, supports, tokenize

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


def best_sentence(c: Chunk, q: Query) -> str:
    if q.attribute == "hours":                          # schedules only make sense whole
        return c.text.replace("\n", "; ")[:400]
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
    if not pool:
        return REFUSAL, []
    top = pool[0]
    if top.is_row:
        # Tiers the question did not pin down (e.g. loan amount for "30-year mortgage") are shown as a
        # range: every row of the same table that matches all qualifiers the question did give.
        hidden, lines = {d.id for d in ev.dropped}, other_lines(q, top)
        same_table = [c for c in view.pages[top.key].page.chunks if c.is_row and c is not top
                      and c.table_span == top.table_span and c.id not in hidden and supports(c, q)
                      and qualify(c, q)[0] != "mismatch" and not on_other_line(c, lines)]
        rows = [top] + same_table[:3]
        own = f"{top.label()} {top.section}".lower()
        if q.category and not any(re.search(rf"\b{re.escape(w)}\b", own) for w in q.category):
            # The top row matched a value shared with a different product than the one the question
            # names ("savings"): also give the named product's own row from the evidence, the plainest
            # (shortest) name first, so a value coincidence can't crowd out the product asked about.
            named = [c for c in ev.chunks if c.is_row and c not in rows
                     and any(re.search(rf"\b{re.escape(w)}\b", f"{c.label()} {c.section}".lower()) for w in q.category)]
            if named:
                rows = rows + [min(named, key=lambda c: len(f"{c.label()} {c.section}"))]
        return " ".join(render_row(r) + "." for r in rows), rows
    return best_sentence(top, q), [top]


class Synthesizer:
    def __init__(self):
        self._client = None
        self.enabled = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")) \
            and not os.getenv("KH_NO_LLM")
        if self.enabled:                # one-time setup here: build() is untimed, the first answer is not
            try:
                self._client = _new_client()
            except Exception:           # no SDK (e.g. Python 3.9): _llm keeps its lazy path and falls back
                self._client = None

    def _llm(self, question: str, ev: Evidence, view: View) -> dict | None:
        try:
            if self._client is None:                   # lazy: a missing package must not break build()
                self._client = _new_client()
            excerpts = "\n\n".join(f"[{i}] ({describe(c, view)})\n{c.text}" for i, c in enumerate(_llm_chunks(ev), 1))
            resp = self._client.messages.create(
                model=MODEL,
                max_tokens=1024,
                # the fixed instructions are identical on every call, so they are cached
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content":
                           f"Website snapshot date: {view.taken_on.isoformat()}\n\nExcerpts:\n{excerpts}\n\n"
                           f"Question: {question}"}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                **_thinking_off(MODEL),
            )
            if resp.stop_reason == "refusal":
                return None
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return json.loads(text)
        except Exception:
            return None

    def compose(self, question: str, q: Query, ev: Evidence, view: View) -> tuple[str, list[Chunk], bool, str]:
        """Returns (text, cited chunks, confident, note)."""
        note = _stale_note(ev, view)
        if self.enabled:
            out = self._llm(question, ev, view)
            if out is not None:
                if not out.get("answerable", False):
                    return REFUSAL, [], False, note
                text = _drop_dead_links(_drop_banned(str(out.get("answer", ""))), view)
                shown = _llm_chunks(ev)                # the exact excerpts the model saw, same numbering
                allowed = numbers(question) | numbers(" ".join(f"{c.text} {describe(c, view)}" for c in shown))
                if text and numbers(text) <= allowed:
                    ids = [i for i in out.get("evidence_ids", []) if isinstance(i, int) and 1 <= i <= len(shown)]
                    cited = [shown[i - 1] for i in ids] or ev.chunks[:1]
                    llm_note = str(out.get("note") or "").strip()
                    return text, cited, True, "; ".join(x for x in (llm_note, note) if x)
        text, cited = extractive(q, ev, view)
        return text, cited, text != REFUSAL, note


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
    uncrawled deep link is a dead one. One-level vanity links (host/short-name) and other hosts are
    redirects the crawl can't check, so they are kept. The hosts come from the crawled pages themselves."""
    hosts = sorted({k.split("/", 1)[0] for k in view.pages})
    if not hosts:
        return text
    site_link = re.compile(r"(?:https?://)?(?:www\.)?(" + "|".join(map(re.escape, hosts)) +
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
