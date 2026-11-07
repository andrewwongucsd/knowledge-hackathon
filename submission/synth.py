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
from retrieve import Evidence, Query, qualify, supports, tokenize

REFUSAL = "That isn't listed on our website, so I can't confirm it. Please contact us and we can help."
# The prompt calls the context "excerpts"; a sentence that talks about them instead of the site is dropped.
BANNED = re.compile(r"\bevidence\b|\bexcerpts?\b|\bdocuments?\b", re.I)
NUM_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
MODEL = os.getenv("KH_MODEL", "claude-haiku-5-5")
TIMEOUT_S = float(os.getenv("KH_LLM_TIMEOUT", "8"))

SYSTEM = """You answer a credit union member's question using ONLY the numbered website excerpts provided.

Rules:
1. Answer whenever an excerpt states the fact. A range, an "as low as" value, a value under a different label, and a general answer to a general question all count. If excerpts disagree, choose by rule 3; do not decline. Set answerable=false only if no excerpt states it for the specific branch, product, term or tier the question names. A conclusion that follows directly from a stated fact and the snapshot date is answerable.
2. Copy numbers, dates, times, names, phone numbers and URLs exactly as written in the excerpts (do not reformat times, dates or spelled-out ordinals).
3. Excerpts are ordered most-trustworthy first. If they disagree, prefer: a newer "effective" date, then a newer "published" date, then a live page over one "not re-fetched" or "removed", then rate/product pages over calculators, promotions, blogs and press releases. Ignore incidental mentions inside bios or anecdotes. Never mention a value you rejected; explain it in "note" instead.
4. Tiers: if the question names a term, balance, loan amount or model year, answer for exactly that tier. Otherwise give the standard (not jumbo), personal (not business) product's main tier, plus the range across that same product's tiers. Never mix in other products' values.
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
        hidden = {d.id for d in ev.dropped}
        same_table = [c for c in view.pages[top.key].page.chunks if c.is_row and c is not top
                      and c.table_span == top.table_span and c.id not in hidden and supports(c, q)
                      and qualify(c, q)[0] != "mismatch"]
        rows = [top] + same_table[:3]
        return " ".join(render_row(r) + "." for r in rows), rows
    return best_sentence(top, q), [top]


class Synthesizer:
    def __init__(self):
        self._client = None
        self.enabled = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")) \
            and not os.getenv("KH_NO_LLM")

    def _llm(self, question: str, ev: Evidence, view: View) -> dict | None:
        try:
            if self._client is None:
                import anthropic                       # lazy: a missing package must not break build()
                self._client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=1)
            excerpts = "\n\n".join(f"[{i}] ({describe(c, view)})\n{c.text}" for i, c in enumerate(_llm_chunks(ev), 1))
            resp = self._client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM,
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


SITE_LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?dupaco\.com((?:/[A-Za-z0-9_\-.]+){2,})", re.I)


def _drop_dead_links(text: str, view: View) -> str:
    """Drop sentences linking to a multi-level dupaco.com page that is not in this snapshot.

    Old articles still link to pages that have since moved (e.g. a retired locator path), and the
    crawl covers the site's pages, so an uncrawled deep link is a dead one. One-level vanity links
    (dupaco.com/homeloans) and other hosts are redirects the crawl can't check, so they are kept.
    """
    def dead(s: str) -> bool:
        return any(url_key("dupaco.com" + m.group(1).rstrip(".")) not in view.pages for m in SITE_LINK_RE.finditer(s))
    return " ".join(s for s in _sentences(text) if not dead(s)).strip()


def _stale_note(ev: Evidence, view: View) -> str:
    rows = [c for c in ev.dropped if c.is_row][:3]
    if not rows:
        return ""
    parts = [f"{'/'.join(v for _, v in c.values() if '%' in v)} on {view.pages[c.key].page.url}"
             + (f" (effective {c.effective.isoformat()})" if c.effective else "") for c in rows]
    return "Ignored older or less authoritative values: " + "; ".join(parts)
