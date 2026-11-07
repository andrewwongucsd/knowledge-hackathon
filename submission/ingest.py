"""Turn snapshot pages into typed, dated chunks, and keep a date-pinned view per snapshot.

A *view* is "the website as it stood on that day": the newest copy of every URL seen up to that
snapshot. A page that vanished from a later crawl is kept (flagged ``missing``) because the
assignment asks us to keep history, not just the latest copy.
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from urllib.parse import urlparse

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
MONTH_RE = "|".join(m.capitalize() for m in MONTHS)

# Lower rank = more authoritative when two sources give different values for the same fact.
KIND_RANK = {"product": 0, "document": 1, "article": 2, "promo": 3, "calculator": 3}
STATUS_RANK = {"fresh": 0, "carried": 1, "missing": 2}

LISTING_RE = re.compile(r"/(tag|author|page)/")
LONG_DATE_RE = re.compile(rf"\b({MONTH_RE})\s+(\d{{1,2}}),\s+(20\d\d)\b")
SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
ISO_DATE_RE = re.compile(r"\b(20\d\d)-(\d{2})-(\d{2})\b")
ANY_DATE = rf"(?:\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|20\d\d-\d{{2}}-\d{{2}}|(?:{MONTH_RE})\s+\d{{1,2}},\s+20\d\d)"
EFFECTIVE_RE = re.compile(rf"effective[^\n|]{{0,40}}?({ANY_DATE})", re.I)
OFFER_END_RE = re.compile(
    rf"(?:offer|promotion|special|deal)[^\n]{{0,60}}?(?:ended|ends|expired|expires|valid (?:through|until))"
    rf"\s*(?:on\s*)?({ANY_DATE})", re.I)
# Footnote markers glued onto values: "3.25%1", "6.10% APR1", "Rate (As Low As) 1"
FOOTNOTE_RE = re.compile(r"(%|\bAP[RY])\s?\d{1,2}\b|(\))\s?\d\b")
VALUE_HEADER_RE = re.compile(r"rate|apy|apr|yield|dividend|fee|payment|price|cost", re.I)
LABEL_HEADER_RE = re.compile(r"min|balance|tier|amount|term|program|type|product|account|loan|plan|point", re.I)


def url_key(u: str) -> str:
    """Same normalisation the scorer uses for source matching: host + path, no query/fragment."""
    p = urlparse(u if "//" in u else "https://" + u)
    return p.netloc.lower().removeprefix("www.") + p.path.rstrip("/").lower()


def parse_date(s: str) -> date | None:
    try:
        if m := LONG_DATE_RE.search(s):
            return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        if m := ISO_DATE_RE.search(s):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m := SLASH_DATE_RE.search(s):
            y = int(m.group(3))
            return date(y + 2000 if y < 100 else y, int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None
    return None


def page_kind(key: str) -> str:
    path = key.split("/", 1)[1] if "/" in key else ""
    if path.startswith("calculators/") or path.endswith("-calculator"):
        return "calculator"
    if re.search(r"\.(pdf|xlsx?|csv|docx?)$", path):
        return "document"
    if path.startswith(("newsroom", "learn/", "business/resources/")) or re.match(r"20\d\d/", path):
        return "article"
    if path.endswith("-lp") or "offer" in path:
        return "promo"
    return "product"


def strip_footnotes(cell: str) -> str:
    return FOOTNOTE_RE.sub(lambda m: m.group(1) or m.group(2), cell).strip()


def clean_inline(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)                      # images

    def link(m):
        label, url = m.group(1).strip(), m.group(2)
        key = url_key(url) if "dupaco.com" in url and "hubspot" not in url else ""
        return f"{label} ({key})" if key and key not in label else label
    text = re.sub(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)", link, text)
    text = text.replace("\\*", "*").replace("**", "").replace("__", "")
    return re.sub(r"[ \t]+", " ", text).strip()


@dataclass
class Chunk:
    id: int
    key: str                      # page url_key
    kind: str                     # page kind (product/document/article/promo/calculator)
    text: str                     # what we show/quote
    index_text: str               # what we search (adds title + section context)
    section: str = ""
    is_row: bool = False
    header: list[str] = field(default_factory=list)
    cells: list[str] = field(default_factory=list)
    value_cols: list[int] = field(default_factory=list)
    effective: date | None = None
    expired_on: date | None = None
    table_span: tuple | None = None          # (first, last) line of the source table
    tf: dict = field(default_factory=dict)   # filled by the retriever
    length: int = 0

    def label(self) -> str:
        """The non-value cells of a table row, e.g. '2024-27 Model Year | 60 Months'."""
        return " | ".join(c for i, c in enumerate(self.cells) if i not in self.value_cols and c)

    def values(self) -> list[tuple[str, str]]:
        return [(self.header[i] if i < len(self.header) else "", self.cells[i])
                for i in self.value_cols if i < len(self.cells) and self.cells[i]]


@dataclass
class ParsedPage:
    key: str
    url: str
    title: str
    kind: str
    publish: date | None
    chunks: list[Chunk]


@dataclass
class ViewPage:
    page: ParsedPage
    snapshot_id: str
    fetched: date
    status: str                   # fresh | carried (not re-fetched this crawl) | missing (gone from crawl)


@dataclass
class View:
    snapshot_id: str
    taken_on: date
    pages: dict[str, ViewPage]


class Parser:
    """Markdown page -> chunks. Stateless apart from the running chunk id."""

    def __init__(self):
        self._next_id = 0

    def _chunk(self, key, title, section, text, **kw) -> Chunk:
        # Short directory-style entries ("100 Main St., Springfield") mean nothing without their
        # heading (a branch vs a car dealer), so the heading becomes part of the text.
        if section and len(text) < 160 and not kw.get("is_row") and re.search(r"\d", text) \
                and section.lower() not in text.lower():
            text = f"{section} — {text}"
        c = Chunk(self._next_id, key, self._kind, text, f"{title} {section} {text}", section=section, **kw)
        self._next_id += 1
        return c

    def parse(self, key: str, url: str, title: str, md: str) -> ParsedPage:
        kind = self._kind = page_kind(key)
        title = re.sub(r"\s*\|\s*Dupaco.*$", "", title or "").strip()
        # PDF conversions leave HTML entities, sometimes double-escaped ("&amp;#45;" for "-").
        lines = html.unescape(html.unescape(md)).splitlines()
        chunks: list[Chunk] = []
        section, caption = "", ""
        i, short_buf = 0, []

        def flush_short():
            if short_buf:
                text = "\n".join(short_buf)
                chunks.append(self._chunk(key, title, section, text, expired_on=_offer_end(text)))
                short_buf.clear()

        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("|"):
                flush_short()
                j = i
                while j < len(lines) and lines[j].strip().startswith("|"):
                    j += 1
                chunks.extend(self._table(key, title, section, caption, lines, i, j))
                caption = ""
                i = j
                continue
            if not line:
                i += 1
                continue
            # collect one blank-line-separated block
            j = i
            block = []
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("|"):
                block.append(lines[j].strip())
                j += 1
            i = j
            if block[0].startswith("#"):
                flush_short()
                section = clean_inline(block[0].lstrip("#"))
                if re.search(r"\d", section):           # headings like "...with N% APY" carry facts
                    chunks.append(self._chunk(key, title, section, section))
                block = block[1:]
                if not block:
                    continue
            text = clean_inline(" ".join(block))
            if len(re.findall(r"[A-Za-z]{2,}", text)) < 2 and not re.search(r"\d", text):
                continue                                # "\# # #", lone bullets, stray markup
            if len(text) < 60 and not re.search(r"\d", text) and len(block) == 1:
                caption = text                          # e.g. "Home Loans" right above a table
            money_lines = [b for b in block if re.search(r"\$\s?\d|\d%", b)]
            if len(block) >= 3 and len(money_lines) * 2 >= len(block):
                flush_short()                           # fee schedules: one fact per line
                for b in block:
                    b = clean_inline(b)
                    if len(b) > 3:
                        chunks.append(self._chunk(key, title, section, b, expired_on=_offer_end(b)))
                continue
            if len(text) < 90:                          # hours lists etc.: keep short lines together
                short_buf.append(text)
                if sum(len(s) for s in short_buf) > 500:
                    flush_short()
                continue
            flush_short()
            for piece in _split_long(text):
                chunks.append(self._chunk(key, title, section, piece, expired_on=_offer_end(piece)))
        flush_short()
        _attach_effective_dates(lines, chunks)
        return ParsedPage(key, url, title, kind, _publish_date(kind, lines), chunks)

    def _table(self, key, title, section, caption, lines, a, b):
        out, header = [], []
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines[a:b]]
        for n, cells in enumerate(rows):
            while cells and not cells[0]:
                cells.pop(0)
            while cells and not cells[-1]:
                cells.pop()
            if not cells or all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue
            nxt = rows[n + 1] if n + 1 < len(rows) else []
            if any(re.fullmatch(r":?-{3,}:?", c.strip()) for c in nxt if c.strip()):
                # "Col2"-style headers are converter placeholders, not labels.
                header = ["" if re.fullmatch(r"Col\d+", h) else h
                          for h in (strip_footnotes(clean_inline(c)) for c in cells)]
                continue
            cells = [strip_footnotes(clean_inline(c)) for c in cells]
            value_cols = [k for k, c in enumerate(cells)
                          if re.search(r"\d\s*%|\$\s?\d", c) and (
                              k >= len(header) or VALUE_HEADER_RE.search(header[k])
                              or not LABEL_HEADER_RE.search(header[k]))
                          and not re.search(r"[-–]\s*\$|\+$|or less", c)]
            ctx = " · ".join(x for x in (section, caption) if x)
            pairs = "; ".join(f"{header[k]}: {c}" if k < len(header) and header[k] else c
                              for k, c in enumerate(cells) if c)
            text = f"{ctx} — {pairs}" if ctx else pairs
            out.append(self._chunk(key, title, ctx, text, is_row=True, header=header, cells=cells,
                                   value_cols=value_cols, expired_on=_offer_end(text), table_span=(a, b)))
        return out


def _split_long(text: str, limit: int = 700) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(sent) > 500:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def _offer_end(text: str) -> date | None:
    m = OFFER_END_RE.search(text)
    return parse_date(m.group(1)) if m else None


def _attach_effective_dates(lines: list[str], chunks: list[Chunk]) -> None:
    """A rate table's effective date is the first 'effective <date>' line after it (before the next
    table); failing that the nearest one before it. Undated tables stay undated (ranked last)."""
    eff = [(n, parse_date(m.group(1))) for n, ln in enumerate(lines) if (m := EFFECTIVE_RE.search(ln))]
    spans = sorted({c.table_span for c in chunks if c.table_span})
    for k, (a, b) in enumerate(spans):
        nxt = spans[k + 1][0] if k + 1 < len(spans) else len(lines)
        after = [d for n, d in eff if b <= n < nxt and d]
        before = [d for n, d in eff if n < a and d]
        when = after[0] if after else (before[-1] if before else None)
        for c in chunks:
            if c.table_span == (a, b):
                c.effective = when
    for c in chunks:                                    # prose that states its own effective date
        if not c.is_row and (m := EFFECTIVE_RE.search(c.text)):
            c.effective = parse_date(m.group(1))


def _publish_date(kind: str, lines: list[str]) -> date | None:
    """Articles print their date just above the title; listing pages were dropped earlier."""
    if kind != "article":
        return None
    h1 = next((n for n, ln in enumerate(lines) if ln.startswith("# ")), None)
    window = range(max(0, h1 - 3), h1) if h1 is not None else range(min(40, len(lines)))
    for n in window:
        if LONG_DATE_RE.search(lines[n]):
            return parse_date(lines[n])
    return None


class Store:
    """All parsed pages plus one view per ingested snapshot (oldest first)."""

    def __init__(self):
        self.parser = Parser()
        self.views: list[View] = []
        self._parsed: dict[tuple[str, str], ParsedPage] = {}

    def add_snapshot(self, snapshot) -> View:
        taken = date.fromisoformat(snapshot.taken_on[:10])
        current: dict[str, tuple] = {}
        for p in snapshot.pages:
            key = url_key(p.url)
            if LISTING_RE.search(key):
                continue
            prev = current.get(key)
            if prev is None or ("?" in prev[0].url and "?" not in p.url):   # prefer the query-less URL
                current[key] = (p, key)
        pages: dict[str, ViewPage] = {}
        if self.views:                                   # carry forward pages that left the crawl
            for key, vp in self.views[-1].pages.items():
                pages[key] = ViewPage(vp.page, vp.snapshot_id, vp.fetched, "missing")
        for key, (p, _) in current.items():
            digest = hashlib.sha1(p.markdown.encode()).hexdigest()
            parsed = self._parsed.get((key, digest))
            if parsed is None:
                parsed = self.parser.parse(key, p.url.split("?")[0], p.title, p.markdown)
                self._parsed[(key, digest)] = parsed
            fetched = date.fromisoformat((p.fetched_at or snapshot.taken_on)[:10])
            status = "fresh" if fetched >= taken - timedelta(days=2) else "carried"
            pages[key] = ViewPage(parsed, snapshot.snapshot_id, fetched, status)
        view = View(snapshot.snapshot_id, taken, pages)
        self.views.append(view)
        return view

    def view_for(self, as_of: str | None) -> int | None:
        if not self.views:
            return None
        if not as_of:
            return len(self.views) - 1
        try:
            when = date.fromisoformat(str(as_of)[:10])
        except ValueError:
            return len(self.views) - 1
        idx = [n for n, v in enumerate(self.views) if v.taken_on <= when]
        return idx[-1] if idx else None
