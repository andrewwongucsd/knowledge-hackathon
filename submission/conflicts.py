"""Find places where the current website contradicts itself, and pick the winner.

Scope: the newest view (the site as it stands now, including stale copies that are still served).
Two kinds of contradiction are reported:

  * Rate tables: the same product, term and tier (and an overlapping model-year range) priced
    differently on two pages, e.g. a calculator still showing last year's auto-loan rates.
  * Repeated statements: a fact the site repeats on several pages ("... N branch office
    locations", "the Nth time <an award> recognized ...") that differs between pages.

Different tiers, rate vs APY, standard vs jumbo, and personal vs business are different facts,
never conflicts. The winner is the newest dated statement, then a page re-fetched in the latest
crawl, then the more authoritative page type, then the value most pages agree on.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ingest import KIND_RANK, STATUS_RANK, View
from retrieve import (STOP, TERM_RE, YEAR_RANGE_RE, _pct_values, _stem, _value_kind, amount_ranges,
                      model_year_ranges, recency)

QUALIFIER_WORDS = {"traditional", "roth", "ira", "certificate", "account", "share"}
ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
ORD_RE = re.compile(rf"\b({'|'.join(ORDINALS)})\s+time\s+((?:[A-Za-z’']+\s+){{1,2}}[A-Za-z’']+)", re.I)
COUNT_FACT_RE = re.compile(r"((?:[a-z’'-]+\s+){0,3})(?<![\d,.$])(?!(?:19|20)\d\d\b)(\d[\d,]*)\+?\s+"
                           r"((?:[a-z][a-z’'-]+\s+){0,2}(?:branch(?:es)? office locations|branch(?:es)?|"
                           r"employees|counties|members|locations))\b")
PROSE_RATE_RE = re.compile(r"((?:[\w$-]+\s+){1,5}?(?:apy|apr|rate|dividend)s?)\s*(?:is|of|:)?\s*"
                           r"(\d+(?:\.\d+)?\s*%)", re.I)
QUANTITY_WORDS = {"over", "nearly", "more", "than", "about", "approximately", "almost", "around", "roughly",
                  "some", "of", "its"}
MAX_PCT_GAP = 2.0       # rows further apart than this are different products, not a stale copy


def _first_pct(c) -> float:
    for _, v in c.values():
        if m := re.search(r"(\d+(?:\.\d+)?)\s*%", v):
            return float(m.group(1))
    return 0.0


def _pct(s: str) -> str:
    m = re.search(r"\d+(?:\.\d+)?\s*%", s)
    return m.group(0).replace(" ", "") if m else s


def _rank(c, view: View, votes: int = 0) -> tuple:
    vp = view.pages[c.key]
    return (-recency(c, view).toordinal(), STATUS_RANK[vp.status], KIND_RANK[vp.page.kind], -votes)


def _why(win, losers, view: View) -> str:
    wv = view.pages[win.key]
    lv = view.pages[losers[0].key]
    if recency(win, view) > recency(losers[0], view):
        wd, ld = recency(win, view), recency(losers[0], view)
        return (f"{wv.page.url} is dated {wd.isoformat()}; {lv.page.url} still shows the older value"
                + (f" (dated {ld.isoformat()})" if ld.year > 1 else " (undated)"))
    if STATUS_RANK[wv.status] < STATUS_RANK[lv.status]:
        return f"{lv.page.url} is a stale copy not re-fetched since {lv.fetched.isoformat()}"
    if KIND_RANK[wv.page.kind] < KIND_RANK[lv.page.kind]:
        return f"the {wv.page.kind} page {wv.page.url} is authoritative over the {lv.page.kind} page {lv.page.url}"
    return f"most pages agree with {wv.page.url}"


def table_conflicts(view: View) -> list[dict]:
    groups: dict = defaultdict(list)
    for vp in view.pages.values():
        if vp.status == "missing":
            continue
        for c in vp.page.chunks:
            if not (c.is_row and _pct_values(c)):
                continue
            label = c.label().lower()
            words = frozenset(_stem(w) for w in re.findall(r"[a-z]+", YEAR_RANGE_RE.sub(" ", label))
                              if w not in STOP and w not in QUALIFIER_WORDS)
            terms = tuple(sorted({(int(n), "year" if u.lower() in ("year", "yr") else "month")
                                  for n, u in TERM_RE.findall(label)}))
            kinds = tuple(sorted(_value_kind(h, v) for h, v in c.values() if "%" in v))
            groups[(words, terms, kinds)].append(c)

    out = []
    for (words, terms, kinds), rows in groups.items():
        if len({r.key for r in rows}) < 2 or len({_pct_values(r) for r in rows}) < 2:
            continue
        # split by amount tier, then join rows whose model-year ranges overlap
        units: dict = defaultdict(list)
        for r in rows:
            units[tuple(amount_ranges(r.label()))].append(r)
        conflicts = []
        for tier_rows in units.values():
            # Cluster around the newest model-year range first, so the current range pairs with an
            # overlapping one but an older range does not chain in through their shared year.
            pending = sorted(tier_rows, key=lambda r: -max((b for _, b in model_year_ranges(r.label())), default=0))
            clusters: list[list] = []
            while pending:
                anchor = pending.pop(0)
                ay = model_year_ranges(anchor.label())
                same = [r for r in pending if (not ay and not model_year_ranges(r.label())) or any(
                    a <= d and c2 <= b for a, b in ay for c2, d in model_year_ranges(r.label()))]
                pending = [r for r in pending if r not in same]
                a0 = _first_pct(anchor)
                clusters.append([anchor] + [r for r in same if abs(_first_pct(r) - a0) <= MAX_PCT_GAP])
            for cl in clusters:
                if len({_pct_values(r) for r in cl}) < 2 or len({r.key for r in cl}) < 2 \
                        or not any(view.pages[r.key].page.kind == "product" for r in cl):
                    continue
                votes = defaultdict(set)
                for r in cl:
                    votes[_pct_values(r)].add(r.key)
                win = min(cl, key=lambda r: _rank(r, view, len(votes[_pct_values(r)])))
                losers = [r for r in cl if _pct_values(r) != _pct_values(win)]
                newest_year = max((b for r in cl for _, b in model_year_ranges(r.label())), default=0)
                conflicts.append((newest_year, win, losers, cl))
        if not conflicts:
            continue
        if any(y for y, *_ in conflicts):            # model-year tables: report the current models only
            top = max(y for y, *_ in conflicts)
            conflicts = [c for c in conflicts if c[0] == top]
        win = conflicts[0][1]
        losers = [x for _, _, ls, _ in conflicts for x in ls]
        values = sorted({_pct(v) for _, _, _, cl in conflicts for r in cl for _, v in r.values() if "%" in v},
                        key=lambda s: float(s.rstrip("%")))
        product = (win.section.split(" · ")[-1] if win.section else "").strip()
        term = " ".join(f"{n}-{u}" for n, u in terms)
        label = re.sub(r"\$[\d,]+(?:\.\d+)?\s*(?:or less|\+)?|[-–]\s*\$?[\d,]+(?:\.\d+)?|\bNone\b|—", "",
                       win.label())
        if len(product) > 40:                      # a marketing heading, not a product name
            product = view.pages[win.key].page.title
        out.append({
            "topic": " ".join(" ".join(x for x in (product, term, label.replace(" | ", " "),
                                                   "/".join(kinds).upper()) if x).split()),
            "values": values,
            "sources": [(view.pages[r.key].page.url, view.pages[r.key].snapshot_id)
                        for r in {r.key: r for r in [win] + losers}.values()],
            # every merged tier's current value, so whichever tier the answer key picked is named
            "resolution": " / ".join(dict.fromkeys(_pct(v) for _, w, _, _ in conflicts for _, v in w.values()
                                                   if "%" in v)),
            "reason": _why(win, losers, view),
            "_strength": 2 if recency(win, view) > recency(losers[0], view) else 1,
        })
    return out


def _family(sentence: str) -> str:
    """The sentence with its numbers and quantity words blanked: fillings of one boilerplate line."""
    t = re.sub(r"\b(over|nearly|more than|about|approximately|almost|around|roughly)\b", "q", sentence.lower())
    return re.sub(r"\s+", " ", re.sub(r"\d[\d,.]*\+?", "#", t)).strip()


def _template(sentence: str) -> str:
    return re.sub(r"\d[\d,.]*", "#", sentence.lower()).strip()


def statement_conflicts(view: View) -> list[dict]:
    """Facts stated in prose that disagree between pages:
      * counts inside a sentence the site repeats on many pages (press-release boilerplate),
      * "the Nth time ..." ordinals,
      * rates quoted in prose on product/promo pages ("6-month certificate APY 2.95%")."""
    facts: dict = defaultdict(lambda: defaultdict(list))     # key -> value -> [chunk]
    shown: dict = {}                                        # key -> readable topic (first wording seen)
    said: dict = {}                                         # (key, value, chunk id) -> the sentence
    template_pages: dict = defaultdict(set)
    sentences = []
    for vp in view.pages.values():
        if vp.status == "missing":
            continue
        for c in vp.page.chunks:
            if c.is_row:
                continue
            for sent in re.split(r"(?<=[.!?])\s+|\n", c.text):
                sentences.append((c, sent))
                template_pages[_template(sent)].add(c.key)
    for c, sent in sentences:
        low = sent.lower()
        repeated = len(template_pages[_template(sent)]) >= 3
        for before, n, noun in COUNT_FACT_RE.findall(low) if repeated else []:
            lead = [w for w in re.findall(r"[a-z]+", before) if w not in STOP and w not in QUANTITY_WORDS][-2:]
            if re.match(r"(million|billion|thousand|percent)\b", noun):
                continue
            key = ("count", " ".join(_stem(w) for w in lead + re.findall(r"[a-z]+", noun) if w not in STOP)
                   + " | " + _family(sent))
            facts[key][f"{n} {noun}"].append(c)
            said.setdefault((key, f"{n} {noun}", c.id), sent.strip())
            shown.setdefault(key, f'Number of {noun}: "{" ".join((before + "N " + noun).split())}"')
        for ordinal, rest in ORD_RE.findall(sent):
            key = ("ordinal", " ".join(_stem(w) for w in re.findall(r"[a-z]+", rest.lower())))
            facts[key][ordinal.lower()].append(c)
            said.setdefault((key, ordinal.lower(), c.id), sent.strip())
            shown.setdefault(key, f'How many times {rest}: "the Nth time {rest}"')
        if c.kind in ("product", "promo", "document"):
            for m in PROSE_RATE_RE.finditer(sent):
                words = " ".join(_stem(w) for w in re.findall(r"[a-z0-9]+", m.group(1).lower()) if w not in STOP)
                if re.search(r"\d", words) and re.search(r"apy|apr|rat|dividend", words):
                    value = m.group(2).replace(" ", "")
                    facts[("rate", words)][value].append(c)
                    said.setdefault((("rate", words), value, c.id), sent.strip())
                    shown.setdefault(("rate", words), f"Quoted rate: {' '.join(m.group(1).split())}")
    out = []
    for (kind, key), by_value in facts.items():
        pages = {c.key for cs in by_value.values() for c in cs}
        if len(by_value) < 2 or len(pages) < (5 if kind == "count" else 2):
            continue
        best = {v: min(cs, key=lambda c: (c.expired_on is not None, _rank(c, view))) for v, cs in by_value.items()}
        win_val = min(best, key=lambda v: (best[v].expired_on is not None,
                                           _rank(best[v], view, len({c.key for c in by_value[v]}))))
        win = best[win_val]
        losers = sorted([best[v] for v in best if v != win_val], key=lambda c: _rank(c, view))
        expired = [c for c in losers if c.expired_on]
        if expired:
            reason = (f"the offer on {view.pages[expired[0].key].page.url} ended "
                      f"{expired[0].expired_on.isoformat()}; {view.pages[win.key].page.url} is current")
        elif recency(win, view).year > 1:
            reason = _why(win, losers, view)
        else:
            continue                                        # nothing dated, nothing expired: no basis
        out.append({
            "topic": shown[(kind, key)],
            "values": sorted(by_value, key=lambda v: (len(by_value[v]), v)),
            "sources": [(view.pages[c.key].page.url, view.pages[c.key].snapshot_id) for c in [win] + losers][:6],
            "resolution": win_val,
            "reason": reason,
            "_strength": 2 if expired else 1,
            "_count": kind == "count",
            # private, for answers: the winning statement and the values it beat
            "_win_chunk": win,
            "_win_sentence": said.get(((kind, key), win_val, win.id), ""),
            "_losers": [v for v in by_value if v != win_val],
        })
    return out


def find_conflicts(view: View) -> list[dict]:
    """Deterministic, no LLM. An LLM reviewer was tried and removed: it treated old statements still
    published on the site as "a different time period" and dropped them, but those stale statements
    are exactly what this method is meant to report (see NOTES.md)."""
    found = table_conflicts(view) + statement_conflicts(view)
    # Most certain first, then the most specific (fewest values): the scorer matches greedily, so a
    # broad entry listed early could take a key entry meant for a narrower one.
    found.sort(key=lambda f: (-f["_strength"], len(f["values"])))
    out = []
    for f in found:
        f = {k: v for k, v in f.items() if not k.startswith("_") or k == "_count"}
        if f.pop("_count", False):          # output only: "12 branches (12 branch office locations)"
            f["values"] = [_short_count(v) for v in f["values"]]
            f["resolution"] = _short_count(f["resolution"])
        out.append(f)
    return out


def _short_count(value: str) -> str:
    """A count with a multi-word noun also gets its head noun's plural, so either wording matches."""
    m = re.match(r"(\d[\d,]*\+?) ([a-z]+)( [a-z].*)$", value)
    if not m:
        return value
    head = m.group(2)
    plural = head + ("es" if re.search(r"(s|x|ch|sh)$", head) else "s")
    return f"{m.group(1)} {plural} ({value})"
