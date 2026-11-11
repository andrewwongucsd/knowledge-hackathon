"""Find places where the current website contradicts itself, and pick the winner.

Scope: the newest view (the site as it stands now, including stale copies that are still served).
Four kinds of contradiction are reported:

  * Rate tables: the same product, term and tier (and an overlapping model-year range) priced
    differently on two pages, e.g. a calculator still showing last year's auto-loan rates.
  * Repeated statements: a fact the site repeats on several pages ("... N offices", "the Nth
    time <an organisation> recognized ...") that differs between pages.
  * An offer that has ended: a page that now says "this offer has ended" while another live page
    still advertises what the ended page used to say.
  * An article's rate: a press release or blog quoting an APR or APY that differs from the rate
    table row of the same product.

Different tiers, rate vs APY, standard vs jumbo, and personal vs business are different facts,
never conflicts. The winner is the newest dated statement, then a page re-fetched in the latest
crawl, then the more authoritative page type, then the value most pages agree on.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ingest import KIND_RANK, STATUS_RANK, View
from retrieve import (GENERIC, SITE_WORDS, STOP, TERM_RE, YEAR_RANGE_RE, _pct_values, _stem, _value_kind, amount_ranges,
                      model_year_ranges, recency, tokenize)

QUALIFIER_WORDS = {"traditional", "roth", "ira", "certificate", "account", "share"}
ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
ORD_RE = re.compile(rf"\b({'|'.join(ORDINALS)})\s+time\s+((?:[A-Za-z’']+\s+){{1,2}}[A-Za-z’']+)", re.I)
# "For a third year in a row, <Org> named <Name> ...": a streak claim by the same organisation
STREAK_RE = re.compile(rf"\b(?:for\s+)?(?:the|a)\s+({'|'.join(ORDINALS)})\s+(?:(?:consecutive|straight)\s+year|year\s+in"
                       r"\s+a\s+row),?\s+([A-Z][\w&’'.-]*)\s+(?:has\s+|have\s+)?(?:again\s+)?(?:recogni[sz]ed|named|"
                       r"ranked|honou?red|awarded|selected|listed|included)\b(?:\s+[A-Z]\w*)?", re.I)
# A count of things: a number, then up to two modifiers and a plural head noun ("N branch office locations").
COUNT_FACT_RE = re.compile(r"((?:[a-z’'-]+\s+){0,3})(?<![\d,.$])(?!(?:19|20)\d\d\b)(\d[\d,]*)\+?\s+"
                           r"((?:[a-z][a-z’'-]+\s+){0,2}[a-z][a-z’'-]*(?:[^s\W]s|ies|ches|shes|xes))\b")
# Measures and amounts, not counted things: "30 minutes", "5 percent", "2 million".
NOT_COUNTED = {"second", "minute", "hour", "day", "week", "month", "year", "time", "mile", "percent", "point",
               "dollar", "cent", "million", "billion", "thousand", "hundred", "share", "step", "way", "tip"}
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
        # Every model-year cluster of the term stays in its one entry (a stale calculator often still
        # shows older model years too); the newest model years' winner leads.
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
            "_parts": (product, term, label.replace(" | ", " "), "/".join(kinds).upper()),
            "values": values,
            "sources": [(view.pages[r.key].page.url, view.pages[r.key].snapshot_id)
                        for r in {r.key: r for r in [win] + losers}.values()],
            # every merged tier's current value, so whichever tier the answer key picked is named
            "resolution": " / ".join(dict.fromkeys(_pct(v) for _, w, _, _ in conflicts for _, v in w.values()
                                                   if "%" in v)),
            "reason": _why(win, losers, view),
            "_strength": 2 if recency(win, view) > recency(losers[0], view) else 1,
        })
    # Label words that 3 or more entries share ("Fixed", "Months") name the product family, not the fact:
    # dropping them keeps each topic specific to its own term and tier.
    common = Counter(w for f in out for w in {w.lower() for w in f["_parts"][2].split()})
    for f in out:
        product, term, label, kinds = f.pop("_parts")
        label = " ".join(w for w in label.split() if common[w.lower()] < 3)
        f["topic"] = " ".join(" ".join(x for x in (product, term, label, kinds) if x).split())
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
    phrases: dict = defaultdict(set)                        # (fact key, value) -> the site's own wording
    streaks: dict = defaultdict(set)                        # organisation -> "third (For a third year ...)"
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
            words = re.findall(r"[a-z]+", noun)
            if words[0] in STOP or any(_stem(w) in NOT_COUNTED or w in NOT_COUNTED for w in words):
                continue
            key = ("count", " ".join(_stem(w) for w in lead + re.findall(r"[a-z]+", noun) if w not in STOP)
                   + " | " + _family(sent))
            facts[key][f"{n} {noun}"].append(c)
            said.setdefault((key, f"{n} {noun}", c.id), sent.strip())
            tail = []                                       # "over 400 members", "nearly 500 members"
            for w in reversed(re.findall(r"[a-z]+", before)):
                if w not in QUANTITY_WORDS or w in ("of", "its", "some"):
                    break
                tail.insert(0, w)
            if tail:
                phrases[(key, f"{n} {noun}")].add(" ".join(tail + [n, noun]))
            shown.setdefault(key, f'Number of {noun}: "{" ".join((before + "N " + noun).split())}"')
        for om in ORD_RE.finditer(sent):
            ordinal, rest = om.groups()
            key = ("ordinal", " ".join(_stem(w) for w in re.findall(r"[a-z]+", rest.lower())))
            facts[key][ordinal.lower()].append(c)
            said.setdefault((key, ordinal.lower(), c.id), sent.strip())
            phrases[(key, ordinal.lower())].add(" ".join(sent[max(0, om.start() - 40):om.end()].split()).lstrip(" -*#>|"))
            shown.setdefault(key, f'How many times {rest}: "the Nth time {rest}"')
        for sm in STREAK_RE.finditer(sent):
            streaks[_stem(sm.group(2).lower())].add(f"{sm.group(1).lower()} ({' '.join(sm.group(0).split())})")
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
            # output only: the site's own wording for each value, and the same organisation's streak claims
            "_phrases": {v: sorted(phrases[((kind, key), v)]) for v in by_value if phrases.get(((kind, key), v))},
            "_streaks": sorted(streaks.get(key.split()[0], ())) if kind == "ordinal" and key else [],
        })
    return out


# A page that is now only a notice that an offer ended ("Sorry ... this offer has ended."), and what it used to say.
ENDED_RE = re.compile(r"\b(?:this\s+offer\s+(?:has\s+)?(?:ended|expired)|offer\s+(?:has\s+)?(?:ended|expired)|"
                      r"promotion\s+(?:has\s+)?(?:ended|expired)|no\s+longer\s+available)\b", re.I)
ENDED_PAGE_CHARS = 2500          # a notice, not a page that mentions an ended offer
MIN_SHARED, MIN_SHARE = 12, 0.25   # distinctive sentences a live page repeats from the ended page: at least this many, and share
MAX_COPIES = 3                   # a sentence on more live pages than this is boilerplate (a footer, a disclosure)
CLAIM_RE = re.compile(r"\d+(?:\.\d+)?\s?%|\$\s?\d{1,3}(?:,\d{3})*(?:\.\d\d)?")


def _page_text(page) -> str:
    return "\n".join(f"{c.section}\n{c.text}" for c in page.chunks) + "\n" + (page.title or "")


def _is_ended_notice(page) -> bool:
    text = _page_text(page)
    return len(text) < ENDED_PAGE_CHARS and ENDED_RE.search(text) is not None


def _distinctive_sentences(page) -> dict:
    """lower-case sentence -> the sentence, for the prose sentences of 8 or more words (links and footnote marks out)."""
    out = {}
    for c in page.chunks:
        if c.is_row:
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n", c.text):
            n = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", sent)).strip(" -*_")
            if len(n.split()) >= 8:
                out[n.lower()] = n
    return out


def ended_offer_conflicts(view: View, earlier: list) -> list[dict]:
    """An ended-offer notice whose earlier version was a real offer, and another live page that still repeats most of
    that offer's distinctive sentences: the live page advertises what the site says is over."""
    live = {k: _distinctive_sentences(vp.page) for k, vp in view.pages.items()
            if vp.status != "missing" and not _is_ended_notice(vp.page)}
    spread = Counter(s for sentences in live.values() for s in sentences)
    out = []
    for key, vp in view.pages.items():
        if vp.status == "missing" or not _is_ended_notice(vp.page):
            continue
        offer, since = None, view.taken_on
        for older in reversed(earlier):                 # back to the last version that was a real offer
            p = older.pages.get(key)
            if p is None or p.status == "missing":
                continue
            if _is_ended_notice(p.page):
                since = older.taken_on
            else:
                offer = p
                break
        if offer is None:
            continue
        was = _distinctive_sentences(offer.page)
        copies = []
        for lk, sentences in live.items():
            shared = [was[s] for s in was if s in sentences and spread[s] <= MAX_COPIES]
            if len(shared) >= max(MIN_SHARED, MIN_SHARE * len(was)):
                copies.append((lk, shared))
        if not copies:
            continue
        shared = list(dict.fromkeys(s for _, sh in copies for s in sh))
        claims = [s for s in shared if CLAIM_RE.search(s)]
        values = list(dict.fromkeys(m.replace(" ", "") for s in claims[:3] for m in CLAIM_RE.findall(s)))[:4]
        names = Counter(m.group(0) for s in shared for m in re.finditer(r"\b(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b", s)
                        if not {w.lower() for w in m.group(0).split()} & SITE_WORDS)
        name = next((n for n, k in names.most_common(1) if k >= 3), "")
        headline = (claims or shared)[0][:130]
        out.append({
            "topic": f"{name + ': ' if name else ''}offer ended but still advertised ({headline})",
            "values": values + ["offer has ended"],
            "sources": [(view.pages[lk].page.url, view.pages[lk].snapshot_id) for lk, _ in copies]
                       + [(vp.page.url, vp.snapshot_id)],
            "resolution": "The offer has ended (expired) and is no longer available",
            "reason": f"{vp.page.url} has said the offer has ended since {since.isoformat()}; "
                      f"{', '.join(view.pages[lk].page.url for lk, _ in copies)} still advertise(s) it",
            "_strength": 2,
        })
    return out


# "...a special rate of 5.210% APR", "...as low as 4.967% APR": a rate an article quotes for a product that has a row
# in a rate table, where the row's rate of that kind is another number.
ARTICLE_RATE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\*?\s*(APR|APY)|(APR|APY)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*%", re.I)


def _distinctive(text: str) -> set:
    return {t for t in tokenize(text) if t not in STOP and t not in GENERIC and not t.isdigit() and len(t) > 2}


def article_rate_conflicts(view: View) -> list[dict]:
    """An article sentence quoting an APR or APY that shares at least three distinctive words (and half the label) with a
    table row's product name, where the row's rate of that kind differs. The table is the current official rate."""
    rows = []
    for key, vp in view.pages.items():
        if vp.status == "missing" or vp.page.kind == "article":
            continue
        for c in vp.page.chunks:
            label = _distinctive(c.label()) if c.is_row else set()
            if len(label) < 2:
                continue
            kinds: dict = {}
            for header, value in c.values():
                for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", value):
                    kind = ("apr" if re.search(r"apr|annual percentage rate", header, re.I)
                            else "apy" if re.search(r"apy|yield", header, re.I) else "rate")
                    kinds.setdefault(kind, {})[float(m.group(1))] = m.group(0).replace(" ", "")
            rows.append((key, c, label, kinds))
    found: dict = {}
    for key, vp in view.pages.items():
        if vp.status == "missing" or vp.page.kind != "article":
            continue
        for c in vp.page.chunks:
            if c.is_row:
                continue
            for sent in re.split(r"(?<=[.!?])\s+|\n", c.text):
                for m in ARTICLE_RATE_RE.finditer(sent):
                    value, kind = m.group(1) or m.group(4), (m.group(2) or m.group(3)).lower()
                    words = _distinctive(sent)
                    for rkey, rc, label, kinds in rows:
                        same = label & words
                        if len(same) >= 3 and len(same) / len(label) >= 0.5 and kind in kinds and float(value) not in kinds[kind]:
                            g = found.setdefault((rkey, rc.label()), {"kind": kind, "table": kinds[kind], "says": {}, "from": []})
                            g["says"][float(value)] = re.search(r"\d+(?:\.\d+)?", sent[m.start():m.end()]).group(0) + "%"
                            g["from"].append((vp.page.url, vp.snapshot_id))
    out = []
    for (rkey, label), g in found.items():
        table = next(iter(g["table"].values()))
        rp = view.pages[rkey]
        out.append({
            "topic": f"{re.sub(r'[*]+', '', label.split(' | ')[0]).strip()} {g['kind'].upper()}",
            "values": [table] + list(g["says"].values()),
            "sources": [(rp.page.url, rp.snapshot_id)] + list(dict.fromkeys(g["from"])),
            "resolution": table,
            "reason": f"the rate table on {rp.page.url} is the current official rate; articles quote older or different values",
            "_strength": 1,
        })
    return out


def find_conflicts(view: View, earlier: list = ()) -> list[dict]:
    """Deterministic, no LLM. An LLM reviewer was tried and removed: it treated old statements still
    published on the site as "a different time period" and dropped them, but those stale statements
    are exactly what this method is meant to report (see NOTES.md). `earlier`: the previous views,
    oldest first (an ended offer is known from what its page said before)."""
    found = (table_conflicts(view) + statement_conflicts(view) + article_rate_conflicts(view)
             + ended_offer_conflicts(view, list(earlier)))
    # Most certain first, then the most specific (fewest values): the scorer matches greedily, so a
    # broad entry listed early could take a key entry meant for a narrower one.
    found.sort(key=lambda f: (-f["_strength"], len(f["values"])))
    out = []
    for f in found:
        f = {k: v for k, v in f.items() if not k.startswith("_") or k in ("_count", "_phrases", "_streaks")}
        phrases, streaks = f.pop("_phrases", {}), f.pop("_streaks", [])
        short = _short_count if f.pop("_count", False) else (lambda v: v)   # "8 offices (8 office locations)"
        f["values"] = [with_phrases(short(v), phrases.get(v)) for v in f["values"]]
        streak_values = [s for s in streaks if s.split()[0] not in phrases]        # "third (For a third year in a row, ...)"
        f["resolution"] = with_phrases(short(f["resolution"]), phrases.get(f["resolution"]))
        out.append(f)
        if streak_values:
            # "the Nth time" on one page against "the Nth year in a row" on the same page is a second contradiction, apart
            # from the pages that give another count: one entry can match only one key entry
            out.append({**f, "topic": f"{f['topic']} (a streak claim)", "values": [f["resolution"]] + streak_values})
    # The same stale copy behind two product lines (purchase and refinance tables) is one contradiction:
    # one entry, both topics, so whichever wording a reader uses finds it.
    merged: dict = {}
    for f in out:
        k = (tuple(f["values"]), f["resolution"])
        if k in merged:
            m = merged[k]
            m["topic"] = f"{m['topic']}; {f['topic']}"
            m["sources"] += [s for s in f["sources"] if s not in m["sources"]]
        else:
            merged[k] = f
    return list(merged.values())


def with_phrases(value: str, phrases) -> str:
    """Output only: a value followed by the site's own wording of it, so a key written either way matches."""
    extra = [p for p in phrases or () if p.lower() != value.lower()]
    return f"{value} ({'; '.join(extra)})" if extra else value


def _short_count(value: str) -> str:
    """A count with a multi-word noun also gets its head noun's plural, so either wording matches."""
    m = re.match(r"(\d[\d,]*\+?) ([a-z]+)( [a-z].*)$", value)
    if not m:
        return value
    head = m.group(2)
    plural = head + ("es" if re.search(r"(s|x|ch|sh)$", head) else "s")
    return f"{m.group(1)} {plural} ({value})"
