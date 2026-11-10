"""Search one date-pinned view and pick the evidence an answer may use.

Order of operations for a question:
  1. BM25 over the view's chunks (stdlib, inverted index built at ingest).
  2. Drop chunks that contradict the question's qualifiers (term, model year, balance tier,
     jumbo/business) and offers whose stated end date has passed.
  3. Suppress stale duplicates: when two sources state the same fact with different values, keep the
     one with the newest effective/publish date (then the more authoritative page). The stale value
     never reaches the answer step, so it cannot be repeated.
  4. At most two chunks per page, top-k.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from ingest import KIND_RANK, MONTH_RE, STATUS_RANK, Chunk, View
from places import CITY_COMMA, CURRENCY, FOREIGN_SET, GIVEN_NAMES, NATIONAL_BANK, PLACE_RE, PROVINCE_SET

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMWORD_RE = re.compile(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ](one|two|three|four|five|six|seven|eight|nine)\b|\b(" + "|".join(NUM_WORDS) + r")\b", re.I)
STOP = set("""a an the of for on in at to and or is are was were be been do does did can could would should
what whats which who whom how when where why your you our my me i we it its this that these those with
from by as any all there their them they about tell give get please provide current
currently now right today know like want need would there's many much have has had still offer""".split())
# General banking synonyms: members say "mortgage", rate tables usually say "home loans".
SYNONYMS = {"mortgage": ["home", "loan"], "cd": ["certificate"], "car": ["auto", "vehicle"],
            "vehicle": ["auto"], "heloc": ["home", "equity", "line"],
            # standard US banking equivalents: a cashier's check is also called an official check; a wire abroad is international
            "cashier": ["official"], "foreign": ["international"], "overseas": ["international"]}
# How-to only: the verbs for editing something are interchangeable ("update", "change", "edit" ...). Never
# for a fee, rate or count question, where these common words would pull in unrelated amounts.
EDIT_VERBS = ("update", "change", "edit", "modify", "correct", "amend")
HOWTO_SYNONYMS = {v: [w for w in EDIT_VERBS if w != v] for v in EDIT_VERBS}
# Standard written abbreviations ("my email info", "acct no."): read as the full word.
ABBREVIATIONS = {"info": "information", "acct": "account", "appt": "appointment", "amt": "amount",
                 "bal": "balance", "dept": "department", "intl": "international"}
GENERIC = STOP | {"month", "months", "year", "years", "rate", "rates", "interest", "new", "standard",
                  "annual", "percentage", "fixed"}
# capitalised words before "branch" that are not a branch's name ("Saturday branch hours")
BRANCH_NOT_NAMES = {"the", "your", "a", "our", "lobby", "drive-up", "drive", "monday", "tuesday",
                    "wednesday", "thursday", "friday", "saturday", "sunday"}
# The site's own name is never written here: Store derives it from the crawl (page titles and hosts) and
# KnowledgeSystem.ingest hands it to use_site_words before indexing.
SITE_WORDS: set = set()
_BASE_STOP, _BASE_GENERIC, _BASE_BRANCH = frozenset(STOP), frozenset(GENERIC), frozenset(BRANCH_NOT_NAMES)
# Words the site only ever writes capitalised mid-sentence ("Elmwood"), never in lower case ("nearest"):
# its proper names, derived from the crawl at ingest. They let "the elmwood branch" name a branch.
PROPER_NAMES: set = set()


def proper_names(chunks) -> set:
    cap, low = Counter(), Counter()
    for c in chunks:
        for sent in re.split(r"(?<=[.!?])\s+|\n|\s[-–—|•]\s", c.text):
            for w in re.findall(r"[A-Za-z][\w'’-]*", sent)[1:]:
                (cap if w[0].isupper() else low)[w.lower()] += 1
    return {w for w, n in cap.items() if n >= 2 and not low[w] and len(w) > 2
            and w not in STOP and w not in BRANCH_NOT_NAMES}


def use_proper_names(names) -> None:
    PROPER_NAMES.clear()
    PROPER_NAMES.update(names)


def use_site_words(words) -> None:
    """The site's name is not a search word, not a branch name, and may sit in front of a counted noun or
    a product list ("how many <Name> branches", "what does <Name> offer"). Replaces any earlier site's
    words; the sets are updated in place because other modules hold references to them."""
    global COUNT_Q, COUNT_CUE, LIST_Q
    words = {w.lower() for w in words}
    SITE_WORDS.clear()
    SITE_WORDS.update(words)
    extra = words | {w + "s" for w in words}            # "Name's" tokenizes as "names"
    for target, base, add in ((STOP, _BASE_STOP, extra), (GENERIC, _BASE_GENERIC, extra),
                              (BRANCH_NOT_NAMES, _BASE_BRANCH, words)):
        target.clear()
        target.update(base | add)
    alt = "|".join(sorted(map(re.escape, words)))
    COUNT_Q = re.compile(rf"\bhow many\s+(?:(?:{alt})\s+)?([a-z-]+)" if alt else r"\bhow many\s+([a-z-]+)", re.I)
    opt = rf"(?:(?:{alt})(?:['’]s)?\s+)?" if alt else ""
    COUNT_CUE = re.compile(rf"(?:^|\bthe\s+|\btotal\s+)number of\s+{opt}([a-z-]+)", re.I)
    who, does = (f"you|{alt}", f"do you|does (?:{alt})") if alt else ("you", "do you")
    LIST_Q = re.compile(rf"\bwhat (?:\w+ ){{0,3}}do (?:{who}) (?:offer|have)\b|"
                        rf"\bwhich (?:\w+ ){{0,3}}(?:{does}) offer\b|\bwhat (?:kinds|types) of\b|{LIST_ASK}", re.I)


LIST_ASK = r"^\s*(?:please\s+)?(?:list|show me all)\b"     # "list your card rates"
OPTIONS_Q = re.compile(r"\boptions\b", re.I)
PRODUCT_HEADS = r"loans?|mortgages?|certificates?|cds?|accounts?|cards?|savings|checking|visa|mastercard|discover|amex|heloc|line"

RATE_Q = re.compile(r"\b(rates?|apr|apy|interest|dividends?|yield)\b", re.I)
# "what does a savings account pay?" asks about a rate without the word (never "…pay in fees")
DEPOSIT_PAYS = re.compile(r"\bwhat\s+(?:does|did|do|will|would)\s+(?:a|an|the|my|your)?\s*(?:[\w-]+\s+){0,3}"
                          r"(?:savings|checking|certificates?|cds?|money\s+market|accounts?)\s+(?:account\s+)?(?:pay|earn|yield)s?\b|"
                          r"\bwhat\s+(?:is|was|are|were)\s+(?:a|an|the|my|your)?\s*(?:[\w-]+\s+){0,4}"
                          r"(?:savings|checking|certificates?|cds?|money\s+market|accounts?|specials?)\s+(?:paying|earning|yielding)\b", re.I)
# "does any savings tier pay 0.45%?" asks about a rate without the word (cash back and points are not rates)
PLAIN_DESCRIPTORS = {"regular", "basic", "standard", "plain", "normal", "ordinary", "simple", "everyday", "share", "personal"}
PCT_PAID = re.compile(r"\b(?:pay|pays|paying|paid|earn|earns|earning)\s+(?:a\s+|an\s+)?\d+(?:\.\d+)?\s*(?:%|percent\b)"
                      r"(?!.*\b(?:cash\s*back|reward(?:s)?|points)\b)", re.I)
FEE_Q = re.compile(r"\b(fees?|cost|charge[sd]?|price)\b|\bhow much (is|are|was|were|does|do|did|for|will)\b|"
                   r"\bhow much to (?:wire|send|mail|replace|rush|stop|cash|certify|transfer)\b", re.I)
HOURS_Q = re.compile(r"\bhours?\b", re.I)
COUNT_Q = re.compile(r"\bhow many\s+([a-z-]+)", re.I)          # rebuilt with the site's name: use_site_words
# Hours and counts asked without "hours" or "how many": "office hrs pls", "when is the walk-up window open?",
# "total number of offices". Used only when nothing else claimed the question.
HOURS_CUE = re.compile(r"\bhrs\b|\bwhat time\b(?!\s+(?:frame|period|limit)s?\b)|^\s*when (?:is|are)\b(?=.*\b(?:lobby|"
                       r"lobbies|branch(?:es)?|drive[- ]?(?:up|thru|through)s?|offices?|locations?|you)\b).*\bopen\b", re.I)
# Rate tables say "adjustable-rate", members say "ARM"; only for rate questions, so "what is a 5/1 ARM?"
# still reaches the articles that explain ARMs.
ARM_RE = re.compile(r"\bARMs?\b")
# A product's name starts after the last determiner or possessive ("the sheet says its certificate"), and
# words that judge a product ("the best certificate", "a good rate") never name one.
DETERMINERS = {"the", "a", "an", "its", "your", "our", "my", "their", "his", "her", "this", "that", "these", "those",
               "any", "each", "every", "some"}
EVALUATIVE = {"best", "better", "top", "good", "great", "cheapest", "cheaper", "lowest", "lower", "highest", "higher",
              "competitive", "excellent", "decent", "attractive", "favorable", "nice"}
# Question-frame words are never part of a product's name ("where can I find checking rates").
FRAME_WORDS = {"find", "get", "see", "use", "earn", "pay", "need", "open", "apply", "compare", "show", "list",
               "kind", "kinds", "type", "types", "available", "charge", "much", "can", "online"}
COUNT_CUE = re.compile(r"(?:^|\bthe\s+|\btotal\s+)number of\s+([a-z-]+)", re.I)
WHEN_Q = re.compile(r"^\s*when\b|what date|\bby when\b", re.I)
LIST_Q = re.compile(r"\bwhat (?:\w+ ){0,3}do (?:you) (?:offer|have)\b|"
                    r"\bwhich (?:\w+ ){0,3}(?:do you) offer\b|\bwhat (?:kinds|types) of\b|" + LIST_ASK, re.I)
RATE_EV = re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:APR|APY|annual percentage|fixed|variable|interest|dividend)|"
                     r"(?:APR|APY|rate of|rates? (?:as low as|start(?:ing)? at|from)|dividend rate)\s*(?:is|of|:)?\s*"
                     r"\d+(?:\.\d+)?\s*%", re.I)
ADDRESS_Q = re.compile(r"\baddress\b|\bwhere is\b|\blocated\b", re.I)
# The member editing their own address is a how-to question, not "where is it": an edit verb on "my/our
# (mailing...) address", or "change of address", and no branch, office or other place named.
OWN_ADDRESS = re.compile(r"\b(?:updat|chang|edit|correct)\w*\s+(?:my|our)\s+(?:(?:home|mailing|email|e-mail|street|"
                         r"billing|physical|contact)\s+)?address\b|\bchange of address\b", re.I)
PLACE_WORDS = re.compile(r"\b(?:branch|branches|location|locations|office|offices|lobby|lobbies)\b", re.I)
ADDRESS_EV = re.compile(r"\b\d{2,6}\s+(?:[A-Z][\w.'’-]*\s+){1,4}(?:St|Street|Ln|Lane|Ave|Avenue|Dr|Drive|Rd|Road|"
                        r"Blvd|Way|Ct|Pl|Pkwy|Hwy|Highway|Square)\b\.?")
DATE_EV = re.compile(rf"\b(?:{MONTH_RE})\s+\d{{1,2}},\s+20\d\d\b")
FEE_EV = re.compile(r"\$\s?\d")
HOURS_EV = re.compile(r"\b\d{1,2}(?::\d{2})?\s*[ap]\.?\s?m\b|\bnoon\b", re.I)
TERM_RE = re.compile(r"\b(\d{1,3})\s*-?\s*(month|mo|year|yr)s?\b", re.I)


def terms_in(text: str) -> set:
    """Loan and deposit terms, in months: a "5-year" loan is the "60 Months" row."""
    return {(int(n) * (12 if u.lower() in ("year", "yr") else 1), "month") for n, u in TERM_RE.findall(text)}


def native_terms(text: str) -> set:
    return {(int(n), "year" if u.lower() in ("year", "yr") else "month") for n, u in TERM_RE.findall(text)}


# Head words every product line shares: they never say which line a row belongs to.
GENERIC_HEADS = {"loan", "loans", "account", "accounts", "card", "cards", "line", "home"}
LINE_PHRASES = {"mortgage": ("home loan",)}        # standard banking usage: a mortgage is a home loan


def names_line(low: str, q) -> bool:
    """Does this (lower-case) text name the product line the question asks about: one of its descriptor
    words, or a specific product word ("certificate" for a CD)? A term converted from years to months
    counts only on such a row, so a "5-year CD" never matches an auto loan's 60-month row."""
    words = {w for w in set(q.descriptors) | q.category if w not in GENERIC_HEADS}
    words |= {s for w in list(words) for s in SYNONYMS.get(w, ()) if s not in GENERIC_HEADS}
    words |= {p for w in q.category for p in LINE_PHRASES.get(w.rstrip("s"), ())}
    return any(re.search(rf"\b{re.escape(w.rstrip('s'))}s?\b", low) for w in words)
YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d\d)\s*[-–]\s*(\d{2}(?:\d{2})?)\b(?!-)")
AND_OLDER_RE = re.compile(r"\b((?:19|20)\d\d)\s+(?:model\s+years?\s*)?\(?\s*(?:and|or|&)\s+(?:older|earlier|prior|before)\b\)?", re.I)
AMT_RANGE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*[-–]\s*\$?\s?([\d,]+(?:\.\d+)?)")
AMT_PLUS_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:\+|and (?:up|above|over))", re.I)
AMT_LESS_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s+or less", re.I)
# "a balance under $X" is a balance just below X: the tier that ends below X, not the one starting at X
AMT_UNDER_RE = re.compile(r"\b(?:under|less\s+than|below)\s+\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?\b", re.I)
# ...and "over $X" is just above X
AMT_OVER_RE = re.compile(r"\b(?:over|above|more\s+than|greater\s+than)\s+\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?\b", re.I)
# an amount per period, a payment, an income or a down payment is not a balance or loan amount
AMT_NOT_BALANCE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?\b(?=\s*(?:(?:a|per|each|every)\s+(?:month|year|week|paycheck)\b"
                             r"|(?:monthly|annually|yearly|weekly)\b|(?:monthly\s+)?payments?\b|(?:in\s+)?(?:income|salary)\b|down\b))"
                             r"|\b(?:earn|earns|earning|make|makes|making|income\s+of|salary\s+of)\s+(?:about\s+|around\s+)?\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?\b",
                             re.I)
# a down payment against a price leaves the loan amount: "$250,000 home with $50,000 down", "20% down"
DOWN_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?\s+(?:as\s+(?:a\s+|the\s+)?)?down\b|"
                     r"\bdown\s+payment\s+(?:of\s+|is\s+|will\s+be\s+)?(?:about\s+|around\s+)?\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?\b", re.I)
PCT_DOWN = re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s*(?:%|percent)\s+down\b|\bdown\s+payment\s+of\s+(\d{1,2}(?:\.\d+)?)\s*(?:%|percent)", re.I)
# ...but an amount stated as the loan itself is already the loan amount ("a $240,000 mortgage", "borrow $X")
LOAN_AMT = re.compile(r"\$\s?[\d,]+(?:\.\d+)?\s*[kK]?\s+(?:(?:home|auto|car|boat|personal)\s+)?(?:loan|mortgage|heloc)\b|"
                      r"\b(?:loan|mortgage|borrow\w*|financ\w*)\s+(?:of\s+|amount\s+of\s+)?(?:about\s+|around\s+)?\$", re.I)
# a table cell under a "Min." header holding a bare dollar amount: the least a row's rate applies to
MIN_HEADER = re.compile(r"\bmin(?:imum|\.)?\b", re.I)
AMT_CELL = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
SOFT_QUALIFIERS = ("jumbo", "business", "traditional", "roth", "ira", "refinance", "hsa", "health")
# A loan-type spec ("7/1", "10/6" adjustable rates): two small numbers, never part of a date.
SPEC_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])")
# "the credit union" names the institution asking about, like the site's own name: never a product or search word.
INSTITUTION = re.compile(r"\b(?:the|your|this|our|my|a)\s+credit\s+unions?(?:['’]s)?\b", re.I)
# "Who can join?", "do I qualify for membership?", "requirements to join": asking who may join asks who is
# eligible, so the search also looks for "eligible" (and "join"). Only for open questions about joining or membership;
# "who can I call" and "who can help" ask for a contact, not eligibility.
ELIGIBILITY_Q = re.compile(
    r"\bwho(?:['’]s|\s+is|\s+are)?\s+(?:can|could|may|allowed\s+to|able\s+to|eligible|qualif(?:y|ies))\b"
    r"(?!\s+(?:i|we|you|me|us)\b)(?!\s+(?:help|assist|answer|explain|tell|talk|speak|contact|call|reach)\b)"
    r"|\b(?:can|could|may)\s+(?:anyone|anybody|everyone|everybody)\b"
    r"|\b(?:am\s+i|are\s+we|is\s+my\s+\w+)\s+(?:able|allowed|eligible)\s+to\b"
    r"|\bqualif(?:y|ies|ied|ication|ications)\b|\beligibility\b"
    r"|\b(?:requirements?|criteria)\s+(?:to|for)\b"
    r"|\bopen\s+to\s+(?:anyone|anybody|everyone|everybody|all|the\s+public)\b|\bwho\b[^?]*\bopen\s+to\b"
    r"|\bwhat\s+does\s+it\s+take\s+to\b", re.I)
MEMBERSHIP_Q = re.compile(r"\b(?:join(?:ing)?|membership|(?:be|become|becoming)\s+an?\s+(?:\w+\s+)?member)\b", re.I)
# An introductory (teaser) rate is its own kind of rate: "the intro APR on X" is not X's standard APR, and an
# intro offer that has ended leaves nothing to quote.
RATE_KINDS = r"intro(?:ductory)?|teaser"
KIND_RATE = re.compile(rf"\b(?:{RATE_KINDS})\s+(?:(?:balance[- ]transfer|dividend|interest|annual percentage)\s+)?"
                       r"(?:apr|apy|rates?|yield|dividend)\b", re.I)
# ...and so is a rate stated "for the first N months" (a rate word just before it, so a payment holiday is not)
KIND_EV = re.compile(rf"\b(?:{RATE_KINDS})\b|\b(?:rates?|ap[ry]s?|yields?|dividends?)\b[^.;|\n]{{0,30}}?"
                     r"\bfor\s+(?:the\s+)?(?:1st|first)\s+\d+\s+months?\b", re.I)
# ...unless the member names the offer itself ("the intro rate on the certificate special"): then the site's
# "special" or "promo" on that product's own row or heading is the offer meant. Only "special" as a noun
# counts, not "a special intro rate".
OFFER_NOUN = re.compile(r"\b(?:specials?|promos?|promotions?)\b"
                        r"(?!\s+(?:intro(?:ductory)?|teaser|rates?|ap[ry]s?|yields?|dividends?|interest)\b)", re.I)
OFFER_LABEL = re.compile(r"\b(?:specials?|promos?|promotions?)\b", re.I)
MONEY_MARKET = re.compile(r"\bmoney\s+markets?\b", re.I)
KIND_RATE_EV = re.compile(rf"\d+(?:\.\d+)?\s*%\s*\d?\s*(?:(?:{RATE_KINDS})\s+)?(?:APR|APY|annual percentage|dividend)|"
                          rf"(?:{RATE_KINDS})\s+(?:\w+\s+){{0,3}}(?:rate|apr|apy)\b[^.]{{0,40}}?\d+(?:\.\d+)?\s*%", re.I)
# "not the jumbo", "not for my business", "except for business": a line ruled out ("…or not for a jumbo" asks about it)
NEGATED_QUALIFIER = re.compile(r"(?<!\bor\s)\b(?:not|non|excluding|except|other than|besides|without|rather than|instead of)"
                               r"[\s-]+(?:for\s+(?:(?:the|a|an|any|my|our)\s+)?|(?:the|a|an|any)\s+)?(%s)\b"
                               % "|".join(SOFT_QUALIFIERS), re.I)
# After a question ("...? I keep my savings with you"): an aside that asks something too, opens a second request,
# tells what the member has in mind, or tells about the member.
ASIDE_ASKS = re.compile(r"\W*(?:also|and|or|plus|what|which|how|when|where|is|are|do|does|can|could|vs|versus)\b")
ABOUT_ME = re.compile(r"\W*(?:i|we|my|our|i['’](?:ve|m|d|ll)|we['’](?:ve|re|d))\b")
ASIDE_INTENT = re.compile(
    r"\W*(?:i['’]?m\s+(?:looking|thinking|interested|considering|asking|planning|hoping|after|buying|financing|getting)"
    r"|i\s+(?:want|need|mean|would|wanna|plan|prefer|am\s+(?:looking|buying|getting))|i['’]d\s+(?:want|like|prefer|need|be)"
    r"|we\s+(?:want|need|would|are\s+(?:looking|buying))|we['’](?:d|re)\s+(?:want|like|looking|buying|thinking))\b")
# a line the site may write either way: an "HSA" row says "Health Savings Account"
LINE_ALIAS = {"hsa": r"\bhealth savings\b", "health": r"\bhsa\b"}


def normalize_numbers(text: str) -> str:
    def repl(m):
        if m.group(1):
            return str(NUM_WORDS[m.group(1).lower()] + NUM_WORDS[m.group(2).lower()])
        return str(NUM_WORDS[m.group(3).lower()])
    return _NUMWORD_RE.sub(repl, text)


def _stem(w: str) -> str:
    """Crude suffix stripping so 'closure'/'closing'/'close' and 'branches'/'branch' meet."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes")):
        return w[:-2]
    for suf in ("ings", "ing", "ures", "ure", "ed", "es", "s", "e"):
        if len(w) - len(suf) >= 4 and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)]
    if len(w) == 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


_SYN = {_stem(k): [_stem(x) for x in v] for k, v in SYNONYMS.items()}
_HOWTO_SYN = {_stem(k): [_stem(x) for x in v] for k, v in HOWTO_SYNONYMS.items()}
_SYN.update({_stem(k): [_stem(v)] for k, v in ABBREVIATIONS.items()})


def _num_token(t: str) -> str:
    return t.rstrip("0").rstrip(".") if "." in t else t


def tokenize(text: str) -> list[str]:
    text = normalize_numbers(text.lower()).replace(",", "")
    text = re.sub(r"(?<=[a-z])\$(?=[a-z])", "s", text)        # a brand spelled with "$" for "s"
    out = []
    for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text):
        if t[0].isdigit():
            out.append(_num_token(t))
        elif t not in STOP and len(t) > 1:
            out.append(_stem(t))
    return out


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def model_year_ranges(text: str) -> list[tuple[int, int]]:
    if not re.search(r"model[- ]year|vehicle", text, re.I):
        return []
    out = []
    for a, b in YEAR_RANGE_RE.findall(text):
        lo, hi = int(a), int(b)
        if hi < 100:
            hi += lo // 100 * 100
        out.append((min(lo, hi), max(lo, hi)))
    out += [(0, int(y)) for y in AND_OLDER_RE.findall(text)]
    return out


def amount_ranges(text: str) -> list[tuple[float, float]]:
    out = [(_f(a), _f(b)) for a, b in AMT_RANGE_RE.findall(text)]
    out += [(_f(a), math.inf) for a in AMT_PLUS_RE.findall(text)]
    out += [(0.0, _f(a)) for a in AMT_LESS_RE.findall(text)]
    return out


@dataclass
class Query:
    raw: str
    tokens: list[str]
    terms: set = field(default_factory=set)          # {(60, 'month')}
    years: set = field(default_factory=set)          # model years, e.g. {2025}
    amounts: list = field(default_factory=list)      # dollar figures in the question
    qualifiers: set = field(default_factory=set)     # jumbo/business/... explicitly asked for
    attribute: str | None = None                     # rate | fee | hours | count | when
    count_noun: str | None = None                    # "office" for "how many offices"
    entity_is_branch: bool = False
    entity: str | None = None                        # named branch, e.g. "elm"
    place: str | None = None                         # a branch the site has, named in a rate or fee question
    descriptors: set = field(default_factory=set)    # words naming the product, e.g. {"tractor"}
    category: set = field(default_factory=set)       # the product-line word itself, e.g. {"savings"}
    list_intent: bool = False                        # "what cards do you offer": show every product line
    subject: list = field(default_factory=list)      # rare question words a fee/count answer must contain
    native_terms: set = field(default_factory=set)   # terms in the question's own unit, {(5, 'year')}
    excluded: set = field(default_factory=set)       # lines the question rules out ("not the jumbo")
    text: str = ""                                   # the question without the lines it rules out
    specs: set = field(default_factory=set)          # loan-type specs ("7/1"), each must be on the row
    expansion: list = field(default_factory=list)    # search words the wording implies ("eligible" for "who can join?")
    rate_kind: str | None = None                     # "intro": every rate asked about is an introductory rate;
                                                     # "offer": ...of a special or promo the question names

    @property
    def numbers(self) -> set:
        return {n for n, _ in self.terms} | self.years | {int(a) for a in self.amounts}


def parse_query(question: str) -> Query:
    raw = question                       # the model still sees the member's own words
    if RATE_Q.search(question) and ARM_RE.search(question):
        question = ARM_RE.sub("adjustable-rate mortgage", question)
    question = INSTITUTION.sub(" ", question)
    excluded = {m.group(1).lower() for m in NEGATED_QUALIFIER.finditer(question)}
    question = NEGATED_QUALIFIER.sub(" ", question)     # "not the jumbo": the standard product is asked about
    norm = normalize_numbers(question)
    low = norm.lower()
    asked = re.match(r".*\?", question, re.S)           # "...? Asking for a friend." searches the question only
    q = Query(raw, tokenize(asked.group(0) if asked else question))
    q.text, q.excluded = question, excluded
    # "...? I've been a member for 10 years": an aside after a question that has words of its own. Its product
    # words never count, and when it only tells about the member (and shares no word with the question) neither
    # do its terms, model years or product lines. An aside that asks something too, says what the member has
    # in mind ("I'm looking at the 24-month term", "Five years"), or shares a word ("...auto loan? My car is a
    # 2025") still counts; a bare question ("What's the rate? It's for a certificate") reads it all.
    context = False
    head = normalize_numbers(asked.group(0)).lower() if asked else low
    tail = low[len(head):] if asked and low.startswith(head) else ""
    product_text = low
    if tail.strip() and not ASIDE_ASKS.match(tail) and not RATE_Q.search(tail) and not FEE_Q.search(tail):
        quiet = {_stem(w) for w in GENERIC | FRAME_WORDS} | set(CHAT_FILLERS)
        if any(t not in quiet and not t[0].isdigit() and not RATE_Q.fullmatch(t) and not FEE_Q.search(t) for t in q.tokens):
            product_text = head
            def own(text):
                ws = set(tokenize(text))
                ws |= {s for w in list(ws) for s in _SYN.get(w, ())}
                return {w for w in ws if not w[0].isdigit()} - quiet - {_stem(w) for w in GENERIC_HEADS} - STOP
            context = bool(ABOUT_ME.match(tail)) and not ASIDE_INTENT.match(tail) and not (own(tail) & own(head))
    if tail.strip() and product_text == low:
        q.tokens = tokenize(question)     # the aside is part of the request ("What's the rate? It's for a CD"): search it too
    pinned = product_text == head and terms_in(head)          # a term in the question itself is never overridden
    q.terms = terms_in(head) if pinned else terms_in(head if context else low)
    q.native_terms = native_terms(head) if pinned else native_terms(head if context else low)
    if re.search(r"model|vehicle|\bcar\b|truck", low):     # model years only, never "as of March 2031"
        q.years = {int(m.group(1)) for m in re.finditer(r"\b((?:19|20)\d\d)\b", norm)
                   if not re.search(rf"(?:{MONTH_RE})\s*(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s*)?$", norm[:m.start()])}
    if q.years and tail.strip():
        mine = {y for y in q.years if str(y) in head}
        q.years = mine if (mine and product_text == head) or context else q.years
    q.amounts = [_f(a) * (1000 if k else 1) for a, k in re.findall(r"\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?\b", norm)]
    other = {_f(m.group(1) or m.group(3)) * (1000 if (m.group(2) or m.group(4)) else 1) for m in AMT_NOT_BALANCE.finditer(norm)}
    downs = [_f(m.group(1) or m.group(3)) * (1000 if (m.group(2) or m.group(4)) else 1) for m in DOWN_RE.finditer(norm)]
    q.amounts = [a for a in q.amounts if a not in other and a not in downs]
    pct = PCT_DOWN.search(norm)
    if (downs or pct) and len(set(q.amounts)) == 1 and q.amounts[0] > sum(downs) and not LOAN_AMT.search(norm):
        price = q.amounts[0]                 # one price and a down payment: the amount borrowed
        q.amounts = [price - sum(downs)] if downs else [round(price * (1 - _f(pct.group(1) or pct.group(2)) / 100), 2)]
    below = {_f(a) * (1000 if k else 1) for a, k in AMT_UNDER_RE.findall(norm)}
    above = {_f(a) * (1000 if k else 1) for a, k in AMT_OVER_RE.findall(norm)}
    q.amounts = [a - 0.01 if a in below else a + 0.01 if a in above else a for a in q.amounts]
    q.qualifiers = {w for w in SOFT_QUALIFIERS if re.search(rf"\b{w}\b", head if context else low)}
    q.excluded -= q.qualifiers          # a line named elsewhere without a negation is still asked about
    if m := COUNT_Q.search(low):
        q.attribute, q.count_noun = "count", _stem(m.group(1))
    elif RATE_Q.search(low) or PCT_PAID.search(low) or (DEPOSIT_PAYS.search(low) and not FEE_Q.search(low)):
        q.attribute = "rate"
    elif FEE_Q.search(low):
        q.attribute = "fee"
    elif HOURS_Q.search(low):
        q.attribute = "hours"
    elif WHEN_Q.search(low):
        q.attribute = "when"
    elif ADDRESS_Q.search(low) and not (OWN_ADDRESS.search(low) and not PLACE_WORDS.search(low)):
        q.attribute = "address"
    if q.attribute == "rate":
        q.specs = {f"{a}/{b}" for a, b in SPEC_RE.findall(question)}
    if q.attribute == "fee":            # "within N days of opening": a condition of the fee, not a term
        q.terms, q.native_terms = set(), set()
    # "what are my options": a list, unless a term or model year already pins one row
    q.list_intent = bool(LIST_Q.search(question)) or (OPTIONS_Q.search(question) is not None
                                                      and not (q.terms or q.years))
    syn = {**_SYN, **_HOWTO_SYN} if q.attribute in (None, "when") else _SYN
    if q.attribute in ("rate", "fee"):
        syn = {**syn, _stem("hsa"): [_stem("health")]}     # a rate row names the line in full
    for t in list(q.tokens):
        q.tokens += [x for x in syn.get(t, []) if x not in q.tokens]
    m = re.search(r"\b((?:[A-Z][\w'’.-]*\s+){0,3}[A-Z][\w'’.-]*)(?:,\s+[A-Z][\w'’.-]*)?\s+[Bb]ranch\b", question)
    if m:                                       # "the Harbor, Ohio branch": the city before the state
        # "<Name>'s Elm branch": the site name's possessive is not the branch name
        words = [re.sub(r"['’]s$", "", w) for w in m.group(1).split()]
        words = [w for w in words if w.strip(".,'’").lower() not in BRANCH_NOT_NAMES]
        q.entity = words[0].strip(".,'’").lower() if words else None
        q.entity_is_branch = q.entity is not None
    if not q.entity and PROPER_NAMES:          # "the elmwood branch / location / office", any case
        m = re.search(r"((?:[a-z][\w'’-]*\s+){1,3})(?:branch|location|office)\b", question.lower())
        name = next((w for w in m.group(1).split() if w in PROPER_NAMES), None) if m else None
        if name:
            q.entity, q.entity_is_branch = name, True
    if not q.entity and q.attribute in ("rate", "fee") and (m := PLACE_NAMED.search(question)) \
            and m.group(1).lower() not in BRANCH_NOT_NAMES | NOT_PLACE_NAMES:
        q.entity, q.entity_is_branch = m.group(1).lower(), True     # "at your Rivertown location": a branch
    if q.attribute is None and (m := COUNT_CUE.search(low)) and m.group(1) not in STOP:
        q.attribute, q.count_noun = "count", _stem(m.group(1))
        q.tokens = [t for t in q.tokens if t not in ("number", "total")]
    # a named branch ("the elm-street branch", any case) keeps its own question type: "what time does it close?"
    named = [w for w in re.findall(r"\b([a-z][\w'’-]*)\s+branch\b", low) if w not in BRANCH_NOT_NAMES | STOP]
    if q.attribute in (None, "when") and not q.entity and not named and HOURS_CUE.search(low):
        q.attribute = "hours"
        if "hour" not in q.tokens:
            q.tokens.append("hour")
    if q.attribute is None and ELIGIBILITY_Q.search(low) and MEMBERSHIP_Q.search(low):
        # who may become a member = who is eligible to join
        q.expansion = [t for t in (_stem("eligible"), _stem("join")) if t not in q.tokens]
        q.tokens += q.expansion
    if q.attribute == "rate":
        kinds = [m.span() for m in KIND_RATE.finditer(low)]
        if kinds and all(any(a <= s and e <= b for a, b in kinds) for s, e in (m.span() for m in RATE_Q.finditer(low))):
            q.rate_kind = "offer" if OFFER_NOUN.search(low) else "intro"
            if _stem("intro") in q.tokens and _stem("introductory") not in q.tokens:
                q.tokens.append(_stem("introductory"))     # "intro" is short for "introductory"
    if q.attribute in ("rate", "fee") and (q.qualifiers & {"hsa"} or re.search(r"\bhealth savings\b", low)):
        q.qualifiers |= {"hsa", "health"}                   # one line, written either way
    if q.attribute in ("rate", "fee"):
        if q.attribute == "fee" and WIRE.search(low):
            product_text = WIRE_ENDS.sub(" ", product_text)    # "from my account to my son's bank account": not the product
        toks = re.findall(r"[a-z]+", product_text.replace("-", " "))
        for k, t in enumerate(toks):
            if re.fullmatch(PRODUCT_HEADS, t):
                win = toks[max(0, k - 2):k]
                cut = [j for j, w in enumerate(win) if w in DETERMINERS]   # "the sheet says its certificate"
                win = win[cut[-1] + 1:] if cut else win
                q.descriptors |= {w for w in win if w not in GENERIC and w not in EVALUATIVE and len(w) > 2
                                  and w not in FRAME_WORDS and not re.fullmatch(PRODUCT_HEADS, w)
                                  and not RATE_Q.fullmatch(w)}       # "what APY does savings earn": APY is no product
                q.category.add(t)
    if q.rate_kind and not q.category and MONEY_MARKET.search(low):
        # an intro or special rate on "the money market": that account is the product asked about
        q.category, q.descriptors = {"market"}, set(q.descriptors) | {"money"}
    return q


def count_re(noun: str):
    """'12 offices', '350 members' — a count, never a year like 'In 2025,'."""
    return re.compile(rf"(?<![\d,.$])(?!(?:19|20)\d\d\b)\d[\d,]*\+?\s+(?:[a-z’'-]+\s+){{0,2}}{noun}")


def qualify(c: Chunk, q: Query) -> tuple[str, float]:
    """'mismatch' if the chunk is about a different term/model year/tier than asked; else a boost."""
    text = c.index_text if c.is_row else c.text
    low = text.lower()
    factor, matched = 1.0, False
    if q.terms:
        units = {u for _, u in q.native_terms}
        same_unit = {t for t in native_terms(low) if t[1] in units}
        converted = terms_in(low) if names_line(low, q) else set()
        if same_unit or converted:
            if q.native_terms & same_unit or q.terms & converted:
                factor, matched = factor * 1.6, True
            elif c.is_row:
                return "mismatch", 0.0
            else:
                factor *= 0.5
    if q.years:
        ranges = model_year_ranges(text)
        if ranges:
            if all(any(lo <= y <= hi for lo, hi in ranges) for y in q.years):
                factor, matched = factor * 1.6, True
            elif c.is_row:
                return "mismatch", 0.0
    if q.amounts:
        ranges = amount_ranges(c.label() if c.is_row else text)
        if ranges:
            if any(all(lo <= a <= hi for a in q.amounts) for lo, hi in ranges):
                factor, matched = factor * 1.6, True
            elif c.is_row:
                return "mismatch", 0.0
        elif (minimums := row_minimums(c)) and any(a < m for a in q.amounts for m in minimums):
            factor *= 0.6                       # the amount asked is below what this row's rate needs
    for w in SOFT_QUALIFIERS:
        present = re.search(rf"\b{w}\b", low) is not None
        if w in q.qualifiers and not (present or (w in LINE_ALIAS and re.search(LINE_ALIAS[w], low))) and c.is_row:
            return "mismatch", 0.0
        if w not in q.qualifiers and present:
            factor *= 0.6
    return ("match" if matched else "neutral"), factor


def row_minimums(c: Chunk) -> list[float]:
    """A row's minimums: bare dollar cells under a "Min." / "Minimum" header ("Min. to Earn $100,000"). A $0
    minimum says nothing."""
    out = []
    for h, cell in zip(c.header or [], c.cells or []):
        m = AMT_CELL.fullmatch((cell or "").strip())
        if c.is_row and h and MIN_HEADER.search(h) and m and _f(m.group(1)) > 0:
            out.append(_f(m.group(1)))
    return out


def tier_width(c: Chunk, q: Query) -> float | None:
    """How narrow the tier is that a row gives the asked amount: the width of its narrowest matching range, or
    minus its largest minimum at or below the amount (a higher minimum is a narrower tier). None: no tier."""
    if not (c.is_row and q.amounts):
        return None
    spans = [hi - lo for lo, hi in amount_ranges(c.label()) if all(lo <= a <= hi for a in q.amounts)]
    if spans:
        return min(spans)
    floors = [m for m in row_minimums(c) if all(a >= m for a in q.amounts)]
    return -max(floors) if floors else None


def supports(c: Chunk, q: Query) -> bool:
    """Does this chunk actually state the attribute asked for, about the entity/product asked about?"""
    text = c.index_text.lower()
    if q.rate_kind:
        # an introductory rate: the chunk says "intro", and its own title, heading or row label names the
        # product asked about (an intro offer on another product mentioned in passing doesn't count)
        label = ((c.label() if c.is_row else c.section) or "").lower()
        offer = (q.rate_kind == "offer" and OFFER_LABEL.search(label) is not None
                 and (not (q.category or q.descriptors) or names_line(label, q)))   # "9 Month Certificate Special"
        if not offer:
            if not KIND_EV.search(text):
                return False
            own = c.index_text[:max(0, len(c.index_text) - len(c.text))] + " " + (c.label() if c.is_row else "")
            if q.category and not names_line(own.lower(), q):
                return False
    if q.entity and (q.entity not in text or (q.entity_is_branch and "branch" not in text)):
        return False
    # a descriptor is present as itself or as the site's word for it ("car" loan -> "Auto" rows)
    if q.descriptors and not any(d in text or any(s in text for s in SYNONYMS.get(d, ())) for d in q.descriptors):
        return False
    if any(w not in text for w in q.qualifiers & {"jumbo", "business"}):
        return False
    if q.specs and not any(sp in text for sp in q.specs):      # "7/1": only a row naming that loan type
        return False
    if q.subject:                       # fee/count: every rare word asked about (or a synonym) is there
        have = c.tf if c.tf else Counter(tokenize(c.index_text))
        if not all(any(w in have for w in alts) for alts in q.subject):
            return False
    if q.attribute == "fee" and (WIRE.search(q.raw) or SEND_Q.search(q.raw)):
        way = wire_direction(q.raw)
        if way in ("out", "abroad") and re.search(r"\bincoming\b", text):
            return False                # sending a wire is never the incoming fee
        if way == "in" and WIRE.search(q.raw) and not re.search(r"\b(?:incoming|inbound|receiv\w*)\b", text):
            return False                # ...and receiving one is only the incoming fee, from anywhere
        if WIRE.search(q.raw) and (not WIRE_FEE_LINE.search(c.text) or (way == "out" and re.search(r"\binternational\b", text))):
            return False                # the wire's own fee line; a wire within the country is not the international one
    if q.terms:                         # some line must be about the asked term, not just "a term"
        lines = c.text.split("\n") if not c.is_row else [c.index_text]
        named = names_line(text, q)
        def has_term(s):
            return bool(native_terms(s.lower()) & q.native_terms) or (named and bool(terms_in(s.lower()) & q.terms))
        if not any(has_term(s) for s in lines):
            return False
    if q.attribute == "rate":           # published rates live in rate tables or product pages,
        if c.is_row:                    # not in blog posts or press releases
            return any("%" in v for _, v in c.values())
        # an intro offer is often worded its own way ("1.99% intro APR", "2.90%1 APR")
        return c.kind != "article" and bool(RATE_EV.search(c.text) or (q.rate_kind and KIND_RATE_EV.search(c.text)))
    if q.attribute == "count":
        return bool(count_re(q.count_noun).search(c.text.lower()))
    if q.attribute == "address":
        return bool(ADDRESS_EV.search(c.text))
    if q.attribute == "when":
        return bool(DATE_EV.search(c.text))
    if q.attribute == "fee":            # like rates: fees are published in fee schedules and product
        return c.kind != "article" and bool(FEE_EV.search(c.text))   # pages, not blog posts
    if q.attribute == "hours":
        return bool(HOURS_EV.search(c.text))
    return True


class Index:
    """BM25 over every chunk visible in one view."""

    def __init__(self, view: View):
        self.view = view
        self.chunks: list[Chunk] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for vp in view.pages.values():
            for c in vp.page.chunks:
                if not c.tf:
                    toks = tokenize(c.index_text)
                    c.tf, c.length = Counter(toks), len(toks)
                pos = len(self.chunks)
                self.chunks.append(c)
                for t, n in c.tf.items():
                    self.postings.setdefault(t, []).append((pos, n))
        self.avgdl = sum(c.length for c in self.chunks) / max(1, len(self.chunks))

    def search(self, tokens: list[str], top: int = 150) -> list[tuple[float, Chunk]]:
        n_docs, scores = len(self.chunks), {}
        for t in dict.fromkeys(tokens):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for pos, tf in plist:
                dl = self.chunks[pos].length
                scores[pos] = scores.get(pos, 0.0) + idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / self.avgdl))
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return [(s, self.chunks[pos]) for pos, s in best]


def recency(c: Chunk, view: View) -> date:
    vp = view.pages[c.key]
    return c.effective or vp.page.publish or date.min


def authority(c: Chunk, view: View) -> tuple:
    """Sort key, smaller = more trustworthy: newest dated statement, then live page, then page type."""
    vp = view.pages[c.key]
    return (-recency(c, view).toordinal(), STATUS_RANK[vp.status], KIND_RANK[vp.page.kind])


def _row_signature(c: Chunk, q: Query):
    label = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", c.label().lower())   # "1st Choice": a name, not a tier
    words = frozenset(_stem(w) for w in re.findall(r"[a-z]+", label) if w not in STOP)
    label_wo_years = YEAR_RANGE_RE.sub(" ", label) if model_year_ranges(c.index_text) else label
    nums = tuple(sorted(_num_token(n) for n in re.findall(r"\d+(?:\.\d+)?", label_wo_years.replace(",", ""))))
    kinds = tuple(sorted(_value_kind(h, v) for h, v in c.values() if "%" in v))
    return words, nums, kinds


def _value_kind(header: str, cell: str) -> str:
    s = f"{cell} {header}".lower()
    for kind in ("apy", "apr"):
        if kind in cell.lower():
            return kind
    if "yield" in s or "apy" in s:
        return "apy"
    if "apr" in s or "annual percentage rate" in s:
        return "apr"
    return "rate"


def _pct_values(c: Chunk) -> tuple:
    return tuple(re.sub(r"[^\d.%$]", "", v) for _, v in c.values() if "%" in v)


def suppress_stale(cands: list[tuple[float, Chunk]], view: View, q: Query) -> tuple[list, list]:
    """Return (kept, dropped). Drops (a) table rows that state the same tier as a better-dated row
    with a different value, and (b) older copies of near-identical prose (press-release boilerplate)."""
    kept, dropped = [], []
    rows = [c for _, c in cands if c.is_row and _pct_values(c)]
    groups: dict = {}
    for c in rows:
        groups.setdefault(_row_signature(c, q), []).append(c)
    loser_ids = set()
    for sig, members in groups.items():
        if len({_pct_values(m) for m in members}) < 2:
            continue
        best = min(members, key=lambda m: authority(m, view))
        for m in members:
            if _pct_values(m) != _pct_values(best):
                loser_ids.add(m.id)
    # (a2) promo/calculator/stale-copy rows that repeat a product row's tier with an older date
    for c in rows:
        vp = view.pages[c.key]
        if c.id in loser_ids or (KIND_RANK[vp.page.kind] < 3 and vp.status == "fresh"):
            continue
        _, nums, kinds = _row_signature(c, q)
        for o in rows:
            ovp = view.pages[o.key]
            if o.key == c.key or KIND_RANK[ovp.page.kind] != 0 or ovp.status != "fresh":
                continue
            _, onums, okinds = _row_signature(o, q)
            if nums and nums == onums and kinds == okinds and _pct_values(o) != _pct_values(c) \
                    and recency(o, view) >= recency(c, view):
                loser_ids.add(c.id)
                break
    seen_prose: list[tuple[frozenset, Chunk]] = []
    order = sorted(cands, key=lambda sc: authority(sc[1], view))
    prose_drop = set()
    for _, c in order:
        if c.is_row or HOURS_EV.search(c.text):       # schedules are per branch, not boilerplate
            continue
        words = frozenset(w for w in re.findall(r"[a-z]{4,}", c.text.lower()))
        if len(words) < 6:
            continue
        for w2, _keep in seen_prose:
            if len(words & w2) / len(words | w2) >= 0.8:
                prose_drop.add(c.id)
                break
        else:
            seen_prose.append((words, c))
    for s, c in cands:
        (dropped if c.id in loser_ids or c.id in prose_drop else kept).append((s, c))
    return kept, dropped


@dataclass
class Evidence:
    chunks: list[Chunk]
    dropped: list[Chunk]          # stale values we deliberately hid (used for conflict notes)
    supported: bool               # some chunk states the asked attribute for the asked entity/product
    copies: dict = field(default_factory=dict)   # chunk id -> other pages repeating the same row
    context: list = field(default_factory=list)  # chunks plus nearby prose from the same page (LLM only)


def _row_key(c: Chunk) -> tuple:
    return (re.sub(r"\W+", " ", c.label().lower()).strip(), _pct_values(c))


def site_branch(index: Index, name: str) -> bool:
    """A branch the site has: a live page that isn't an article names it with an address or opening hours."""
    view = index.view
    for pos, _ in index.postings.get(_stem(name), []):
        c = index.chunks[pos]
        if c.kind != "article" and view.pages[c.key].status != "missing" and (ADDRESS_EV.search(c.text) or HOURS_EV.search(c.text)):
            return True
    return False


def select(index: Index, q: Query, k: int = 8) -> Evidence:
    view = index.view
    # A product word the site never uses ("share certificate") is a synonym, not a requirement.
    q.descriptors = {d for d in q.descriptors if _stem(d) in index.postings}
    if q.category and q.descriptors & PLAIN_DESCRIPTORS:
        # "a regular canoe savings account": a plain word counts only if a row of the asked kind names itself with it
        kinds = asked_kinds(q)
        def says(d):
            for pos, _ in index.postings.get(_stem(d), []):
                c = index.chunks[pos]
                name = product_name(c) if c.is_row else ""
                if name and kinds_in(name) & kinds and re.search(rf"\b{re.escape(d)}", name.lower()):
                    return True
            return False
        q.descriptors = {d for d in q.descriptors if d not in PLAIN_DESCRIPTORS or says(d)}
    if q.attribute in ("rate", "fee") and q.entity and q.entity_is_branch and site_branch(index, q.entity):
        # rates and fees are the same at every branch: a branch the site has doesn't narrow the answer
        q.place, q.entity, q.entity_is_branch = q.entity, None, False
        q.tokens = [t for t in q.tokens if t not in (_stem(q.place), _stem("branch"), _stem("location"), _stem("office"))]
    if q.attribute in ("fee", "count") and not q.subject:
        q.subject = rare_subject(index, q)
    hits = index.search(q.tokens, top=400)
    if q.attribute == "count" and q.count_noun:
        # BM25 ranks long boilerplate ("... N members and M offices") low for a
        # one-word query, so scan every chunk containing the counted noun for an actual count.
        pat, have = count_re(q.count_noun), {c.id for _, c in hits}
        hits += [(_bm25_floor(hits), index.chunks[pos]) for pos, _ in index.postings.get(q.count_noun, [])
                 if index.chunks[pos].id not in have and pat.search(index.chunks[pos].text.lower())]
    if q.attribute == "rate" and q.category:
        hits += product_page_rates(index, q, hits)
    scored = []
    for s, c in hits:
        if c.expired_on and c.expired_on < view.taken_on:
            continue
        if c.is_row and q.excluded and any(re.search(rf"\b{w}\b", f"{c.label()} {c.section}".lower()) for w in q.excluded):
            continue                    # a line the question rules out
        status, factor = qualify(c, q)
        if status == "mismatch":
            continue
        if q.attribute in ("rate", "fee") and c.is_row:
            factor *= 1.3
            if c.effective:             # a table re-dated recently is the live rate sheet
                age = (view.taken_on - c.effective).days
                factor *= 1.5 if age <= 60 else (0.7 if age > 180 else 1.0)
        if supports(c, q):
            factor *= 1.4
        scored.append((s * factor, c))
    scored.sort(key=lambda sc: (-sc[0], sc[1].id))
    if q.attribute in ("rate", "fee"):  # rate tables must not be crowded out by prose that says "mortgage"
        rows = [sc for sc in scored if sc[1].is_row][:40]
        scored = sorted(rows + [sc for sc in scored if not sc[1].is_row][:30], key=lambda sc: (-sc[0], sc[1].id))
    else:
        scored = scored[:60]
    kept, dropped = suppress_stale(scored, view, q)

    # The same row copied onto several pages (a rate table is often on several pages) is one piece of evidence.
    copies: dict = {}
    first_by_key: dict = {}
    deduped = []
    for s, c in kept:
        if c.is_row and _pct_values(c):
            rk = _row_key(c)
            if rk in first_by_key:
                copies.setdefault(first_by_key[rk], []).append(c.key)
                continue
            first_by_key[rk] = c.id
        deduped.append((s, c))
    kept = deduped

    if q.attribute == "hours" and not q.entity:
        # "General" hours = what most branches share: weight a schedule by how common its times are.
        def times(c):
            return {re.sub(r"\W", "", m.lower()) for m in HOURS_EV.findall(c.text)}
        freq: Counter = Counter()
        for key, ts in {(c.key, frozenset(times(c))) for _, c in kept if supports(c, q)}:
            freq.update(ts)
        kept = sorted(((s * (1 + 0.1 * sum(freq[t] for t in times(c))) if supports(c, q) else s, c)
                       for s, c in kept), key=lambda sc: (-sc[0], sc[1].id))
    if q.attribute in ("count", "when"):
        # Facts that change over time: among chunks relevant enough to the question, newest first.
        # "Relevant enough" = covers as many of the question's words as the best chunk does, so a
        # boilerplate staff count can't outrank the figure the question is about.
        def coverage(c):
            return len(set(q.tokens) & set(c.tf))
        sup = [sc for sc in kept if supports(sc[1], q)]
        best = max((coverage(c) for _, c in sup), default=0)
        sup = sorted([sc for sc in sup if coverage(sc[1]) >= best], key=lambda sc: authority(sc[1], view))
        kept = sup + [sc for sc in kept if sc not in sup]
        if q.attribute == "count" and sup:
            # The newest count wins; older counts of the same thing leave the evidence (rule 5), so a
            # count from last year's press release can't be repeated. Only when the winning chunk
            # states a single count, so a multi-figure recap can't evict real data.
            pat = count_re(q.count_noun)
            def counts(c):
                return {re.match(r"[\d,]+", m.group(0)).group(0).replace(",", "") for m in pat.finditer(c.text.lower())}
            win = counts(sup[0][1])
            if len(win) == 1:
                stale = [sc for sc in kept if counts(sc[1]) and not counts(sc[1]) & win]
                dropped += stale
                kept = [sc for sc in kept if sc not in stale]
    per_page: Counter = Counter()
    chosen, overflow = [], []
    for s, c in kept:
        if per_page[c.key] >= (4 if c.is_row else 2):
            if not c.is_row:
                overflow.append(c)      # relevant prose cut only to keep the top-k diverse
            continue
        per_page[c.key] += 1
        chosen.append(c)
        if len(chosen) >= k:
            break
    if q.attribute and not any(supports(c, q) for c in chosen):
        # Nothing in the top k states what was asked: look further down the whole search before declining.
        gone = {c.id for _, c in dropped}
        deeper = [c for _, c in hits if c.id not in gone and supports(c, q) and qualify(c, q)[0] != "mismatch"
                  and not (c.expired_on and c.expired_on < view.taken_on)
                  and (not q.category or any(_stem(w) in c.tf for w in q.category))   # of the line asked about
                  and not (c.is_row and any(re.search(rf"\b{w}\b", f"{c.label()} {c.section}".lower())
                                            for w in q.excluded))]
        if deeper:
            chosen = list({c.id: c for c in reversed(deeper)}.values())[::-1][:k]
    lines = other_lines(q, next((c for c in chosen if c.is_row and supports(c, q)), None))
    chosen = [c for c in chosen if not on_other_line(c, lines)]
    supported = any(supports(c, q) for c in chosen) if q.attribute else bool(chosen)
    dropped_chunks = [c for _, c in dropped]
    context = [c for c in with_context(chosen, dropped_chunks, view, q, overflow) if not on_other_line(c, lines)]
    if q.list_intent:
        context += list_rows(hits, context, dropped_chunks, view, q)
    # The model reads rows of a line the question didn't ask for (business, jumbo, ...) last, after the
    # standard product's own evidence; the extractive answer's evidence is unchanged.
    way = wire_direction(q.raw) if q.attribute == "fee" and WIRE.search(q.raw) else None
    if way in ("out", "abroad") or (way == "in" and names_abroad(q.raw)):
        # the model never reads another direction's wire fee ("receive an international wire" is the incoming fee)
        context = [c for c in context if not (WIRE_FEE_LINE.search(c.text) and not supports(c, q))]
    other = [c for c in context if c.is_row and any(re.search(rf"\b{w}\b", f"{c.label()} {c.section} {c.key}".lower())
                                                    for w in SOFT_QUALIFIERS if w not in q.qualifiers)]
    context = [c for c in context if c not in other] + other
    return Evidence(chosen, dropped_chunks, supported, copies, context)


# The kind of product a row's own name says it is ("Regular Certificate" under a "Kayak Savings Rates" heading is a
# certificate). A CD is a certificate; a name saying "money market" is a money market, whatever else it says.
KIND_NAME = re.compile(r"\b(loan|mortgage|certificate|cd|card|savings|checking|heloc|money\s+market)s?\b")
KIND_SAME = {"cd": "certificate"}
LOAN_KINDS = {"loan", "mortgage", "heloc"}
HOME_LOAN = re.compile(r"\bhome\s+loans?\b", re.I)          # a home loan is a mortgage (a loan on home equity is not)
KNOWN_KINDS = {"loan", "mortgage", "certificate", "card", "savings", "checking", "heloc", "money market"}


def kinds_in(text: str) -> set:
    text = HOME_LOAN.sub("mortgage", text.lower())
    kinds = {KIND_SAME.get(k, k) for k in (re.sub(r"\s+", " ", m.group(1)) for m in KIND_NAME.finditer(text))}
    return {"money market"} if "money market" in kinds else kinds


def product_name(c: Chunk) -> str:
    """The row's own product name: its first label cell, else the last part of its heading, else the heading."""
    for t in (c.label().split(" | ")[0], c.section.split(" · ")[-1], c.section):
        if kinds_in(t):
            return t
    return ""


def asked_kinds(q: Query) -> set:
    """The kinds of product the question names ("account" alone names none in particular)."""
    words = (q.category - {"account", "accounts"}) or q.category
    kinds = {KIND_SAME.get(k, k) for k in (w if w == "savings" else w.rstrip("s") for w in words)}
    kinds = {"heloc" if k == "line" else k for k in kinds}                  # a line of credit
    kinds |= {"money market"} if re.search(r"\bmoney\s+market", q.raw, re.I) else set()
    kinds = (kinds - {"loan"}) | {"mortgage"} if HOME_LOAN.search(q.raw) else kinds
    return kinds & KNOWN_KINDS


def other_kind(c: Chunk, q: Query) -> bool:
    """The row's own name says another kind of product than the one asked ("loan" alone takes any kind of loan)."""
    asked, kinds = asked_kinds(q), kinds_in(product_name(c))
    if not asked:                                   # no kind the names can be compared with
        return False
    if "loan" in asked:
        kinds = {"loan" if k in LOAN_KINDS else k for k in kinds}
    return bool(kinds - asked)


def _asked(w: str, q: Query) -> bool:
    low = (q.text or q.raw).lower()     # "refinancing", "IRAs" and "refi" ask for that line too
    return (w in q.qualifiers or re.search(rf"\b{re.escape(_stem(w))}", low) is not None
            or (w == "refinance" and re.search(r"\brefi\b", low) is not None))


def other_lines(q: Query, top: Chunk | None) -> set:
    """Product lines (jumbo, IRA, business, refinance, ...) the question didn't ask about and the best row
    isn't on. Only for a question that pins a term or model year, gives no amount (an amount can imply
    the jumbo tier) and isn't a list question: there, a neighbouring line's row is another tier's value,
    and a jumbo rate on one date can equal the standard rate on the next."""
    if top is None or not top.is_row or not (q.terms or q.years) or q.list_intent or q.amounts:
        return set()
    text = top.index_text.lower()
    return {w for w in SOFT_QUALIFIERS if not _asked(w, q) and not re.search(rf"\b{w}\b", text)}


def on_other_line(c: Chunk, lines: set) -> bool:
    return bool(lines) and c.is_row and any(re.search(rf"\b{w}\b", c.index_text.lower()) for w in lines)


def product_page_rates(index: Index, q: Query, hits) -> list:
    """A product's own page is the authority for its rates. Its rate-stating paragraphs join the candidates
    even when the member's words ("interest", "charge") rank them below blog posts: product pages whose
    address names every product word of the question ("credit card" -> /.../credit-cards)."""
    want = {_stem(w) for w in q.category | q.descriptors}
    paths = getattr(index, "_paths", None)
    if paths is None:                        # built once per view
        paths = {k: set(tokenize(vp.page.url.split("/", 3)[-1])) for k, vp in index.view.pages.items()}
        index._paths = paths
    have = {c.id for _, c in hits}
    floor = hits[min(len(hits) - 1, 29)][0] if hits else 1.0
    out = []
    for key, vp in index.view.pages.items():
        if vp.status == "missing" or vp.page.kind != "product" or not want <= paths[key]:
            continue
        for c in vp.page.chunks:
            if c.id not in have and not c.is_row and RATE_EV.search(c.text):
                out.append((floor, c))
                have.add(c.id)
    return out


def list_rows(hits, context: list[Chunk], dropped: list[Chunk], view, q: Query, limit: int = 4) -> list[Chunk]:
    """"What cards do you offer?" covers every product line. Rows for a line the question didn't
    name (business, jumbo, ...) are down-weighted for tier questions, so for list questions the best
    of them go to the LLM context. The extractive answer and its evidence are unchanged."""
    taken = {c.id for c in context} | {c.id for c in dropped}
    out = []
    for _, c in hits:
        if len(out) >= limit:
            break
        if (not c.is_row or c.id in taken or (c.expired_on and c.expired_on < view.taken_on)
                or qualify(c, q)[0] == "mismatch"):
            continue
        low = c.index_text.lower()
        if any(re.search(rf"\b{w}\b", low) for w in SOFT_QUALIFIERS if w not in q.qualifiers):
            taken.add(c.id)
            out.append(c)
    return out


CONTEXT_RADIUS = 1          # paragraphs on each side (2 let more stale values into the test batteries' evidence; 0 lost coverage)
CONTEXT_CHARS = 6000        # budget for added text (~1.5k tokens), spent on the best-ranked chunks first
OVERFLOW_MAX = 6            # extra prose from pages that hit the per-page cap


def with_context(chosen: list[Chunk], dropped: list[Chunk], view, q: Query, overflow=()) -> list[Chunk]:
    """Evidence chunks in rank order, each surrounded by its neighbouring paragraphs on the same page,
    followed by the best prose the per-page cap left out.

    A fact is often one paragraph away from the words that retrieve it (a program's dollar amount
    after its name, a phone number after "update your address"), or is the third relevant paragraph
    of the one page that matters most. Only prose is added: table rows and anything suppressed as
    stale stay out, so rule 5 (losing values never reach the answer) holds, and a paragraph about a
    different term or tier is skipped. The extractive (no-key) answer still uses `chosen` alone.
    """
    taken = {c.id for c in chosen} | {c.id for c in dropped}
    budget, out = CONTEXT_CHARS, []
    for c in chosen:
        page = view.pages[c.key].page.chunks
        pos = next((j for j, x in enumerate(page) if x.id == c.id), None)
        window = [c] if pos is None else page[max(0, pos - CONTEXT_RADIUS): pos + CONTEXT_RADIUS + 1]
        for x in window:
            if x is c:
                out.append(c)
                continue
            if (x.id in taken or x.is_row or len(x.text) > budget
                    or (x.expired_on and x.expired_on < view.taken_on) or qualify(x, q)[0] == "mismatch"):
                continue
            taken.add(x.id)
            budget -= len(x.text)
            out.append(x)
    extra = [c for c in overflow if c.id not in taken and not c.is_row][:OVERFLOW_MAX]
    return out + extra


RARE_SHARE = 0.005       # a word on under 0.5% of chunks names something specific (a place, a service)
FEE_WORDS = {_stem(w) for w in ("fee", "fees", "cost", "costs", "charge", "charges", "price", "pay")}
SEND_WORDS = {_stem(w) for w in ("send", "sending", "sends")}
SEND_Q = re.compile(r"\bsend(?:s|ing)?\b", re.I)
# A wire's recipient ("to my son's bank in Rivertown") says where the money goes, not what is charged: those words
# aren't required, but the fee line must be about a wire. ("The fee for an incoming wire" names the wire itself.)
WIRE = re.compile(r"\bwir(?:e|es|ed|ing)\b", re.I)
WIRE_ENDS = re.compile(r"\b(?:to|from|into)\s+(?:a|an|the|my|his|her|their|our|your|someone|somebody|another)\b[^.?!;]*", re.I)
RECIPIENT = re.compile(r"\bto\s+(?:(?:a|an|the|my|his|her|their|our|your|someone|somebody|another)\b[^.?!;]*|"
                       r"[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*)")
RECEIVE_Q = re.compile(r"\b(?:receiv\w*|incoming|inbound)\b", re.I)
# Which way a wire goes (wire_direction). Several kinds asked about together are all answered.
_WIRE_KIND = r"(?:incoming|inbound|outgoing|domestic|international|send(?:ing)?|receiv(?:e|ing))"
WIRE_ALL = re.compile(rf"\b{_WIRE_KIND}\b\s*(?:,|/|\band\b|\bor\b|\bvs\.?|\bversus\b)\s*(?:\w+\s+)?{_WIRE_KIND}\b|"
                      r"\bwire(?:\s+transfer)?\s+fees\b|\bfees\s+(?:for|on)\s+(?:\w+\s+)?wires\b", re.I)
WIRE_IN = re.compile(r"\b(?:receiv\w*|incoming|inbound|coming\s+in|(?:wire[sd]?|wiring|send(?:s|ing)?|sent)\s+(?:me|us)|"
                     r"(?:get|gets|getting|got|deposit\w*)\s+(?:a|an|the)\s+(?:\w+\s+)?wire)\b", re.I)
# someone else sends the wire ("a wire my employer sends to my account"): it is received
WIRE_OTHER_SENDS = re.compile(r"\b(?:(?:my|our|his|her|their)\s+(?!own\b)[a-z]+|someone|somebody|they|he|she)\s+(?:is\s+|will\s+|wants\s+to\s+|"
                              r"plans\s+to\s+)?(?:send|sends|sending|sent|wire|wires|wiring|wired)\b", re.I)
WIRE_OUT = re.compile(r"\b(?:outgoing|send(?:s|ing)?\b(?!\s+(?:me|us)\b)|from\s+my\s+(?:\w+\s+)?(?:accounts?|acct)\b)", re.I)
# "to my (own) checking account" or "to me" is the member's own; "to my account at another bank" is not
WIRE_OWN = re.compile(r"\b(?:to|into)\s+(?:me|us)\b|\b(?:to|into)\s+(?:my|our)\s+(?:own\s+)?"
                      r"(?:(?:\w+\s+)?(?:checking|savings|share|money\s+market)\s+)?(?:accounts?|acct)\b(?!\s+(?:at|with|in)\b)", re.I)
WIRE_FROM_ABROAD = re.compile(r"\bfrom\s+(?:overseas|abroad|another\s+country|out\s+of\s+the\s+country)\b", re.I)
WIRE_ABROAD = re.compile(r"\b(?:internationa\w*|intl|int['’]l|abroad|overseas|foreign|another\s+country|out\s+of\s+the\s+country|"
                         r"outside\s+(?:of\s+)?the\s+(?:us|u\.s\.|united\s+states|country))\b", re.I)
# a capitalised name before "location" or "office", not at the start of the question
PLACE_NAMED = re.compile(r"(?<=\s)([A-Z][\w'’-]+)\s+(?:location|office)\b")
NOT_PLACE_NAMES = {"main", "home", "head", "corporate", "post", "box", "back", "front", "branch"}


# Chat shorthand for "please" and "thanks": never a thing the question asks about.
CHAT_FILLERS = {_stem(w) for w in ("pls", "plz", "thx", "thanks", "thank", "ty", "asap")}
# who ("someone who isn't a member") and how soon ("a debit card fast") are never what a fee is charged for
NOT_FEE_SUBJECT = {_stem(w) for w in ("someone", "somebody", "anyone", "anybody", "everyone", "everybody", "person",
                                      "people", "fast", "quick", "quickly", "soon", "urgent", "urgently", "immediately",
                                      "yesterday", "today", "tonight", "tomorrow", "recently", "earlier", "already")}
# A relative clause after the thing a fee is asked about ("a check I wrote", "a check my wife wrote", "the card I've
# used"): its verb, and its subject noun, say what was done with it. They are not required fee words unless the site
# states a fee next to them ("…lost key $100": "a debit card I lost" still needs a fee for a lost card).
REL_CLAUSE = re.compile(r"\b(?:a|an|the|this|that|my|your|our|his|her|their)\s+(?:[a-z][\w'’-]*\s+){0,2}?([a-z]+)\s+"
                        r"(?:that\s+|which\s+)?(?:(?:i|we|you|he|she|they)(?:['’](?:ve|d|ll))?|(?:my|our|his|her|their)\s+([a-z]+))\s+"
                        r"(?!(?:can|could|would|should|will|do|does|did|am|is|are|was|were|have|has|had|need|want|wants|might|"
                        r"must|may|plan|hope|get|got)\b)([a-z]+)\b", re.I)
TIME_NOUNS = {"time", "times", "day", "days", "moment", "way", "week", "month", "year", "night", "morning"}
QUESTION_START = r"(?:how|what|what['’]?s|when|where|which|who|why|is|are|do|does|did|can|could|will|would|should)\b"
OPENER = re.compile(rf"^(?:(?:[^?.!:]*[!:]|[^?,]*,|[^?]*?\s[-–—]\s)\s*)+?(?={QUESTION_START})", re.I)


def asking_clause(text: str) -> str:
    """The question without a leading greeting or aside ("Hi! Quick question: how many ...", "I was
    wondering, what ..."): openers ending in "!", ":", "," or a spaced dash, up to a question word; and
    without anything after its last question mark ("... have? Asking for a friend.")."""
    text = OPENER.sub("", text)
    asked = re.match(r".*\?", text, re.S)
    return asked.group(0) if asked else text


def rare_subject(index: Index, q: Query) -> list[list[str]]:
    """The specific things a fee or count question is about ("Springfield", "cashier's check"), each with
    its synonyms. A fee line or count that names none of them answers a different question, so the
    honest answer is "isn't listed". Common words ("lost", "card") are not required."""
    limit = RARE_SHARE * max(1, len(index.chunks))
    text = asking_clause(q.raw)                      # a greeting names nothing asked about
    # words in parentheses restate the thing ("a teller's check (bank draft)"); a branch the site has doesn't
    # narrow a fee (see select)
    skip = {t for m in re.finditer(r"\(([^)]*)\)", text) for t in tokenize(m.group(1))} | ({_stem(w) for w in (q.place, "branch", "location", "office")} if q.place else set())
    fee_said = _fee_line_words(index)
    for m in REL_CLAUSE.finditer(text):
        if m.group(1).lower() not in TIME_NOUNS:
            skip |= {_stem(w.lower()) for w in (m.group(2), m.group(3)) if w and _stem(w.lower()) not in fee_said}
    wire = bool(WIRE.search(text))
    if wire:
        recipient = {t for m in RECIPIENT.finditer(text) for t in tokenize(m.group(0))} - {_stem("wire")}
        sent = {_stem("money"), _stem("funds")}                   # what is sent, not what is charged
        skip |= recipient | sent
        q.tokens = [t for t in q.tokens if t not in recipient | sent]
    out = []
    for t in dict.fromkeys(tokenize(text)):
        if t in GENERIC or t in FEE_WORDS or t in CHAT_FILLERS or t == q.count_noun or t[0].isdigit() or t in SEND_WORDS \
                or t in skip or (q.attribute == "fee" and t in NOT_FEE_SUBJECT):
            continue                                # "send a wire": the fee line says what is sent, not "send"
        if len(index.postings.get(t, ())) < limit:
            out.append([t] + _SYN.get(t, []))
    if wire:
        # only a word a wire's fee line can say ("international", "incoming") narrows a wire question; the rest
        # ("my brother", "for his wedding", "in Rivertown") is who and why
        said = _wire_line_words(index)
        out = [alts for alts in out if any(w in said for w in alts)]
        if not any(_stem("wire") in alts for alts in out):
            out.append([_stem("wire")])
        way = wire_direction(q.raw)
        kinds = {_stem(w) for w in ("international", "domestic", "incoming", "outgoing", "foreign", "overseas")}
        # "wiring" and "wired" are searched as "wire", with the kind of wire it is
        said = {"in": "incoming", "abroad": "international", "out": "domestic"}.get(way)
        q.tokens += [t for t in (_stem("wire"), _stem(said) if said else None) if t and t not in q.tokens]
        if way == "all":                                 # every kind asked about: none of them is required
            out = [a for a in out if not set(a) & kinds]
        elif way == "in":                                # an incoming wire costs the same from anywhere
            out = [a for a in out if not set(a) & kinds] + [[_stem(w) for w in ("incoming", "inbound", "receive", "receiving")]]
        elif way == "abroad":
            out = [a for a in out if not set(a) & kinds] + [[_stem(w) for w in ("international", "foreign", "overseas")]]
    return out


def _foreign_frames(text: str) -> set:
    """The frames ("to", "in", "from") in which a wire question names a foreign place: a country, capital or big city
    ("in London", "to the Philippines", "to Europe"), or a city with a foreign country or province after a comma
    ("Lyon, France"). A city with anything else after the comma stays domestic ("Paris, Texas"), and a place that is
    also a given name counts only after "in" ("to Chad" is a person)."""
    commas = [(m.group(1).lower(), m.group(2).lower(), m.start()) for m in CITY_COMMA.finditer(text)]
    domestic = {city for city, after, _ in commas if after not in PROVINCE_SET}
    frames = set()
    for m in PLACE_RE.finditer(text):
        frame, name = m.group(1).lower(), m.group(2).lower()
        if name in domestic or (frame == "to" and name in GIVEN_NAMES):
            continue
        frames.add(frame)
    for city, after, at in commas:
        if after in PROVINCE_SET and city not in FOREIGN_SET:
            words = text[:at].lower().split()
            frames.add(words[-1] if words and words[-1] in ("to", "in", "from") else "in")
    return frames


def wire_direction(text: str) -> str:
    """Which wire a wire question asks about: "all" (several kinds named together, or "wire fees"), "in" (received:
    "incoming", "wire me", "into my checking account", "from abroad"), "abroad" (sent abroad: "internationally",
    "to my sister in London", "a Canadian bank", "in euros") or "out" (sent within the country, the default).
    Sending ("send", "outgoing", "from my account") outranks "to my account"."""
    for w in SITE_WORDS:                       # "into my <site name> checking account" is the member's own
        text = re.sub(rf"\b{re.escape(w)}\b\s*", "", text, flags=re.I)
    if WIRE_ALL.search(text):
        return "all"
    frames = _foreign_frames(text)
    sending = bool(WIRE_OUT.search(text)) and not (WIRE_OTHER_SENDS.search(text) and WIRE_OWN.search(text))
    from_abroad = frames == {"from"} or (WIRE_FROM_ABROAD.search(text) and not frames - {"from"})
    if WIRE_IN.search(text) or (not sending and (WIRE_OWN.search(text) or from_abroad)):
        return "in"
    return "abroad" if names_abroad(text, frames) else "out"


def names_abroad(text: str, frames=None) -> bool:
    """The wire question points abroad: a word ("internationally", "overseas"), a foreign place, a nationality's
    bank or a foreign currency."""
    frames = _foreign_frames(text) if frames is None else frames
    return bool(WIRE_ABROAD.search(text) or frames - {"from"} or NATIONAL_BANK.search(text) or CURRENCY.search(text))


# a wire's own fee line: the wire named next to its dollar amount ("Domestic wire $25.00 each"), not an offer's
# condition that happens to mention wires and an amount
WIRE_FEE_LINE = re.compile(r"\bwir(?:e|es)\b[^\n.$]{0,40}?\$\s?\d|\$\s?\d[\d,.]*[^\n.$]{0,15}\bwir(?:e|es)\b", re.I)


def _fee_line_words(index: Index) -> set:
    """Every word the site writes just before a dollar amount on one line ("Stop payment … $30.00", "lost key
    $100.00"): the things it states a fee or price for. Built once per view."""
    words = getattr(index, "_fee_words", None)
    if words is None:
        words = set()
        for c in index.chunks:
            if c.kind == "article" or "$" not in c.text:
                continue
            for line in c.text.split("\n"):
                for m in re.finditer(r"\$\s?\d", line):
                    words |= set(tokenize(line[max(0, m.start() - 40):m.start()]))
        index._fee_words = words
    return words


def _wire_line_words(index: Index) -> set:
    """Every word on a chunk that names a wire and a dollar amount. Built once per view."""
    words = getattr(index, "_wire_words", None)
    if words is None:
        words = set()
        for pos, _ in index.postings.get(_stem("wire"), []):
            c = index.chunks[pos]
            if "$" in c.text:
                words |= set(c.tf) if c.tf else set(tokenize(c.index_text))
        index._wire_words = words
    return words


def _bm25_floor(hits) -> float:
    return hits[min(len(hits) - 1, 20)][0] if hits else 1.0
