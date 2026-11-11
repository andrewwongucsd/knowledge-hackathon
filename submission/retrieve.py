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
from datetime import date, timedelta

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
    who = f"you|{alt}" if alt else "you"
    LIST_Q = re.compile(rf"\b(?:what|which) (?:\w+ ){{0,3}}(?:do|does) (?:{who}) (?:offer|have|carry|provide)\b|"
                        rf"\b(?:what|which) (?:\w+ ){{0,3}}(?:are|is) (?:available|offered)\b|\bwhat (?:kinds|types) of\b|"
                        rf"\blist (?:the|your|all|every|out)\b|\bshow me all\b|{LIST_ASK}", re.I)


LIST_ASK = r"^\s*(?:please\s+)?(?:list|show me all)\b"     # "list your card rates"
OPTIONS_Q = re.compile(r"\boptions\b", re.I)
PRODUCT_HEADS = r"loans?|mortgages?|certificates?|cds?|accounts?|cards?|savings|checking|visa|mastercard|discover|amex|heloc|line"

RATE_Q = re.compile(r"\b(rates?|apr|apy|interest|dividends?|yield)\b", re.I)
# "what does a savings account pay?" / "how much does a 1-year certificate earn?" asks about a rate without the word
# (never "…pay for", "…pay in fees")
_DEPOSIT = r"(?:savings|checking|certificates?|cds?|money\s+market|accounts?|specials?|shares?|hsas?|iras?)"
_NOT_PAY_FOR = r"(?!\s+(?:for|out|off|back)\b)"
DEPOSIT_PAYS = re.compile(rf"\b(?:what|how\s+much)\s+(?:does|did|do|will|would|could|can)\s+(?:a|an|the|my|your|our)?\s*(?:[\w-]+\s+){{0,4}}?"
                          rf"{_DEPOSIT}\s+(?:account\s+)?(?:pay|earn|yield)s?\b{_NOT_PAY_FOR}|"
                          rf"\bwhat\s+(?:is|was|are|were)\s+(?:a|an|the|my|your)?\s*(?:[\w-]+\s+){{0,4}}{_DEPOSIT}\s+(?:paying|earning|yielding)\b", re.I)
# ...or refers back to a deposit product named elsewhere in the question ("a 2-year CD — what does it earn?")
VALUE_BACKREF = re.compile(rf"\b(?:what|how\s+much)\s+(?:does|did|do|will|would|could|can)\s+(?:it|that|this|they)\s+(?:pay|earn|yield)\b{_NOT_PAY_FOR}|"
                           r"\bwhat(?:['’]s|\s+is|\s+are|\s+was)\s+(?:it|that|this|they)\s+(?:paying|earning|yielding)\b|"
                           r"\b(?:what|how\s+much)\s+(?:\w+\s+){0,3}(?:would|will|could|can|do|did|should)\s+(?:i|we)\s+earn\b", re.I)
DEPOSIT_NAMED = re.compile(r"\b(?:savings|checking|certificates?|cds?|money\s+market|hsas?|iras?)\b", re.I)
FEE_WORD = re.compile(r"\b(?:fees?|costs?|charges?|charged|price)\b", re.I)
# a rate question naming no product line but a vehicle is about a loan on it; never one about saving
DEPOSIT_CUE = re.compile(r"\b(?:apy|dividends?|sav(?:e|es|ing|ings)|certificates?|cds?|deposits?|money\s+market|hsas?|iras?)\b", re.I)
VEHICLE = re.compile(r"\b(cars?|vehicles?|trucks?|pickups?|suvs?|vans?|motorcycles?|boats?|snowmobiles?|atvs?|utvs?|watercraft)\b", re.I)
RV = re.compile(r"\b(?:rvs?|recreational\s+vehicles?|campers?|motor\s*homes?)\b", re.I)
# "does any savings tier pay 0.45%?" asks about a rate without the word (cash back and points are not rates)
PLAIN_DESCRIPTORS = {"regular", "basic", "standard", "plain", "normal", "ordinary", "simple", "everyday", "share", "personal"}
PCT_PAID = re.compile(r"\b(?:pay|pays|paying|paid|earn|earns|earning)\s+(?:a\s+|an\s+)?\d+(?:\.\d+)?\s*(?:%|percent\b)"
                      r"(?!.*\b(?:cash\s*back|reward(?:s)?|points)\b)", re.I)
FEE_Q = re.compile(r"\b(fees?|cost|charge[sd]?|price)\b|\bhow much (is|are|was|were|does|do|did|for|will)\b|"
                   r"\bhow much to (?:wire|send|mail|replace|rush|stop|cash|certify|transfer|get|have|order|buy|purchase|use|"
                   r"notarize|fax|copy|drill|re-?key|reopen|print|exchange)\b|\bhow much would (?:it|that|you|they)\b|"
                   r"\b(?:have|need)\s+to\s+pay\b(?!\s+(?:off|back|down))|\bfree\s+of\s+charge\b|\bat\s+no\s+(?:charge|cost)\b|"
                   # "is it free to …", "are money orders free for members?", "what would I pay for …"
                   r"\bis\s+it\s+free\b|\b(?:is|are)\s+[^?,.]{1,40}?\s+free(?=\s+(?:for|to|at|with|if|when)\b|\s*\?|\s*$)|"
                   r"\bwhat\s+(?:would|will|do|did|should|must|might)\s+(?:i|we)\s+(?:have\s+to\s+|need\s+to\s+)?pay\s+(?:for|to)\b", re.I)
HOURS_Q = re.compile(r"\bhours?\b|\b(?:lobby|lobbies|drive[- ]?(?:up|thru|through))\s+(?:times?|schedules?)\b|\bopening\s+times?\b", re.I)
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
LIST_Q = re.compile(r"\b(?:what|which) (?:\w+ ){0,3}(?:do|does) (?:you) (?:offer|have|carry|provide)\b|"
                    r"\b(?:what|which) (?:\w+ ){0,3}(?:are|is) (?:available|offered)\b|\bwhat (?:kinds|types) of\b|"
                    r"\blist (?:the|your|all|every|out)\b|\bshow me all\b|" + LIST_ASK, re.I)
RATE_EV = re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:APR|APY|annual percentage|fixed|variable|interest|dividend)|"
                     r"(?:APR|APY|rate of|rates? (?:as low as|start(?:ing)? at|from)|dividend rate)\s*(?:is|of|:)?\s*"
                     r"\d+(?:\.\d+)?\s*%", re.I)
ADDRESS_Q = re.compile(r"\baddress\b|\bwhere is\b|\blocated\b", re.I)
# The member editing their own address is a how-to question, not "where is it": an edit verb on "my/our
# (mailing...) address", or "change of address", and no branch, office or other place named.
OWN_ADDRESS = re.compile(r"\b(?:updat|chang|edit|correct)\w*\s+(?:my|our)\s+(?:(?:home|mailing|email|e-mail|street|"
                         r"billing|physical|contact)\s+)?address\b|\bchange of address\b|"
                         # "change the address or phone number on my account", "update my phone number",
                         # "the address on file", "I have a new address. How do I update it?"
                         r"\b(?:updat|chang|edit|correct)\w*\s+(?:the|my|our)\s+[^?.]*\b(?:address|phone\s+number|e-?mail)\b[^?.]*"
                         r"\b(?:on|in|for)\s+(?:my|our)\s+(?:\w+\s+)?(?:accounts?|file|profile|records?)\b|"
                         r"\b(?:updat|chang|edit|correct)\w*\s+(?:my|our)\s+(?:\w+\s+){0,2}?(?:(?:tele)?phone|cell|mobile|e-?mail)\b|"
                         r"\b(?:address|phone\s+number|e-?mail)\s+(?:you\s+have\s+)?on\s+file\b|"
                         r"\bnew\s+(?:address|phone\s+number|e-?mail)\b[^?]*\b(?:updat|chang)\w*\b", re.I)
PLACE_WORDS = re.compile(r"\b(?:branch|branches|location|locations|office|offices|lobby|lobbies)\b", re.I)
ADDRESS_EV = re.compile(r"\b\d{2,6}\s+(?:[A-Z][\w.'’-]*\s+){1,4}(?:St|Street|Ln|Lane|Ave|Avenue|Dr|Drive|Rd|Road|"
                        r"Blvd|Way|Ct|Pl|Pkwy|Hwy|Highway|Square)\b\.?")
DATE_EV = re.compile(rf"\b(?:{MONTH_RE})\s+\d{{1,2}},\s+20\d\d\b")
# a fee is a dollar amount or a share of the amount ("1% of Amount", "5% of total for non-members")
PCT_OF = r"\d+(?:\.\d+)?\s?%\*?\s+of\s+(?:the\s+|each\s+)?(?:[Aa]mount|[Tt]otal|[Tt]ransaction|[Pp]urchase|[Bb]alance|[Aa]dvance)s?\b"
FEE_EV = re.compile(rf"\$\s?\d|{PCT_OF}")
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
    period: object = None                            # a count asked of a year: "last", "this", ... or a calendar year
    series: list = field(default_factory=list)       # (year, sentence, chunk) per year the site states the count
    no_year: bool = False                            # a year was asked and the site counts nothing for it
    count_said: set = field(default_factory=set)     # words of how a count was asked that are no requirement ("staff": the
                                                     # site says "employees"; "number" of; "team members")
    focus: tuple | None = None                       # what a fee question is about (see fee_focus)
    places: list = field(default_factory=list)       # hours: the towns and streets asked about, each a set of search words
    compare: bool = False                            # hours: "do all branches keep the same hours?"
    schedules: list = field(default_factory=list)    # hours: the schedules of the place or places asked about
    soft_hours: bool = False                         # hours asked without the word: a schedule answers it, or it is
    loose_hours: bool = False                        # ...read as it was before (read_as: attribute, qualifiers)
    need_schedule: bool = False
    read_as: tuple | None = None

    @property
    def numbers(self) -> set:
        return {n for n, _ in self.terms} | self.years | {int(a) for a in self.amounts}


def parse_query(question: str) -> Query:
    question = written_amounts(question)  # "15k", "fifteen thousand dollars", "between 10k and 25k": dollar amounts
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
    if re.search(r"model|vehicle|\bcar\b|truck|pickup|\bsuv|\bvan\b|motorcycle", low):   # model years only, never "as of March 2031"
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
    elif RATE_Q.search(low) or PCT_PAID.search(low) or (DEPOSIT_PAYS.search(low) and not FEE_Q.search(low)) \
            or ((DEPOSIT_PAYS.search(head) or (VALUE_BACKREF.search(head) and DEPOSIT_NAMED.search(head)))
                and not FEE_WORD.search(head)):         # what a deposit pays or earns, even "how much does … pay"
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
        # the site's own name, possessive or not, is never a branch ("<Name>'s branch lobby hours")
        name = next((w for w in m.group(1).split() if w in PROPER_NAMES
                     and re.sub(r"['’]s$", "", w) not in BRANCH_NOT_NAMES), None) if m else None
        if name:
            q.entity, q.entity_is_branch = name, True
    if not q.entity and q.attribute in ("rate", "fee") and (m := PLACE_NAMED.search(question)) \
            and m.group(1).lower() not in BRANCH_NOT_NAMES | NOT_PLACE_NAMES:
        q.entity, q.entity_is_branch = m.group(1).lower(), True     # "at your Rivertown location": a branch
    if q.attribute is None and (m := COUNT_CUE.search(low)) and m.group(1) not in STOP:
        q.attribute, q.count_noun = "count", _stem(m.group(1))
        q.tokens = [t for t in q.tokens if t not in ("number", "total")]
    # a named branch ("the elm-street branch", any case) keeps its own question type: "what time does it close?"
    named = [w for w in re.findall(r"\b([a-z][\w'’-]*)\s+branch\b", low)
             if re.sub(r"['’]s$", "", w) not in BRANCH_NOT_NAMES | STOP]
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
    if q.attribute == "rate":
        # "what does savings pay / earn": the verb asks for the rate, it names nothing to search for
        q.tokens = list(dict.fromkeys(_stem("rate") if t in VALUE_VERB_STEMS else t for t in q.tokens))
        if not q.category and not DEPOSIT_CUE.search(head):
            # "finance a 2030 car for 7 years": a loan on that vehicle (an RV, camper or motorhome is an RV loan)
            if RV.search(head):
                q.category.add("loan")
                q.descriptors = set(q.descriptors) | {"rv"}
            elif v := VEHICLE.search(head):
                word = v.group(1).lower()
                word = word[:-1] if word.endswith("s") else word
                q.category.add("loan")
                q.descriptors = set(q.descriptors) | {word} | ({"vehicle"} if word in ("car", "vehicle", "truck", "pickup", "suv", "van") else set())
    if q.attribute == "fee":
        for t in list(q.tokens):                        # "overdraw" is searched as "overdraft" too
            q.tokens += [x for x in FEE_SYN.get(t, []) if x not in q.tokens]
        q.descriptors = {d for d in q.descriptors if _stem(d.lower()) not in HAPPENED}   # "a lost card": no product
    raw_low = q.raw.lower()
    _read_hours(q, raw_low)
    _read_counts(q, raw_low)
    _read_vehicle_year(q, raw_low)
    _read_refinance(q, raw_low)
    if q.list_intent and q.attribute == "rate" and (q.terms or q.years or q.amounts) and VALUE_ASKED.search(raw_low) \
            and not re.match(LIST_ASK, q.raw, re.I):
        q.list_intent = False                           # a value asked of one pinned row is not a list
    _read_contact(q, raw_low)
    return q


# A vehicle word names the loan when a model year is asked: "a 72-month loan on a 2016 or older vehicle" is the auto
# table's question (it has no such row), not a boat or RV loan's.
def _read_vehicle_year(q: Query, low: str) -> None:
    if q.attribute == "rate" and q.years and q.category == {"loan"} and not q.descriptors and (v := VEHICLE.search(low)):
        word = v.group(1).lower()
        word = word[:-1] if word.endswith("s") else word
        q.descriptors = {word} | ({"vehicle"} if word in ("car", "vehicle", "truck", "pickup", "suv", "van") else set())


# "refinancing", "refinanced", "refi" ask for the refinance rows of a rate question, unless the member says they are not
# ("purchase, not refinancing", "I'm not refinancing", "without refinancing", "instead of a refinance").
REFINANCE_WORD = re.compile(r"\brefinanc\w*|\brefi\b", re.I)
REFINANCE_NEGATED = re.compile(
    r"\b(?:not|non|no|never|excluding|except|without|rather\s+than|instead\s+of|other\s+than)\s+(?:(?:a|an|the|my|any)\s+)?(?:\w+\s+)?"
    r"(?:refinanc\w*|refi)\b|\b(?:i['’]?m|am|are|we['’]?re)\s+not\s+(?:\w+\s+)?(?:refinanc\w*|refi)\b", re.I)


def _read_refinance(q: Query, low: str) -> None:
    if q.attribute == "rate" and REFINANCE_WORD.search(low) and not REFINANCE_NEGATED.search(low):
        q.qualifiers = set(q.qualifiers) | {"refinance"}


def _read_contact(q: Query, low: str) -> None:
    """An address, phone number or email on the member's own account is their contact information: a how-to,
    searched as such."""
    if q.attribute is None and OWN_ADDRESS.search(low) and not PLACE_WORDS.search(low):
        frame = {_stem(w) for w in ("file", "profile", "record", "records", "number", "new")}
        q.tokens = [t for t in q.tokens if t not in frame]
        add = [t for t in (_stem("contact"), _stem("information"), _stem("personal")) if t not in q.tokens]
        q.expansion, q.tokens = q.expansion + add, q.tokens + add


# ---- Hours: asked of a place, without the word, or of every branch ----------------------------------------------
# A branch's schedule sits in a block that names neither its town nor its street; the page's lead paragraph does
# (a dateline, "located at …"). So a schedule is bound to its page's lead (see _about) and a place asked about is
# matched against both. A question about hours with no place named gets one schedule, the same for every wording.
_DAY = r"(?:mon|tues|wednes|thurs|fri|satur|sun)days?"
_VISIT = r"(?:open|close[sd]?|closing|when|until|use|available|stop|visit|come|go)"
_FAC = r"(?:lobby|lobbies|drive[- ]?(?:up|thru|through)s?|branch(?:es)?|offices?|locations?)"
_OPENS = r"(?:open|opens|opened|close|closes|closed|closing|shut|shuts)"
_WHEN_DAY = rf"(?:{_DAY}|today|tomorrow|tonight|weekends?|weekdays?|late|later|early|earlier|until|till|after|before|past|at\s+\d|holidays?)"
FACILITY = re.compile(rf"\b{_FAC}\b", re.I)
# "what time is it open on Fridays": a weaker cue, trusted only when the question also names a place
HOURS_LOOSE = re.compile(rf"\bwhat time\b(?!\s+(?:frame|period|limit)s?\b)|\bhow (?:late|early)\b|"
                         rf"\b{_DAY}\b.*\b{_VISIT}\b|\b{_VISIT}\b.*\b{_DAY}\b|"
                         r"\bwhen\s+(?:is|are|can|could|may)\b[^?]*\b(?:open|use|visit|stop|come|go)\b", re.I)
STATUS_EV = re.compile(r"\b(?:will\s+close|closed|closes|closing|closure|will\s+end|ended|ends|expired|expires|discontinued|"
                       r"no\s+longer|reopen\w*|will\s+open|opened|last\s+day|final\s+day)\b", re.I)
STATUS_DATE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+20\d\d\b")
_PLACE_NOUN = r"(?:branch(?:es)?|locations?|offices?|lobb(?:y|ies)|drive[- ]?ups?)"
SAME_Q = re.compile(r"\b(?:same|identical|differ\w*|vary|varies|varying|uniform|consistent|change|depend\w*)\b", re.I)
ALL_PLACES = re.compile(rf"\b(?:all|every|each|any|both|different|other|various)\s+(?:of\s+)?(?:[\w'’-]+\s+){{0,2}}?{_PLACE_NOUN}\b|"
                        rf"\b(?:by|per|across|between|among|depending\s+on|based\s+on)\s+(?:the\s+|each\s+|which\s+)?{_PLACE_NOUN}\b|"
                        r"\b(?:branch|location|office)\s+to\s+(?:branch|location|office)\b|\beverywhere\b|\b(?:branches|locations|offices)\b", re.I)
STREET_WORDS = {"avenue", "ave", "road", "rd", "street", "st", "blvd", "boulevard", "drive", "dr", "lane", "ln", "highway",
                "hwy", "way", "court", "ct", "place", "pl", "e", "w", "n", "s"}
MONTH_WORDS = set("january february march april may june july august september october november december".split())
NOT_PLACE = {"i", "lobby", "drive-up", "drive", "atm", "atms", "branch", "branches", "office", "location", "hours"}
DAY_LINE = re.compile(r"(?mi)^(?:[A-Za-z-]+\s)?(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day(?:\s*[-–—]\s*\w+day)?\s*:")
LOCATIVE = {"in", "at", "on", "near", "inside", "outside", "around", "by", "off"}
AFTER_PLACE = {"branch", "branches", "location", "locations", "office", "offices", "lobby", "lobbies", "drive-up", "drive-thru",
               "drive", "hours", "hrs", "atm", "atms", "is", "are", "open", "opens", "close", "closes", "closed", "area"}
NUMBERED_ROADS = {"highway", "hwy", "route", "rt", "rte", "road", "rd", "county", "interstate"}
ABBREV = set("jan feb mar apr jun jul aug sep sept oct nov dec mon tue tues wed thu thur thurs fri sat sun am pm".split())
_STRONG: tuple | None = None


def hours_strong(low: str) -> bool:
    """Hours asked without the word: when the lobby, the drive-up, a branch, "you" or the site itself opens or closes."""
    global _STRONG
    if _STRONG is None or _STRONG[0] != frozenset(SITE_WORDS):
        site = "|".join(sorted(map(re.escape, SITE_WORDS))) or r"(?!x)x"
        fac = rf"(?:(?:the|your|our|a|an|that|this|any|each|every|all|(?:{site})(?:['’]s)?)\s+)?(?:[\w.'’-]+\s+){{0,8}}?{_FAC}"
        subj = (rf"(?:(?:you|we)\s+(?:[\w-]+\s+)?|(?:{site})\s+(?:(?:in|on|at|near|inside)\s+(?:[\w.,'’-]+\s+){{1,7}}?)?|"
                rf"{fac}\s+(?:[\w-]+\s+){{0,2}}?)")
        _STRONG = (frozenset(SITE_WORDS), re.compile(
            rf"\bhrs\b|"
            rf"\b(?:what\s+time|how\s+(?:late|early|long)|when|(?:un)?till?\s+when)\s+(?:do|does|is|are|will|can|could|did)\s+{subj}{_OPENS}\b|"
            rf"\b(?:is|are|will)\s+{subj}(?:open|closed)\b(?=[^?]*\b{_WHEN_DAY}\b)|"
            rf"\b(?:can|could|may)\s+(?:i|we)\s+(?:visit|come|stop|go|use|get)\b[^?]*\b{_FAC}\b[^?]*\b{_WHEN_DAY}\b|"
            rf"\bwhen\s+(?:can|could|may)\s+(?:i|we)\s+(?:visit|come|stop|go|use|get)\b[^?]*\b{_FAC}\b|"
            rf"\b(?:do|does)\s+{subj}(?:open|close)\b[^?]*\b(?:same|different)\s+times?\b|"
            rf"\bwhat\s+time\b(?!\s+(?:frame|period|limit)s?\b)[^?]*\b{_FAC}\b", re.I))
    return _STRONG[1].search(low) is not None


def place_phrases(question: str) -> list:
    """The towns and streets a question names, each as a set of search words: capitalised runs and street words
    ("Elm Point Road", "Harbor") that the wording marks as a place: by a street word, by "in / at / on / near" before
    it (past "the" or "your"), or by branch, lobby, drive-up, hours, "is" or "opens" after it. A numbered road
    keeps its number ("Highway 12"); each town of a list ("A, B, or C") is one. A name in a greeting, a weekday or month, AM/PM, the site's own name, the
    region after a town and its comma ("Elmville, Ohio": unless a list of towns goes on), and anything in an
    all-caps or title-case question mark nothing."""
    words = re.findall(r"[A-Za-z][\w'’-]*\.?|\d+|[,.?!:;—–-]", question)
    alpha = [w for w in words if w[0].isalpha() and len(w.strip(".")) > 3]
    if alpha and all(w[0].isupper() for w in alpha):
        return []
    plain = lambda w: re.sub(r"['’]s$", "", w.lower().strip("."))
    out, run, start = [], [], True
    region_from = skip_to = last_end = -1

    def flush(end):
        nonlocal last_end
        if run:
            first = end - len(run)
            every = {t for w in run for t in tokenize(w)}
            core = {t for w in run for t in tokenize(w) if plain(w) not in STREET_WORDS}
            j = first - 1
            while j >= 0 and plain(words[j]) in ("the", "a", "an", "your", "our", "my", "that", "this"):
                j -= 1
            before = plain(words[j]) if j >= 0 else ""
            nxt = [plain(w) for w in words[end:end + 2] if w[0].isalnum()] if end < len(words) and words[end][0].isalnum() else []
            after = next((w for w in nxt if not re.fullmatch(_DAY, w)), "")
            item = last_end >= 0 and (first - 1 == last_end and words[last_end] == "," or
                                      first - 2 == last_end and words[first - 1].lower() in ("or", "and", "&"))   # "A, B, or C"
            if core and (core != every or before in LOCATIVE or after in AFTER_PLACE or item
                         or run[-1].lower().endswith(("'s", "’s"))):
                out.append(core)
                last_end = end
            run.clear()
    for i, w in enumerate(words):
        if w[0].isdigit():
            if run and plain(run[-1]) in NUMBERED_ROADS:
                run.append(w)
            else:
                flush(i)
            start = False
            continue
        if not w[0].isalpha():
            had_run = bool(run)
            flush(i)
            if w == "," and had_run:
                region_from = i + 1
            start = start or w in ".?!:"
            continue
        if i == region_from and w[0].isupper():
            j = i
            while j < len(words) and words[j][0].isalpha() and words[j][0].isupper():
                j += 1
            if (words[j].lower() if j < len(words) else "") not in (",", "or", "and", "&"):
                skip_to = j                     # "<Town>, <Region>": the region is no place; "A, B, or C" is a list
        if i < skip_to:
            start = False
            continue
        low = plain(w)
        nxt_raw = words[i + 1] if i + 1 < len(words) else ""
        nxt = nxt_raw.lower().strip(".")
        proper = low in PROPER_NAMES or (w[0].isupper() and not start and not (w.isupper() and len(w) > 1))
        street = low in STREET_WORDS and bool(run)
        skip = (low in BRANCH_NOT_NAMES or low in MONTH_WORDS or low in NOT_PLACE or low in STOP or low in ABBREV
                or re.fullmatch(_DAY, low) is not None)
        if low in STREET_WORDS and not run and w[0].isupper() and (nxt_raw[:1].isupper() or nxt in PROPER_NAMES
                                                                   or (nxt.isdigit() and low in NUMBERED_ROADS)):
            run.append(w)
        elif (proper and not skip) or street:
            run.append(w)
        else:
            flush(i)
        start = False
    flush(len(words))
    return [p for p in out if p]


def _prose(c: Chunk) -> bool:
    """A paragraph of sentences, not a line of links."""
    return not c.is_row and len(re.findall(r"[A-Za-z]{2,}", re.sub(r"\([^)]*\)", " ", c.text))) >= 15


def _about(index: Index) -> None:
    """Bind each schedule (a block with two or more weekday lines and clock times) to its page's lead paragraph, which
    names the town and the street, and to the block above it; the dateline town is the schedule's branch name. Only
    on a page whose schedules form one run after the lead. Fills index._schedules (those on live pages)."""
    if getattr(index, "_about_done", False):
        return
    index._schedules = []
    for vp in index.view.pages.values():
        chunks = vp.page.chunks
        hrs = [i for i, c in enumerate(chunks) if not c.is_row and len(DAY_LINE.findall(c.text)) >= 2 and HOURS_EV.search(c.text)]
        if not hrs:
            continue
        lead = next((i for i, c in enumerate(chunks) if _prose(c) and len(DAY_LINE.findall(c.text)) < 2), None)
        one = hrs[-1] - hrs[0] <= 2 and lead is not None and lead < hrs[0]
        for i in hrs:
            prev = chunks[i - 1].text if i else ""
            chunks[i].about = f"{chunks[lead].text if one else ''} {prev}"
            m = re.match(r"\s*([A-Z][A-Z .'’-]{2,}?),\s", chunks[lead].text) if one else None
            chunks[i].branch = m.group(1).title() if m else ""
            if vp.status != "missing":
                index._schedules.append(chunks[i])
    index._about_done = True


def dated_status(index: Index, q: Query) -> bool:
    """Does a passage naming the asked place state, with a full date, that it opens, closes or ends?"""
    named = set().union(*q.places)
    for pos in sorted({pos for t in named for pos, _ in index.postings.get(t, [])}):
        if any(STATUS_EV.search(s) and STATUS_DATE.search(s) for s in re.split(r"(?<=[.!?])\s+", index.chunks[pos].text)):
            return True
    return False


def _read_hours(q: Query, low: str) -> None:
    """Hours asked without the word ("when does the lobby close?": answered from a schedule, else read as before: see
    select), the place asked about, and a comparison of every branch."""
    strict = hours_strong(low)
    site = any(re.search(rf"\b{re.escape(w)}\b", low) for w in SITE_WORDS)
    loose = bool(not strict and (FACILITY.search(low) or site) and HOURS_LOOSE.search(low) and place_phrases(q.raw))
    if q.attribute in (None, "when", "hours") and not HOURS_Q.search(low) and not FEE_Q.search(low) and (strict or loose):
        q.soft_hours, q.loose_hours, q.read_as = True, loose, ("when" if WHEN_Q.search(low) else None, set(q.qualifiers))
        q.attribute = "hours"
        if "hour" not in q.tokens:
            q.tokens.append("hour")
    q.places = place_phrases(q.raw) if q.attribute == "hours" else []
    if q.attribute == "hours" and not q.places and q.entity and q.entity_is_branch:
        q.places = [{_stem(q.entity)}]               # "the harbor branch", in any case
    if q.places:                                     # "Highway 12" is not a product qualifier
        q.qualifiers = {w for w in q.qualifiers if not any(_stem(w) in p for p in q.places)}
    q.compare = bool(q.attribute == "hours" and not q.places and SAME_Q.search(low) and ALL_PLACES.search(low))


# Amounts written out: "15k", "15 thousand", "fifteen thousand dollars", "five grand", "7,500 dollars" (never "401k");
# a range is one interval whose round upper bound is exclusive ("between $20,000 and $50,000" is the tier that ends below
# $50,000; "20 to 50k" carries the unit to the bare small number). Plain numbers ("12 to 24 months") are no range.
_AMT_N = r"(\d[\d,]*(?:\.\d+)?)"
_THOUSAND = r"([kK]\b|\s*thousand\b)?"
RANGE_Q = re.compile(rf"\bbetween\s+\$?{_AMT_N}\s*{_THOUSAND}\s+(?:and|to)\s+\$?{_AMT_N}\s*{_THOUSAND}(\s+dollars?\b)?|"
                     rf"(\$)?{_AMT_N}\s*{_THOUSAND}\s*(?:-|–|\bto\b)\s*(\$)?{_AMT_N}\s*{_THOUSAND}(\s+dollars?\b)?", re.I)
BARE_AMT = re.compile(r"(?<![$\d.,])\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s?([kK])\b|\s?(thousand|grand)\b(?:\s+dollars?\b)?|"
                      r"(?=\s*(?:dollars?|bucks)\b))", re.I)
RETIREMENT_PLANS = ("401", "403", "457")


def _range(m) -> str:
    if m.group(1):
        a, ka, b, kb = m.group(1), m.group(2), m.group(3), m.group(4)
    else:
        a, ka, b, kb = m.group(7), m.group(8), m.group(10), m.group(11)
        if not (m.group(6) or m.group(9) or ka or kb or m.group(12)):
            return m.group(0)
    lo = _f(a) * (1000 if ka or (kb and _f(a) < 1000) else 1)
    hi = _f(b) * (1000 if kb or (ka and _f(b) < 1000) else 1)
    if hi <= lo:
        return m.group(0)
    if hi % 1000 == 0:
        hi -= 0.01
    return f"between ${lo:,.2f} and ${hi:,.2f}"


def _bare_amount(m) -> str:
    if (m.group(2) or m.group(3)) and m.group(1) in RETIREMENT_PLANS:
        return m.group(0)
    return f"${m.group(1)}k" if (m.group(2) or m.group(3)) else f"${m.group(1)}"


def written_amounts(question: str) -> str:
    text = normalize_numbers(question) if re.search(r"\b(?:thousand|grand)\b", question, re.I) else question
    return BARE_AMT.sub(_bare_amount, RANGE_Q.sub(_range, text))


def count_re(noun: str):
    """'12 offices', '350 members' — a count, never a year like 'In 2025,'."""
    return re.compile(rf"(?<![\d,.$])(?!(?:19|20)\d\d\b)\d[\d,]*\+?\s+(?:[a-z’'-]+\s+){{0,2}}{noun}")


# A count asked of a year ("how many … last year", "… in 2023"): the site states such counts as dated sentences, one per
# year, so the answer is the sentence for the asked year (see select and count_sentence).
PERIOD_Q = re.compile(r"\b(last|this|past|previous|prior)\s+year\b|\b(?:in|during|for|of)\s+((?:19|20)\d\d)\b", re.I)
YEAR_IN = re.compile(r"\b(?:in|during|for)\s+((?:19|20)\d\d)\b", re.I)
PERIOD_WORDS = {_stem(w) for w in ("last", "this", "past", "previous", "prior", "year")}
CLAUSE_END = re.compile(r",\s+|;\s+|\s[—–]\s")
# what a member calls the people who work there, and the site's word for them
COUNT_SYN = {_stem(w): _stem("employees") for w in ("staff", "staffers", "workers", "personnel")}
COUNT_OF = re.compile(r"\bhow many\s+of\s+(?:the|your|our|its|their|those|these)\s+([a-z-]+)", re.I)
WHAT_NUMBER = re.compile(r"\bwhat\s+number\s+of\s+(?:(?:the|your|our)\s+)?((?:[a-z-]+\s+)?[a-z-]+)", re.I)
TEAM_MEMBERS = re.compile(r"\bhow many\s+(?:of\s+(?:the|your|our)\s+)?(?:[a-z]+\s+)?(?:team|staff|crew)\s+members?\b", re.I)
# a value (rate, APY) asked of one row a term, model year or amount pins is no list ("what APY does <Name> offer on its
# 17-month special?")
VALUE_ASKED = re.compile(r"\b(?:what|which)\s+(?:\w+\s+){0,2}?(?:rates?|apr|apy|dividends?|yield|interest)\b", re.I)


def _spells(acronym: str, text: str) -> bool:
    """Does the text spell the acronym out ("KRC" = "Kayak Repair Crew")?"""
    return re.search(r"\b" + r"[\w'’-]*\s+".join(re.escape(ch) for ch in acronym.upper()) + r"[\w'’-]*", text) is not None


def _acronyms(q: Query) -> dict:
    return {_stem(w.lower()): w for w in re.findall(r"\b[A-Z]{2,5}\b", q.raw)}


def count_sentence(c: Chunk, q: Query) -> tuple:
    """(the sentence or bullet of the chunk that counts the asked noun, the year it is stated for, how many of the
    question's words the counting clause itself names). A count in one clause and the topic in another doesn't
    qualify: "40 workers paddled a total of 900 hours, with many using their <benefit>" counts paddlers."""
    pat = count_re(q.count_noun)
    parts = [p.strip().lstrip("-• ").strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])|\n|\s+[-•]\s+(?=[A-Z0-9])", c.text)]
    cand = [p for p in parts if p and pat.search(p.lower())]
    if not cand:
        return None, None, 0
    words = [t for t in dict.fromkeys(q.tokens) if t not in GENERIC and t not in PERIOD_WORDS and t not in LIGHT_VERBS
             and t != q.count_noun and not t.isdigit()]
    acr = _acronyms(q)

    def cover(p):
        m = pat.search(p.lower())
        cuts = [0] + [x.end() for x in CLAUSE_END.finditer(p)] + [len(p) + 1]
        clause = next(p[a:b] for a, b in zip(cuts, cuts[1:]) if a <= m.start() < b)
        have = set(tokenize(clause))
        return sum(t in have or (t in acr and _spells(acr[t], clause)) for t in words)
    best = max(cand, key=cover)
    m = YEAR_IN.search(best) or YEAR_IN.search(c.section or "")
    return re.sub(r"\s*\([^()]*$", "", best), (int(m.group(1)) if m else None), cover(best)


def _read_counts(q: Query, low: str) -> None:
    """How many …: the counted noun as the site says it, and the year asked."""
    if q.attribute is None and (m := WHAT_NUMBER.search(low)):
        noun = next((w for w in m.group(1).split() if w not in SITE_WORDS and w not in STOP), None)
        if noun:
            q.attribute, q.count_noun = "count", _stem(noun)
            q.count_said.add(_stem("number"))
            q.tokens = [t for t in q.tokens if t != _stem("number")]
    if q.attribute == "count" and TEAM_MEMBERS.search(low):
        q.count_noun = _stem("staff")                # "team members": the people who work there
        q.count_said |= {_stem(w) for w in ("team", "crew", "member")}
        q.tokens = [t for t in q.tokens if t not in q.count_said]
    m = COUNT_OF.search(low)
    if m and q.attribute == "count" and q.count_noun in (_stem("of"), "of"):
        q.count_noun = _stem(m.group(1))
    if q.attribute == "count" and q.count_noun in COUNT_SYN:
        asked, q.count_noun = q.count_noun, COUNT_SYN[q.count_noun]
        q.count_said.add(asked)
        q.tokens = list(dict.fromkeys(q.count_noun if t == asked else t for t in q.tokens))
        if q.count_noun not in q.tokens:
            q.tokens.append(q.count_noun)
    m = PERIOD_Q.search(low) if q.attribute == "count" else None
    q.period = (int(m.group(2)) if m.group(2) else m.group(1)) if m else None


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
    if q.attribute == "hours" and q.places:
        # a schedule block whose own text plus its page's lead names every word of the place asked about
        named = set(tokenize(f"{c.index_text} {c.about}"))
        if len(DAY_LINE.findall(c.text)) < 2 or not all(p <= named for p in q.places):
            return False
    elif q.attribute == "hours" and q.need_schedule:
        if len(DAY_LINE.findall(c.text)) < 2:           # asked without the word "hours": only a schedule answers
            return False
    elif q.entity and (q.entity not in text or (q.entity_is_branch and "branch" not in text)):
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
        acr = _acronyms(q)
        if not all(any(w in have or (w in acr and _spells(acr[w], c.text)) for w in alts) for alts in q.subject):
            return False
        # a state word ("lost") and the thing it happened to ("money order") on one fee item's own label
        on_label = [a for a in q.subject if ON_LABEL in a or set(a) & HAPPENED]
        if q.attribute == "fee" and on_label and not fee_evidence(c, q, on_label):
            return False
    if q.attribute == "fee" and (WIRE.search(q.raw) or SEND_Q.search(q.raw)):
        way = wire_direction(q.raw)
        if way == "abroad" and re.search(r"\bdomestic\b", text):
            return False                # a wire sent abroad is never the domestic fee
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
    if q.attribute == "fee":            # like rates: fees are published in fee schedules and product pages, not
        # blog posts; a fee item, never a rate row's balance tier nor a threshold ("less than $5") unless one is asked
        return c.kind not in ("article", "calculator") and not rate_row(c) and fee_evidence(c, q)
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


# Words that describe a product rather than name it. "special" is plain English in "a special mortgage rate" unless the
# site sells specials of that kind; "regular" means "the standard one" in "the regular 17-month special".
ADJECTIVES = {"special", "promotional", "promo", "limited", "featured", "current", "latest", "new"}
OFFER_WORDS = {"special", "promotional", "promo", "limited", "featured"}       # one family: a row with any carries them all
SPECIAL_ASKED = re.compile(r"\b(?:special|promo\w*)\b", re.I)
NOT_SPECIAL = re.compile(r"\bnot\s+(?:the\s+)?(?:special|promo\w*)", re.I)


def drop_describing_words(index: Index, q: Query) -> set:
    """The question's descriptors without the ones that only describe. An adjective goes when no row of the asked kind
    carries it (or its family) at any term; so "a 6-month special certificate" still declines where the site has
    specials of other terms. A plain word ("regular") goes when a special is asked for and a special row that agrees
    with the asked term, year and amount lacks it."""
    desc = set(q.descriptors)
    if q.attribute != "rate" or not desc or not q.category:
        return desc
    kind = [{pos for pos, _ in index.postings.get(_stem(w), [])} for w in q.category]
    rows = [(c, c.index_text.lower()) for pos in set.intersection(*kind) if (c := index.chunks[pos]).is_row]
    if not rows:
        return desc
    carries = lambda low, d: d in low or any(x in low for x in SYNONYMS.get(d, ()))
    special = SPECIAL_ASKED.search(q.raw) and not NOT_SPECIAL.search(q.raw)
    agreeing = [low for c, low in rows if re.search(r"special|promo", low) and qualify(c, q)[0] != "mismatch"] if special else []
    drop = set()
    for d in desc:
        if d in ADJECTIVES and not any(carries(low, w) for _, low in rows for w in (OFFER_WORDS if d in OFFER_WORDS else {d})):
            drop.add(d)
        if d in PLAIN_DESCRIPTORS and agreeing and not any(carries(low, d) for low in agreeing):
            drop.add(d)
    return desc - drop


def select(index: Index, q: Query, k: int = 8) -> Evidence:
    view = index.view
    if q.soft_hours:
        _about(index)
        q.need_schedule = True
        if not any(supports(c, q) for c in index._schedules) and (q.loose_hours or not q.places or dated_status(index, q)):
            # no schedule answers it, and the wording or a dated statement about the place says it asks something
            # else ("when is the <Name> branch open until?"): the question as it read before
            q.attribute, q.qualifiers = q.read_as
            q.places, q.compare, q.need_schedule, q.soft_hours = [], False, False, False
            q.tokens = [t for t in q.tokens if t != "hour"]
    q.descriptors = drop_describing_words(index, q)
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
        if q.count_said:                    # "staff": the site says "employees"; the asked word is no requirement
            q.subject = [a for a in q.subject if not q.count_said & set(a)]
    hits = index.search(q.tokens, top=400)
    if q.attribute == "count" and q.count_noun:
        # BM25 ranks long boilerplate ("... N members and M offices") low for a
        # one-word query, so scan every chunk containing the counted noun for an actual count.
        pat, have = count_re(q.count_noun), {c.id for _, c in hits}
        hits += [(_bm25_floor(hits), index.chunks[pos]) for pos, _ in index.postings.get(q.count_noun, [])
                 if index.chunks[pos].id not in have and pat.search(index.chunks[pos].text.lower())]
    if q.attribute == "rate" and q.category:
        hits += product_page_rates(index, q, hits)
    focus = fee_focus(index, q) if q.attribute == "fee" and not WIRE.search(q.raw) else None
    if focus and focus[0]:
        # fee items naming what was asked, wherever the word search ranked them; below every real hit on score, so
        # they win only by naming more of what was asked
        have, where = {c.id for _, c in hits}, _fee_where(index)
        low = min((s for s, _ in hits), default=1.0) * 0.5
        for pos in sorted({p for a, _ in focus[0].values() for x in a for p in where.get(x, ())}):
            if index.chunks[pos].id not in have:
                hits.append((low, index.chunks[pos]))
    if q.attribute == "hours":
        _about(index)
        if q.places:                    # the schedules of the place asked about, wherever the word search ranked them
            have, floor = {c.id for _, c in hits}, _bm25_floor(hits)
            hits += [(floor, c) for c in index._schedules if c.id not in have and supports(c, q)]
        q.schedules = (list(index._schedules) if q.compare
                       else [c for c in index._schedules if supports(c, q)] if q.places else [])
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
    cover = {}
    if focus and focus[0]:
        cover = {c.id: chunk_cover(c, q, focus)[:3] for _, c in scored if supports(c, q)}
        cover = {i: v for i, v in cover.items() if v[1]}
    rank = lambda sc: (tuple(-x for x in cover.get(sc[1].id, (0, 0, 0))), -sc[0], sc[1].id)
    scored.sort(key=rank)
    every = list(scored)                # every hit, for the counts of each year (see below)
    if q.attribute in ("rate", "fee"):  # rate tables must not be crowded out by prose that says "mortgage"
        rows = [sc for sc in scored if sc[1].is_row][:40]
        scored = sorted(rows + [sc for sc in scored if not sc[1].is_row][:30], key=rank)
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

    if q.attribute == "hours" and not q.entity and not q.places:
        # "General" hours = what most branches share: weight a schedule by how common its times are.
        def times(c):
            return {re.sub(r"\W", "", m.lower()) for m in HOURS_EV.findall(c.text)}
        freq: Counter = Counter()
        for key, ts in {(c.key, frozenset(times(c))) for _, c in kept if supports(c, q)}:
            freq.update(ts)
        kept = sorted(((s * (1 + 0.1 * sum(freq[t] for t in times(c))) if supports(c, q) else s, c)
                       for s, c in kept), key=lambda sc: (-sc[0], sc[1].id))
    if q.attribute == "hours" and not q.entity and not q.places and not q.compare:
        top = next((c for _, c in kept if supports(c, q)), None) or next((c for _, c in hits if supports(c, q)), None)
        if q.need_schedule or (top is not None and len(DAY_LINE.findall(top.text)) >= 2):
            # one schedule for every wording: the one whose times the most branches share
            clock = lambda c: {re.sub(r"\W", "", m.lower()) for m in HOURS_EV.findall(c.text)}
            every_s = [c for c in index._schedules if supports(c, q)]
            share = Counter(t for c in every_s for t in clock(c))
            if every_s:
                best = min(every_s, key=lambda c: (-sum(share[t] for t in clock(c)), c.id))
                kept = [(kept[0][0] if kept else 1.0, best)] + [sc for sc in kept if sc[1].id != best.id]
                q.need_schedule = True
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
    if q.attribute == "count" and q.period:
        # Counts stated per year ("N employees … in 2024", "M … in 2023") are not stale copies of each other: the
        # asked year's sentence answers; a relative year ("last year") the site doesn't count gets the years it does,
        # newest first; a calendar year it doesn't count gets none.
        found = []
        for s, c in every:
            if supports(c, q):
                sent, year, cov = count_sentence(c, q)
                if sent and year:
                    found.append((cov, year, sent, s, c))
        top = max((f[0] for f in found), default=0)
        by_year = {}
        for cov, year, sent, s, c in found:
            if cov and cov == top:
                by_year.setdefault(year, (sent, s, c))
        if by_year:
            now = (view.taken_on + timedelta(days=30)).year
            want = q.period if isinstance(q.period, int) else {"last": now - 1, "past": now - 1, "previous": now - 1,
                                                               "prior": now - 1, "this": now}[q.period]
            years = [want] if want in by_year else [] if isinstance(q.period, int) else sorted(by_year, reverse=True)[:3]
            q.no_year = not years
            q.series = [(y, by_year[y][0], by_year[y][2]) for y in years]
            ids = {c.id for _, _, c in q.series}
            dropped = [sc for sc in dropped if sc[1].id not in ids]
            kept = [(by_year[y][1], by_year[y][2]) for y in years] + [sc for sc in kept if sc[1].id not in ids]
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
# two kinds asked together or compared: "domestic and international", "higher than domestic", "domestically versus
# internationally", "an international wire? And a domestic one?"
_WIRE_KIND = r"(?:incoming|inbound|outgoing|domestic(?:ally)?|international(?:ly)?|abroad|overseas|send(?:ing)?|receiv(?:e|ing))"
WIRE_ALL = re.compile(rf"\b{_WIRE_KIND}\b(?:\s+[\w'’-]+){{0,4}}?\s*(?:,|/|\band\b|\bor\b|\bvs\.?|\bversus\b|\bthan\b|"
                      rf"\bcompared\s+(?:to|with)\b|[?.]\s*(?:and|also|what\s+about|how\s+about)\b)"
                      rf"\s*(?:(?:a|an|the|for|to|with)\s+)?(?:\w+\s+)?{_WIRE_KIND}\b|"
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
                                      "something", "anything", "everything", "stuff",
                                      "people", "fast", "quick", "quickly", "soon", "urgent", "urgently", "immediately",
                                      "yesterday", "today", "tonight", "tomorrow", "recently", "earlier", "already",
                                      "ago", "few", "couple", "several", "later", "last", "next", "week", "weeks", "day",
                                      "days", "hi", "hello", "hey", "howdy")}
# words that never say what a fee is for, nor name one
FUNCTION_WORDS = {_stem(w) for w in ("after", "before", "during", "until", "since", "while", "because", "if", "when",
                                     "then", "also", "just", "only", "even", "about", "around", "per", "each", "every",
                                     "both", "either", "neither", "some", "more", "most", "less", "least", "other",
                                     "another", "same", "such", "than", "very", "really", "into", "onto", "over",
                                     "under", "up", "down", "out", "off", "again", "once")}
QUIET = STOP | GENERIC | FEE_WORDS | CHAT_FILLERS | SEND_WORDS | NOT_FEE_SUBJECT | FUNCTION_WORDS \
    | {_stem(w) for w in ("will", "shall", "might", "must", "may", "she", "he", "him", "her", "ones")}
# verbs that say little about what is asked ("what does it cost to take …", "how many used …"): never required
LIGHT_VERBS = {_stem(w) for w in """take takes took taken taking use uses used using get gets got gotten getting make makes
made making do does did done doing have has had having give gives gave given giving go goes went gone going receive receives
received receiving put puts putting utilize utilizes utilized utilizing""".split()}
# never what a fee is charged for, however rarely the site writes them ("if", "will", "under")
NEVER_SUBJECT = FUNCTION_WORDS | {_stem(w) for w in ("will", "shall", "might", "must", "may", "ones")} | LIGHT_VERBS
# A relative clause after the thing a fee is asked about ("a check I wrote", "a check my wife wrote", "the card I've
# used"), a passive one ("a check written by me") or one without a subject ("a check that bounced"): its verb, and
# its subject noun, say what was done with it (see rare_subject). The noun is never a function word ("a fee if I lose").
REL_CLAUSE = re.compile(r"\b(?:a|an|the|this|that|my|your|our|his|her|their)\s+(?:[a-z][\w'’-]*\s+){0,2}?([a-z]+)\s+"
                        r"(?:that\s+|which\s+)?(?:(?:i|we|you|he|she|they)(?:['’](?:ve|d|ll))?|(?:my|our|his|her|their)\s+([a-z]+))\s+"
                        r"(?!(?:can|could|would|should|will|do|does|did|am|is|are|was|were|have|has|had|need|want|wants|might|"
                        r"must|may|plan|hope|get|got)\b)([a-z]+)\b", re.I)
PASSIVE_BY = re.compile(r"\b(?:a|an|the|this|that|my|your|our|his|her|their)\s+(?:[a-z][\w'’-]*\s+){0,2}?([a-z]+)\s+"
                        r"(?:that\s+(?:was|were|is|has\s+been)\s+)?([a-z]+)\s+by\s+((?:me|us|you|him|her|them)\b|"
                        r"(?:my|our|his|her|their|your|a|an|the)\s+[a-z]+)", re.I)
THAT_VERB = re.compile(r"\b(?:a|an|the|this|my|your|our|his|her|their)\s+(?:[a-z][\w'’-]*\s+){0,2}?([a-z]+)\s+(?:that|which)\s+"
                       r"(?:(?:was|were|got|gets|is|are|has\s+been|have\s+been)\s+)?(?!(?:can|could|would|should|will|do|does|did|"
                       r"am|is|are|was|were|have|has|had|need|needs|want|wants|might|must|may|i|we|you|he|she|they|my|our|"
                       r"his|her|their)\b)([a-z]+)\b", re.I)
NOT_HEAD = STOP | {"if", "when", "whenever", "because", "since", "so", "and", "or", "but", "then", "after", "before",
                   "while", "unless", "until", "once", "though", "although", "where"}
TIME_NOUNS = {"time", "times", "day", "days", "moment", "way", "week", "month", "year", "night", "morning"}
# What happened to the thing ("a lost card", "a check that bounced", "my wallet was stolen"): a different service from
# the thing's own fee, so the fee item must say it too (see rare_subject)
HAPPENED = {_stem(w) for w in ("lost", "lose", "loses", "losing", "stolen", "stole", "bounced", "bounce", "bounces",
                               "bouncing", "returned", "damaged", "forged", "expired")}
_NP_WORDS = r"((?:(?!(?:and|or|but|so|if|when|to|for|on|in|at|with|from|after|before|because|while|how|what|is|was|are|" \
            r"were)\b)[a-z][\w'’-]*\s*){1,5})"
HAPPENED_ADJ = re.compile(r"\b(lost|lose|loses|losing|stolen|stole|bounced|bounce|bounces|bouncing|returned|damaged|forged|"
                          r"expired)\s+" + _NP_WORDS, re.I)
# Who and why: a person after a determiner ("my son", "a friend"), a who-clause ("my sister, who lives in …"), a purpose
# phrase after the thing ("a cashier's check for his apartment deposit")
PERSON = {_stem(w) for w in """son sons daughter daughters child children kid kids sister sisters brother brothers sibling
siblings mother mom mum father dad parent parents wife husband spouse partner fiance fiancee boyfriend girlfriend grandson
granddaughter grandchild grandchildren grandmother grandfather grandma grandpa grandparent grandparents aunt uncle cousin
cousins nephew niece friend friends roommate neighbor neighbour tenant tenants landlord landlady contractor employer
employee boss coworker colleague client customer family relative relatives caregiver babysitter nanny""".split()}
PERSON_AFTER = re.compile(r"\b(?:my|your|his|her|our|their|a|an)\s+([a-z]+)(?:\s+([a-z]+))?", re.I)
WHO_CLAUSE = re.compile(r"\b[a-z]+,?\s+(?:who|whom|whose)\b([^,.;?!—–]*)", re.I)
PURPOSE = re.compile(r"\b([a-z][\w'’-]*)\s+for\s+(?:my|your|his|her|our|their|a|an|the)\s+((?:[a-z][\w'’-]*\s*){1,3})", re.I)
NOT_PURPOSE_HEAD = FEE_WORDS | {_stem(w) for w in ("paid", "much", "is", "are", "was", "there", "it")}
# Equivalents for fee questions only: what the member says and what a fee schedule writes
FEE_SYN = {_stem(k): [_stem(x) for x in v] for k, v in {
    "lose": ["lost"], "loses": ["lost"], "losing": ["lost"], "overdraw": ["overdraft"], "overdraws": ["overdraft"],
    "overdrawn": ["overdraft"], "overdrew": ["overdraft"], "overdrawing": ["overdraft"], "expedite": ["rush"],
    "expedited": ["rush"], "expediting": ["rush"], "overnight": ["rush"], "overnighted": ["rush"], "stopping": ["stop"],
    "stopped": ["stop"], "keyed": ["key"], "bounced": ["returned"], "bounce": ["returned"], "bounces": ["returned"],
    "bouncing": ["returned"], "notarize": ["notary"], "notarized": ["notary"], "notarizing": ["notary"],
    "garnished": ["garnishment"], "garnish": ["garnishment"], "levied": ["levy"], "dormant": ["inactive"],
    "overseas": ["foreign"], "foreign": ["overseas"], "abroad": ["foreign", "international"]}.items()}
VALUE_VERB_STEMS = {_stem(w) for w in ("pay", "pays", "paying", "paid", "earn", "earns", "earning", "earned", "yield",
                                       "yields", "yielding")}
UNPRICED = "\x00unpriced"     # a required word the site prices nothing under: no chunk can carry it
ON_LABEL = "\x00label"        # a group that must sit on the fee item's own label, with the state word
QUESTION_START = r"(?:how|what|what['’]?s|when|where|which|who|why|is|are|do|does|did|can|could|will|would|should)\b"
# a comma before a relative "who" or "which" ("my sister, who lives in …") ends no aside
_RELATIVE = r"(?:who|whom|whose|which|where|when)\b"
OPENER = re.compile(rf"^(?:(?:[^?.!:]*[!:]|[^?,]*,(?!\s*{_RELATIVE})|[^?]*?\s[-–—]\s)\s*)+?(?={QUESTION_START})", re.I)
CLAUSE_SPLIT = re.compile(rf"[;?!]|\.(?=\s+[A-Za-z]{{2,}})|\s[–—]\s|\s-\s(?={QUESTION_START})|"
                          r",?\s+\b(?:so|and|but|because|if|when|since|then)\b", re.I)


def asking_clause(text: str) -> str:
    """The question without a leading greeting or aside ("Hi! Quick question: how many ...", "I was
    wondering, what ..."): openers ending in "!", ":", "," or a spaced dash, up to a question word; and
    without anything after its last question mark ("... have? Asking for a friend."). When what is left names
    nothing ("…, who lives in Germany — what will that cost me?"), it refers back: the whole question is read."""
    full = re.match(r".*\?", text, re.S)
    full = full.group(0) if full else text
    stripped = OPENER.sub("", text)
    asked = re.match(r".*\?", stripped, re.S)
    clause = asked.group(0) if asked else stripped
    if clause != full and not any(t not in QUIET and not t[0].isdigit() for t in tokenize(clause)):
        return full
    return clause


def rare_subject(index: Index, q: Query) -> list[list[str]]:
    """The specific things a fee or count question is about ("Springfield", "cashier's check"), each with
    its synonyms. A fee line or count that names none of them answers a different question, so the
    honest answer is "isn't listed". Common words ("card") are not required; who, why and when never are; what
    happened to the thing ("lost", "bounced") is, on the same fee item as the thing, unless a specific fee is named
    elsewhere in the question ("stop payment on a lost check")."""
    limit = RARE_SHARE * max(1, len(index.chunks))
    full = re.match(r".*\?", q.raw, re.S)
    full = full.group(0) if full else q.raw
    text = asking_clause(q.raw)                      # a greeting names nothing asked about
    fee_said = _fee_line_words(index)
    priced = lambda t: t in fee_said or any(x in fee_said for x in _SYN.get(t, []) + FEE_SYN.get(t, []))
    clause_words = None                              # the asking clause's own words, when the lead-in is read too
    if text != full and text in full:
        if (set(tokenize(full[:full.index(text)])) - QUIET) & fee_said:
            clause_words, text = set(tokenize(text)), full   # "For a cashier's check, what's the fee…": the topic
    elif text == full and OPENER.sub("", full) != full:
        clause_words = set(tokenize(OPENER.sub("", full)))   # "…, who lives in Germany — what will that cost me?"
    # words in parentheses restate the thing ("a teller's check (bank draft)"); a branch the site has doesn't
    # narrow a fee (see select)
    skip = {t for m in re.finditer(r"\(([^)]*)\)", text) for t in tokenize(m.group(1))} | ({_stem(w) for w in (q.place, "branch", "location", "office")} if q.place else set())
    if clause_words is not None:                     # a lead-in word counts only if the site prices it
        skip |= {t for t in set(tokenize(full)) - clause_words if not priced(t) and t not in HAPPENED}
    amounts = _fee_amounts(index)
    words = lambda s: set(tokenize(s or ""))
    # a word naming one specific fee: its fee items carry one or two amounts ("stop" -> $30)
    named = {t for t in tokenize(text) if t not in QUIET and t not in HAPPENED and not t[0].isdigit()
             and any(1 <= len(amounts.get(x, ())) <= 2 for x in [t] + FEE_SYN.get(t, []))}
    for m in PERSON_AFTER.finditer(text):           # "my son", "a friend": who, not what
        for g in (1, 2):
            if m.group(g) and _stem(m.group(g).lower()) in PERSON:
                skip.add(_stem(m.group(g).lower()))
                break
    for m in WHO_CLAUSE.finditer(text):             # "my sister, who lives in Germany"
        skip |= words(m.group(1)) - fee_said
    for m in PURPOSE.finditer(text):                # "a cashier's check for his apartment deposit"
        if _stem(m.group(1).lower()) not in NOT_PURPOSE_HEAD and m.group(1).lower() not in STOP:
            skip |= words(m.group(2)) - fee_said
    covered = set()
    for pat, groups in ((REL_CLAUSE, (2, 3)), (PASSIVE_BY, (2, 3)), (THAT_VERB, (2,))):
        for m in pat.finditer(text):
            if m.group(1).lower() in TIME_NOUNS or m.group(1).lower() in NOT_HEAD:
                continue
            np = words(text[m.start():m.end(1)])
            said = set().union(*(words(m.group(g)) for g in groups))
            covered |= said & HAPPENED
            if named - np - said:                    # "stop payment on a check I wrote / that bounced"
                skip |= said
            elif np & named:                         # "temporary drafts I wrote" (never "…I lost")
                skip |= (said - HAPPENED) - fee_said
    for m in HAPPENED_ADJ.finditer(text):           # "stop payment on a lost check", "drill my box open if I lost the key"
        np, said = words(m.group(2)), words(m.group(1))
        if said & covered:
            continue
        covered |= said
        if named - np - said:
            skip |= said
    clauses = [set(tokenize(p)) for p in CLAUSE_SPLIT.split(text) if p and p.strip()]
    for t in set(tokenize(text)) & HAPPENED - covered:
        elsewhere = set().union(*[c for c in clauses if t not in c])
        if named & elsewhere:                        # "My wallet was stolen. How much is a rush debit card?"
            skip.add(t)
    wire = bool(WIRE.search(text))
    if wire:
        recipient = {t for m in RECIPIENT.finditer(text) for t in tokenize(m.group(0))} - {_stem("wire")}
        sent = {_stem("money"), _stem("funds")}                   # what is sent, not what is charged
        skip |= recipient | sent
        q.tokens = [t for t in q.tokens if t not in recipient | sent]
    fee = q.attribute == "fee"
    out = []
    for t in dict.fromkeys(tokenize(text)):
        if t in GENERIC or t in FEE_WORDS or t in CHAT_FILLERS or t == q.count_noun or t[0].isdigit() or t in SEND_WORDS \
                or t in skip or t in NEVER_SUBJECT or (fee and t in NOT_FEE_SUBJECT) or (q.period and t in PERIOD_WORDS):
            continue                                # "send a wire": the fee line says what is sent, not "send"
        if len(index.postings.get(t, ())) < limit or (fee and t in HAPPENED):
            out.append([t] + _SYN.get(t, []) + (FEE_SYN.get(t, []) if fee else []))
    if fee and any(set(a) & HAPPENED for a in out):
        # what it happened to ("a lost money order"): on the same fee item as the state word ("lost key" won't do)
        req = {w for a in out for w in a}
        things = {t for t in tokenize(text) if t in fee_said and t not in req and t not in skip and t not in QUIET
                  and not t[0].isdigit()}
        if things:
            out.append(sorted(things) + [ON_LABEL])
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
    if fee:
        # a word the site prices nothing under: the fee isn't listed (a product's name, "HSA", may sit in its paragraph)
        product = {_stem(w.lower()) for w in set(q.category) | set(q.descriptors) | set(q.qualifiers)}
        product |= {s for w in list(product) for s in _SYN.get(w, [])}
        if any(not set(a) & fee_said and not set(a) & product for a in out):
            out.append([UNPRICED])
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


# A fee item: one amount ("$N", "N% of the amount", a "Free" value inside a block) with its label (the text since the
# previous item, from its last ";" or ". ", at most 60 characters in prose) and the rest of its clause, the lowercase
# words, numbers and amounts up to ";", ". " or a capitalised word ("$10.00 or 3.00% … whichever is greater", "a $4 service
# fee unless you keep a balance of $2,000+", "$10.00 for non-members"). A one-line block of a fee schedule ("Rush fee for
# debit card $50.00* Foreign transaction fee 1% of Amount …") is several items.
_FEE_AMOUNT = rf"(?:\$\s?\d[\d,]*(?:\.\d+)?|{PCT_OF}|(?<=[a-z]\s)Free(?=\s+[A-Z]))"
FEE_ITEM = re.compile(_FEE_AMOUNT + r"\*?(?:\s*(?:[Ee]ach\b|[Pp]er\s+[\w/]+(?:\s+[a-z/]+)*))?(?:\s*\([^)]*\))?"
                      r"((?:\s*(?:,(?!\s*[A-Z])|[a-z][\w'’/-]*\*?|\d[\d,.]*\s?%?|\$\s?\d[\d,.]*\*?|\([^)]*\)))*)")
# a label ending this way makes its amount a condition ("…for transactions less than $5"), not a fee
THRESHOLD = re.compile(r"\b(?:less|fewer|more|greater)\s+than\s*$|\b(?:under|below|over|above|at\s+least|minimum(?:\s+of)?|"
                       r"min\.?)\s*$", re.I)
ASKS_THRESHOLD = re.compile(r"\b(?:under|less\s+than|below|over|more\s+than|above|at\s+least)\s+\$|\b(?:maximum|minimum)\b", re.I)
RATE_HEAD = re.compile(r"\b(?:apr|apy|rate|dividend|yield|annual\s+percentage)", re.I)


def rate_row(c: Chunk) -> bool:
    """A table row whose percentage sits under a rate header (APR, APY, dividend); "Foreign transaction fee | 1% of
    Amount" is a fee row."""
    return c.is_row and any("%" in v and RATE_HEAD.search(f"{h} {v}") for h, v in c.values())


def fee_items(line: str) -> list[tuple[str, str, bool]]:
    """(label words, item text, is a threshold) for each fee item of one line."""
    out, start = [], 0
    for m in FEE_ITEM.finditer(line):
        label = line[start:m.start()].rstrip(" ;:|—–-")
        cut = max(label.rfind(";"), label.rfind(". "))
        label = label[cut + 1:] if cut >= 0 else label
        if len(label) > 60:
            label = label[-60:].split(" ", 1)[-1]
        start = m.end()
        out.append((f"{label} {m.group(1) or ''}", f"{label} {line[m.start():m.end()]}".strip(), bool(THRESHOLD.search(label))))
    return out


def _labels(c: Chunk):
    """(label tokens, item text, is a threshold) for each fee item of a chunk. A table row's label cell is its label,
    whole; a label under three words also takes the line before it ("Foreign currency exchange fee / (includes
    Canadian checks) $35.00")."""
    if c.is_row:
        vals = " ".join(v for _, v in c.values())
        if not rate_row(c) and FEE_EV.search(vals):
            yield tokenize(f"{c.label()} {vals}"), f"{c.label()} {vals}", False
        elif c.free_row:                                    # "Debit card transactions at <our> ATMs | Free"
            vals = " ".join(c.cells[1:])
            yield tokenize(f"{c.cells[0]} {vals}"), f"{c.cells[0]} {vals}", False
        return
    lines = c.text.split("\n")
    for i, line in enumerate(lines):
        for label, item, threshold in fee_items(line):
            toks = tokenize(label)
            if len(toks) < 3 and i:
                toks = toks + tokenize(lines[i - 1])
            yield toks, item, threshold


NO_CHARGE = re.compile(r"\bfree\b|\bno\s+(?:charge|cost|fee)\b", re.I)


def _mark_free_rows(index: Index) -> None:
    """In a fee table (two or more rows with a dollar amount), a row whose other cells say "Free" or "no charge" is
    itself a fee line: what costs nothing is priced too."""
    for vp in index.view.pages.values():
        tables: dict = {}
        for c in vp.page.chunks:
            if c.is_row and not rate_row(c):
                tables.setdefault(c.table_span, []).append(c)
        for rows in tables.values():
            if sum(bool(FEE_EV.search(" ".join(v for _, v in r.values()))) for r in rows) >= 2:
                for r in rows:
                    rest = " ".join(r.cells[1:])
                    if len(r.cells) >= 2 and not FEE_EV.search(rest) and NO_CHARGE.search(rest) and len(rest.split()) <= 10:
                        r.free_row = True


def _fee_line_words(index: Index) -> set:
    """Every word of every fee item on the site ("Stop payment … $30.00", "Re-key box due to lost key $100.00"): the
    things it states a fee or price for; and, per word, the distinct amounts under $1,000 it is priced at. Built once
    per view."""
    words = getattr(index, "_fee_words", None)
    if words is None:
        _mark_free_rows(index)
        words, amounts = set(), {}
        for c in index.chunks:
            if c.kind in ("article", "calculator") or rate_row(c):     # a calculator's figures are examples
                continue
            for toks, item, threshold in _labels(c):
                words |= set(toks)
                if threshold:
                    continue
                m = re.search(r"\$\s?(\d[\d,]*(?:\.\d+)?)|(\d+(?:\.\d+)?)\s?%", item)
                if not m:                                   # "Free": a price of nothing is no amount
                    continue
                v = float(m.group(1).replace(",", "")) if m.group(1) else "%" + m.group(2)
                if isinstance(v, float) and v >= 1000:      # a balance or loan amount, not a fee
                    continue
                for w in set(toks):
                    amounts.setdefault(w, set()).add(v)
        index._fee_words, index._fee_amounts = words, amounts
    return words


def _fee_amounts(index: Index) -> dict:
    _fee_line_words(index)
    return index._fee_amounts


def fee_evidence(c: Chunk, q: Query, groups=()) -> bool:
    """A fee item of this chunk (a threshold only when the question asks one) whose label carries every group."""
    allow = bool(ASKS_THRESHOLD.search(q.raw))
    return any((not threshold or allow) and all(any(w in toks for w in alts) for alts in groups)
               for toks, _, threshold in _labels(c))


# What a fee question is about. Its words that some fee line prices split into the thing asked ("money order",
# "foreign transaction": primary, double weight in the asking clause, not the lead-in) and who or which product
# ("non-member", "debit card", "a box of": secondary). A fee line is ranked by the primary words it names, then by
# the side it is on, then by the word search: "a money order for a non-member" is the money order's line, not the
# line that happens to say "non-member".
WHO_WORDS = PERSON | {_stem(w) for w in ("member", "members", "nonmember", "nonmembers")}
NEG_WORDS = {"non", "not"}
NEGATED = re.compile(r"\bnon\b|\bnot\b|n['’]t\b|\banother\b|\bother\b|\bsomeone\s+else", re.I)
QUANTITY_OF = re.compile(r"\b(box|book|pack|set|sheet|bunch|couple|pair|lot|number|kind|type|sort)e?s?\s+of\b", re.I)
FOCUS_QUIET = QUIET | LIGHT_VERBS | {_stem(w) for w in FRAME_WORDS} | {_stem(w) for w in ("free", "new", "need", "needs", "want")}


def fee_focus(index: Index, q: Query) -> tuple:
    """(primary {word: (alternatives, weight)}, secondary {word: alternatives}, side): side is "own" ("a <Name> ATM",
    "your own"), "other" ("non", "not", "another", "someone else's"), "both", or None when the question states none."""
    if q.focus is not None:
        return q.focus
    fee_said = _fee_line_words(index)
    full = re.match(r".*\?", q.raw, re.S)
    full = full.group(0) if full else q.raw
    clause = set(tokenize(OPENER.sub("", full)))
    product = {_stem(w.lower()) for w in set(q.category) | set(q.descriptors) | set(q.qualifiers) | PLAIN_DESCRIPTORS}
    product |= {s for w in list(product) for s in _SYN.get(w, [])}
    quantity = {_stem(m.group(1).lower()) for m in QUANTITY_OF.finditer(full)}
    alts = lambda t: [t] + _SYN.get(t, []) + FEE_SYN.get(t, [])
    primary, secondary = {}, {}
    for t in dict.fromkeys(tokenize(full)):
        if t in FOCUS_QUIET or t[0].isdigit() or t in NEG_WORDS or not any(x in fee_said for x in alts(t)):
            continue
        if t in product or t in WHO_WORDS or t in quantity:
            secondary[t] = alts(t)
        else:
            primary[t] = (alts(t), 2 if t in clause else 1)
    said = [re.sub(r"['’]s$", "", w) for w in re.findall(r"[a-z][\w'’-]*", full.lower())]
    mine = {"your", "our", "own"} | set(SITE_WORDS)
    # "a <Name> ATM", "your ATM"; "a non-<Name> ATM" is the other side
    own = any(_stem(w) in primary and mine & {p for x in said[max(0, i - 2):i] if not x.startswith("non-") for p in x.split("-")}
              for i, w in enumerate(said))
    own = own or re.search(r"\b(?:your|our)\s+own\b|-owned\b", full.lower()) is not None
    both = own and re.search(r"\b(?:another|other)\b|\bnon-?[a-z]", full.lower()) is not None
    q.focus = (primary, secondary, "both" if both else "other" if NEGATED.search(full) else "own" if own else None)
    return q.focus


def fee_cover(toks, focus: tuple, priced: bool = True) -> tuple:
    """(weight of the primary words the fee item names, how many, whether it is on the side asked, secondary words).
    With no side asked, a priced item is on it and a free one is not."""
    primary, secondary, side = focus
    toks = set(toks)
    hit = [w for a, w in primary.values() if any(x in toks for x in a)]
    sec = sum(any(x in toks for x in a) for a in secondary.values())
    return (sum(hit), len(hit), int(("non" in toks) == (side == "other")) if side in ("own", "other") else int(priced), sec)


def chunk_cover(c: Chunk, q: Query, focus: tuple) -> tuple:
    allow = bool(ASKS_THRESHOLD.search(q.raw))
    return max((fee_cover(toks, focus, bool(re.search(r"[$%]", item))) for toks, item, thr in _labels(c) if not thr or allow),
               default=(0, 0, 0, 0))


def _fee_where(index: Index) -> dict:
    """word -> positions of the chunks whose fee items carry it. Built once per view."""
    where = getattr(index, "_fee_pos", None)
    if where is None:
        _fee_line_words(index)
        where = {}
        for pos, c in enumerate(index.chunks):
            if c.kind in ("article", "calculator") or rate_row(c):
                continue
            for toks, _, thr in _labels(c):
                for w in set(toks):
                    where.setdefault(w, set()).add(pos)
        index._fee_pos = where
    return where


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
