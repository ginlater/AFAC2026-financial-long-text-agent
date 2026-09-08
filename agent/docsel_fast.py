"""Deterministic high-confidence document routing.

The selector uses the current question surface together with source-derived
identities and lexical matching. ``None`` delegates selection to Qwen.

Public entry point::

    decision = select_docs_fast(question)
    if decision is not None:
        picked, diagnostics = decision

This module is separate from :mod:`agent.doc_select` so the two routing stages
remain independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
import unicodedata
from typing import Iterable, Optional

from . import retrieval
from .paths import PROCESSED_DIR


_PUNCT_RE = re.compile(r"[^0-9a-zA-Z一-鿿%]+")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return _PUNCT_RE.sub("", text).lower()


def _qtext(q: dict) -> str:
    """Return the normalized visible question surface."""
    opts = q.get("options") or {}
    values = opts.values() if isinstance(opts, dict) else opts
    return (q.get("question") or "") + "\n" + "\n".join(str(v) for v in values)


_CORP_RE = re.compile(
    r"([一-鿿A-Za-z0-9（）()]{2,42}?"
    r"(?:股份有限公司|集团有限公司|有限责任公司|有限公司))")
_SHORT_RE = re.compile(
    r"(?:股票|证券|公司)简称\s*[:：]\s*([A-Za-z0-9一-鿿]{2,16})")
_CORP_SUFFIX_RE = re.compile(r"(?:股份有限公司|集团有限公司|有限责任公司|有限公司)$")
_BAD_ALIAS = {
    "股份有限公司", "有限公司", "集团有限公司", "上市公司",
    "公司", "本公司", "发行人", "上市公司", "标的公司", "目标公司",
    "交易对方", "标的资产", "交易标的", "债务人", "债权人",
    "主承销商", "独立财务顾问", "年度报告", "募集说明书",
}

# Contract documents commonly identify a target company in their glossary
# rather than in the cover-page issuer name.  Treat a line-oriented
# ``short name -> legal name`` definition as stronger evidence than an
# incidental body-text mention.
_DEFINED_COMPANY_RE = re.compile(
    r"(?m)^\s*([^\n]{2,48})\s*\n\s*指\s+"
    r"([^\n]{2,80}?(?:股份有限公司|集团有限公司|有限责任公司|有限公司))"
)


def _clean_company(raw: str) -> str:
    # Regexes can start at a nearby label because Chinese has no word boundary.
    raw = re.sub(r"^(?:标题|股票简称|证券简称|公司简称)[:：]?", "", raw)
    return raw.strip("。；;,: ")


def _company_aliases(company: str, short_names: Iterable[str]) -> tuple[str, ...]:
    company = _clean_company(company)
    base = _CORP_SUFFIX_RE.sub("", company)
    aliases = {company, base}
    for short in short_names:
        short = short.strip()
        aliases.add(short)
    return tuple(sorted({_norm(a) for a in aliases
                         if len(_norm(a)) >= 3 and _norm(a) not in _BAD_ALIAS},
                        key=lambda x: (-len(x), x)))


def _defined_company_aliases(text: str) -> tuple[str, ...]:
    """Extract explicit glossary aliases for legally named companies.

    Only aliases coupled to a full legal company name by a visible ``指``
    definition are accepted.  Generic glossary labels are discarded.
    """
    aliases: set[str] = set()
    for lhs, legal_name in _DEFINED_COMPANY_RE.findall(text):
        legal_base = _norm(_CORP_SUFFIX_RE.sub("", legal_name))
        # The left side also contains role labels such as "董事会" and
        # "标的公司".  Accept a short form only when it is literally part of
        # the legal company name; this retains a source-defined short name while rejecting
        # generic governance/transaction vocabulary.
        left_aliases = []
        for candidate in re.split(r"[、,，/；;]", lhs):
            norm = _norm(candidate.strip())
            if norm and (norm in legal_base or legal_base in norm):
                left_aliases.append(candidate)
        candidates = left_aliases + [legal_name, _CORP_SUFFIX_RE.sub("", legal_name)]
        for candidate in candidates:
            norm = _norm(candidate.strip())
            if len(norm) >= 3 and norm not in _BAD_ALIAS:
                aliases.add(norm)
    return tuple(sorted(aliases, key=lambda x: (-len(x), x)))


@dataclass(frozen=True)
class _Identity:
    doc_id: str
    domain: str
    company_key: str
    aliases: tuple[str, ...]
    year: Optional[str]
    surface: str


@lru_cache(maxsize=None)
def _identity(doc_id: str) -> _Identity:
    meta = retrieval.docs_meta()[doc_id]
    domain = meta["domain"]
    # Identity fields occur near the front.  The larger report window also
    # handles reports whose cover pages consist only of artwork.
    if domain == "financial_reports":
        head_limit = 50_000
    elif domain == "financial_contracts":
        # Contract glossaries can begin after several cover/notice pages.
        head_limit = 20_000
    else:
        head_limit = 8_000
    head = retrieval.doc_path(doc_id).read_text(encoding="utf-8")[:head_limit]
    title = str(meta.get("title") or "")
    combined = title + "\n" + head
    companies = [_clean_company(x) for x in _CORP_RE.findall(combined)]
    primary = companies[0] if companies else ""
    shorts = _SHORT_RE.findall(combined[:12_000])
    aliases = set(_company_aliases(primary, shorts) if primary else ())
    # A target/counterparty defined in the contract glossary is an explicit
    # document identity too.  Limit this to financial contracts: reports have
    # long abbreviation tables whose incidental company references are not a
    # safe single-report routing signal.
    if domain == "financial_contracts":
        aliases.update(_defined_company_aliases(head[:20_000]))
    company_key = _norm(_CORP_SUFFIX_RE.sub("", primary))
    ym = re.search(r"(20\d{2})\s*年", title + "\n" + head[:800])
    if not ym:
        ym = re.search(r"_(20\d{2})_", doc_id)
    return _Identity(
        doc_id=doc_id,
        domain=domain,
        company_key=company_key or doc_id,
        aliases=tuple(sorted(aliases, key=lambda x: (-len(x), x))),
        year=ym.group(1) if ym else None,
        surface=_norm(combined[:12_000]),
    )


@lru_cache(maxsize=None)
def _domain_ids(domain: str) -> tuple[str, ...]:
    return tuple(d for d, m in retrieval.docs_meta().items()
                 if m["domain"] == domain)


def _quoted_title_pick(q: dict, domain: str) -> Optional[tuple[list[str], dict]]:
    anchors = [a for a in re.findall(r"《([^》]{8,160})》", q.get("question") or "")
               if len(_norm(a)) >= 10]
    if not anchors:
        return None
    picked = []
    matches = {}
    for anchor in anchors:
        na = _norm(anchor)
        hits = [d for d in _domain_ids(domain)
                if na in _identity(d).surface]
        if len(hits) != 1:
            return None
        picked.append(hits[0])
        matches[anchor] = hits[0]
    picked = list(dict.fromkeys(picked))
    return picked, {
        "method": "exact_quoted_title",
        "confidence": 1.0,
        "anchors": matches,
        "fallback": False,
    }


def _matched_company_groups(q: dict, domain: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return matched company groups and the aliases that triggered them."""
    nq = _norm(_qtext(q))
    groups: dict[str, list[_Identity]] = {}
    for doc_id in _domain_ids(domain):
        ident = _identity(doc_id)
        groups.setdefault(ident.company_key, []).append(ident)
    triggers: dict[str, str] = {}
    docs: dict[str, list[str]] = {}
    for key, identities in groups.items():
        aliases = sorted({a for ident in identities for a in ident.aliases},
                         key=lambda a: (-len(a), a))
        hit = next((a for a in aliases if a in nq), None)
        if hit:
            triggers[key] = hit
            docs[key] = [x.doc_id for x in identities]
    return triggers, docs


_MULTI_DOC_RE = re.compile(
    r"(?:两|三|四|几|多)(?:家|份|款|类|个)[^\n。？?]{0,18}"
    r"(?:公司|产品|文件|报告|机构|募集说明书|合同)")


def _requires_multiple(q: dict) -> bool:
    text = _qtext(q)
    return bool(_MULTI_DOC_RE.search(text) or
                re.search(r"以下三份|各文件|与各文件|两家公司|四家", text))


def _company_pick(q: dict, domain: str) -> Optional[tuple[list[str], dict]]:
    triggers, group_docs = _matched_company_groups(q, domain)
    if not triggers:
        return None
    # A no-option extraction/calculation is assumed to have one source unless
    # its wording explicitly requests multiple documents.  Several matched
    # company groups therefore indicate ambiguity, not permission to guess.
    if not q.get("options") and not _requires_multiple(q) and len(triggers) != 1:
        return None
    years = set(re.findall(r"20\d{2}", _qtext(q)))
    picked = []
    for key in triggers:
        docs = group_docs[key]
        if domain == "financial_reports" and years:
            dated = [d for d in docs if _identity(d).year in years]
            docs = dated or docs
        picked.extend(docs)
    picked = list(dict.fromkeys(picked))
    if _requires_multiple(q) and len(picked) < 2:
        return None
    # More than eight identities is a sign that a generic alias slipped
    # through.  Let Qwen resolve it rather than bloating the evidence context.
    if not picked or len(picked) > 8:
        return None
    return picked, {
        "method": "explicit_company_aliases",
        "confidence": 0.99,
        "matched_groups": len(triggers),
        "aliases": sorted(triggers.values()),
        "years": sorted(years),
        "fallback": False,
    }


@lru_cache(maxsize=1)
def _insurance_catalog() -> dict:
    path = PROCESSED_DIR / "insurance_titles.json"
    return json.loads(path.read_text(encoding="utf-8"))


_INSURANCE_LEGAL_SUFFIX_RE = re.compile(
    r"(?:保险股份有限公司|保险有限责任公司|股份有限公司|"
    r"有限责任公司|集团有限公司|有限公司)$"
)
_INSURANCE_SECTOR_SUFFIX_RE = re.compile(
    r"(?:财产保险|人寿保险|健康保险|养老保险|保险|财产|人寿|健康|养老|在线|集团)$"
)


def _insurance_brands(info: dict) -> tuple[str, ...]:
    """Derive insurer-name variants only from the current source identity."""

    company = _norm(info.get("company") or "")
    product = _norm(info.get("product") or "")
    values = {company}
    values.update(_norm(value) for value in info.get("company_aliases", []))
    core = _INSURANCE_LEGAL_SUFFIX_RE.sub("", company)
    values.add(core)
    reduced = core
    while reduced:
        shorter = _INSURANCE_SECTOR_SUFFIX_RE.sub("", reduced)
        if shorter == reduced:
            break
        reduced = shorter
        values.add(reduced)
    for value in tuple(values):
        without_country = re.sub(r"^(?:中华人民共和国|中国)", "", value)
        if without_country != value:
            values.add(without_country)
    for length in range(min(6, len(product)), 1, -1):
        prefix = product[:length]
        if prefix and prefix in company:
            values.add(prefix)
            break
    return tuple(sorted(
        {value for value in values if 2 <= len(value) <= 32},
        key=lambda value: (-len(value), value),
    ))


def _insurance_pick(q: dict) -> Optional[tuple[list[str], dict]]:
    nq = _norm(_qtext(q))
    catalog = _insurance_catalog()
    alias_owners: dict[str, list[str]] = {}
    for doc_id, info in catalog.items():
        for alias in info.get("alias", []):
            alias_owners.setdefault(_norm(alias), []).append(doc_id)

    picked, matches = [], {}
    for doc_id, info in catalog.items():
        brands = _insurance_brands(info)
        product = _norm(info.get("product") or "")
        aliases = [_norm(x) for x in info.get("alias", []) if len(_norm(x)) >= 2]
        hit = None
        if product and product in nq:
            hit = product
        else:
            for alias in sorted(aliases, key=lambda x: -len(x)):
                owners = alias_owners.get(alias, [])
                branded = next(
                    (brand + alias for brand in brands if brand + alias in nq),
                    "",
                )
                if branded:
                    hit = branded
                    break
                # A unique product alias is safe without a company qualifier.
                if len(owners) == 1 and alias in nq:
                    hit = alias
                    break
                # If the shared alias is genuinely bare (none of its owners'
                # brands occurs next to it), all matching products are needed.
                owner_brands = [brand for owner in owners
                                for brand in _insurance_brands(catalog[owner])]
                if alias in nq and not any((brand + alias) in nq
                                           for brand in owner_brands):
                    hit = alias
                    break
        if hit:
            picked.append(doc_id)
            matches[doc_id] = hit
    picked = list(dict.fromkeys(picked))
    if _requires_multiple(q) and len(picked) < 2:
        return None
    if not picked or len(picked) > 10:
        return None
    return picked, {
        "method": "insurance_product_aliases",
        "confidence": 0.995,
        "matched_products": matches,
        "fallback": False,
    }


def select_docs_fast(q: dict) -> Optional[tuple[list[str], dict]]:
    """Return a zero-token document decision, or ``None`` for Qwen fallback.

    The result is invariant to ``q['qid']``: only domain, question and options
    are inspected.  Picks preserve corpus order for reproducibility.
    """
    domain = q.get("domain")
    if domain not in {"financial_contracts", "financial_reports", "insurance",
                      "regulatory", "research"}:
        return None
    exact = _quoted_title_pick(q, domain)
    if exact is not None:
        return exact
    if domain == "insurance":
        return _insurance_pick(q)
    if domain in {"regulatory", "research"}:
        return None
    company = _company_pick(q, domain)
    if company is not None:
        return company
    return None


__all__ = ["select_docs_fast"]
