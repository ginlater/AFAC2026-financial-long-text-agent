#!/usr/bin/env python3
"""Build portable insurance identity cards from parsed policy text.

The lexical parser extracts insurer and product identities from
``processed_data/insurance/*.txt`` for source routing.

The parser handles common PDF extraction artifacts:

* the insurer and title are joined on one line;
* OCR spaces split a title (``A 款`` / ``2025 版``);
* dot leaders split the title over several lines; and
* the useful title appears after a reading guide rather than on page one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


_PAGE_TAG_RE = re.compile(r"^\[P\d+\]$")
_COMPANY_SUFFIX_RE = re.compile(r"(?:保险股份有限公司|保险有限责任公司)$")
_COMPANY_ANY_RE = re.compile(
    r"[\u4e00-\u9fff]{2,32}(?:保险股份有限公司|保险有限责任公司)"
)
_EXPLICIT_ALIAS_RE = re.compile(
    r"险种简称\s*[:：]\s*[“\"']?([^\s，。；;”\"']{2,24})"
)
_COMPANY_ALIAS_FIELD_RE = re.compile(
    r"(?:公司中文简称|公司的中文简称|公司简称|企业简称)\s*[:：]?\s*"
    r"[“\"']?([^\s，。；;”\"']{2,24})"
)

# Longer, more specific product types must precede their shorter suffixes.
_PRODUCT_TYPES = (
    "专属商业养老保险",
    "家庭财产综合保险",
    "商业保险示范条款",
    "养老年金保险",
    "重大疾病保险",
    "意外伤害保险",
    "家庭财产保险",
    "终身寿险",
    "医疗保险",
    "责任保险",
    "商业保险",
    "财产保险",
    "年金保险",
    "健康保险",
)

_TITLE_NOISE = (
    "本保险合同由",
    "保险合同由",
    "本保险条款、",
    "以下简称",
    "内容的解释",
    "本阅读指引",
    "我们提供的保障",
    "产品提供",
    "合同",
    "保险公司",
    "保险责任",
)

_SENTENCE_HINT_RE = re.compile(
    r"(?:适用|依据|若|未经|包括|取得补偿|赔付|结算|颁布|定义|给付|另有|"
    r"被保险人|投保人|受益人|我们|本公司)"
)

_GENERIC_ALIASES = {
    "保险",
    "保险条款",
    "商业",
    "财产",
    "责任",
    "医疗",
    "意外伤害",
    "重大疾病",
    "养老年金",
    "终身寿",
    "个人",
    "团体",
}
_GENERIC_COMPANY_ALIASES = _GENERIC_ALIASES | {
    "本公司", "公司", "集团", "股份", "有限", "在线",
}
_COMPANY_SECTOR_SUFFIXES = (
    "保险", "财产", "人寿", "健康", "养老", "在线", "集团",
)


def _natural_doc_key(path: Path) -> tuple[int, int | str]:
    """Sort numeric doc ids numerically, then all other ids lexically."""

    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _compact(text: str) -> str:
    """Normalize title-like text without changing substantive characters."""

    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    # PDF underlines often arrive as a run of U+FF0E between every character.
    text = text.replace("．", "").replace("…", "")
    text = re.sub(r"[\t\r\n ]+", "", text)
    text = re.sub(r"([A-Za-z0-9])款", r"\1款", text)
    text = re.sub(r"(20\d{2})版", r"\1版", text)
    return text.strip("·•◆◇*-—_，,。；;：:")


def _plain_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or _PAGE_TAG_RE.fullmatch(raw):
            continue
        compact = _compact(raw)
        if compact:
            lines.append(compact)
    return lines


def _clean_company_candidate(candidate: str) -> str:
    candidate = _compact(candidate).strip("（）()【】[]“”\"'")
    # A company in explanatory prose is commonly introduced by one of these
    # phrases.  Retain only the text after the last such boundary.
    for marker in ("均指", "我们指", "本公司指", "保险人指", "是指", "指"):
        if marker in candidate:
            candidate = candidate.rsplit(marker, 1)[-1]
    match = _COMPANY_ANY_RE.search(candidate)
    if not match:
        return ""
    company = match.group(0)
    # Avoid an over-greedy Chinese prefix from surrounding prose.
    for boundary in ("为", "由", "和", "与", "，", "。", "（", "("):
        if boundary in company:
            tail = company.rsplit(boundary, 1)[-1]
            if _COMPANY_SUFFIX_RE.search(tail) and len(tail) >= 8:
                company = tail
    return company


def _extract_company(lines: list[str]) -> str:
    # Exact company-only lines are the least ambiguous evidence.
    for line in lines:
        if _COMPANY_SUFFIX_RE.search(line) and _COMPANY_ANY_RE.fullmatch(line):
            return line

    candidates: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        for match in _COMPANY_ANY_RE.finditer(line):
            company = _clean_company_candidate(match.group(0))
            if not company:
                continue
            # Earlier mentions and shorter (less prose-contaminated) strings win.
            candidates.append((index, len(company), company))
    if not candidates:
        return ""
    return min(candidates)[2]


def _strip_company_prefix(title: str, company: str) -> str:
    if company and title.startswith(company):
        title = title[len(company) :]
    return title.lstrip("：:，,。-—")


def _canonical_title(candidate: str, company: str) -> str:
    title = _compact(candidate)
    title = re.sub(r"^(?:第?[〇零一二三四五六七八九十百\d]+页|-?\d+-?)", "", title)
    title = _strip_company_prefix(title, company)
    title = title.strip("【】[]“”\"'，,。；;：:")

    # Remove extraction-only page annotations while retaining a genuine 条款.
    title = re.sub(r"（第[〇零一二三四五六七八九十百\d]+页）$", "", title)
    title = re.sub(r"（第[〇零一二三四五六七八九十百\d]+页$", "", title)
    title = re.sub(r"(?:利益)?条款（第[〇零一二三四五六七八九十百\d]+页）$", "", title)

    # A quoted title can be followed by prose when OCR loses the closing quote.
    for stop in ("内容的解释", "以下简称", "本阅读指引", "注册号"):
        if stop in title:
            title = title.split(stop, 1)[0]
    title = re.sub(r"利益条款$", "", title)
    return title.strip("【】[]“”\"'，,。；;：:")


def _looks_like_product(title: str) -> bool:
    if not (5 <= len(title) <= 90):
        return False
    if not any(kind in title for kind in _PRODUCT_TYPES):
        return False
    if _COMPANY_SUFFIX_RE.fullmatch(title):
        return False
    if _COMPANY_ANY_RE.search(title) or title.endswith(("（", "(")):
        return False
    if re.search(r"[，。；：“”]", title):
        return False
    if _SENTENCE_HINT_RE.search(title):
        return False
    return not any(noise in title for noise in _TITLE_NOISE)


def _title_score(title: str, line_index: int, source: str) -> int:
    # An explicitly quoted title spanning OCR-broken lines is stronger than a
    # short fragment which merely happens to end in “保险”.
    score = 100 if source == "line" else 95
    score += max(0, 15 - min(line_index, 15))
    score += max(len(kind) for kind in _PRODUCT_TYPES if kind in title)
    score += min(len(title), 30)
    if len(title) < 12:
        score -= (12 - len(title)) * 5
    if title.endswith("条款") or re.search(r"[）)]条款$", title):
        score += 12
    if re.search(r"20\d{2}版|[A-ZＥ]款|互联网|分红型|万能型", title):
        score += 6
    # Long lines are more likely to have swallowed nearby prose.
    score -= max(0, len(title) - 55)
    return score


def _quoted_candidates(text: str) -> Iterable[str]:
    # OCR may put dot leaders and newlines between every title character, so
    # search a compacted prefix as well as individual source lines.
    prefix = _compact(text[:16000])
    for match in re.finditer(r"[“\"]([^”\"]{5,100}(?:保险|寿险)[^”\"]{0,30})[”\"]", prefix):
        yield match.group(1)


def _extract_product(text: str, lines: list[str], company: str) -> str:
    candidates: list[tuple[int, int, str]] = []

    for index, line in enumerate(lines):
        title = _canonical_title(line, company)
        if _looks_like_product(title):
            candidates.append((_title_score(title, index, "line"), -index, title))

    for quoted in _quoted_candidates(text):
        title = _canonical_title(quoted, company)
        if _looks_like_product(title):
            candidates.append((_title_score(title, 25, "quote"), -25, title))

    if not candidates:
        return ""
    # Stable tie breaking: score, then earlier occurrence, then shorter title.
    candidates.sort(key=lambda row: (-row[0], -row[1], len(row[2]), row[2]))
    return candidates[0][2]


def _is_ordered_subsequence(needle: str, haystack: str) -> bool:
    cursor = 0
    for char in haystack:
        if cursor < len(needle) and char == needle[cursor]:
            cursor += 1
    return cursor == len(needle)


def _source_company_aliases(text: str, company: str) -> set[str]:
    """Extract only aliases explicitly coupled to the source company."""
    sample = text[:20_000]
    aliases = {match.group(1) for match in _COMPANY_ALIAS_FIELD_RE.finditer(sample)}
    if company:
        contextual = re.compile(
            re.escape(company) +
            r".{0,80}?(?:以下简称|简称(?:为)?)\s*[“\"']([^\s，。；;”\"']{2,24})",
            re.S,
        )
        aliases.update(match.group(1) for match in contextual.finditer(sample))
    return {
        alias for raw in aliases
        if 2 <= len(alias := _compact(raw)) <= 24
        and alias not in _GENERIC_COMPANY_ALIASES
    }


def _brand_prefixes(company: str, product: str = "", text: str = "") -> list[str]:
    """Derive removable organization prefixes from the current source only."""
    company = _compact(company)
    product = _compact(product)
    core = _COMPANY_SUFFIX_RE.sub("", company)
    variants = {company, core, *_source_company_aliases(text, company)}

    reduced = core
    while reduced:
        matched = False
        for suffix in _COMPANY_SECTOR_SUFFIXES:
            if reduced.endswith(suffix):
                reduced = reduced[:-len(suffix)]
                variants.add(reduced)
                matched = True
                break
        if not matched:
            break

    for value in tuple(variants):
        without_country = re.sub(r"^(?:中华人民共和国|中国)", "", value)
        if without_country != value:
            variants.add(without_country)

    # If a product begins with an undeclared short organization form, accept
    # it only when its characters occur in source-company order.
    for length in range(min(6, len(product)), 1, -1):
        prefix = product[:length]
        if (prefix in company or
                (length <= 4 and _is_ordered_subsequence(prefix, company))):
            variants.add(prefix)
            break

    return sorted(
        {value for value in variants
         if 2 <= len(value) <= 32 and value not in _GENERIC_COMPANY_ALIASES},
        key=lambda value: (-len(value), value),
    )


def _product_stem(product: str, company: str, text: str = "") -> str:
    stem = _strip_company_prefix(_compact(product), company)
    stem = re.sub(r"（[^（）]{1,30}）", "", stem)
    stem = re.sub(r"条款$", "", stem)
    for prefix in _brand_prefixes(company, product, text):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break

    # Peel generic insurance taxonomy from the right, leaving the distinctive
    # product name or insured subject.  Repetition handles e.g. 责任保险条款.
    suffixes = (
        r"专属商业养老保险",
        r"养老年金保险",
        r"终身寿险",
        r"重大疾病保险",
        r"意外伤害保险",
        r"医疗保险",
        r"综合保险",
        r"责任保险",
        r"商业保险",
        r"保险",
        r"团体",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            new = re.sub(rf"(?:{suffix})$", "", stem)
            if new != stem:
                stem, changed = new, True
    return stem.strip("（）()【】[]-—_，,。")


def _extract_aliases(text: str, product: str, company: str) -> list[str]:
    aliases: list[str] = []

    for match in _EXPLICIT_ALIAS_RE.finditer(text):
        aliases.append(match.group(1))

    stem = _product_stem(product, company, text)
    if stem:
        aliases.append(stem)
        # Audience words are not normally part of a product's memorable name.
        trimmed = re.sub(r"^(?:个人|团体)", "", stem)
        if trimmed != stem:
            aliases.append(trimmed)

    # Reattach a generic category word when the compact stem is otherwise too
    # terse.  Both pieces are derived from this product title.
    for product_type in _PRODUCT_TYPES:
        if product_type not in product:
            continue
        category = re.sub(r"(?:综合|商业|示范条款|保险)$", "", product_type)
        if stem and category and not stem.endswith(category):
            aliases.append(stem + category)
        break

    result: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        alias = _compact(alias)
        if not (2 <= len(alias) <= 24):
            continue
        if alias in _GENERIC_ALIASES or alias in seen or alias == product or alias == company:
            continue
        seen.add(alias)
        result.append(alias)
    return result


def build(
    processed_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, dict[str, str | list[str]]]:
    """Build and write ``insurance_titles.json``.

    Args:
        processed_dir: Directory containing an ``insurance`` text directory.
        output_path: Optional destination.  Defaults to
            ``processed_dir/insurance_titles.json``.

    Returns:
        The deterministic document-id keyed identity mapping.

    Raises:
        FileNotFoundError: if no parsed insurance documents are present.
        ValueError: if a source document has no recoverable company or title.
    """

    processed_dir = Path(processed_dir).expanduser().resolve()
    insurance_dir = processed_dir / "insurance"
    paths = sorted(insurance_dir.glob("*.txt"), key=_natural_doc_key)
    if not paths:
        raise FileNotFoundError(f"no parsed insurance texts found under {insurance_dir}")

    records: dict[str, dict[str, str | list[str]]] = {}
    failures: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lines = _plain_lines(text)
        company = _extract_company(lines)
        product = _extract_product(text, lines, company)
        if not company or not product:
            missing = ",".join(
                name for name, value in (("company", company), ("product", product)) if not value
            )
            failures.append(f"{path.name}:{missing}")
            continue
        records[path.stem] = {
            "company": company,
            "company_aliases": sorted(_source_company_aliases(text, company)),
            "product": product,
            "alias": _extract_aliases(text, product, company),
        }

    if failures:
        raise ValueError("insurance identity extraction failed: " + "; ".join(failures))

    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else processed_dir / "insurance_titles.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(records, ensure_ascii=False, indent=1, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic insurer/product identity cards from parsed insurance text."
    )
    parser.add_argument(
        "--processed",
        required=True,
        type=Path,
        help="processed_data directory containing insurance/*.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSON path (default: PROCESSED/insurance_titles.json)",
    )
    args = parser.parse_args()
    records = build(args.processed, args.output)
    alias_count = sum(len(record["alias"]) for record in records.values())
    print(f"wrote {len(records)} insurance identities ({alias_count} aliases)")


if __name__ == "__main__":
    main()
