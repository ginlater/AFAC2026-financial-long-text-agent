"""Deterministic, source-derived memory capsules for insurance documents.

The companion builder segments parsed policy text into verbatim clause cards;
query-time selection ranks those cards by lexical overlap.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

from .paths import PROCESSED_DIR


SCHEMA_VERSION = "insurance_capsule.v1"
DEFAULT_PATH = PROCESSED_DIR / "insurance_capsules.json"

# A small domain ontology helps balance evidence without encoding particular
# policy values or exception scenarios. Every pattern is a broad term common
# to insurance contracts; unmatched clauses remain available as generic cards.
TOPIC_RULES = (
    ("coverage", "保障与给付", r"保险责任|保障责任|保险金|给付|赔偿"),
    ("exclusion", "责任限制", r"责任免除|除外责任|不承担|不负责|限制"),
    ("eligibility", "主体与资格", r"投保范围|投保条件|投保人|被保险人|受益人"),
    ("duration", "期间与时间", r"保险期间|等待期|宽限期|期限|届满|生效|终止"),
    ("financial_terms", "金额与费用", r"保险费|现金价值|账户价值|利率|金额|限额|免赔|比例|费用"),
    ("contract_changes", "合同变更", r"解除|变更|中止|恢复|转让|领取|借款"),
    ("claims", "申请与理赔", r"理赔|索赔|申请|通知|证明|材料|审核"),
    ("definitions", "定义与解释", r"释义|定义|是指|本条款"),
)

_TOPIC_REGEX = tuple((key, label, re.compile(pattern))
                     for key, label, pattern in TOPIC_RULES)
TOPIC_LABELS = {key: label for key, label, _pattern in TOPIC_RULES}
TOPIC_LABELS["other_clause"] = "其他条款"

_NUMBER_RE = re.compile(
    r"(?:\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?)|"
    r"(?:-?\d[\d,]*(?:\.\d+)?\s*(?:%|％|个工作日|工作日|周岁|岁|个月|月|日|天|"
    r"小时|分钟|元|万元|亿元|倍|份|次|项|年))|"
    r"(?:[零〇一二三四五六七八九十百千万两]+(?:个)?(?:工作日|周岁|岁|年|月|日|天|次|项|份))"
)
_LEX_RE = re.compile(r"\d+(?:\.\d+)?%?|[A-Za-z]+|[一-鿿]+")
_GENERIC_QUERY_WORDS = {
    "保险", "责任", "条款", "产品", "公司", "明确", "列明", "规定",
    "约定", "包含", "关于", "说法", "正确", "下列", "选项", "处理",
}


def _literal_query_matches(query: str, text: str) -> set[str]:
    """Return maximal non-generic literal phrases shared by two strings."""
    matches = set()
    compact_query = re.sub(r"\s+", "", query or "")
    for run in re.findall(r"[A-Za-z0-9.%％一-鿿]+", compact_query):
        for length in range(2, min(10, len(run)) + 1):
            for start in range(len(run) - length + 1):
                phrase = run[start:start + length]
                if phrase in text and phrase not in _GENERIC_QUERY_WORDS:
                    matches.add(phrase)
    return {
        phrase for phrase in matches
        if not any(phrase != other and phrase in other for other in matches)
    }


def _quoted_query_anchors(query: str) -> set[str]:
    """Extract literal target terms explicitly quoted in a query."""
    anchors = set()
    for quoted in re.findall(r"[“\"]([^”\"]{2,80})[”\"]", query or ""):
        for part in re.split(r"[、,，/]|(?:或)|(?:以及)|(?:等)", quoted):
            part = re.sub(r"^[\s‘’'（(]+|[\s‘’'）)]+$", "", part)
            if len(part) >= 2 and part not in _GENERIC_QUERY_WORDS:
                anchors.add(part)
    return anchors


def topic_scores(text: str, title: str = "") -> dict[str, int]:
    """Return broad topic-match counts, with a deterministic title boost."""
    scores = {}
    for key, _label, pattern in _TOPIC_REGEX:
        body_count = len(pattern.findall(text or ""))
        title_count = len(pattern.findall(title or ""))
        if body_count or title_count:
            scores[key] = body_count + 3 * title_count
    return scores


def infer_topics(text: str, title: str = "") -> list[str]:
    """Infer ordered broad topics from source wording."""
    scores = topic_scores(text, title)
    order = {key: index for index, (key, _label, _pattern)
             in enumerate(TOPIC_RULES)}
    return sorted(scores, key=lambda key: (-scores[key], order[key]))


def extract_numbers(text: str) -> list[str]:
    """Extract value-and-unit strings in source order, de-duplicated."""
    values = []
    seen = set()
    for match in _NUMBER_RE.finditer((text or "").replace("％", "%")):
        value = re.sub(r"\s+", "", match.group(0))
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _lexical_tokens(text: str) -> Counter:
    tokens = []
    for raw in _LEX_RE.findall((text or "").replace("％", "%")):
        if re.fullmatch(r"[一-鿿]+", raw):
            tokens.extend(raw)
            tokens.extend(raw[index:index + 2]
                          for index in range(len(raw) - 1))
        else:
            tokens.append(raw.lower())
    return Counter(tokens)


_CACHE: dict[str, tuple[int, dict]] = {}


def load_capsules(path: str | pathlib.Path | None = None, *, force=False) -> dict:
    """Load and validate a capsule artifact, cached by path and mtime."""
    source = pathlib.Path(path or DEFAULT_PATH).resolve()
    if not source.exists():
        return {"schema_version": SCHEMA_VERSION, "documents": {}, "stats": {}}
    stamp = source.stat().st_mtime_ns
    cached = _CACHE.get(str(source))
    if cached and cached[0] == stamp and not force:
        return cached[1]
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported insurance capsule schema: {data.get('schema_version')!r}"
        )
    if not isinstance(data.get("documents"), dict):
        raise ValueError("insurance capsule artifact has no documents mapping")
    _CACHE[str(source)] = (stamp, data)
    return data


def _query_text(question: dict) -> str:
    options = question.get("options") or {}
    return ((question.get("question") or "") + " " +
            " ".join(str(value) for value in options.values()))


def _score(card: dict, identity: dict, query_tokens: Counter,
           query_numbers: set[str], query_topics: set[str], query: str) -> float:
    source_text = " ".join((
        identity.get("company", ""), identity.get("product", ""),
        " ".join(identity.get("aliases") or []), card.get("clause_title", ""),
        card.get("verbatim", ""),
    ))
    card_tokens = _lexical_tokens(source_text)
    overlap = 0.0
    for token, count in query_tokens.items():
        if token not in card_tokens:
            continue
        weight = 1.8 if len(token) == 2 and "一" <= token[0] <= "鿿" else 0.6
        if token[:1].isdigit() or token.endswith("%"):
            weight = 4.0
        overlap += min(count, card_tokens[token]) * weight

    card_topics = {card.get("topic"), *(card.get("tags") or [])}
    topic_bonus = 16.0 * len(query_topics & card_topics)
    number_bonus = 6.0 * len(query_numbers & set(card.get("numbers") or []))
    title = re.sub(r"[\s：:（）()]+", "", card.get("clause_title", ""))
    compact_query = re.sub(r"\s+", "", query or "")
    title_bonus = 24.0 if len(title) >= 2 and title in compact_query else 0.0
    literal_bonus = sum(4.0 * len(phrase) ** 2
                        for phrase in _literal_query_matches(query, source_text))
    anchor_bonus = 60.0 * sum(
        1 for anchor in _quoted_query_anchors(query) if anchor in source_text
    )
    return overlap + topic_bonus + number_bonus + title_bonus + literal_bonus + anchor_bonus


def select_capsules(question: dict, *, path: str | pathlib.Path | None = None,
                    max_cards: int = 24, per_doc: int = 2,
                    char_budget: int = 8500) -> list[dict]:
    """Select balanced, query-relevant verbatim cards from selected documents."""
    if question.get("domain") != "insurance" or not question.get("doc_ids"):
        return []
    data = load_capsules(path)
    query = _query_text(question)
    query_tokens = _lexical_tokens(query)
    query_numbers = set(extract_numbers(query))
    query_topics = set(infer_topics(query))
    doc_order = [str(value) for value in question.get("doc_ids") or []]
    ranked = {}
    for doc_id in doc_order:
        document = data["documents"].get(doc_id)
        if not document:
            continue
        identity = document.get("identity") or {}
        rows = []
        for card in document.get("capsules") or []:
            score = _score(card, identity, query_tokens, query_numbers,
                           query_topics, query)
            if score > 0:
                rows.append((score, card.get("id", ""), card, identity))
        rows.sort(key=lambda item: (-item[0], item[1]))
        ranked[doc_id] = rows

    selected = []
    seen = set()
    used = 0

    def take(item) -> bool:
        nonlocal used
        score, _card_id, card, identity = item
        card_id = card.get("id")
        cost = len(card.get("verbatim", "")) + 150
        if (card_id in seen or len(selected) >= max_cards or
                used + cost > char_budget):
            return False
        row = dict(card)
        row["identity"] = identity
        row["score"] = round(score, 3)
        selected.append(row)
        seen.add(card_id)
        used += cost
        return True

    # Allocate a small base share to each selected source before global ranking.
    for position in range(max(0, per_doc)):
        for doc_id in doc_order:
            rows = ranked.get(doc_id) or []
            if position < len(rows):
                take(rows[position])

    all_rows = [item for rows in ranked.values() for item in rows]
    all_rows.sort(key=lambda item: (-item[0], item[1]))
    for item in all_rows:
        take(item)
        if len(selected) >= max_cards:
            break
    return selected


def insurance_capsule_block(
        question: dict, *, path: str | pathlib.Path | None = None,
        max_cards: int = 24, per_doc: int = 2,
        char_budget: int = 8500) -> str:
    """Render selected source cards as a compact prompt evidence block."""
    cards = select_capsules(question, path=path, max_cards=max_cards,
                            per_doc=per_doc, char_budget=char_budget)
    if not cards:
        return ""
    parts = ["保险条款词法记忆卡（以下内容均直接摘自当前输入文档）:"]
    for card in cards:
        identity = card.get("identity") or {}
        clause = " ".join(value for value in (
            card.get("clause"), card.get("clause_title")) if value).strip()
        topic = TOPIC_LABELS.get(card.get("topic"), card.get("topic", ""))
        numbers = "、".join(card.get("numbers") or []) or "无显式数字"
        metadata = (
            f"◆doc={card.get('doc_id')}｜{identity.get('product', '')}｜"
            f"P{card.get('page')}｜{clause or '未编号条款'}｜"
            f"主题={topic}｜数字={numbers}"
        )
        entry = metadata + "\n原句：" + (card.get("verbatim") or "").strip()
        if len("\n".join(parts + [entry])) > char_budget:
            break
        parts.append(entry)
    return "\n".join(parts)
