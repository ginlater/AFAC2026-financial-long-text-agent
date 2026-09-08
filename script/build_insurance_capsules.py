#!/usr/bin/env python3
"""Build deterministic typed-memory capsules for parsed insurance clauses.

Inputs (no model calls):
  processed_data/insurance/*.txt
  processed_data/insurance_titles.json

Output:
  processed_data/insurance_capsules.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from collections import Counter

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from agent.insurance_capsules import (  # noqa: E402
    SCHEMA_VERSION, extract_numbers, infer_topics,
)

PD = WORK / "processed_data"
DEFAULT_OUT = PD / "insurance_capsules.json"

PAGE_RE = re.compile(r"^\[P(\d+)\]\s*$")
CN_CLAUSE_RE = re.compile(r"^(第[零〇一二三四五六七八九十百千万两\d]+条)\s*(.*)$")
DEC_CLAUSE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})\s*(.*)$")
DOTS_RE = re.compile(r"[.。·…]{5,}")
SECTION_HEADING_RE = re.compile(
    r"(?:总则|释义|定义|责任|义务|权益|期间|费用|处理|适用|解除|变更)$"
)


def _normal(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").replace("％", "%"))


def _split_pages(text: str) -> list[tuple[int, list[str]]]:
    pages = []
    page = 0
    lines = []
    for raw in text.splitlines():
        match = PAGE_RE.match(raw.strip())
        if match:
            if page or lines:
                pages.append((page, lines))
            page = int(match.group(1))
            lines = []
        else:
            lines.append(raw.rstrip())
    if page or lines:
        pages.append((page, lines))
    return pages


def _is_toc(lines: list[str]) -> bool:
    text = "\n".join(lines)
    headers = sum(bool(CN_CLAUSE_RE.match(line.strip()) or
                       DEC_CLAUSE_RE.match(line.strip())) for line in lines)
    return "条款目录" in text or (headers >= 7 and len(DOTS_RE.findall(text)) >= 3)


def _looks_like_title(rest: str, *, decimal=False) -> bool:
    rest = rest.strip()
    if not rest or len(rest) > (50 if decimal else 32):
        return False
    if re.search(r"[。；，：]", rest):
        return False
    return True


def _is_section_heading(line: str) -> bool:
    """Recognize short structural headings without a fixed title catalogue."""
    value = line.strip()
    # Table values such as "不适用" are clause content, not headings.
    return bool(value != "不适用" and 2 <= len(value) <= 24 and
                not re.search(r"[。；，：:]", value) and
                SECTION_HEADING_RE.search(value))


def _split_chunks(lines: list[str], max_chars: int) -> list[str]:
    atoms = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            atoms.append(line)
            continue
        sentences = [x for x in re.split(r"(?<=[。；！？])", line) if x]
        for sentence in sentences:
            if len(sentence) <= max_chars:
                atoms.append(sentence)
            else:
                atoms.extend(sentence[i:i + max_chars]
                             for i in range(0, len(sentence), max_chars))
    chunks, buf = [], []
    size = 0
    for atom in atoms:
        extra = len(atom) + (1 if buf else 0)
        if buf and size + extra > max_chars:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(atom)
        size += len(atom) + (1 if len(buf) > 1 else 0)
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _parse_sections(text: str) -> list[dict]:
    """Return page-bound clause bodies while propagating headings across pages."""
    groups = []
    clause = ""
    title = ""
    section = ""
    pending_code = ""
    title_open = False

    for page, lines in _split_pages(text):
        if _is_toc(lines):
            continue
        current = None
        seen_page_content = False

        def append_body(value: str):
            nonlocal current
            if not clause or not value:
                return
            key = (page, clause, title, section)
            if current is None or current["key"] != key:
                current = {"key": key, "page": page, "clause": clause,
                           "clause_title": title or section, "lines": []}
                groups.append(current)
            current["lines"].append(value)

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            # Several PDF parsers emit the printed page number as the first
            # line.  Later one-digit lines can be footnote markers and must be
            # retained to keep ``verbatim`` traceable to the raw page.
            if not seen_page_content and re.fullmatch(r"\d+", line):
                seen_page_content = True
                continue
            seen_page_content = True
            if re.fullmatch(r"-\d+-", line):
                continue
            cn = CN_CLAUSE_RE.match(line)
            dec = DEC_CLAUSE_RE.match(line)
            if cn or dec:
                match = cn or dec
                code, rest = match.group(1), match.group(2).strip()
                clause = code
                current = None
                pending_code = ""
                title_open = False
                decimal = dec is not None
                if _looks_like_title(rest, decimal=decimal):
                    title = rest
                    title_open = True
                elif rest:
                    title = section
                    append_body(rest)
                else:
                    title = ""
                    pending_code = code
                continue

            if _is_section_heading(line):
                section = line
                continue

            if pending_code:
                if len(line) <= 50 and not re.search(r"[。；，：]", line):
                    title = line
                    title_open = True
                    pending_code = ""
                    continue
                pending_code = ""
                title = section

            if (title_open and title and len(title) <= 12 and len(line) <= 12 and
                    len(title) + len(line) <= 36 and
                    not re.match(r"^[（(]?[0-9一二三四五六七八九十]+[）)．.]", line) and
                    _looks_like_title(line, decimal=True)):
                title += line
                title_open = False
                current = None
                continue
            title_open = False
            append_body(line)
    return [g for g in groups if g["lines"]]


def _source_hash(raw: str, identity: dict) -> str:
    h = hashlib.sha256()
    h.update(raw.encode("utf-8"))
    h.update(json.dumps(identity, ensure_ascii=False,
                        sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def build_document(doc_id: str, raw: str, identity: dict,
                   max_chars: int) -> tuple[dict, dict]:
    cards = []
    seq = 0
    seen = set()

    def add_card(page, clause, clause_title, verbatim, topics):
        nonlocal seq
        if not verbatim.strip() or not topics:
            return
        key = (page, clause, topics[0], _normal(verbatim))
        if key in seen:
            return
        seen.add(key)
        seq += 1
        cards.append({
            "id": f"{doc_id}:p{page}:{seq:04d}",
            "doc_id": doc_id,
            "page": page,
            "clause": clause,
            "clause_title": clause_title,
            "topic": topics[0],
            "tags": topics[1:],
            "numbers": extract_numbers(verbatim),
            "verbatim": verbatim.strip(),
            "sources": ["raw_text"],
        })

    for group in _parse_sections(raw):
        for chunk in _split_chunks(group["lines"], max_chars):
            topics = infer_topics(chunk, group["clause_title"])
            if not topics:
                topics = ["other_clause"]
            add_card(group["page"], group["clause"],
                     group["clause_title"], chunk, topics)

    doc = {
        "identity": {
            "company": identity.get("company", ""),
            "company_aliases": identity.get("company_aliases", []),
            "product": identity.get("product", ""),
            "aliases": identity.get("alias", identity.get("aliases", [])),
        },
        "source_sha256": _source_hash(raw, identity),
        "capsules": cards,
    }
    stats = {"cards": len(cards)}
    return doc, stats


def build(max_chars: int = 520, processed_dir: pathlib.Path = PD) -> dict:
    processed_dir = pathlib.Path(processed_dir)
    insurance_dir = processed_dir / "insurance"
    titles = json.loads((processed_dir / "insurance_titles.json").read_text(encoding="utf-8"))
    documents = {}
    doc_stats = {}
    topic_counts = Counter()
    topic_membership = Counter()
    lengths = []
    numeric = 0

    def sort_key(path):
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)

    for path in sorted(insurance_dir.glob("*.txt"), key=sort_key):
        doc_id = path.stem
        raw = path.read_text(encoding="utf-8", errors="ignore")
        doc, stats = build_document(
            doc_id, raw, titles.get(doc_id, {}), max_chars)
        documents[doc_id] = doc
        doc_stats[doc_id] = stats
        for card in doc["capsules"]:
            topic_counts[card["topic"]] += 1
            topic_membership.update([card["topic"], *(card.get("tags") or [])])
            lengths.append(len(card["verbatim"]))
            numeric += bool(card["numbers"])

    total = sum(x["cards"] for x in doc_stats.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "builder": "script/build_insurance_capsules.py",
        "built_from": [
            "processed_data/insurance/*.txt",
            "processed_data/insurance_titles.json",
        ],
        "documents": documents,
        "stats": {
            "documents": len(documents),
            "capsules": total,
            "with_numbers": numeric,
            "max_verbatim_chars": max(lengths, default=0),
            "average_verbatim_chars": round(sum(lengths) / max(len(lengths), 1), 1),
            "primary_topics": dict(sorted(topic_counts.items())),
            "topic_membership": dict(sorted(topic_membership.items())),
            "per_document": doc_stats,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=pathlib.Path, default=PD)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--max-chars", type=int, default=520)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    artifact = build(max_chars=args.max_chars, processed_dir=args.processed)
    if not args.stats_only:
        out = args.output or args.processed / "insurance_capsules.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
    print(json.dumps(artifact["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
