#!/usr/bin/env python3
"""Deterministically parse raw AFAC documents into portable plain-text files.

PDF pages receive ``[P<n>]`` markers, HTML metadata is retained, and TXT files
are copied verbatim.  This step uses no semantic model.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import pymupdf as fitz
from bs4 import BeautifulSoup

fitz.TOOLS.mupdf_display_errors(False)


def clean_page(text: str) -> str:
    text = text.replace(" ", " ").replace("", "·")
    lines = [line.rstrip() for line in text.split("\n")]
    merged, buffer = [], []
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 1 and not stripped.isdigit():
            buffer.append(stripped)
            continue
        if buffer:
            merged.append("".join(buffer) if len(buffer) >= 3 else "\n".join(buffer))
            buffer = []
        merged.append(line)
    if buffer:
        merged.append("".join(buffer) if len(buffer) >= 3 else "\n".join(buffer))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(merged)).strip()


def parse_pdf(path: pathlib.Path) -> tuple[str, dict]:
    with fitz.open(path) as document:
        pages = [f"[P{i + 1}]\n{clean_page(page.get_text())}"
                 for i, page in enumerate(document)]
        first_lines = []
        if document.page_count:
            first_lines = [line.strip() for line in document[0].get_text().splitlines()
                           if line.strip()][:8]
        title = max(first_lines, key=len) if first_lines else path.stem
        meta = {"n_pages": document.page_count, "title": title}
    return "\n\n".join(pages), meta


def parse_html(path: pathlib.Path) -> tuple[str, dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    metas = {}
    for item in soup.find_all("meta"):
        name = (item.get("name") or "").strip()
        if name in {"ArticleTitle", "PubDate", "ColumnName", "description", "Description"}:
            value = (item.get("content") or "").strip()
            if value and name not in metas:
                metas[name] = value
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    body = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    title = metas.get("ArticleTitle", path.stem)
    header = (f"标题：{title}\n栏目：{metas.get('ColumnName', '')}\n"
              f"发布日期：{metas.get('PubDate', '')}\n")
    return header + "\n" + body, {
        "n_pages": None,
        "title": title,
        "pub_date": metas.get("PubDate", ""),
        "column": metas.get("ColumnName", ""),
    }


def discover_jobs(raw_root: pathlib.Path) -> list[tuple[str, str, pathlib.Path, str]]:
    raw_root = pathlib.Path(raw_root).resolve()
    jobs = []
    specs = [
        ("insurance", raw_root / "insurance", "*.pdf", "pdf"),
        ("financial_contracts", raw_root / "financial_contracts", "*.pdf", "pdf"),
        ("financial_reports", raw_root / "financial_reports", "*.[pP][dD][fF]", "pdf"),
        ("research", raw_root / "research", "*.pdf", "pdf"),
        ("regulatory", raw_root / "regulatory" / "txt", "*.txt", "txt"),
        ("regulatory", raw_root / "regulatory" / "html", "*.html", "html"),
        ("regulatory", raw_root / "regulatory" / "attachments", "*.pdf", "pdf"),
    ]
    for domain, directory, pattern, kind in specs:
        for path in sorted(directory.glob(pattern), key=lambda value: value.as_posix()):
            jobs.append((domain, path.stem, path, kind))
    return jobs


def build(raw_root: pathlib.Path, output_dir: pathlib.Path, *, strict: bool = True) -> dict:
    """Parse all discovered documents and return ``docs_meta``."""
    raw_root = pathlib.Path(raw_root).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_meta, errors = {}, []
    jobs = discover_jobs(raw_root)
    for domain, doc_id, path, kind in jobs:
        try:
            if kind == "pdf":
                text, meta = parse_pdf(path)
            elif kind == "html":
                text, meta = parse_html(path)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                quoted = re.findall(r"《([^》]{4,80})》", text[:4000])
                lines = [line.strip() for line in text[:4000].splitlines()
                         if 5 <= len(line.strip()) <= 80]
                title = quoted[0] if quoted else (
                    lines[0] if lines else path.stem)
                meta = {"n_pages": None, "title": title[:80]}
            directory = output_dir / domain
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            meta.update({
                "domain": domain,
                "doc_id": doc_id,
                # Raw-root relative paths make the artifact relocatable.
                "src": path.relative_to(raw_root).as_posix(),
                "n_chars": len(text),
            })
            if doc_id in all_meta:
                raise ValueError(f"duplicate doc_id: {doc_id}")
            all_meta[doc_id] = meta
        except Exception as exc:  # collect every source error before failing
            errors.append(f"{path}: {exc}")
    if errors and strict:
        raise RuntimeError("document parsing failed:\n" + "\n".join(errors))
    (output_dir / "docs_meta.json").write_text(
        json.dumps(all_meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return all_meta


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path,
                        default=root / "dataset" / "raw")
    parser.add_argument("--output", type=pathlib.Path,
                        default=root / "processed_data")
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()
    meta = build(args.input, args.output, strict=not args.allow_errors)
    by_domain = {}
    for item in meta.values():
        count, chars = by_domain.setdefault(item["domain"], [0, 0])
        by_domain[item["domain"]] = [count + 1, chars + item["n_chars"]]
    print(json.dumps({"documents": len(meta), "domains": by_domain},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
