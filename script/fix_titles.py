#!/usr/bin/env python3
"""Recompute document titles using deterministic PDF layout/text rules."""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import pymupdf as fitz

fitz.TOOLS.mupdf_display_errors(False)

SUFFIX = re.compile(
    r"(条款|办法|规定|决定|报告书|说明书|年度报告|全文|摘要|指引|准则|细则|"
    r"保险|年金|寿险|研究报告|点评|深度报告)$")
NOISE = re.compile(r"^(第[一二三四五六七八九十百]+[条章]|目\s*录|[\d\s.]+$|"
                   r"注册号|阅读指引|总\s*则)")
PRODUCT = re.compile(
    r"[一-鿿A-Za-z0-9]{2,18}(?:养老年金保险|年金保险|终身寿险|两全保险|养老保险|"
    r"人寿保险|财产保险|责任保险|医疗保险|疾病保险|意外伤害保险)"
    r"(?:（[^）]{1,14}）){0,2}")
DOCNAME = re.compile(r"^[^。；]{0,40}(募集说明书|重大资产重组报告书|报告书)[^。；]{0,10}$")


def pdf_title(path: pathlib.Path) -> str | None:
    with fitz.open(path) as document:
        best = []
        for pno in range(min(2, document.page_count)):
            data = document[pno].get_text("dict")
            for block in data["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if not text or len(text) < 5 or len(text) > 60 or NOISE.match(text):
                        continue
                    size = max(span["size"] for span in line["spans"])
                    best.append((round(size, 1), pno, block["number"], text))
            if best:
                break
    if not best:
        return None
    max_size = max(item[0] for item in best)
    top = [item for item in best if item[0] >= max_size - 0.6]
    title = "".join(item[3] for item in top[:3])
    return title[:70] if len(title) >= 5 else None


def insurance_product_name(text: str) -> str | None:
    from collections import Counter
    names = Counter(PRODUCT.findall(text))
    candidates = [(count * len(name), name) for name, count in names.items()
                  if count >= 3 and "本合同" not in name and "条款" not in name]
    return max(candidates)[1] if candidates else None


def resolve_source(raw_root: pathlib.Path, meta: dict) -> pathlib.Path:
    """Resolve a validated raw-root-relative ``src`` path."""
    source = pathlib.PurePosixPath(str(meta.get("src", "")))
    if (not source.parts or source.is_absolute() or
            any(part in {"", ".", ".."} for part in source.parts)):
        raise ValueError(f"unsafe source path: {meta.get('src')!r}")
    candidate = raw_root / pathlib.Path(*source.parts)
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"cannot resolve source {meta.get('src')!r} below {raw_root}")


def update_titles(raw_root: pathlib.Path, processed_dir: pathlib.Path) -> int:
    raw_root = pathlib.Path(raw_root).resolve()
    processed_dir = pathlib.Path(processed_dir).resolve()
    meta_path = processed_dir / "docs_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    changed = 0
    for doc_id, item in meta.items():
        source = resolve_source(raw_root, item)
        new_title = None
        text_path = processed_dir / item["domain"] / f"{doc_id}.txt"
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        if source.suffix.lower() == ".pdf":
            new_title = pdf_title(source)
            if not new_title:
                lines = [line.strip() for line in text[:1500].splitlines()
                         if line.strip() and not line.strip().startswith("[P")][:14]
                candidates = [line for line in lines if SUFFIX.search(line)
                              and not NOISE.match(line) and 6 <= len(line) <= 60]
                new_title = max(candidates, key=len) if candidates else None
        elif source.suffix.lower() == ".txt":
            quoted = re.findall(r"《([^》]{4,80})》", text[:4000])
            if quoted:
                new_title = quoted[0]
            else:
                lines = [line.strip() for line in text[:4000].splitlines()
                         if 5 <= len(line.strip()) <= 80
                         and not line.strip().startswith("[P")
                         and not NOISE.match(line.strip())]
                new_title = lines[0] if lines else source.stem

        if item["domain"] == "insurance":
            current = new_title or item["title"]
            if not PRODUCT.search(current):
                product = insurance_product_name(text)
                if product:
                    new_title = product + "条款"
        elif item["domain"] == "financial_reports":
            year = re.search(r"(20\d\d)", doc_id)
            base = (new_title or item["title"]).split("20")[0]
            if year:
                new_title = f"{base}{year.group(1)}年年度报告"
        elif item["domain"] == "financial_contracts" and new_title:
            if "说明书" not in new_title and "报告书" not in new_title:
                lines = [line.strip() for line in text[:2000].splitlines()
                         if line.strip()][:20]
                extra = next((line for line in lines if DOCNAME.match(line)), None)
                if extra and extra not in new_title:
                    new_title = (new_title + extra)[:80]

        if new_title and new_title != item["title"]:
            item["title"] = new_title
            changed += 1
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    return changed


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path,
                        default=root / "dataset" / "raw")
    parser.add_argument("--output", type=pathlib.Path,
                        default=root / "processed_data")
    args = parser.parse_args()
    print(f"titles updated: {update_titles(args.input, args.output)}")


if __name__ == "__main__":
    main()
