#!/usr/bin/env python3
"""Copy source HTML metadata into ``docs_meta.json``."""
from __future__ import annotations

import argparse
import json
import pathlib
import re

from bs4 import BeautifulSoup

try:
    from fix_titles import resolve_source
except ImportError:  # package-style import used by tests/orchestrator
    from .fix_titles import resolve_source


def add_metadata(raw_root: pathlib.Path, processed_dir: pathlib.Path) -> int:
    """Copy each source page's own ``meta description`` when present."""

    raw_root = pathlib.Path(raw_root).resolve()
    processed_dir = pathlib.Path(processed_dir).resolve()
    meta_path = processed_dir / "docs_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    count = 0
    for item in meta.values():
        source = resolve_source(raw_root, item)
        if source.suffix.lower() != ".html":
            continue
        soup = BeautifulSoup(
            source.read_text(encoding="utf-8", errors="ignore"), "lxml"
        )
        description = ""
        for tag in soup.find_all("meta"):
            if (tag.get("name") or "").lower() == "description":
                content = (tag.get("content") or "").strip()
                if len(content) > len(description):
                    description = content
        if description:
            item["source_description"] = re.sub(r"\s+", " ", description)[:160]
            count += 1
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return count


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=pathlib.Path, default=root / "dataset" / "raw"
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=root / "processed_data"
    )
    args = parser.parse_args()
    print(f"HTML source descriptions copied: {add_metadata(args.input, args.output)}")


if __name__ == "__main__":
    main()
