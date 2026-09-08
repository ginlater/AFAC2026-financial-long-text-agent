#!/usr/bin/env python3
"""Rebuild every required ``processed_data`` artifact from AFAC raw inputs.

The deterministic pipeline performs PDF/HTML parsing, layout recovery and
source-derived table and clause structuring.

Examples::

    python script/rebuild_processed.py \
      --input dataset/raw \
      --output processed_data

    python script/rebuild_processed.py \
      --input dataset/raw --check-only

For safety an existing output path is never overwritten or deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import shutil
import sys
import tempfile
from collections import Counter

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import add_html_metadata
import build_financial_facts
import build_insurance_capsules
import build_insurance_titles
import fix_titles
import parse_docs

PIPELINE_VERSION = "afac-processed-v1"
DERIVED_ARTIFACTS = (
    "docs_meta.json",
    "insurance_titles.json",
    "financial_facts.json",
    "insurance_capsules.json",
)
STEPS = (
    "parse_docs",
    "fix_titles",
    "add_html_metadata",
    "build_insurance_titles",
    "build_financial_facts",
    "build_insurance_capsules",
)
REQUIRED_LAYOUT = (
    "insurance",
    "financial_contracts",
    "financial_reports",
    "research",
    "regulatory/txt",
    "regulatory/html",
    "regulatory/attachments",
)


def resolve_raw_root(path: pathlib.Path) -> pathlib.Path:
    """Accept either the direct ``raw`` directory or its dataset parent."""
    candidate = pathlib.Path(path).expanduser().resolve()
    if not (candidate / "insurance").is_dir() and (candidate / "raw").is_dir():
        candidate = candidate / "raw"
    return candidate.resolve()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_record(root: pathlib.Path, paths: list[pathlib.Path]) -> dict:
    digest = hashlib.sha256()
    total_bytes = 0
    unique = sorted(set(path.resolve() for path in paths),
                    key=lambda path: path.relative_to(root).as_posix())
    for path in unique:
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"files": len(unique), "bytes": total_bytes,
            "tree_sha256": digest.hexdigest()}


def inspect_input(raw_root: pathlib.Path) -> dict:
    raw_root = resolve_raw_root(raw_root)
    missing = [item for item in REQUIRED_LAYOUT if not (raw_root / item).is_dir()]
    if missing:
        raise FileNotFoundError("raw input is missing directories: " + ", ".join(missing))
    jobs = parse_docs.discover_jobs(raw_root)
    if not jobs:
        raise FileNotFoundError(f"no supported documents below {raw_root}")
    counts = Counter(domain for domain, _doc_id, _path, _kind in jobs)
    kind_counts = Counter(kind for _domain, _doc_id, _path, kind in jobs)
    sources = [path for _domain, _doc_id, path, _kind in jobs]
    record = _tree_record(raw_root, sources)
    return {
        "raw_root_layout": "raw/",
        **record,
        "documents_by_domain": dict(sorted(counts.items())),
        "files_by_kind": dict(sorted(kind_counts.items())),
        "dependencies": {
            "PyMuPDF": importlib.metadata.version("PyMuPDF"),
            "beautifulsoup4": importlib.metadata.version("beautifulsoup4"),
        },
    }


def _domain_record(processed_dir: pathlib.Path, domain: str, meta: dict) -> dict:
    paths = sorted((processed_dir / domain).glob("*.txt"))
    record = _tree_record(processed_dir, paths)
    entries = [item for item in meta.values() if item["domain"] == domain]
    return {**record, "documents": len(entries),
            "characters": sum(int(item["n_chars"]) for item in entries)}


def validate_output(raw_root: pathlib.Path, processed_dir: pathlib.Path) -> dict:
    for name in DERIVED_ARTIFACTS:
        if not (processed_dir / name).is_file():
            raise RuntimeError(f"pipeline did not produce {name}")
    meta = json.loads((processed_dir / "docs_meta.json").read_text(encoding="utf-8"))
    parsed = list(processed_dir.glob("*/*.txt"))
    if len(parsed) != len(meta):
        raise RuntimeError(f"parsed text/meta mismatch: {len(parsed)} != {len(meta)}")
    for doc_id, item in meta.items():
        if pathlib.PurePosixPath(item["src"]).is_absolute():
            raise RuntimeError(f"non-portable absolute src for {doc_id}")
        if not (processed_dir / item["domain"] / f"{doc_id}.txt").is_file():
            raise RuntimeError(f"missing parsed text for {doc_id}")
        fix_titles.resolve_source(raw_root, item)

    identities = json.loads((processed_dir / "insurance_titles.json").read_text(
        encoding="utf-8"))
    insurance_ids = {item["doc_id"] for item in meta.values()
                     if item["domain"] == "insurance"}
    if set(identities) != insurance_ids:
        raise RuntimeError("insurance identity coverage does not match parsed insurance docs")
    capsules = json.loads((processed_dir / "insurance_capsules.json").read_text(
        encoding="utf-8"))
    if set(capsules.get("documents", {})) != insurance_ids:
        raise RuntimeError("insurance capsule coverage does not match parsed insurance docs")
    return meta


def make_manifest(raw_root: pathlib.Path, processed_dir: pathlib.Path,
                  input_record: dict) -> dict:
    meta = validate_output(raw_root, processed_dir)
    domains = sorted({item["domain"] for item in meta.values()})
    output_files = sorted(path for path in processed_dir.rglob("*") if path.is_file()
                          and path.name != "processed_manifest.json")
    key_artifacts = {
        name: {"bytes": (processed_dir / name).stat().st_size,
               "sha256": _sha256(processed_dir / name)}
        for name in DERIVED_ARTIFACTS
    }
    return {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "deterministic": True,
        "semantic_models_used": [],
        "steps": list(STEPS),
        "input": input_record,
        "output": {
            **_tree_record(processed_dir, output_files),
            "documents": len(meta),
            "domains": {domain: _domain_record(processed_dir, domain, meta)
                        for domain in domains},
            "artifacts": key_artifacts,
        },
    }


def run_pipeline(raw_root: pathlib.Path, processed_dir: pathlib.Path) -> dict:
    parse_docs.build(raw_root, processed_dir, strict=True)
    fix_titles.update_titles(raw_root, processed_dir)
    add_html_metadata.add_metadata(raw_root, processed_dir)
    build_insurance_titles.build(processed_dir)
    build_financial_facts.build(
        raw_root, processed_dir / "financial_facts.json")
    capsules = build_insurance_capsules.build(processed_dir=processed_dir)
    (processed_dir / "insurance_capsules.json").write_text(
        json.dumps(capsules, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return validate_output(raw_root, processed_dir)


def rebuild(raw_root: pathlib.Path, output_dir: pathlib.Path, *,
            check_only: bool = False) -> dict:
    raw_root = resolve_raw_root(raw_root)
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    input_record = inspect_input(raw_root)
    if check_only:
        return {"status": "ok", "mode": "check-only", "input": input_record,
                "planned_steps": list(STEPS)}
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_dir}; choose a new --output")
    if output_dir == raw_root or raw_root in output_dir.parents:
        raise ValueError("output must not be inside the raw input tree")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".afac-rebuild-",
                                     dir=output_dir.parent) as temporary:
        staging = pathlib.Path(temporary) / "processed_data"
        staging.mkdir()
        run_pipeline(raw_root, staging)
        manifest = make_manifest(raw_root, staging, input_record)
        (staging / "processed_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        # Destination was proven absent above; copytree does not remove or merge
        # any user data.  A failed build leaves the destination untouched.
        shutil.copytree(staging, output_dir)
    return {"status": "ok", "mode": "rebuilt", "output": str(output_dir),
            "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=pathlib.Path,
                        help="raw/ directory or a parent containing raw/")
    parser.add_argument("--output", type=pathlib.Path,
                        help="new processed-data directory (must not already exist)")
    parser.add_argument("--check-only", action="store_true",
                        help="validate input/dependencies and print the plan; write nothing")
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only is used")
    output = args.output or pathlib.Path.cwd() / ".afac-check-only-unused"
    result = rebuild(args.input, output, check_only=args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
