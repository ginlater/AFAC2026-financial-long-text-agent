#!/usr/bin/env python3
"""Assemble one run into a self-contained ``evidence.json`` audit trail.

The script joins artifacts written by the same run:

* question, option and recorded follow-up query surfaces;
* selected documents and exact cited chunks;
* every full Qwen request/response/retry with API-returned usage;
* raw candidate answers, schema formatting and final slots;
* the exact usage returned for calls made for that question.

Usage: ``python script/build_evidence.py OUTPUT_DIR --qdir QUESTIONS``
"""
import argparse
import json
import pathlib
import re
import sys

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from agent import retrieval, submission_schema  # noqa: E402


def _jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)
    return rows


def _slot_answer(slots):
    return "；".join(str(x) for x in (slots or []) if str(x).strip())


def _question_map(config, qdir_override=None):
    qdir = qdir_override or (config.get("input_paths") or {}).get("qdir") \
        or config["arguments"]["qdir"]
    path = pathlib.Path(qdir).expanduser()
    if not path.is_absolute():
        # Schema-2 paths are repository-relative in a development checkout and
        # package-root-relative after extraction.  Resolve both layouts without
        # depending on the caller's current working directory.
        candidates = (WORK.parent / path, WORK / path)
        path = next((candidate for candidate in candidates
                     if candidate.exists()), path)
    return {q["qid"]: q
            for q in submission_schema.load_questions(path)}


def _supplement_queries(calls):
    pat = re.compile(r"补充检索[:：]\s*([^\n]+)")
    out = []
    for call in calls:
        response = call.get("response") or {}
        for value in (response.get("content"),
                      response.get("reasoning_content")):
            for match in pat.findall(value or ""):
                query = match.strip()
                if query and query not in out:
                    out.append(query)
    return out


def _public_api_record(row):
    """Keep the complete key-free API record and expose its source line."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main(outdir, qdir=None):
    outdir = pathlib.Path(outdir).expanduser().resolve()
    required = (
        "answers.json", "reasonings.json", "reasoning_sources.json",
        "run_config.json", "api_calls.jsonl", "run_log.jsonl",
        "docsel_log.jsonl", "token_ledger.json",
    )
    missing = [name for name in required if not (outdir / name).is_file()]
    if missing:
        raise SystemExit(f"cannot build evidence; missing: {missing}")

    config = json.load(open(outdir / "run_config.json", encoding="utf-8"))
    questions = _question_map(config, qdir)
    answers = json.load(open(outdir / "answers.json", encoding="utf-8"))
    reasonings = json.load(open(outdir / "reasonings.json", encoding="utf-8"))
    sources = json.load(open(outdir / "reasoning_sources.json", encoding="utf-8"))
    ledger = json.load(open(outdir / "token_ledger.json", encoding="utf-8"))
    api_rows = _jsonl(outdir / "api_calls.jsonl")
    run_rows = _jsonl(outdir / "run_log.jsonl")
    docsel_rows = _jsonl(outdir / "docsel_log.jsonl")

    by_run = {}
    for row in run_rows:
        if row.get("qid"):
            by_run.setdefault(row["qid"], []).append(row)
    by_docsel = {}
    for row in docsel_rows:
        if row.get("qid"):
            by_docsel.setdefault(row["qid"], []).append(row)

    calls_by_qid = {qid: [] for qid in questions}
    for row in api_rows:
        qid = row.get("qid")
        if qid in calls_by_qid:
            calls_by_qid[qid].append(row)

    ledger_calls = {qid: [] for qid in questions}
    for idx, call in enumerate(ledger.get("calls", []), 1):
        qid = call.get("qid")
        if qid in ledger_calls:
            item = dict(call)
            item["ledger_call"] = idx
            ledger_calls[qid].append(item)

    order = list(config.get("question_ids") or questions)

    chunk_cache = {}

    def cited_chunk(cid):
        doc_id = str(cid).split("#", 1)[0]
        if doc_id not in chunk_cache:
            chunk_cache[doc_id] = {
                c["id"]: c for c in retrieval.chunk_doc(doc_id)
            }
        chunk = chunk_cache[doc_id].get(cid)
        if not chunk:
            return {"evidence_id": cid, "missing": True}
        return {
            "evidence_id": cid,
            "doc_id": chunk["doc_id"],
            "page": chunk.get("page"),
            "quoted_clause": chunk["text"],
        }

    out = []
    for qid in order:
        q = questions[qid]
        if qid not in answers:
            continue
        run = by_run.get(qid, [])
        evidence_ids = []
        expanded_docs = []
        for row in run:
            evidence_ids.extend(row.get("evidence_ids") or [])
            expanded = row.get("doc_expanded")
            if isinstance(expanded, list):
                expanded_docs.extend(str(value) for value in expanded)
            elif expanded:
                expanded_docs.append(str(expanded))
        evidence_ids = list(dict.fromkeys(evidence_ids))
        picked = []
        for row in by_docsel.get(qid, []):
            picked.extend(str(value) for value in row.get("picked") or [])
        picked = list(dict.fromkeys(picked + expanded_docs))

        calls = calls_by_qid.get(qid, [])
        query_surfaces = [str(q.get("question") or "")]
        query_surfaces.extend(
            str(value) for value in (q.get("options") or {}).values()
        )
        query_surfaces.extend(_supplement_queries(calls))
        query_surfaces = [x for x in dict.fromkeys(query_surfaces) if x]

        trace_info = sources.get(qid) or {}
        candidates = []
        for trace in trace_info.get("traces") or []:
            candidates.append({
                "stage": trace.get("stage"),
                "parsed_answer": trace.get("answer"),
                "visible_output": trace.get("content"),
                "evidence_ids": trace.get("evidence_ids") or [],
            })

        q_usage = ledger.get("per_qid", {}).get(qid, [0, 0])
        q_usage = [int(q_usage[0]), int(q_usage[1])]
        entry = {
            "qid": qid,
            "question": q.get("question"),
            "options": q.get("options") or {},
            "answer_format": q.get("answer_format"),
            "answer_slots": answers[qid],
            "answer": _slot_answer(answers[qid]),
            "reasoning": reasonings.get(qid, ""),
            "reasoning_source_stage": trace_info.get("reasoning_stage"),
            "model_traces": candidates,
            "postprocessing": {
                "raw_answer": trace_info.get("raw_answer"),
                "formatted_slots": answers[qid],
            },
            "retrieval": {
                "query_surfaces": query_surfaces,
                "selected_doc_ids": picked or list(q.get("doc_ids") or []),
                "evidence_ids": evidence_ids,
                "evidence_retrieval": [cited_chunk(cid)
                                       for cid in evidence_ids],
            },
            "token_accounting": {
                "prompt_tokens": int(q_usage[0]),
                "completion_tokens": int(q_usage[1]),
                "total_tokens": int(q_usage[0]) + int(q_usage[1]),
                "ledger_calls": ledger_calls.get(qid, []),
            },
            "api_attempts": [dict(_public_api_record(row),
                                  audit_line=row["_line"])
                             for row in calls],
            "run_log_records": [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in run
            ],
        }
        out.append(entry)

    payload = {
        "format_version": 2,
        "provenance": {
            "run_config": config,
            # evidence.json lives in the source run directory; expressing that
            # relationship relative to this file avoids leaking a host-specific
            # absolute path and remains true after ZIP extraction/relocation.
            "source_directory": ".",
            "note": "Deterministic assembly of same-run artifacts.",
        },
        "questions": out,
    }
    with open(outdir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"evidence.json written: {len(out)} entries -> {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--qdir")
    args = parser.parse_args()
    main(args.output_dir, args.qdir)
