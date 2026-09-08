#!/usr/bin/env python3
"""Strict, zero-API validation of one reproduction output directory.

The checker joins independently written artifacts. A directory is accepted
only when the submitted rows, final answer
objects, reasoning provenance, raw API audit, token ledger, execution logs and
evidence bundle describe the same run.
"""
import argparse
import csv
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import json
import math
import pathlib
import re
import sys


WORK = pathlib.Path(__file__).resolve().parents[1]
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

from agent import (answerer, calc, doc_select, docsel_fast, retrieval,  # noqa: E402
                   submission_schema)
from agent.qwen_client import is_allowed_model  # noqa: E402
from agent.repro import (build_input_manifest,  # noqa: E402
                         validate_complete_run_config,
                         verify_runtime_manifest)


REQUIRED = (
    "answer.csv", "answers.json", "reasonings.json", "reasoning_sources.json",
    "run_config.json", "api_calls.jsonl", "run_log.jsonl",
    "docsel_log.jsonl", "token_ledger.json", "evidence.json",
)

def fail(message):
    raise SystemExit(f"REPRO CHECK FAILED: {message}")


def load_json(path, label=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid {label or pathlib.Path(path).name}: {exc}")


def jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                fail(f"blank JSONL line {path.name}:{no}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL {path.name}:{no}: {exc}")
            if not isinstance(row, dict):
                fail(f"JSONL row is not an object {path.name}:{no}")
            rows.append(row)
    return rows


def token_int(value, label):
    # bool is an int subclass but is never a valid raw token count.
    if isinstance(value, bool):
        fail(f"invalid integer for {label}: {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        fail(f"invalid integer for {label}: {value!r}")
    if str(value).strip() not in {str(number), f"+{number}"}:
        fail(f"non-integral token value for {label}: {value!r}")
    if number < 0:
        fail(f"negative token value for {label}: {number}")
    return number


def token_pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        fail(f"invalid token pair for {label}: {value!r}")
    return (token_int(value[0], f"{label}.prompt_tokens"),
            token_int(value[1], f"{label}.completion_tokens"))


def normalize_reasoning(value):
    return str(value or "").replace("\n", " ").strip()


def answer_slots(value, label, slot_count):
    if (not isinstance(slot_count, int) or slot_count < 1 or
            not isinstance(value, list) or not 1 <= len(value) <= slot_count):
        fail(f"{label} must be a list with 1-{slot_count} answer slots")
    slots = [str(x if x is not None else "").strip() for x in value]
    if not any(slots):
        fail(f"empty answer slots for {label}")
    return tuple(slots + [""] * (slot_count - len(slots)))


_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(%?)$")
_SUPPLEMENT_QUERY_RE = re.compile(r"补充检索[:：]\s*([^\n]+)")
_LEAKED_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_UNREDACTED_HEADER_RE = re.compile(
    r"(?i)(?:authorization|x-api-key|api[_ -]?key)[\"']?\s*[:=]\s*"
    r"[\"']?(?!\[REDACTED\])[^\s,;}'\"]+"
)
_TRACE_TAGS = {
    "r1": "r1",
    "r1b": "r1b",
    "r2": "r2",
    "r3": "r3",
    "primary": "calc_primary",
    "evidence_retry": "calc_evidence_retry",
    "verify": "calc_verify",
    "arbitrate": "calc_arbitrate",
}


def _value_equivalent(left, right):
    """Compare a parsed raw answer with one final formatted slot.

    The final JSON/CSV comparisons are byte-level after trimming.  This looser
    comparison is only for an upstream trace or run-log answer: a parser may
    omit insignificant trailing zeroes that the schema formatter restores.
    """
    a = re.sub(r"\s+", "", str(left or "")).strip("'\"。")
    b = re.sub(r"\s+", "", str(right or "")).strip("'\"。")
    a = re.sub(r"^(?:答案|answer)[:：]", "", a, flags=re.I)
    b = re.sub(r"^(?:答案|answer)[:：]", "", b, flags=re.I)
    a = a.replace("＞", ">").replace("％", "%")
    b = b.replace("＞", ">").replace("％", "%")
    if a == b:
        return True

    # Multiple-choice ordering is semantically irrelevant, although the final
    # formatted answer is checked separately for canonical storage.
    if re.fullmatch(r"[A-D]+", a) and re.fullmatch(r"[A-D]+", b):
        return "".join(sorted(set(a))) == "".join(sorted(set(b)))

    ma, mb = _NUMBER.fullmatch(a), _NUMBER.fullmatch(b)
    if ma and mb and bool(ma.group(1)) == bool(mb.group(1)):
        try:
            return Decimal(a.rstrip("%").replace(",", "")) == \
                Decimal(b.rstrip("%").replace(",", ""))
        except InvalidOperation:
            pass

    def date_parts(text):
        match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if match:
            return tuple(map(int, match.groups()))
        match = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
        return tuple(map(int, match.groups())) if match else None

    da, db = date_parts(a), date_parts(b)
    return da is not None and da == db


def answer_equivalent(raw, final_slots):
    slots = [slot for slot in final_slots if slot]
    if not slots:
        return False
    if isinstance(raw, list):
        pieces = [str(x).strip() for x in raw]
    else:
        text = str(raw or "").strip()
        pieces = re.split(r"[；;]", text) if len(slots) > 1 else [text]
    if len(pieces) != len(slots):
        return False
    return all(_value_equivalent(piece, slot)
               for piece, slot in zip(pieces, slots))


def evidence_entries(path):
    """Return qid-keyed entries from supported evidence layouts."""
    payload = load_json(path, "evidence.json")
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = None
        for key in ("questions", "items", "evidence"):
            if isinstance(payload.get(key), list):
                entries = payload[key]
                break
        if entries is None:
            entries = []
            for qid, value in payload.items():
                if not isinstance(value, dict):
                    fail("evidence keyed-object values must be objects")
                entry = dict(value)
                entry.setdefault("qid", qid)
                entries.append(entry)
    else:
        fail("evidence.json must contain a list or object")

    result = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(f"evidence entry {i} is not an object")
        qid = str(entry.get("qid") or "").strip()
        if not qid:
            fail(f"evidence entry {i} has no qid")
        if qid in result:
            fail(f"duplicate qid in evidence.json: {qid}")
        if not any(v not in (None, "", [], {}) for k, v in entry.items()
                   if k != "qid"):
            fail(f"evidence entry has no payload: {qid}")
        result[qid] = entry
    return payload, result


def validate_complete_evidence(entry, qid):
    """Require concrete, non-missing cited clauses for a schema-2 question."""

    retrieval = entry.get("retrieval")
    if not isinstance(retrieval, dict):
        fail(f"schema2 evidence has no retrieval object for {qid}")
    selected = retrieval.get("selected_doc_ids")
    evidence_ids = retrieval.get("evidence_ids")
    chunks = retrieval.get("evidence_retrieval")
    if (not isinstance(selected, list) or not selected or
            any(not str(value).strip() for value in selected)):
        fail(f"schema2 evidence has no selected documents for {qid}")
    if (not isinstance(evidence_ids, list) or not evidence_ids or
            any(not str(value).strip() for value in evidence_ids)):
        fail(f"schema2 evidence has no evidence_ids for {qid}")
    if not isinstance(chunks, list) or not chunks:
        fail(f"schema2 evidence has no cited chunks for {qid}")
    seen = []
    for pos, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            fail(f"schema2 evidence chunk is not an object for {qid}#{pos}")
        if chunk.get("missing"):
            fail(f"schema2 evidence contains missing chunk for {qid}#{pos}")
        for field in ("evidence_id", "doc_id", "quoted_clause"):
            if not str(chunk.get(field) or "").strip():
                fail(f"schema2 evidence chunk lacks {field} for {qid}#{pos}")
        if str(chunk["doc_id"]) not in {str(value) for value in selected}:
            fail(f"schema2 evidence cites an unselected document for {qid}#{pos}")
        seen.append(str(chunk["evidence_id"]))
    if seen != [str(value) for value in evidence_ids]:
        fail(f"schema2 evidence_ids/chunks order mismatch for {qid}")


def validate_schema2_api_record(record, line):
    """Validate one direct, key-free Qwen attempt record."""

    status = record.get("status")
    if status not in {"ok", "request_error"}:
        fail(f"unsupported API attempt status at line {line}: {status!r}")
    if not is_allowed_model(record.get("model")):
        fail(f"unsupported Qwen model in API audit at line {line}")
    if not str(record.get("qid") or "").strip():
        fail(f"API attempt has no qid at line {line}")
    attempt = record.get("attempt")
    if (not isinstance(attempt, int) or isinstance(attempt, bool)
            or attempt < 1):
        fail(f"API attempt number is invalid at line {line}")
    started = record.get("started_at")
    finished = record.get("finished_at")
    if (not isinstance(started, (int, float)) or isinstance(started, bool)
            or not math.isfinite(started)
            or not isinstance(finished, (int, float))
            or isinstance(finished, bool) or not math.isfinite(finished)
            or finished < started):
        fail(f"API attempt timestamps are invalid at line {line}")
    common = {
        "status", "qid", "tag", "model", "attempt", "started_at",
        "finished_at", "request",
    }
    allowed = common | ({"response", "usage"} if status == "ok"
                        else {"error"})
    if set(record) != allowed:
        fail(f"API attempt has unexpected fields at line {line}")
    request = record.get("request")
    if (not isinstance(request, dict)
            or not isinstance(request.get("messages"), list)
            or not request["messages"]):
        fail(f"API attempt has no request messages at line {line}")
    if set(request) != {"messages", "max_tokens", "temperature", "extra_body"}:
        fail(f"API attempt request has unexpected fields at line {line}")
    if (not isinstance(request.get("max_tokens"), int)
            or isinstance(request.get("max_tokens"), bool)
            or request["max_tokens"] < 1
            or not isinstance(request.get("temperature"), (int, float))
            or isinstance(request.get("temperature"), bool)
            or not math.isfinite(request["temperature"])
            or not isinstance(request.get("extra_body"), dict)):
        fail(f"API attempt request controls are invalid at line {line}")
    if status == "request_error":
        if record.get("usage") not in (None, {}):
            fail(f"failed API request unexpectedly has usage at line {line}")
        error = record.get("error")
        if (not isinstance(error, dict)
                or set(error) != {"type", "message"}
                or not str(error.get("type") or "").strip()
                or not isinstance(error.get("message"), str)):
            fail(f"failed API request has no error object at line {line}")
        message = error["message"]
        if (_LEAKED_KEY_RE.search(message)
                or _UNREDACTED_HEADER_RE.search(message)):
            fail(f"failed API request exposes credential-like text at line {line}")


def validate_schema2_evidence_provenance(payload, config):
    """Bind portable evidence metadata to the exact checked run config."""

    if not isinstance(payload, dict):
        fail("schema2 evidence must use the object layout")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        fail("schema2 evidence has no provenance object")
    if provenance.get("source_directory") != ".":
        fail("schema2 evidence source_directory must be portable '.'")
    if provenance.get("run_config") != config:
        fail("schema2 evidence run_config differs from run_config.json")


def _resolve_config_input(raw):
    path = pathlib.Path(str(raw or "")).expanduser()
    if path.is_absolute():
        return path
    candidates = (WORK.parent / path, WORK / path)
    return next((candidate for candidate in candidates if candidate.exists()),
                path)


def validate_schema2_inputs(config, question_ids, *, qdir_override=None,
                            submit_override=None):
    """Re-read the configured inputs and verify their order and fingerprints."""

    paths = config.get("input_paths")
    if not isinstance(paths, dict):
        fail("schema2 run_config has no input_paths object")
    qdir = (pathlib.Path(qdir_override).expanduser().resolve()
            if qdir_override else _resolve_config_input(paths.get("qdir")))
    submit = (pathlib.Path(submit_override).expanduser().resolve()
              if submit_override
              else _resolve_config_input(paths.get("submit_template")))
    try:
        loaded_qids = [str(q["qid"])
                       for q in submission_schema.load_questions(qdir)]
        template_schema = submission_schema.load_schema(submit)
        current_manifest = build_input_manifest(qdir, submit)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            json.JSONDecodeError) as exc:
        fail(f"schema2 inputs cannot be verified: {exc}")
    if (len(loaded_qids) != len(set(loaded_qids)) or
            set(loaded_qids) != set(question_ids)):
        fail("schema2 configured question set differs from answer.csv")
    if list(template_schema.question_ids) != list(question_ids):
        fail("schema2 template order differs from answer.csv")
    if config.get("inputs") != current_manifest:
        fail("schema2 input manifest differs from configured input files")


def _call_signature(qid, model, tag, usage):
    return (
        str(qid or ""),
        str(model or ""),
        str(tag or ""),
        json.dumps(usage, ensure_ascii=False, sort_keys=True),
    )


def _api_attached_to(record, qid):
    """Whether one direct API attempt belongs to ``qid``."""

    return record.get("qid") == qid


def _response_contains_reasoning(record, reasoning):
    """Require the submitted explanation to be literal visible API output."""

    if record.get("status") != "ok" or not reasoning:
        return False
    response = record.get("response")
    if not isinstance(response, dict):
        return False
    return any(reasoning in str(response.get(key) or "")
               for key in ("content", "reasoning_content"))


def _query_surfaces(question, records):
    """Rebuild the question, option and follow-up surfaces in evidence."""

    values = [str(question.get("question") or "")]
    values.extend(
        str(value) for value in (question.get("options") or {}).values()
    )
    for record in records:
        response = record.get("response") or {}
        for field in ("content", "reasoning_content"):
            for match in _SUPPLEMENT_QUERY_RE.findall(
                    str(response.get(field) or "")):
                query = match.strip()
                if query:
                    values.append(query)
    return [value for value in dict.fromkeys(values) if value]


def _embedded_record(record, marker, label):
    if not isinstance(record, dict):
        fail(f"{label} reference is not an object")
    line = token_int(record.get(marker), f"{label}.{marker}")
    if line < 1:
        fail(f"{label}.{marker} must be 1-based")
    clean = dict(record)
    clean.pop(marker, None)
    return line, clean


def main(out, *, qdir=None, submit_template=None):
    out = pathlib.Path(out)
    for name in REQUIRED:
        if not (out / name).is_file():
            fail(f"missing {name}")

    # Load the run contract first so the checked input/template can define the
    # expected question set and output layout.
    cfg_text = (out / "run_config.json").read_text(encoding="utf-8")
    if "DASHSCOPE_API_KEY" in cfg_text or "sk-" in cfg_text:
        fail("secret-like text found in run_config.json")
    try:
        config = json.loads(cfg_text)
    except json.JSONDecodeError as exc:
        fail(f"invalid run_config.json: {exc}")
    if not isinstance(config, dict):
        fail("run_config.json must contain an object")
    if config.get("schema_version") != 2:
        fail("run_config schema_version must be 2")
    input_paths = config.get("input_paths")
    if not isinstance(input_paths, dict):
        fail("run_config has no input_paths object")
    template_path = (pathlib.Path(submit_template).expanduser().resolve()
                     if submit_template else _resolve_config_input(
                         input_paths.get("submit_template")))
    question_path = (pathlib.Path(qdir).expanduser().resolve()
                     if qdir else _resolve_config_input(
                         input_paths.get("qdir")))
    try:
        question_objects = submission_schema.load_questions(question_path)
        question_by_qid = {
            str(question.get("qid") or ""): question
            for question in question_objects
        }
        template_schema = submission_schema.refine_schema_from_questions(
            submission_schema.load_schema(template_path), question_objects
        )
    except (OSError, ValueError, KeyError, TypeError,
            json.JSONDecodeError) as exc:
        fail(f"submission template cannot be read: {exc}")
    answer_columns = list(template_schema.answer_columns)

    # ------------------------------------------------------------------ CSV
    with open(out / "answer.csv", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    expected_header = submission_schema.output_columns(template_schema)
    if header != expected_header:
        fail("unexpected answer.csv header")
    if not rows or rows[0]["qid"] != "summary":
        fail("first data row must be summary")
    questions = rows[1:]
    question_ids = [str(row.get("qid") or "").strip() for row in questions]
    if (not questions or any(not qid for qid in question_ids) or
            len(set(question_ids)) != len(question_ids)):
        fail("answer.csv must contain unique nonempty question rows")
    if question_ids != list(template_schema.question_ids):
        fail("answer.csv question order differs from submission template")
    qid_set = set(question_ids)
    try:
        validate_complete_run_config(config, question_ids)
        verify_runtime_manifest(config.get("runtime_manifest"), WORK)
    except RuntimeError as exc:
        fail(str(exc))
    validate_schema2_inputs(
        config, question_ids, qdir_override=qdir,
        submit_override=submit_template)

    csv_answers = {}
    csv_reasonings = {}
    csv_usage = {}
    for qid, row in zip(question_ids, questions):
        slots = tuple((row[column] or "").strip()
                      for column in answer_columns)
        if not any(slots):
            fail(f"empty answer: {qid}")
        expected_slot_count = len(template_schema[qid])
        if any(slots[expected_slot_count:]):
            fail(f"answer occupies an undefined template slot: {qid}")
        reasoning = (row["reasoning"] or "").strip()
        if not reasoning:
            fail(f"empty reasoning: {qid}")
        csv_answers[qid] = slots
        csv_reasonings[qid] = reasoning

        rp = token_int(row["prompt_tokens"], f"{qid}.prompt_tokens")
        rc = token_int(row["completion_tokens"], f"{qid}.completion_tokens")
        rt = token_int(row["total_tokens"], f"{qid}.total_tokens")
        if rt != rp + rc:
            fail(f"row token sum mismatch for {qid}: {rp}+{rc}!={rt}")
        csv_usage[qid] = (rp, rc, rt)

    qp = sum(v[0] for v in csv_usage.values())
    qc = sum(v[1] for v in csv_usage.values())
    summary = rows[0]
    declared = (
        token_int(summary["prompt_tokens"], "summary.prompt_tokens"),
        token_int(summary["completion_tokens"], "summary.completion_tokens"),
        token_int(summary["total_tokens"], "summary.total_tokens"),
    )
    if declared != (qp, qc, qp + qc):
        fail(f"CSV token sum mismatch: {declared} vs {(qp, qc, qp + qc)}")

    # ----------------------------------------- answers/reasonings/provenance
    answers = load_json(out / "answers.json")
    reasonings = load_json(out / "reasonings.json")
    sources = load_json(out / "reasoning_sources.json")
    for label, obj in (("answers.json", answers),
                       ("reasonings.json", reasonings),
                       ("reasoning_sources.json", sources)):
        if not isinstance(obj, dict):
            fail(f"{label} must contain an object")
        missing = sorted(qid_set - set(obj))
        extra = sorted(set(obj) - qid_set)
        if missing or extra:
            fail(f"{label} qid mismatch: missing={missing}, extra={extra}")

    normalized_answers = {}
    for qid in question_ids:
        normalized_answers[qid] = answer_slots(
            answers[qid], f"answers.json.{qid}", len(answer_columns))
        if normalized_answers[qid] != csv_answers[qid]:
            fail(f"answers.json/CSV answer mismatch for {qid}: "
                 f"JSON={normalized_answers[qid]} CSV={csv_answers[qid]}")
        if not isinstance(reasonings[qid], str):
            fail(f"reasonings.json value is not text for {qid}")
        if normalize_reasoning(reasonings[qid]) != csv_reasonings[qid]:
            fail(f"reasonings.json/CSV reasoning mismatch for {qid}")
        source = sources[qid]
        if not isinstance(source, dict):
            fail(f"reasoning source is not an object for {qid}")
        if source.get("reasoning") != reasonings[qid]:
            fail(f"reasoning_sources/reasonings mismatch for {qid}")
        stage = str(source.get("reasoning_stage") or "").strip()
        traces = source.get("traces")
        if not stage or not isinstance(traces, list) or not traces:
            fail(f"reasoning source has no selected stage/traces for {qid}")
        selected = []
        for pos, trace in enumerate(traces, 1):
            if not isinstance(trace, dict):
                fail(f"reasoning trace is not an object for {qid}#{pos}")
            if not str(trace.get("stage") or "").strip():
                fail(f"reasoning trace has no stage for {qid}#{pos}")
            if not isinstance(trace.get("content"), str):
                fail(f"reasoning trace content is not text for {qid}#{pos}")
            trace_evidence = trace.get("evidence_ids")
            if (not isinstance(trace_evidence, list)
                    or any(not str(value).strip()
                           for value in trace_evidence)):
                fail(f"reasoning trace has invalid evidence for {qid}#{pos}")
            if (trace.get("stage") == stage
                    and trace.get("content") == reasonings[qid]):
                selected.append(trace)
        if not selected:
            fail(f"selected reasoning does not correspond to a trace for {qid}")
        if not any(answer_equivalent(t.get("answer"), csv_answers[qid])
                   for t in selected):
            fail(f"selected reasoning trace disagrees with final answer for {qid}")
        source_evidence = source.get("evidence_ids")
        if (not isinstance(source_evidence, list) or not source_evidence
                or any(not str(value).strip() for value in source_evidence)):
            fail(f"reasoning source has no selected evidence for {qid}")
        if not any(
            [str(value) for value in trace.get("evidence_ids") or []]
            == [str(value) for value in source_evidence]
            for trace in selected
        ):
            fail(f"selected reasoning/evidence mismatch for {qid}")
        if source.get("raw_answer") not in (None, "", []):
            if not answer_equivalent(source["raw_answer"], csv_answers[qid]):
                fail(f"reasoning source raw_answer disagrees with final for {qid}")

    # ---------------------------------------------------------- API + ledger
    audit = jsonl(out / "api_calls.jsonl")
    successes = []
    api_signatures = []
    api_prompt = 0
    api_completion = 0
    attempt_groups = defaultdict(list)
    for line, row in enumerate(audit, 1):
        validate_schema2_api_record(row, line)
        if row.get("qid") not in qid_set:
            fail(f"API attempt is not bound to an input qid at line {line}")
        attempt_groups[(row.get("qid"), row.get("tag"))].append((
            line,
            row.get("attempt"),
            row.get("status"),
            row.get("model"),
            json.dumps(row.get("request"), ensure_ascii=False,
                       sort_keys=True),
        ))
        if row.get("status") != "ok":
            continue
        successes.append(row)
        usage = row.get("usage")
        if not isinstance(usage, dict):
            fail(f"successful API call has no usage at line {line}")
        ap = token_int(usage.get("prompt_tokens"),
                       f"api_calls.jsonl:{line}.prompt_tokens")
        ac = token_int(usage.get("completion_tokens"),
                       f"api_calls.jsonl:{line}.completion_tokens")
        at = token_int(usage.get("total_tokens"),
                       f"api_calls.jsonl:{line}.total_tokens")
        if at != ap + ac:
            fail(f"API token sum mismatch at line {line}: {ap}+{ac}!={at}")
        response = row.get("response")
        if not isinstance(response, dict) or not any(
                str(response.get(key) or "").strip()
                for key in ("content", "reasoning_content")):
            fail(f"successful API call has no visible response at line {line}")
        api_signatures.append(_call_signature(
            row.get("qid"), row.get("model"), row.get("tag"), usage))
        api_prompt += ap
        api_completion += ac

    for (qid, tag), attempts in attempt_groups.items():
        numbers = [number for _line, number, _status, _model, _request
                   in attempts]
        if numbers != list(range(1, len(attempts) + 1)):
            fail(
                f"API retry sequence is not consecutive for {qid}/{tag}: "
                f"{numbers}"
            )
        statuses = [status for _line, _number, status, _model, _request
                    in attempts]
        if statuses[-1] != "ok" or any(
                status != "request_error" for status in statuses[:-1]):
            fail(f"API retry sequence has invalid status order for {qid}/{tag}")
        contracts = {(model, request) for _line, _number, _status,
                     model, request in attempts}
        if len(contracts) != 1:
            fail(f"API retry changed model or request for {qid}/{tag}")

    allowed_tags = {"docsel", *_TRACE_TAGS.values()}
    unknown_tags = sorted({str(row.get("tag")) for row in audit
                           if row.get("tag") not in allowed_tags})
    if unknown_tags:
        fail(f"API audit contains unknown call tags: {unknown_tags}")

    api_trace_keys = Counter()
    success_by_trace = {}
    for row in successes:
        tag = row.get("tag")
        if tag == "docsel":
            continue
        content = str((row.get("response") or {}).get("content") or "")
        key = (str(row.get("qid")), str(tag), content)
        api_trace_keys[key] += 1
        success_by_trace[key] = row

    source_trace_keys = Counter()
    chunk_cache = {}
    calculation_stages = {
        "primary", "evidence_retry", "verify", "arbitrate"
    }
    for qid in question_ids:
        question = question_by_qid[qid]
        answer_format = question.get("answer_format")
        for position, trace in enumerate(
                sources[qid].get("traces") or [], 1):
            stage = str(trace.get("stage") or "")
            tag = _TRACE_TAGS.get(stage)
            if tag is None:
                fail(f"reasoning trace has unknown stage for {qid}#{position}")
            if (answer_format == "calc") != (stage in calculation_stages):
                fail(f"reasoning trace stage/type mismatch for {qid}#{position}")
            content = str(trace.get("content") or "")
            parsed = (calc.parse_calc(content) if answer_format == "calc"
                      else answerer.parse_answer(content, answer_format))
            if parsed != str(trace.get("answer") or ""):
                fail(f"reasoning trace parsed answer mismatch for {qid}#{position}")
            key = (qid, tag, content)
            source_trace_keys[key] += 1
            call = success_by_trace.get(key)
            if call is None:
                fail(f"reasoning trace has no matching API response for {qid}#{position}")
            prompt = "\n".join(
                str(message.get("content") or "")
                for message in call["request"]["messages"]
                if isinstance(message, dict)
            )
            for evidence_id in trace.get("evidence_ids") or []:
                evidence_id = str(evidence_id)
                doc_id = evidence_id.split("#", 1)[0]
                try:
                    if doc_id not in chunk_cache:
                        chunk_cache[doc_id] = {
                            chunk["id"]: chunk
                            for chunk in retrieval.chunk_doc(doc_id)
                        }
                    chunk = chunk_cache[doc_id].get(evidence_id)
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    fail(f"cannot resolve trace evidence {evidence_id}: {exc}")
                if chunk is None:
                    fail(f"reasoning trace cites missing evidence {evidence_id}")
                label = (f"{chunk['doc_id']} P{chunk['page']}"
                         if chunk.get("page") else chunk["id"])
                rendered = f"【{label}】{chunk['text']}"
                if rendered not in prompt:
                    fail(
                        f"reasoning trace evidence was not sent to Qwen: "
                        f"{qid}#{position}/{evidence_id}"
                    )
    if source_trace_keys != api_trace_keys:
        missing = api_trace_keys - source_trace_keys
        extra = source_trace_keys - api_trace_keys
        fail(
            "reasoning traces/API responses differ: "
            f"missing={list(missing.items())[:3]}, "
            f"extra={list(extra.items())[:3]}"
        )

    ledger = load_json(out / "token_ledger.json")
    if not isinstance(ledger, dict):
        fail("token_ledger.json must contain an object")
    per_qid = ledger.get("per_qid")
    calls = ledger.get("calls")
    if not isinstance(per_qid, dict) or not isinstance(calls, list):
        fail("token_ledger.json must contain per_qid object and calls list")
    clean_ledger = {str(qid): token_pair(usage, f"ledger.{qid}")
                    for qid, usage in per_qid.items()}
    if set(clean_ledger) != qid_set:
        fail("ledger must contain direct usage for exactly the input qids")

    rebuilt = defaultdict(lambda: [0, 0])
    ledger_signatures = []
    for line, call in enumerate(calls, 1):
        if not isinstance(call, dict):
            fail(f"ledger call {line} is not an object")
        cp = token_int(call.get("prompt_tokens"),
                       f"ledger.calls.{line}.prompt_tokens")
        cc = token_int(call.get("completion_tokens"),
                       f"ledger.calls.{line}.completion_tokens")
        ct = token_int(call.get("total_tokens"),
                       f"ledger.calls.{line}.total_tokens")
        if ct != cp + cc:
            fail(f"ledger call token sum mismatch at line {line}")
        qid = str(call.get("qid") or "").strip()
        if qid not in qid_set:
            fail(f"ledger call {line} is not bound to an input qid")
        if set(call) != {
            "qid", "model", "tag", "prompt_tokens", "completion_tokens",
            "total_tokens", "usage", "ts",
        }:
            fail(f"ledger call {line} has unexpected fields")
        if not is_allowed_model(call.get("model")):
            fail(f"unsupported Qwen model in ledger call {line}")
        raw_usage = call.get("usage")
        if not isinstance(raw_usage, dict):
            fail(f"ledger call {line} has no raw usage object")
        raw_prompt = token_int(
            raw_usage.get("prompt_tokens"),
            f"ledger.calls.{line}.usage.prompt_tokens",
        )
        raw_completion = token_int(
            raw_usage.get("completion_tokens"),
            f"ledger.calls.{line}.usage.completion_tokens",
        )
        raw_total = token_int(
            raw_usage.get("total_tokens"),
            f"ledger.calls.{line}.usage.total_tokens",
        )
        if (raw_prompt, raw_completion, raw_total) != (cp, cc, ct):
            fail(f"ledger call {line} differs from its raw usage object")
        rebuilt[qid][0] += cp
        rebuilt[qid][1] += cc
        ledger_signatures.append(_call_signature(
            qid, call.get("model"), call.get("tag"), raw_usage))

    rebuilt_clean = {qid: tuple(value) for qid, value in rebuilt.items()}
    if rebuilt_clean != clean_ledger:
        keys = sorted(set(rebuilt_clean) | set(clean_ledger))
        bad = [(qid, rebuilt_clean.get(qid), clean_ledger.get(qid))
               for qid in keys if rebuilt_clean.get(qid) != clean_ledger.get(qid)]
        fail(f"ledger per_qid cannot be rebuilt from calls: {bad[:5]}")
    if Counter(api_signatures) != Counter(ledger_signatures):
        missing = Counter(api_signatures) - Counter(ledger_signatures)
        extra = Counter(ledger_signatures) - Counter(api_signatures)
        fail(f"API/ledger per-call mismatch: missing={list(missing.items())[:3]}, "
             f"extra={list(extra.items())[:3]}")

    lp = sum(v[0] for v in clean_ledger.values())
    lc = sum(v[1] for v in clean_ledger.values())
    if declared != (lp, lc, lp + lc):
        fail(f"ledger mismatch: {declared} vs {(lp, lc, lp + lc)}")
    if declared != (api_prompt, api_completion,
                    api_prompt + api_completion):
        fail(
            "API usage mismatch: "
            f"{declared} vs "
            f"{(api_prompt, api_completion, api_prompt + api_completion)}"
        )

    for qid in question_ids:
        direct_p, direct_c = clean_ledger[qid]
        expected = (direct_p, direct_c, direct_p + direct_c)
        if csv_usage[qid] != expected:
            fail(f"per-qid ledger mismatch for {qid}: "
                 f"CSV={csv_usage[qid]} expected={expected}")

    # ----------------------------------------------------- execution coverage
    run_rows = jsonl(out / "run_log.jsonl")
    docsel_rows = jsonl(out / "docsel_log.jsonl")
    by_run, by_docsel = defaultdict(list), defaultdict(list)
    for row in run_rows:
        by_run[str(row.get("qid") or "")].append(row)
    for row in docsel_rows:
        by_docsel[str(row.get("qid") or "")].append(row)
    for label, mapping in (("run_log.jsonl", by_run),
                           ("docsel_log.jsonl", by_docsel)):
        missing = sorted(qid_set - set(mapping))
        extra = sorted(set(mapping) - qid_set)
        if missing or extra:
            fail(f"{label} qid coverage mismatch: "
                 f"missing={missing}, extra={extra}")
    runtime_environment = config.get("environment") or {}

    def expected_fast_decision(question):
        if runtime_environment.get("AFAC_FAST_DOCSEL") != "1":
            return None
        if question.get("answer_format") == "calc":
            if runtime_environment.get("AFAC_FAST_DOCSEL_CALC") != "1":
                return None
            domains = {
                value.strip() for value in
                runtime_environment.get(
                    "AFAC_FAST_DOCSEL_CALC_DOMAINS", ""
                ).split(",") if value.strip()
            }
            if domains and question.get("domain") not in domains:
                return None
        return docsel_fast.select_docs_fast(question)

    for qid in question_ids:
        finals = [row for row in by_run[qid]
                  if row.get("final") not in (None, "", [])]
        if not finals:
            fail(f"run_log has no final result for {qid}")
        if not answer_equivalent(finals[-1]["final"], csv_answers[qid]):
            fail(f"run_log final disagrees with submitted answer for {qid}")
        if not any(isinstance(row.get("picked"), list) and row["picked"]
                   for row in by_docsel[qid]):
            fail(f"docsel_log has no nonempty picked documents for {qid}")

        records = by_docsel[qid]
        if len(records) != 1:
            fail(f"docsel_log must have one decision for {qid}")
        decision = records[0]
        if decision.get("qid") != qid:
            fail(f"docsel_log qid mismatch for {qid}")
        if decision.get("kinds") != list(template_schema[qid]):
            fail(f"docsel_log answer schema mismatch for {qid}")
        selector = decision.get("selector")
        allowed_fields = {"qid", "picked", "kinds", "selector"}
        if selector == "code":
            allowed_fields.add("diagnostics")
        if set(decision) != allowed_fields:
            fail(f"docsel_log has unexpected fields for {qid}")

        question = question_by_qid[qid]
        docsel_calls = [
            row for row in successes
            if row.get("qid") == qid and row.get("tag") == "docsel"
        ]
        if question.get("doc_ids"):
            if selector != "input" or docsel_calls:
                fail(f"input document routing provenance mismatch for {qid}")
            expected_picked = list(question["doc_ids"])
        elif selector == "code":
            if docsel_calls:
                fail(f"code document routing unexpectedly called Qwen for {qid}")
            expected_decision = expected_fast_decision(question)
            if expected_decision is None:
                fail(f"code document routing cannot be reproduced for {qid}")
            expected_picked, expected_diagnostics = expected_decision
            if decision.get("diagnostics") != expected_diagnostics:
                fail(f"code document routing diagnostics mismatch for {qid}")
        elif selector == "qwen":
            if expected_fast_decision(question) is not None:
                fail(f"Qwen routing bypassed a high-confidence route for {qid}")
            candidates = doc_select.coarse_candidates(question, k=12)
            if len(candidates) <= 2:
                if docsel_calls:
                    fail(f"small-corpus routing unexpectedly called Qwen for {qid}")
                expected_picked = candidates
            else:
                if len(docsel_calls) != 1:
                    fail(f"Qwen document routing call mismatch for {qid}")
                content = str(
                    (docsel_calls[0].get("response") or {}).get("content") or ""
                )
                parsed = doc_select._parse_selected_ids(content, candidates, 4)
                expected_picked = doc_select._finalize_picks(
                    question, parsed, candidates, 4
                )
        else:
            fail(f"docsel_log has unknown selector for {qid}: {selector!r}")
        if decision.get("picked") != expected_picked:
            fail(f"document routing decision mismatch for {qid}")

    # ---------------------------------------------------------- evidence join
    evidence_payload, evidence = evidence_entries(out / "evidence.json")
    validate_schema2_evidence_provenance(evidence_payload, config)
    missing_ev = sorted(qid_set - set(evidence))
    extra_ev = sorted(set(evidence) - qid_set)
    if missing_ev or extra_ev:
        fail(f"evidence qid mismatch: missing={missing_ev}, extra={extra_ev}")

    # Index which global raw records belong to each question.  Evidence must
    # reference exactly these records by their 1-based source line/call index.
    expected_api_refs = defaultdict(set)
    for line, row in enumerate(audit, 1):
        qid = row.get("qid")
        if qid in qid_set:
            expected_api_refs[qid].add(line)
    expected_ledger_refs = defaultdict(set)
    for line, call in enumerate(calls, 1):
        qid = call.get("qid")
        if qid in qid_set:
            expected_ledger_refs[qid].add(line)

    for qid in question_ids:
        entry = evidence[qid]
        validate_complete_evidence(entry, qid)
        question = question_by_qid[qid]
        if entry.get("question") != question.get("question"):
            fail(f"evidence question differs from input for {qid}")
        if entry.get("options") != (question.get("options") or {}):
            fail(f"evidence options differ from input for {qid}")
        if entry.get("answer_format") != question.get("answer_format"):
            fail(f"evidence answer format differs from input for {qid}")

        expected_selected = []
        for row in by_docsel[qid]:
            expected_selected.extend(str(value)
                                     for value in row.get("picked") or [])
        for row in by_run[qid]:
            expanded = row.get("doc_expanded")
            if isinstance(expanded, list):
                expected_selected.extend(str(value) for value in expanded)
            elif expanded:
                expected_selected.append(str(expanded))
        expected_selected = list(dict.fromkeys(expected_selected))
        retrieval_info = entry.get("retrieval") or {}
        actual_selected = [str(value) for value in
                           retrieval_info.get("selected_doc_ids") or []]
        if actual_selected != expected_selected:
            fail(f"evidence selected documents differ from run for {qid}")
        qid_calls = [row for row in audit if row.get("qid") == qid]
        if retrieval_info.get("query_surfaces") != \
                _query_surfaces(question, qid_calls):
            fail(f"evidence query surfaces differ from run for {qid}")
        ev_slots = answer_slots(entry.get("answer_slots"),
                                f"evidence.{qid}.answer_slots",
                                len(answer_columns))
        if ev_slots != csv_answers[qid]:
            fail(f"evidence answer_slots mismatch for {qid}")
        if not answer_equivalent(entry.get("answer"), csv_answers[qid]):
            fail(f"evidence answer mismatch for {qid}")
        if entry.get("reasoning") != reasonings[qid]:
            fail(f"evidence reasoning mismatch for {qid}")
        if entry.get("reasoning_source_stage") != \
                sources[qid].get("reasoning_stage"):
            fail(f"evidence reasoning stage mismatch for {qid}")
        expected_postprocessing = {
            "raw_answer": sources[qid].get("raw_answer"),
            "formatted_slots": answers[qid],
        }
        if entry.get("postprocessing") != expected_postprocessing:
            fail(f"evidence postprocessing differs from run for {qid}")
        expected_traces = [
            {
                "stage": trace.get("stage"),
                "parsed_answer": trace.get("answer"),
                "visible_output": trace.get("content"),
                "evidence_ids": trace.get("evidence_ids") or [],
            }
            for trace in sources[qid].get("traces") or []
        ]
        if entry.get("model_traces") != expected_traces:
            fail(f"evidence model traces mismatch for {qid}")
        if [str(value) for value in
                entry.get("retrieval", {}).get("evidence_ids") or []] != [
                    str(value) for value in
                    sources[qid].get("evidence_ids") or []
                ]:
            fail(f"evidence package/selected reasoning mismatch for {qid}")

        accounting = entry.get("token_accounting")
        if not isinstance(accounting, dict):
            fail(f"evidence has no token_accounting for {qid}")
        ev_usage = (
            token_int(accounting.get("prompt_tokens"),
                      f"evidence.{qid}.prompt_tokens"),
            token_int(accounting.get("completion_tokens"),
                      f"evidence.{qid}.completion_tokens"),
            token_int(accounting.get("total_tokens"),
                      f"evidence.{qid}.total_tokens"),
        )
        if ev_usage != csv_usage[qid] or ev_usage[2] != ev_usage[0] + ev_usage[1]:
            fail(f"evidence/CSV token mismatch for {qid}: "
                 f"evidence={ev_usage} CSV={csv_usage[qid]}")
        if set(accounting) != {
            "prompt_tokens", "completion_tokens", "total_tokens",
            "ledger_calls",
        }:
            fail(f"evidence token accounting has unexpected fields for {qid}")
        if ev_usage[:2] != clean_ledger[qid]:
            fail(f"evidence usage differs from direct ledger for {qid}")

        api_refs = entry.get("api_attempts")
        if not isinstance(api_refs, list) or not api_refs:
            fail(f"evidence has no API references for {qid}")
        seen_api = set()
        has_success = False
        for ref in api_refs:
            line, embedded = _embedded_record(
                ref, "audit_line", f"evidence.{qid}.api")
            if line > len(audit):
                fail(f"evidence API reference out of range for {qid}: {line}")
            if line in seen_api:
                fail(f"duplicate evidence API reference for {qid}: {line}")
            seen_api.add(line)
            if embedded != audit[line - 1]:
                fail(f"evidence API record differs from audit line {line} "
                     f"for {qid}")
            has_success |= embedded.get("status") == "ok"
        if seen_api != expected_api_refs[qid]:
            fail(f"evidence API reference set mismatch for {qid}: "
                 f"evidence={sorted(seen_api)} "
                 f"expected={sorted(expected_api_refs[qid])}")
        if not has_success:
            fail(f"evidence has no successful API reference for {qid}")

        ledger_refs = accounting.get("ledger_calls")
        if not isinstance(ledger_refs, list) or not ledger_refs:
            fail(f"evidence has no ledger references for {qid}")
        seen_ledger = set()
        referenced_usage = [0, 0]
        for ref in ledger_refs:
            line, embedded = _embedded_record(
                ref, "ledger_call", f"evidence.{qid}.ledger")
            if line > len(calls):
                fail(f"evidence ledger reference out of range for {qid}: {line}")
            if line in seen_ledger:
                fail(f"duplicate evidence ledger reference for {qid}: {line}")
            seen_ledger.add(line)
            original = calls[line - 1]
            if embedded != original:
                fail(f"evidence ledger record differs from call {line} for {qid}")
            if original.get("qid") != qid:
                fail(f"evidence references another qid's ledger call for {qid}")
            expected_pair = (
                token_int(original.get("prompt_tokens"),
                          f"ledger.calls.{line}.prompt_tokens"),
                token_int(original.get("completion_tokens"),
                          f"ledger.calls.{line}.completion_tokens"),
            )
            referenced_usage[0] += expected_pair[0]
            referenced_usage[1] += expected_pair[1]
        if seen_ledger != expected_ledger_refs[qid]:
            fail(f"evidence ledger reference set mismatch for {qid}: "
                 f"evidence={sorted(seen_ledger)} "
                 f"expected={sorted(expected_ledger_refs[qid])}")
        if tuple(referenced_usage) != clean_ledger[qid]:
            fail(f"evidence ledger references do not sum for {qid}")

        embedded_runs = entry.get("run_log_records")
        if not isinstance(embedded_runs, list) or not embedded_runs:
            fail(f"evidence has no run_log records for {qid}")
        if embedded_runs != by_run[qid]:
            fail(f"evidence run_log records differ from run_log.jsonl for {qid}")

    if set(evidence_payload) != {"format_version", "provenance", "questions"}:
        fail("evidence top-level object has unexpected fields")
    if evidence_payload.get("format_version") != 2:
        fail("evidence format_version must be 2")

    # --------------------------------------------------------------- config
    model_values = [value for key, value in config.items()
                    if "model" in str(key).lower() and value]
    if not model_values or any(
            not is_allowed_model(value)
            for value in model_values if isinstance(value, str)):
        fail("run config model is outside Qwen3.5/3.6/3.7")

    # A trace/run-log join alone is insufficient: deterministic postprocessing
    # could otherwise manufacture a new explanation after the paid call.  The
    # exact selected reasoning must occur in a successful response explicitly
    # attached directly to that question.
    for qid in question_ids:
        if not any(_api_attached_to(row, qid) and
                   _response_contains_reasoning(row, reasonings[qid])
                   for row in successes):
            fail("selected reasoning is not literal output of an attached "
                 f"successful API response for {qid}")

    print(f"reproduction check OK: {len(question_ids)} questions, "
          f"{declared[2]:,} tokens, "
          f"{len(successes)} successful API calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--qdir")
    parser.add_argument("--submit-template")
    args = parser.parse_args()
    main(args.output_dir, qdir=args.qdir,
         submit_template=args.submit_template)
