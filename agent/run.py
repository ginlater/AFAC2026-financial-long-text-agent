"""Run a complete question set with per-question retrieval and answering.

Usage::

    python -m agent.run --output-dir OUTPUT \
        --qdir INPUT/questions --submit-template INPUT/submit.csv
"""

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import answerer, calc, doc_select  # noqa: E402
from agent import submission_schema  # noqa: E402
from agent.paths import PROCESSED_DIR  # noqa: E402
from agent.qwen_client import (  # noqa: E402
    DEFAULT_MODEL,
    LEDGER,
    close_audit,
    configure_audit,
    is_allowed_model,
)
from agent.repro import LockedJsonlWriter, write_run_config  # noqa: E402


def _fast_docsel_enabled(question):
    """Return whether source-derived identity routing applies to a question."""

    if os.environ.get("AFAC_FAST_DOCSEL") != "1":
        return False
    if question.get("answer_format") != "calc":
        return True
    if os.environ.get("AFAC_FAST_DOCSEL_CALC") != "1":
        return False
    raw = os.environ.get("AFAC_FAST_DOCSEL_CALC_DOMAINS", "")
    domains = {value.strip() for value in raw.split(",") if value.strip()}
    return not domains or question.get("domain") in domains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qdir", required=True)
    parser.add_argument("--submit-template", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--verify-model", default="")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    effective_verify_model = (
        args.verify_model
        or os.environ.get("AFAC_VERIFY_MODEL")
        or args.model
    )
    for label, model_name in (
        ("model", args.model),
        ("verify model", effective_verify_model),
    ):
        if not is_allowed_model(model_name):
            parser.error(
                f"{label} must belong to Qwen3.5, Qwen3.6, or Qwen3.7; "
                f"got {model_name!r}"
            )
    answerer.VERIFY_MODEL = effective_verify_model

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    processed_dir = pathlib.Path(PROCESSED_DIR).expanduser().resolve()
    if output_dir == processed_dir or processed_dir in output_dir.parents:
        raise RuntimeError(
            f"output directory cannot be inside processed_data: {output_dir}"
        )
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    # The template defines both row order and answer-slot layout.
    questions = submission_schema.load_questions(args.qdir)
    schema = submission_schema.refine_schema_from_questions(
        submission_schema.load_schema(args.submit_template), questions
    )
    by_qid = {}
    for question in questions:
        qid = str(question.get("qid") or "").strip()
        if not qid or qid in by_qid:
            raise RuntimeError(
                f"question set has an empty or duplicate qid: {qid!r}"
            )
        by_qid[qid] = question
    order = list(schema.question_ids)
    questions_only = sorted(set(by_qid) - set(order))
    template_only = sorted(set(order) - set(by_qid))
    if questions_only or template_only:
        raise RuntimeError(
            "question/template qid mismatch: "
            f"questions_only={questions_only}, template_only={template_only}"
        )
    questions = [by_qid[qid] for qid in order]

    configure_audit(output_dir / "api_calls.jsonl", append=False)
    write_run_config(
        output_dir / "run_config.json",
        args,
        args.qdir,
        args.submit_template,
        question_ids=order,
    )

    results = {}
    reasonings = {}
    reasoning_sources = {}
    run_log = LockedJsonlWriter(output_dir / "run_log.jsonl", "w")
    docsel_log = LockedJsonlWriter(output_dir / "docsel_log.jsonl", "w")

    def checkpoint():
        for name, data in (
            ("answers.json", results),
            ("reasonings.json", reasonings),
            ("reasoning_sources.json", reasoning_sources),
        ):
            temporary = output_dir / (name + ".tmp")
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=1)
            temporary.replace(output_dir / name)

    def accept(qid, slots, info):
        reasoning = str(info.get("reasoning") or "").strip()
        if not reasoning:
            raise RuntimeError(f"{qid}: reasoning is empty")
        results[qid] = slots
        reasonings[qid] = reasoning
        reasoning_sources[qid] = info
        checkpoint()

    def work(question):
        qid = question["qid"]
        kinds = schema.get(qid, ["letter"])
        if question.get("doc_ids"):
            picked = list(question["doc_ids"])
            docsel_log.write(
                json.dumps(
                    {
                        "qid": qid,
                        "picked": picked,
                        "kinds": kinds,
                        "selector": "input",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        else:
            decision = None
            if _fast_docsel_enabled(question):
                from agent.docsel_fast import select_docs_fast

                decision = select_docs_fast(question)
            if decision is not None:
                picked, diagnostics = decision
                selector = "code"
                record = {
                    "qid": qid,
                    "picked": picked,
                    "kinds": kinds,
                    "selector": selector,
                    "diagnostics": diagnostics,
                }
            else:
                picked = doc_select.select_docs(
                    question, qid=qid, model=args.model
                )
                record = {
                    "qid": qid,
                    "picked": picked,
                    "kinds": kinds,
                    "selector": "qwen",
                }
            docsel_log.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            question = dict(question, doc_ids=picked)

        if question["answer_format"] == "calc":
            raw, info = calc.answer_calc(
                question,
                kinds,
                model=args.model,
                log=run_log,
                verify_model=effective_verify_model,
                blind_mode=True,
                return_info=True,
            )
            return qid, submission_schema.split_answer(raw, kinds), info

        answer, info = answerer.answer_question(
            question, args.model, run_log, blind_mode=True
        )
        return qid, [submission_schema.fmt_slot(answer, "letter")], info

    started = time.time()
    try:
        print(
            f"answering {len(questions)} questions; model={args.model} "
            f"verify={effective_verify_model}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(work, question): question
                for question in questions
            }
            for position, future in enumerate(as_completed(futures), 1):
                qid, slots, info = future.result()
                accept(qid, slots, info)
                _prompt, _completion, total = LEDGER.totals()
                print(
                    f"[{position}/{len(questions)}] {qid} -> {slots} "
                    f"({total:,} tok)",
                    flush=True,
                )
    finally:
        run_log.close()
        docsel_log.close()
        LEDGER.dump(output_dir / "token_ledger.json")
        close_audit()

    missing_answers = [qid for qid in order if qid not in results]
    if missing_answers:
        raise RuntimeError(f"incomplete run, missing answers: {missing_answers}")
    unexpected_usage = sorted(set(LEDGER.per_qid) - set(order))
    missing_usage = sorted(set(order) - set(LEDGER.per_qid))
    if unexpected_usage or missing_usage:
        raise RuntimeError(
            "token ledger must contain only direct per-question usage: "
            f"unexpected={unexpected_usage}, missing={missing_usage}"
        )
    checkpoint()

    submission_schema.write_submission(
        output_dir / "answer.csv",
        results,
        schema,
        order,
        LEDGER.per_qid,
        LEDGER.totals(),
        reasonings=reasonings,
    )
    prompt, completion, total = LEDGER.totals()
    print(
        f"done in {time.time() - started:.0f}s; tokens {total:,} "
        f"(p={prompt:,} c={completion:,})"
    )
    print(f"output: {output_dir}/answer.csv")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_audit()
