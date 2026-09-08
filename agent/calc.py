"""Evidence-driven workflow for calculation and extraction questions.

The workflow assembles attributable source evidence, asks Qwen to extract
operands before calculating, and can use an independent verification pass.
Missing evidence triggers one lexical expansion.
"""

from __future__ import annotations

import json
import os
import re

from .answerer import (
    _doc_title,
    expand_docs_if_needed,
    gather_evidence,
)
from .qwen_client import DEFAULT_MODEL, chat


SLOT_DESC = {
    "number": "纯数字，按题目要求保留小数，不带单位或千分位逗号",
    "percent": "百分数，带 % 符号",
    "ranking": "排序结果，使用英文半角 > 连接",
    "date": "完整日期，使用 YYYY年M月D日",
    "text": "题目要求的文本，不加额外说明",
}

ANSWER_RE = re.compile(r"^\s*答案[:：]\s*(.+)$", re.MULTILINE)
SEARCH_RE = re.compile(r"补充检索[:：]\s*(.+)")
_SLOT_PATTERNS = {
    "number": re.compile(r"-?[\d,]+(?:\.\d+)?"),
    "percent": re.compile(r"-?[\d,]+(?:\.\d+)?\s*[%％]"),
    "date": re.compile(r"\d{4}年\d{1,2}月\d{1,2}日"),
    "ranking": re.compile(
        r"[^>＞;；]+(?:\s*[>＞]\s*[^>＞;；]+)+"
    ),
}


def _slot_instructions(kinds: list[str] | tuple[str, ...]) -> str:
    return "\n".join(
        f"  {index + 1}. {SLOT_DESC.get(kind, kind)}"
        for index, kind in enumerate(kinds)
    )


def _answer_template(kinds: list[str] | tuple[str, ...]) -> str:
    placeholders = {
        "number": "<数字>",
        "percent": "<百分数>",
        "ranking": "<对象1> > <对象2> > … > <对象N>",
        "date": "<日期>",
        "text": "<文本>",
    }
    return "；".join(placeholders.get(kind, "<值>") for kind in kinds)


def _instruction(kinds: list[str] | tuple[str, ...]) -> str:
    return f"""请仅根据给定证据完成金融取数或计算。

1. 先列出每个必需原始值，标明文档、页码、期间、主体、口径和单位。
2. 再写公式和计算过程；中间步骤保留精度，最后才按题意舍入。
3. 区分百分比变化率与百分点之差，严格对齐年度、公司、报表范围和币种。
4. 不得用常识、记忆或估算补齐缺失值。如必需证据缺失，另起一行写“补充检索: <关键词>”。

答案槽共 {len(kinds)} 个，按顺序为：
{_slot_instructions(kinds)}

最后一行必须严格写为：
答案: {_answer_template(kinds)}
多个槽位用中文分号“；”分隔，最后一行不写解释。""".strip()


def parse_calc(content: str) -> str:
    """Return the last non-placeholder answer line."""

    for match in reversed(list(ANSWER_RE.finditer(content or ""))):
        value = match.group(1).strip()
        if value and not re.search(r"<[^>]+>", value):
            return value
    return ""


def valid_calc(answer: str, kinds: list[str] | tuple[str, ...]) -> bool:
    """Check slot count and surface format before accepting model output."""

    parts = [part.strip() for part in re.split(r"[；;]", answer or "")
             if part.strip()]
    if len(parts) != len(kinds):
        return False
    for part, kind in zip(parts, kinds):
        pattern = _SLOT_PATTERNS.get(kind)
        if pattern is not None and pattern.fullmatch(part) is None:
            return False
    return True


def _structured_memory(q: dict) -> list[str]:
    """Load only generic, source-derived memory blocks enabled at runtime."""

    blocks: list[str] = []
    if q.get("domain") == "insurance" and os.environ.get("AFAC_INS_CAPSULES") == "1":
        from .insurance_capsules import insurance_capsule_block

        capsule = insurance_capsule_block(
            q,
            char_budget=int(os.environ.get("AFAC_INS_CALC_CAPSULE_BUDGET", "6000")),
        )
        if capsule:
            blocks.append(capsule)
    if q.get("domain") == "financial_reports":
        from .answerer import financial_facts_block

        block = financial_facts_block(q)
        if block:
            blocks.append(block)
    return blocks


def calc_evidence(
    q: dict,
    *,
    model: str = DEFAULT_MODEL,
    extra_queries: tuple[str, ...] = (),
    cap_multiplier: int = 1,
) -> tuple[str, list[str]]:
    """Assemble memory cards and lexical evidence for the current question."""

    blocks: list[str] = []
    domain = str(q.get("domain") or "")
    doc_ids = [str(value) for value in q.get("doc_ids") or ()]
    blocks.append("涉及文档:\n" + "\n".join(
        f"- {doc_id}: 《{_doc_title(doc_id)}》" for doc_id in doc_ids
    ))
    blocks.extend(_structured_memory(q))
    base_cap = int(os.environ.get("AFAC_CALC_EVIDENCE_CHARS", "10000"))
    evidence, kept, _protected = gather_evidence(
        q,
        k_opt=4,
        k_q=5,
        cap=max(1000, base_cap * max(1, cap_multiplier)),
        extra_queries=extra_queries,
    )
    blocks.append("原文片段证据:\n" + evidence)
    return "\n\n".join(block for block in blocks if block), [
        str(chunk.get("id") or "") for chunk in kept
    ]


def _call(
    prompt: str,
    *,
    qid: str,
    model: str,
    tag: str,
    thinking_budget: int,
) -> tuple[str, str]:
    content, _tokens, _usage = chat(
        [{"role": "user", "content": prompt}],
        qid=qid,
        model=model,
        thinking=True,
        thinking_budget=thinking_budget,
        max_tokens=int(os.environ.get("AFAC_CALC_MAX_TOKENS", "3600")),
        tag=tag,
    )
    return content, parse_calc(content)


def answer_calc(
    q: dict,
    kinds: list[str] | tuple[str, ...],
    model: str = DEFAULT_MODEL,
    log=None,
    verify_model: str | None = None,
    blind_mode: bool = False,
    return_info: bool = False,
):
    """Answer one calculation question from current evidence only."""

    qid = str(q["qid"])
    verify_model = verify_model or os.environ.get("AFAC_VERIFY_MODEL") or None
    evidence, evidence_ids = calc_evidence(q, model=model)
    base = evidence + "\n\n题目:\n" + str(q.get("question") or "") + \
        "\n\n" + _instruction(kinds)
    budget = int(os.environ.get("AFAC_CALC_THINKING_BUDGET", "2800"))
    first_text, first_answer = _call(
        base, qid=qid, model=model, tag="calc_primary", thinking_budget=budget
    )
    traces = [{"stage": "primary", "content": first_text,
               "answer": first_answer,
               "evidence_ids": list(evidence_ids)}]
    selected_text, selected_answer, selected_stage = (
        first_text, first_answer, "primary"
    )

    search = SEARCH_RE.search(first_text)
    if search or not valid_calc(first_answer, kinds):
        query = search.group(1).strip() if search else str(q.get("question") or "")
        if blind_mode and query:
            expanded, added = expand_docs_if_needed(q, query, model=model)
            if added:
                q = expanded
                if log is not None:
                    log.write(json.dumps({"qid": qid, "doc_expanded": added},
                                         ensure_ascii=False) + "\n")
        evidence, evidence_ids = calc_evidence(
            q,
            model=model,
            extra_queries=(query,) if query else (),
            cap_multiplier=2,
        )
        base = evidence + "\n\n题目:\n" + str(q.get("question") or "") + \
            "\n\n" + _instruction(kinds)
        retry_text, retry_answer = _call(
            base,
            qid=qid,
            model=model,
            tag="calc_evidence_retry",
            thinking_budget=budget,
        )
        traces.append({"stage": "evidence_retry", "content": retry_text,
                       "answer": retry_answer,
                       "evidence_ids": list(evidence_ids)})
        if valid_calc(retry_answer, kinds):
            selected_text, selected_answer, selected_stage = (
                retry_text, retry_answer, "evidence_retry"
            )

    if os.environ.get("AFAC_CALC_SINGLE") != "1":
        second_text, second_answer = _call(
            base,
            qid=qid,
            model=verify_model or model,
            tag="calc_verify",
            thinking_budget=budget,
        )
        traces.append({"stage": "verify", "content": second_text,
                       "answer": second_answer,
                       "evidence_ids": list(evidence_ids)})
        if not valid_calc(selected_answer, kinds) and valid_calc(second_answer, kinds):
            selected_text, selected_answer, selected_stage = (
                second_text, second_answer, "verify"
            )
        elif (valid_calc(selected_answer, kinds) and
              valid_calc(second_answer, kinds) and
              re.sub(r"\s+", "", selected_answer) != re.sub(r"\s+", "", second_answer)):
            arbitration = (
                base + "\n\n两次独立计算不一致。请回到证据逐项核对原始值、"
                "口径、单位和算式，不要以候选标签投票。\n"
                f"候选一: {selected_answer}\n候选二: {second_answer}\n"
                "请写出独立复算过程，最后一行仍按既定格式输出。"
            )
            arb_text, arb_answer = _call(
                arbitration,
                qid=qid,
                model=verify_model or model,
                tag="calc_arbitrate",
                thinking_budget=budget,
            )
            traces.append({"stage": "arbitrate", "content": arb_text,
                           "answer": arb_answer,
                           "evidence_ids": list(evidence_ids)})
            if valid_calc(arb_answer, kinds):
                selected_text, selected_answer, selected_stage = (
                    arb_text, arb_answer, "arbitrate"
                )

    if not valid_calc(selected_answer, kinds):
        raise RuntimeError(f"{qid}: model produced no schema-valid calculation answer")
    reasoning = selected_text.strip()
    selected_trace = next(
        (
            trace for trace in reversed(traces)
            if trace.get("stage") == selected_stage
            and trace.get("content") == selected_text
        ),
        None,
    )
    if selected_trace is None or not selected_trace.get("evidence_ids"):
        raise RuntimeError(f"{qid}: selected calculation has no attributable evidence")
    selected_evidence_ids = [
        str(value) for value in selected_trace["evidence_ids"]
        if str(value).strip()
    ]
    info = {
        "reasoning": reasoning,
        "reasoning_stage": selected_stage,
        "traces": traces,
        "raw_answer": selected_answer,
        "evidence_ids": selected_evidence_ids,
    }
    if log is not None:
        log.write(json.dumps({"qid": qid, "final": selected_answer, **info},
                             ensure_ascii=False) + "\n")
        log.flush()
    return (selected_answer, info) if return_info else selected_answer


__all__ = ["answer_calc", "calc_evidence", "parse_calc", "valid_calc"]
