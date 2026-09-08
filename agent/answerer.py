"""逐题答题流程：记忆卡 + 定向检索 → 逐项判断 → 复核 → 答案规范化。

AFAC_STABLE=1 环境变量启用稳定模式：非计算域关闭思维链+低温采样（降方差降token）。
"""
import json, os, pathlib, re

from . import retrieval
from .paths import PROCESSED_DIR
from .qwen_client import chat, DEFAULT_MODEL

FMT_NAME = {"mcq": "单选题(唯一正确答案)", "multi": "多选题(一个或多个正确)",
            "tf": "判断题(A/B其一)"}


_INS_TITLES = None


def _doc_title(doc_id):
    # PDF title layers can be duplicated or malformed; prefer the offline
    # company/product identity map when it is available.
    global _INS_TITLES
    meta = retrieval.docs_meta()[doc_id]
    if meta["domain"] == "insurance":
        if _INS_TITLES is None:
            p = PROCESSED_DIR / "insurance_titles.json"
            _INS_TITLES = json.load(open(p)) if p.exists() else {}
        t = _INS_TITLES.get(doc_id)
        if t:
            return f"{t['company']}{t['product']}"
    return meta["title"]


# ---------------- 证据组装 ----------------


def gather_evidence(q, k_opt=2, k_q=3, cap=9000, extra_queries=()):
    doc_ids = q["doc_ids"]
    queries = [q["question"]] + [f"{q['question'][:40]} {t}" for t in q["options"].values()]
    # Numeric and nonnumeric option surfaces provide complementary lexical hits.
    for t in q["options"].values():
        stripped = re.sub(r"[0-9.,%％]+", " ", t)
        if stripped != t and len(stripped.strip()) >= 8:
            queries.append(stripped)
    queries += list(extra_queries)
    # Protect the strongest per-document hit for each quoted phrase.
    qtext = q["question"] + " " + " ".join(q["options"].values())
    hard_kws = list(dict.fromkeys(
        m.group(1).strip()
        for m in re.finditer(r"[‘’“\"《]([^’”\"》]{2,24})[’”\"》]", qtext)
        if m.group(1).strip()
    ))[:8]
    forced = []
    for kw in hard_kws:
        for d in doc_ids:
            hits_kw = retrieval.doc_index(d).search(kw, k=2)
            cands_kw = [c for c, _s in hits_kw if kw in c["text"]]
            if not cands_kw:
                continue
            # Prefer a numeric-bearing line when the same anchor has several hits.
            with_num = [c for c in cands_kw if any(
                kw in ln and re.search(r"[\d％%]", ln)
                for ln in c["text"].split("\n"))]
            forced.append((with_num or cands_kw)[0])
    # Option-local anchors.  ``search_docs`` normalizes scores independently
    # inside each document; in a genuine multi-document comparison several
    # unrelated hits can therefore tie at 1.0 and only the first document's
    # hit receives the global protected slot below.  Search each selected
    # document with the option alone, then protect the strongest raw-BM25 hit
    # (or every document whose title is named explicitly in that option).
    # This is bounded to at most one new chunk per option in the usual case
    # and uses only the current question, selected sources and source titles.
    def _norm_identity(value):
        value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value or "")
        for boiler in ("股份有限公司", "有限责任公司", "有限公司",
                       "募集说明书", "年度报告", "研究报告", "首次覆盖报告",
                       "公开发行", "面向专业投资者", "公司债券"):
            value = value.replace(boiler, "")
        return value

    def _named_in_option(option, doc_id):
        left = _norm_identity(option)
        right = _norm_identity(_doc_title(doc_id))
        if len(left) < 4 or len(right) < 4:
            return False
        # A four-character company/product alias is already discriminative in
        # these corpora; prefer the longest shared literal when one exists.
        for width in range(min(14, len(left), len(right)), 3, -1):
            if any(left[i:i + width] in right
                   for i in range(len(left) - width + 1)):
                return True
        return False

    for option in q.get("options", {}).values():
        local = []
        for d in doc_ids:
            hits = retrieval.doc_index(d).search(option, k=1)
            if hits:
                c, raw_score = hits[0]
                local.append((c, raw_score, _named_in_option(option, d)))
        named = [item for item in local if item[2]]
        chosen = named or (max(local, key=lambda item: item[1])
                           if local else None)
        for c, _score, _is_named in (named if named else
                                      ([chosen] if chosen else [])):
            forced.append(c)
    # 跨查询命中的同一块只保留最高分。
    # 每条查询的 top-1 受保护，预算截断时优先保留。
    best, chunk_by_id, protected = {}, {}, set()
    n_core = 1 + len(q["options"])  # 题干+原始选项查询享受top-1保护
    n_extra = len(list(extra_queries))  # 定向补查的top-1同样保护
    # Directed queries receive the same top-hit protection as core queries.
    for i, query in enumerate(queries):
        k = k_q if i == 0 else k_opt
        hits = retrieval.search_docs(doc_ids, query, k_per_doc=k)
        if hits and (i < n_core or i >= len(queries) - n_extra):
            protected.add(hits[0][0]["id"])
        for c, s in hits:
            cid = c["id"]
            chunk_by_id[cid] = c
            if s > best.get(cid, 0):
                best[cid] = s
    for c in forced:
        cid = c["id"]
        chunk_by_id[cid] = c
        protected.add(cid)
        best[cid] = max(best.get(cid, 0), 1e9)  # 强制块置顶
    out = [(chunk_by_id[cid], s) for cid, s in best.items()]
    # 目录/图表索引块降权（占坑但无正文信息量）
    def _is_toc(c):
        t = c["text"]
        return t.count("……") >= 3 or t.count("...") >= 6 or \
            len(re.findall(r"^[图表]：", t, re.M)) >= 4
    out.sort(key=lambda x: (x[0]["id"] not in protected, _is_toc(x[0]), -x[1]))
    kept, total = [], 0
    for c, s in out:
        piece_len = len(c["text"]) + 20
        # Protected top hits and forced keyword blocks survive the soft cap.
        if c["id"] not in protected and total + piece_len > cap:
            continue
        total += piece_len
        kept.append(c)
    # Retain at least one substantive block from every selected document.
    have = {c["doc_id"] for c in kept}
    for d in doc_ids:
        if d in have:
            continue
        cand = next((c for c, _s in out
                     if c["doc_id"] == d and not _is_toc(c)), None)
        if cand is None:
            continue
        while kept and sum(len(c["text"]) + 20 for c in kept) \
                + len(cand["text"]) + 20 > cap:
            victims = [c for c in kept
                       if c["id"] not in protected and c["doc_id"] != d]
            if not victims:
                break
            kept.remove(victims[-1])
        kept.append(cand)
    kept.sort(key=lambda c: (c["doc_id"], c["page"] or 0,
                             int(c["id"].split("#c")[1])))
    parts = []
    for c in kept:
        tag = f"{c['doc_id']} P{c['page']}" if c["page"] else c["id"]
        parts.append(f"【{tag}】{c['text']}")
    return "\n\n".join(parts), kept, protected


def _render(kept):
    parts = []
    for c in kept:
        tag = f"{c['doc_id']} P{c['page']}" if c["page"] else c["id"]
        parts.append(f"【{tag}】{c['text']}")
    return "\n\n".join(parts)


_FIN_FACTS = None


def financial_facts_block(q):
    """Return source-derived statement cells ranked against the question."""
    global _FIN_FACTS
    if (os.environ.get("AFAC_FIN_FACTS") != "1"
            or q.get("domain") != "financial_reports"):
        return ""
    if _FIN_FACTS is None:
        p = pathlib.Path(__file__).resolve().parents[1] \
            / "processed_data" / "financial_facts.json"
        _FIN_FACTS = json.load(open(p)) if p.exists() else {}
    qtext = q["question"] + " " + " ".join((q.get("options") or {}).values())
    # Character bigrams provide deterministic lexical ranking.
    qgrams = {run[i:i+2] for run in re.findall(r"[一-鿿]+", qtext)
              for i in range(len(run) - 1)}
    rows = []
    for d in q.get("doc_ids") or []:
        for r in _FIN_FACTS.get(d, []):
            label = r.split(":")[0]
            lgrams = {run[i:i+2] for run in re.findall(r"[一-鿿]+", label)
                      for i in range(len(run) - 1)}
            score = len(lgrams & qgrams)
            if score >= 3:
                rows.append((score, f"[{d}]{r}"))
    rows.sort(key=lambda x: -x[0])
    if not rows:
        return ""
    # Round-robin across table names so one statement cannot consume the cap.
    buckets, order = {}, []
    for s, r in rows:
        t = r.split("]")[1] if "]" in r else "?"
        if t not in buckets:
            order.append(t)
        buckets.setdefault(t, []).append(r)
    picked, i = [], 0
    while len(picked) < 40 and any(buckets.values()):
        t = order[i % len(order)]
        if buckets[t]:
            picked.append(buckets[t].pop(0))
        i += 1
        if i > 400:
            break
    return ("报表单元格速查表（保留来源表头、列身份和页码）:\n" +
            "\n".join(picked))


def evidence_block(q, extra_queries=()):
    """返回当前题的证据文本、原文块与受保护块集合。"""
    domain = q["domain"]
    blocks = []
    ff = financial_facts_block(q)
    if ff:
        blocks.append(ff)
    capsule = ""
    if domain == "insurance" and os.environ.get("AFAC_INS_CAPSULES") == "1":
        from .insurance_capsules import insurance_capsule_block
        capsule = insurance_capsule_block(
            q, char_budget=int(os.environ.get("AFAC_INS_CAPSULE_BUDGET",
                                              "4800")))
        if capsule:
            blocks.append(capsule)
    titles = "涉及文档:\n" + "\n".join(
        f"- {d}: 《{_doc_title(d)}》" for d in q["doc_ids"])
    blocks.append(titles)
    if capsule:
        cap = int(os.environ.get("AFAC_INS_RAW_CAP", "1800"))
    else:
        base = int(os.environ.get("AFAC_RAW_EVIDENCE_BASE", "3600"))
        per_source = int(os.environ.get(
            "AFAC_RAW_EVIDENCE_PER_SOURCE", "1000"))
        maximum = int(os.environ.get("AFAC_RAW_EVIDENCE_MAX", "9000"))
        cap = min(maximum, base + per_source * max(
            0, len(set(q.get("doc_ids") or ())) - 2))
    ev, kept, prot = gather_evidence(q, k_opt=2, k_q=2, cap=cap,
                                     extra_queries=extra_queries)
    blocks.append("原文片段证据:\n" + ev)
    return "\n\n".join(blocks), kept, prot


# ---------------- 作答与解析 ----------------

# Accept contiguous or punctuation-separated option letters, then normalize
# both forms through the same answer parser.
ANSWER_RE = re.compile(
    r"答案[:：]\s*([A-D](?:[\s,，、;/；]*[A-D]){0,3})")
SEARCH_RE = re.compile(r"补充检索[:：]\s*(.+)")


def normalize(ans, fmt):
    letters = [c for c in ans.upper() if c in "ABCD"]
    if not letters:
        return ""
    if fmt in ("mcq", "tf"):
        return letters[0]
    return "".join(sorted(set(letters)))


_FALLBACK_BAD = re.compile(r"不选|无法|判断[:：]|入选|分析|证据|复核|标准")


def parse_answer(content, fmt):
    m = list(ANSWER_RE.finditer(content))
    if m:
        return normalize(m[-1].group(1), fmt)
    # Fallback accepts only short letter-only lines and rejects analysis text.
    for line in reversed(content.strip().splitlines()):
        s = line.strip()
        if len(s) > 12 or _FALLBACK_BAD.search(s):
            continue
        cand = normalize(s, fmt)
        if cand:
            return cand
    return ""


def select_reasoning(final, traces, fmt):
    """Return one complete, accounted response and its exact evidence set."""
    target = normalize(final or "", fmt)
    for trace in reversed(traces):
        text = (trace.get("content") or "").strip()
        answer = normalize(trace.get("answer") or "", fmt)
        if text and answer == target:
            evidence_ids = [
                str(value) for value in trace.get("evidence_ids") or ()
                if str(value).strip()
            ]
            if not evidence_ids:
                raise RuntimeError(
                    "selected Qwen response has no attributable evidence"
                )
            return text, trace.get("stage", ""), evidence_ids
    raise RuntimeError("final answer has no matching complete Qwen response")


def _q_text(q):
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    return f"题目({FMT_NAME[q['answer_format']]}):\n{q['question']}\n\n选项:\n{opts}"


JUDGE_STD = (
    "核验规则:\n"
    "1. 只依据给定文档作答，不用模型记忆补充事实。\n"
    "2. 把题目和选项拆成可核对的主张，逐一核对主体、期间、条件、动作、"
    "数值和单位，并引用对应页码。\n"
    "3. 摘要和检索片段可能不完整；证据不足时提出具体补充检索词，不能把"
    "没有检索到直接当作不存在。\n"
    "4. 涉及数值时列出取数与算式、统一单位，并按题目要求舍入。\n"
    "5. 多文档题分别核对各来源；同一事项存在多个版本或口径时，说明所用"
    "版本、期间和适用条件。\n"
    "6. 最后按题型要求输出答案，不输出证据无法支持的选项。"
)

_COMPACT_JUDGE_BASE = (
    "核验规则:\n"
    "1. 仅依据给定文档，逐项核对主体、期间、条件、动作、数值和单位。\n"
    "2. 每个判断引用页码；片段不足时输出'补充检索: 关键词'，不要以常识补全。\n"
    "3. 数值题列出取数与算式、统一单位，并按题目要求舍入。\n"
    "4. 多来源、多版本或多口径内容分别核对并说明采用范围。\n"
    "5. 按题型输出答案，不选择证据无法支持的主张。"
)


def judge_std_for(q_or_qs):
    """Compile generic judge rules from input structure and domain metadata."""
    if os.environ.get("AFAC_COMPACT_JUDGE") != "1":
        return JUDGE_STD
    qs = q_or_qs if isinstance(q_or_qs, (list, tuple)) else [q_or_qs]
    text = " ".join(
        str(q.get("question", "")) + " " +
        " ".join(str(v) for v in (q.get("options") or {}).values())
        for q in qs)
    domains = {q.get("domain") for q in qs}
    extra = []
    if re.search(r"\d|%|％", text):
        extra.append(
            "【数值核验】绑定主体、期间、单位和列头后重算；仅最终一步舍入，"
            "并按题干定义区分相对变化、绝对差值和百分点。")
    if "financial_reports" in domains:
        extra.append(
            "【报表】先绑定报表主体、期间、列头和单位，再读取对应单元格；"
            "不得从同名但不同口径的表格取值。")
    if "insurance" in domains:
        extra.append(
            "【合同】逐份绑定当前合同的责任、条件、例外与计算基础；"
            "不得把其他合同的相似条款移入当前来源。")
    if any(len(set(q.get("doc_ids") or ())) > 1 for q in qs):
        extra.append(
            "【多来源】每个对象均需绑定自己的来源证据；不能用一个来源的结论"
            "代替其他来源，也不能凭单个对象推出整体结论。")
    return _COMPACT_JUDGE_BASE + (("\n" + "\n".join(extra)) if extra else "")


def r1_instruction(q):
    return (
        "你是金融文档审读专家。严格依据上述证据逐项判断，证据不足不得臆断。\n"
        + judge_std_for(q) + "\n"
        "输出格式:\n选择标准: <一句话>\n分析: <每个选项一行，引用页码及理由>\n"
        "判断: A入选/不选 B入选/不选 C入选/不选 D入选/不选\n答案: <字母>\n"
        "若关键证据缺失，最后一行输出: 补充检索: <关键词>"
    )


def r2_instruction(q):
    return (
        "忽略初判结论，独立按选择标准逐项复核。重点检查选择标准、数值日期主体、"
        "漏选、过度严苛和无依据主张。\n" + judge_std_for(q) +
        "\n输出格式:\n选择标准: <一句话>\n复核: <每项一行>\n答案: <字母>"
    )

def expand_docs_if_needed(q, query, model=DEFAULT_MODEL):
    """Expand the source set when a follow-up query has a much stronger hit."""
    from . import doc_select  # 延迟导入避免环
    cur = set(q["doc_ids"])
    idx = doc_select.domain_doc_index(q["domain"])
    # Compare current and external documents in the same document-level BM25
    # index.  Per-document chunk scores are independently normalised and are
    # therefore not comparable with this index's raw scores.
    ranked = idx.search(query, k=len(idx.chunks))
    best_in = max((score for chunk, score in ranked
                   if chunk["doc_id"] in cur), default=0.0)
    ext = [(chunk, score) for chunk, score in ranked
           if chunk["doc_id"] not in cur]
    if ext and ext[0][1] > best_in * 1.5:
        new_doc = ext[0][0]["doc_id"]
        return dict(q, doc_ids=q["doc_ids"] + [new_doc]), new_doc
    return q, None


CALC_DOMAINS = ("insurance", "financial_reports")
STABLE = os.environ.get("AFAC_STABLE") == "1"
SLIM = os.environ.get("AFAC_SLIM") == "1"   # 紧凑模式：单样本与紧证据
STABLE_DOMAINS = ("regulatory",) if not STABLE else \
    ("regulatory", "financial_contracts", "research")
VERIFY_MODEL = os.environ.get("AFAC_VERIFY_MODEL", "")


def _think(q):
    """Choose which domains use the model's extended reasoning mode."""
    return q["domain"] not in STABLE_DOMAINS


def answer_question(q, model=DEFAULT_MODEL, log=None, blind_mode=False):
    qid, fmt = q["qid"], q["answer_format"]
    think_r1 = 2200 if q["domain"] in CALC_DOMAINS else 1900
    if SLIM:
        think_r1 = 1600
    ev, kept, _protected = evidence_block(q)
    ev_ids = [c["id"] for c in kept]
    base = ev + "\n\n" + _q_text(q)
    r1_inst = r1_instruction(q)
    r2_inst = r2_instruction(q)

    c1, _r1_reasoning, _ = chat(
        [{"role": "user", "content": base + "\n\n" + r1_inst}],
        qid=qid, model=model, thinking=_think(q), thinking_budget=think_r1,
        max_tokens=4000, tag="r1")
    ans1 = parse_answer(c1, fmt)
    traces = [{"stage": "r1", "content": c1, "answer": ans1,
               "evidence_ids": list(ev_ids)}]
    # 补充检索一轮
    ms = SEARCH_RE.search(c1)
    if ms:
        supp_q = ms.group(1).strip()
        if blind_mode:  # Allow source expansion when the current evidence is incomplete.
            q, added = expand_docs_if_needed(q, supp_q, model=model)
            if added and log is not None:
                log.write(json.dumps({"qid": qid, "doc_expanded": added},
                                     ensure_ascii=False) + "\n")
        ev2, kept, _protected = evidence_block(q, extra_queries=[supp_q])
        ev_ids = [c["id"] for c in kept]
        base = ev2 + "\n\n" + _q_text(q)
        c1b, _t, _ = chat(
            [{"role": "user", "content": base + "\n\n" + r1_inst.rsplit("\n", 1)[0]}],
            qid=qid, model=model, thinking=_think(q), thinking_budget=think_r1,
            max_tokens=4000, tag="r1b")
        if parse_answer(c1b, fmt):
            c1, ans1 = c1b, parse_answer(c1b, fmt)
        traces.append({"stage": "r1b", "content": c1b,
                       "answer": parse_answer(c1b, fmt),
                       "evidence_ids": list(ev_ids)})

    final, c2, ans2 = ans1, None, None
    # Compact mode reviews only when the first answer could not be parsed.
    need_r2 = (not SLIM and fmt in ("multi", "mcq")) or not ans1
    if need_r2:
        r2_base = base
        c2, _t, _ = chat(
            [{"role": "user", "content": r2_base + "\n\n" + r2_inst}],
            qid=qid, model=VERIFY_MODEL or model, thinking=_think(q),
            thinking_budget=1500, max_tokens=2600, tag="r2")
        ans2 = parse_answer(c2, fmt)
        traces.append({"stage": "r2", "content": c2, "answer": ans2,
                       "evidence_ids": list(ev_ids)})
        if ans1 and ans2 and ans2 != ans1:
            # 定向仲裁：只带分歧选项的针对性证据。最终答案必须
            # 直接来自某次已记账的完整模型输出，不拼接候选结果。
            disputed = [l for l in "ABCD"
                        if (l in (ans1 or "")) != (l in (ans2 or ""))]
            dq = [f"{q['question'][:30]} {q['options'][l]}" for l in disputed
                  if l in q["options"]]
            ev3, k3, _p3 = gather_evidence(q, k_opt=3, k_q=2, cap=5500,
                                           extra_queries=dq)
            dtxt = "\n".join(f"{l}. {q['options'][l]}" for l in disputed
                             if l in q["options"])
            adj = ("原文片段证据:\n" + ev3 + "\n\n" + _q_text(q) +
                   f"\n\n两次独立判断在以下选项上有分歧:\n{dtxt}\n"
                   "请仅针对这些分歧选项逐项核对证据并给出该选项是否入选的结论。\n"
                   + judge_std_for(q) + "\n输出格式:\n仲裁: <分歧选项逐项>\n"
                   "答案: <完整最终答案字母>")
            c3, _t, _ = chat([{"role": "user", "content": adj}],
                             qid=qid, model=VERIFY_MODEL or model,
                             thinking=True, thinking_budget=2600,
                             max_tokens=3000, tag="r3")
            ans3 = parse_answer(c3, fmt)
            traces.append({"stage": "r3", "content": c3,
                           "answer": ans3,
                           "evidence_ids": [c["id"] for c in k3]})
            final = ans3 or ans2
        elif ans2:
            final = ans2
    if not final:
        raise RuntimeError(f"{qid}: model produced no valid answer")

    reasoning, reasoning_stage, ev_ids = select_reasoning(final, traces, fmt)

    if log is not None:
        log.write(json.dumps({
            "qid": qid, "final": final, "r1": ans1, "r2": ans2,
            "c1": c1, "c2": c2, "reasoning": reasoning,
            "reasoning_stage": reasoning_stage,
            "evidence_ids": ev_ids},
            ensure_ascii=False) + "\n")
        log.flush()
    return final, {"r1": ans1, "r2": ans2, "c1": c1,
                   "reasoning": reasoning,
                   "reasoning_stage": reasoning_stage,
                   "evidence_ids": ev_ids,
                   "traces": traces}
