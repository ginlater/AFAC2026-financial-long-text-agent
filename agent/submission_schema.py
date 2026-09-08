"""题型归一并从提交模板占位符推断答案 schema。

模板中的占位符定义答案格式：
  90817.43       → 数值，保留两位小数
  73.19%         → 百分数，带%保留两位小数
  对象甲>对象乙    → 排序类，半角 > 连接
  A              → 选项字母
占位符个数 = 该题需要填写的 answer_i 个数。
"""
import csv, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]

TYPE_MAP = {"多选题": "multi", "单选题": "mcq", "判断题": "tf",
            "计算题": "calc", "抽取题": "calc"}
_ANSWER_COLUMN_RE = re.compile(r"^answer_?(\d+)$", re.IGNORECASE)
_AUDIT_COLUMNS = ("prompt_tokens", "completion_tokens", "total_tokens",
                  "reasoning")


class SubmissionSchema(dict):
    """Dictionary-like qid schema carrying the inferred template layout."""

    def __init__(self, values, *, fieldnames, answer_columns, question_ids):
        super().__init__(values)
        self.fieldnames = tuple(fieldnames)
        self.answer_columns = tuple(answer_columns)
        self.question_ids = tuple(question_ids)
        self.qid_column = "qid"


def output_columns(schema):
    """Return the template columns plus reproducibility audit columns."""

    columns = list(getattr(schema, "fieldnames", ()))
    if not columns:
        answer_columns = list(getattr(schema, "answer_columns", ()))
        columns = ["qid", *answer_columns]
    for column in _AUDIT_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def load_questions(qdir):
    """读取题目目录（.json 与 .jsonl 可混合，文件可带 BOM）。"""
    qs = []
    for f in sorted(pathlib.Path(qdir).iterdir()):
        if f.suffix not in (".json", ".jsonl"):
            continue
        txt = f.read_text(encoding="utf-8-sig")
        data = ([json.loads(l) for l in txt.splitlines() if l.strip()]
                if f.suffix == ".jsonl" else json.loads(txt))
        qs.extend(data)
    for q in qs:
        explicit = str(q.get("answer_format") or "").strip().lower()
        if not q.get("options"):
            q["answer_format"] = "calc"
        elif explicit in {"mcq", "multi", "tf", "calc"}:
            q["answer_format"] = explicit
        else:
            q["answer_format"] = TYPE_MAP.get(q.get("type", ""), "multi")
    return qs


def _slot_kind(ph):
    if ph.endswith("%"):
        return "percent"
    if ">" in ph:
        return "ranking"
    if re.fullmatch(r"[0-9.]+", ph):
        return "number"
    if re.search(r"\d{4}年", ph):
        return "date"
    if ph in ("A", "B", "C", "D") or re.fullmatch(r"[A-D]+", ph):
        return "letter"
    return "text"


def load_schema(submit_csv):
    """Return a qid mapping and layout inferred from the supplied template."""

    values = {}
    question_ids = []
    with open(submit_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or ())
        if "qid" not in fieldnames:
            raise ValueError("submission template has no qid column")
        indexed = []
        for column in fieldnames:
            match = _ANSWER_COLUMN_RE.fullmatch(column)
            if match:
                indexed.append((int(match.group(1)), column))
        if not indexed or len({index for index, _column in indexed}) != len(indexed):
            raise ValueError("submission template has invalid answer columns")
        answer_columns = [column for _index, column in sorted(indexed)]
        for row_no, row in enumerate(reader, 2):
            qid = str(row.get("qid") or "").strip()
            if not qid:
                raise ValueError(f"submission template has empty qid at row {row_no}")
            if qid.casefold() == "summary":
                continue
            if qid in values:
                raise ValueError(f"duplicate qid in submission template: {qid}")
            placeholders = [str(row.get(column) or "").strip()
                            for column in answer_columns]
            active = [index for index, value in enumerate(placeholders) if value]
            if not active:
                raise ValueError(f"submission template has no answer slot for {qid}")
            last = active[-1]
            if any(not placeholders[index] for index in range(last + 1)):
                raise ValueError(
                    f"submission template has a gap in answer slots for {qid}")
            values[qid] = [_slot_kind(value)
                           for value in placeholders[:last + 1]]
            question_ids.append(qid)
    return SubmissionSchema(
        values, fieldnames=fieldnames, answer_columns=answer_columns,
        question_ids=question_ids)


_PERCENT_QUESTION_RE = re.compile(
    r"(?:同比|环比|较[^，。；]{0,16})(?:增长率|下降率|增速)|"
    r"(?:百分比|百分率|毛利率|收益率|增值率)[^，。；]{0,12}(?:多少|为)"
)
_PLAIN_NUMBER_RE = re.compile(r"(?:不带\s*[%％]|不带单位|百分点)")
_DATE_QUESTION_RE = re.compile(r"(?:哪一天|何时|什么日期|日期为多少)")


def refine_schema_from_questions(schema, questions):
    """Use explicit wording to refine an ambiguous numeric placeholder.

    Some templates use a generic numeric placeholder even when a single-slot
    question explicitly asks for a percentage rate or calendar date.  This
    query-driven rule applies uniformly and never consults identifiers or
    answers.
    """
    by_qid = {str(question.get("qid") or ""): question for question in questions}
    for qid, kinds in schema.items():
        question = by_qid.get(qid)
        if not question or kinds != ["number"]:
            continue
        text = str(question.get("question") or "")
        if _DATE_QUESTION_RE.search(text):
            schema[qid] = ["date"]
        elif (_PERCENT_QUESTION_RE.search(text)
              and not _PLAIN_NUMBER_RE.search(text)):
            schema[qid] = ["percent"]
    return schema


# ---------------- 答案格式化 ----------------

_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")


def fmt_slot(value, kind):
    """把模型给出的答案片段规范成模板要求的格式。"""
    v = (value or "").strip().strip("。;；,，")
    if kind == "letter":
        letters = [c for c in v.upper() if c in "ABCD"]
        return "".join(sorted(set(letters)))
    if kind == "ranking":
        # 统一为半角 > 且前后无空格；去掉可能的公司后缀空白
        parts = re.split(r"\s*[>＞]\s*", v)
        parts = [p.strip().strip("。；;,，") for p in parts if p.strip()]
        return ">".join(parts)
    if kind == "date":
        m = re.search(r"(\d{4})\D{1,2}(\d{1,2})\D{1,2}(\d{1,2})", v)
        if m:
            return f"{int(m.group(1))}年{int(m.group(2))}月{int(m.group(3))}日"
        return v
    if kind in ("percent", "number"):
        m = _NUM.search(v.replace("%", ""))
        if not m:
            return v
        num = float(m.group(0).replace(",", ""))
        s = f"{num:.2f}"
        return s + "%" if kind == "percent" else s
    return v


def split_answer(raw, kinds):
    """把模型的一行答案拆成 len(kinds) 个槽位。"""
    raw = (raw or "").strip()
    if len(kinds) == 1:
        return [fmt_slot(raw, kinds[0])]
    # 优先按分号或中文分号切分多个槽位。
    parts = re.split(r"[;；]", raw)
    if len(parts) < len(kinds):
        # 退化：按逗号切，但排序类内部不含逗号才安全
        alt = re.split(r"[,，]", raw)
        if len(alt) >= len(kinds):
            parts = alt
    parts = [p for p in (x.strip() for x in parts) if p]
    if len(parts) < len(kinds):
        # 兜底：模型未按分号分隔时，按顺序抽取全部数字填充数值/百分数槽
        nums = _NUM.findall(raw)
        if len(nums) >= len(kinds) and all(k in ("number", "percent")
                                           for k in kinds):
            parts = nums
    out = []
    for i, kind in enumerate(kinds):
        out.append(fmt_slot(parts[i] if i < len(parts) else "", kind))
    return out


def write_submission(path, results, schema, order, ledger_per_qid,
                     totals, reasonings=None):
    """按当前模板写 CSV，并附加可复核的 token 与 reasoning 字段。

    每题 token 直接汇总该题独占 API 调用的原始 usage；reasoning 必须
    来自运行器收集的非空模型输出。
    """
    p, c, t = totals
    reasonings = reasonings or {}
    answer_columns = list(getattr(schema, "answer_columns", ()))
    if not answer_columns:
        raise ValueError("schema has no inferred answer columns")
    columns = output_columns(schema)
    unexpected = {
        key: value for key, value in ledger_per_qid.items()
        if key not in order and any(int(part) for part in value)
    }
    if unexpected:
        raise ValueError(
            "token ledger contains calls not bound to one template qid: "
            + ", ".join(sorted(unexpected))
        )
    direct_p = sum(int(ledger_per_qid.get(qid, [0, 0])[0]) for qid in order)
    direct_c = sum(int(ledger_per_qid.get(qid, [0, 0])[1]) for qid in order)
    if (direct_p, direct_c, direct_p + direct_c) != (p, c, t):
        raise ValueError("per-question raw usage does not equal summary usage")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        summary = {column: "" for column in columns}
        summary.update({"qid": "summary", "prompt_tokens": p,
                        "completion_tokens": c, "total_tokens": t})
        w.writerow(summary)
        for qid in order:
            slots = list(results.get(qid) or ())
            expected_slots = len(schema.get(qid, ()))
            if expected_slots and len(slots) > expected_slots:
                raise ValueError(f"too many answer slots for {qid}")
            qp, qc = map(int, ledger_per_qid.get(qid, [0, 0]))
            rs = (reasonings.get(qid) or "").replace("\n", " ").strip()
            row = {column: "" for column in columns}
            row.update({"qid": qid, "prompt_tokens": qp,
                        "completion_tokens": qc, "total_tokens": qp + qc,
                        "reasoning": rs})
            for column, value in zip(answer_columns, slots):
                row[column] = value
            w.writerow(row)
