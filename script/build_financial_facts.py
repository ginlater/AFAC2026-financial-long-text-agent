#!/usr/bin/env python3
"""Extract statement cells while preserving scope and period identity.

Word coordinates are clustered into rows and numeric columns, then bound to
headers such as consolidated/company and current/prior period. The output is
``processed_data/financial_facts.json``.
"""
import argparse
import json
import pathlib
import re

import pymupdf as fitz

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw" / "financial_reports"
OUT = ROOT / "processed_data" / "financial_facts.json"

TITLE_RE = re.compile(
    r"(合并及公司|合并及母公司|合并|母公司|公司)(资产负债表|利润表|损益表|现金流量表)")
NUM_RE = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?$|^-$")
YEAR_RE = re.compile(r"20\d{2}")


def page_rows(page):
    """词坐标 → 行（y中心连续分组, 容差4.5pt——中西文基线不同y0会漂）。"""
    words = sorted(page.get_text("words"),
                   key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    out, cur, last_y = [], [], None
    for x0, y0, x1, y1, w, *_ in words:
        yc = (y0 + y1) / 2
        if last_y is not None and yc - last_y > 4.5:
            out.append(sorted(cur))
            cur = []
        cur.append((x0, x1, w))
        last_y = yc
    if cur:
        out.append(sorted(cur))
    return out


def cluster_columns(xs, gap=28):
    """数字单元格x坐标 → 列中心（间距聚类）。"""
    if not xs:
        return []
    xs = sorted(xs)
    centers, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] > gap:
            centers.append(sum(cur) / len(cur))
            cur = [x]
        else:
            cur.append(x)
    centers.append(sum(cur) / len(cur))
    return centers


def extract_statement(page, title):
    """一页主报表 → 事实行列表。"""
    rows = page_rows(page)
    # 真表门槛：审计报告等散文页会误匹配表头字样, 数字单元格<15视为非表
    n_num = sum(1 for row in rows for x0, x1, w in row
                if NUM_RE.match(w) and any(c.isdigit() for c in w))
    if n_num < 15:
        return []
    # 收集全页数字单元格x
    numx = []
    for row in rows:
        for x0, x1, w in row:
            if NUM_RE.match(w) and any(c.isdigit() for c in w):
                numx.append(x1)  # 右缘聚类：财务表右对齐, x0随位数漂移
    cols = cluster_columns(numx)
    if len(cols) < 2:
        return []
    dual = "及" in title  # 合并及公司 → 4值列
    ncols = 4 if dual else 2
    # 取x最大的ncols列为值列（左侧可能有附注列混入数字）
    vcols = cols[-ncols:] if len(cols) >= ncols else cols
    if dual:
        heads = ["合并本期", "合并上期", "公司本期", "公司上期"]
    elif title.startswith("母公司") or title.startswith("公司"):
        heads = ["母公司本期", "母公司上期"]
    else:
        heads = ["合并本期", "合并上期"]
    heads = heads[:len(vcols)]
    facts = []
    for row in rows:
        label_parts, cells = [], {}
        for x0, x1, w in row:
            if (NUM_RE.match(w) and (any(c.isdigit() for c in w) or w == "-")
                    and x1 >= vcols[0] - 60):
                ci = min(range(len(vcols)), key=lambda i: abs(vcols[i] - x1))
                cells.setdefault(ci, w)
            elif not NUM_RE.match(w):
                label_parts.append(w)
        label = "".join(label_parts).strip()
        label = re.sub(r"附注|[一二三四五六七八九十]+\([\d一-鿿()a-z]*\)?", "", label)
        label = label.strip("|、 ")
        if not cells or not label or len(label) > 40:
            continue
        if not re.search(r"[一-鿿]{2}", label):
            continue
        if "。" in label or "审计" in label or "错报" in label:
            continue  # 散文行
        vals = " ".join(f"{heads[i]}={cells[i]}" for i in sorted(cells)
                        if i < len(heads))
        facts.append(f"[{title}] {label}: {vals}")
    return facts


def build(raw_root=ROOT / "dataset" / "raw", output_path=OUT,
          *, verbose=False):
    raw_reports = pathlib.Path(raw_root) / "financial_reports"
    output_path = pathlib.Path(output_path)
    result = {}
    pdfs = sorted(set(raw_reports.glob("*.PDF")) | set(raw_reports.glob("*.pdf")),
                  key=lambda path: path.as_posix())
    for pdf in pdfs:
        doc_id = pdf.stem
        with fitz.open(pdf) as document:
            facts = []
            carry, carry_p = None, -9
            for pno in range(len(document)):
                head = document[pno].get_text()[:400]
                match = TITLE_RE.search(head.replace(" ", ""))
                if match:
                    title = match.group(0)
                    years = YEAR_RE.findall(head)
                    ttl = f"{title}{'/'.join(years[:1])}" if years else title
                    carry, carry_p = ttl, pno
                elif carry and pno - carry_p <= 3:
                    ttl = carry + "(续)"
                else:
                    continue
                got = [fact + f" (P{pno + 1})"
                       for fact in extract_statement(document[pno], ttl)]
                if match and len(got) < 4:
                    carry, carry_p = None, -9
                facts += got
        result[doc_id] = facts
        if verbose:
            print(f"{doc_id}: {len(facts)}行", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=0) + "\n",
                           encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path,
                        default=ROOT / "dataset" / "raw")
    parser.add_argument("--output", type=pathlib.Path, default=OUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = build(args.input, args.output, verbose=args.verbose)
    print(f"→ {args.output} ({len(result)} documents)")


if __name__ == "__main__":
    main()
