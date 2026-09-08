"""词法检索：分块 + 中文 unigram/bigram BM25。"""
import json, math, re
from collections import Counter

from .paths import PROCESSED_DIR

PROC = PROCESSED_DIR

_META = None


def docs_meta():
    global _META
    if _META is None:
        with open(PROC / "docs_meta.json", encoding="utf-8") as f:
            _META = json.load(f)
    return _META


def doc_path(doc_id):
    m = docs_meta()[doc_id]
    return PROC / m["domain"] / f"{doc_id}.txt"


# ---------------- 分块 ----------------

_PAGE_RE = re.compile(r"\[P(\d+)\]\n")
_CLAUSE_RE = re.compile(r"(?=\n第[一二三四五六七八九十百零\d]+条)")

MAX_CHUNK = 1400
MIN_CHUNK = 120


def _split_long(text):
    if len(text) <= MAX_CHUNK:
        return [text]
    parts, buf = [], ""
    for para in text.split("\n"):
        if len(buf) + len(para) > MAX_CHUNK and len(buf) >= MIN_CHUNK:
            parts.append(buf)
            buf = para
        else:
            buf = buf + "\n" + para if buf else para
    if buf:
        parts.append(buf)
    return parts


def chunk_doc(doc_id):
    """返回 [{id, doc_id, page, text}]"""
    text = doc_path(doc_id).read_text(encoding="utf-8")
    chunks = []
    if "[P" in text and _PAGE_RE.search(text):
        pieces = _PAGE_RE.split(text)
        # pieces: [pre, p1, text1, p2, text2, ...]
        it = iter(range(1, len(pieces), 2))
        for i in it:
            page, ptxt = int(pieces[i]), pieces[i + 1]
            for j, sub in enumerate(_split_long(ptxt)):
                if sub.strip():
                    chunks.append({"page": page, "text": sub.strip()})
    else:
        # 法规 txt / html：优先按条款切
        if len(_CLAUSE_RE.findall(text)) >= 5:
            pieces = _CLAUSE_RE.split(text)
        else:
            pieces = _split_long(text)
        for p in pieces:
            for sub in _split_long(p):
                if sub.strip():
                    chunks.append({"page": None, "text": sub.strip()})
    # 合并过小块
    merged = []
    for c in chunks:
        if merged and len(merged[-1]["text"]) < MIN_CHUNK and \
           merged[-1]["page"] == c["page"]:
            merged[-1]["text"] += "\n" + c["text"]
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c["id"] = f"{doc_id}#c{i}"
        c["doc_id"] = doc_id
    return merged


# ---------------- 分词与 BM25 ----------------

_TOKEN_RE = re.compile(
    r"[A-Za-z]+[0-9]+|[0-9]+(?:\.[0-9]+)?%?|[A-Za-z]+|[一-鿿]"
)

_QUERY_ALIAS_GROUPS = (
    ("一季度", "第一季度", "q1", "1-3月", "1至3月"),
    ("二季度", "第二季度", "q2", "4-6月", "4至6月"),
    ("三季度", "第三季度", "q3", "7-9月", "7至9月"),
    ("四季度", "第四季度", "q4", "10-12月", "10至12月"),
    ("增长", "增加", "上升", "提高"),
    ("下降", "减少", "下滑", "回落", "降低"),
    ("现金分红", "分红", "股利", "利润分配", "派息"),
)

def tokenize(text):
    """中文字符 bigram + 数字/英文整词，百分数作为整体保留。"""
    text = text.replace("％", "%").replace("，", ",")
    raw = _TOKEN_RE.findall(text)
    toks = []
    i = 0
    while i < len(raw):
        t = raw[i]
        if "一" <= t <= "鿿":
            j = i
            while j < len(raw) and "一" <= raw[j] <= "鿿":
                j += 1
            run = raw[i:j]
            toks.extend(run)  # unigram
            toks.extend(run[k] + run[k + 1] for k in range(len(run) - 1))
            i = j
        else:
            toks.append(t.lower())
            i += 1
    return toks


def query_tokens(text):
    """为查询补充少量通用金融写法，文档索引保持原文不变。

    扩展只依赖当前查询文本：四位年份补两位写法，季度、变动方向和
    分红术语使用固定对称词组。重复 token 不会被额外加权。
    """
    source = str(text or "")
    lowered = source.casefold().replace("—", "-").replace("–", "-")
    tokens = tokenize(source)
    seen = set(tokens)

    def append_alias(value):
        for token in tokenize(value):
            if token not in seen:
                tokens.append(token)
                seen.add(token)

    for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", source):
        append_alias(year[2:])
    for group in _QUERY_ALIAS_GROUPS:
        if any(alias.casefold() in lowered for alias in group):
            for alias in group:
                append_alias(alias)
    return tokens


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = []
        self.df = Counter()
        self.dl = []
        for c in chunks:
            cnt = Counter(tokenize(c["text"]))
            self.tf.append(cnt)
            self.dl.append(sum(cnt.values()))
            for tok in cnt:
                self.df[tok] += 1
        self.N = len(chunks)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0

    def search(self, query, k=5):
        q = query_tokens(query)
        # 去重但保留权重：查询词频
        qcnt = Counter(q)
        scores = [0.0] * self.N
        for tok, qw in qcnt.items():
            df = self.df.get(tok)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            # bigram 权重更高（更有区分度）
            wt = 1.6 if len(tok) == 2 and "一" <= tok[0] <= "鿿" else 1.0
            if tok[-1:] == "%" or tok[:1].isdigit():
                wt = 2.2  # 数字精确匹配权重最高
            for i in range(self.N):
                f = self.tf[i].get(tok)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += qw * wt * idf * f * (self.k1 + 1) / denom
        order = sorted(range(self.N), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], scores[i]) for i in order if scores[i] > 0]


_INDEX_CACHE = {}


def doc_index(doc_id) -> BM25:
    if doc_id not in _INDEX_CACHE:
        _INDEX_CACHE[doc_id] = BM25(chunk_doc(doc_id))
    return _INDEX_CACHE[doc_id]


def search_docs(doc_ids, query, k_per_doc=3):
    """在指定文档集合内检索，返回 [(chunk, score)]，按分数排序。

    跨文档分数按各文档该查询的 top-1 归一，避免独立 IDF
    空间不可比导致的长文档偏置。"""
    hits = []
    for d in doc_ids:
        dh = doc_index(d).search(query, k=k_per_doc)
        if not dh:
            continue
        top = dh[0][1] or 1.0
        hits.extend((c, s / top) for c, s in dh)
    hits.sort(key=lambda x: -x[1])
    return hits
