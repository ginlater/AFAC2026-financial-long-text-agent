"""Select source documents with lexical recall followed by Qwen reranking."""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

from . import retrieval
from .paths import PROCESSED_DIR
from .qwen_client import DEFAULT_MODEL, chat


_DOC_BM25 = {}
INDEX_HEAD = 30000
SEL_RE = re.compile(r"\[.*?\]", re.S)

# These are broad layout markers used only to avoid spending prompt space on
# document furniture. No source-specific organization or title is embedded.
BOILERPLATE_RE = re.compile(
    r"免责声明|目录|页码|分析师|研究助理|联系人|电子邮件|邮箱|电话|"
    r"声明|提示|执业证书|^\d+$|^P\d+$|@"
)
WEAK_TITLE_RE = re.compile(r"声明|提示|指引|目录|信息披露")


def _question_text(question: dict) -> str:
    options = question.get("options") or {}
    return ((question.get("question") or "") + " " +
            " ".join(str(value) for value in options.values()))


def _display_title(doc_id: str) -> str:
    """Recover a descriptive title when metadata contains page furniture."""
    metadata = retrieval.docs_meta()[doc_id]
    title = re.sub(r"^标题[:：]", "", metadata.get("title", "")).strip()
    if title and not WEAK_TITLE_RE.search(title):
        return title

    raw = retrieval.doc_path(doc_id).read_text(encoding="utf-8")[:8000]
    candidates = re.findall(r"《([^》]{4,80})》", raw)
    candidates.extend(
        line.strip() for line in raw.splitlines()
        if 6 <= len(line.strip()) <= 80
        and not line.lstrip().startswith("[P")
        and not BOILERPLATE_RE.search(line.strip())
    )
    if candidates:
        best = Counter(candidates).most_common(1)[0][0]
        return best
    return title or doc_id


def _doc_card(doc_id: str) -> str:
    metadata = retrieval.docs_meta()[doc_id]
    fields = [f"{doc_id}: 《{_display_title(doc_id)}》"]
    if metadata.get("column"):
        fields.append(f"栏目:{metadata['column']}")
    if metadata.get("pub_date"):
        fields.append(f"日期:{str(metadata['pub_date'])[:10]}")
    return " ".join(fields)


def domain_doc_index(domain: str):
    """Build and cache a document-level lexical index for one domain."""
    if domain not in _DOC_BM25:
        chunks = []
        for doc_id, metadata in retrieval.docs_meta().items():
            if metadata.get("domain") != domain:
                continue
            text = retrieval.doc_path(doc_id).read_text(encoding="utf-8")
            index_text = (metadata.get("title", "") + "\n") * 3 + text[:INDEX_HEAD]
            chunks.append({
                "id": doc_id,
                "doc_id": doc_id,
                "page": None,
                "text": index_text,
            })
        _DOC_BM25[domain] = retrieval.BM25(chunks)
    return _DOC_BM25[domain]


def coarse_candidates(question: dict, k: int = 18) -> list[str]:
    """Return high-recall candidates using corpus size and lexical similarity."""
    domain = question["domain"]
    all_ids = [doc_id for doc_id, metadata in retrieval.docs_meta().items()
               if metadata.get("domain") == domain]
    if len(all_ids) <= max(12, 2 * k):
        return all_ids

    index = domain_doc_index(domain)
    candidates = []
    seen = set()

    def add_results(query: str, limit: int) -> None:
        for chunk, _score in index.search(query, k=limit):
            doc_id = chunk["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                candidates.append(doc_id)

    add_results(_question_text(question), k)
    add_results(question.get("question", ""), max(2, k // 3))
    for option in (question.get("options") or {}).values():
        add_results(str(option), max(1, k // 6))
    return candidates or all_ids


def _content_head(doc_id: str, limit: int = 120) -> str:
    """Return the first informative source lines within a character limit."""
    raw = retrieval.doc_path(doc_id).read_text(encoding="utf-8")[:3000]
    lines = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if (len(line) < 6 or line.startswith("[P") or
                BOILERPLATE_RE.search(line)):
            continue
        lines.append(line)
        if sum(len(value) for value in lines) >= limit:
            break
    return re.sub(r"\s+", " ", " ".join(lines))[:limit]


def _parse_selected_ids(content: str, candidates: list[str],
                        max_docs: int) -> list[str]:
    match = SEL_RE.search(content or "")
    if not match:
        return []
    try:
        values = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(values, list):
        return []

    candidate_set = set(candidates)
    selected = []
    for value in values:
        value = str(value)
        if value in candidate_set:
            selected.append(value)
            continue
        suffix_matches = [doc_id for doc_id in candidates if doc_id.endswith(value)]
        if len(suffix_matches) == 1:
            selected.append(suffix_matches[0])
    return list(dict.fromkeys(selected))[:max_docs]


_MULTI_SOURCE_RE = re.compile(
    r"(?:两|三|四|五|多|若干)(?:家|份|款|类|个|项)[^\n。？?]{0,24}"
    r"(?:主体|产品|文档|文件|报告|机构|合同|规定)|"
    r"(?:分别|各自|逐一|比较|对比)[^\n。？?]{0,30}"
    r"(?:主体|产品|文档|文件|报告|机构|合同|规定)"
)


def _requires_multiple_sources(question: dict) -> bool:
    """Detect explicit multi-source wording in the current input."""
    return bool(_MULTI_SOURCE_RE.search(_question_text(question)))


_RELATED_SUFFIX_RE = re.compile(
    r"(?:[_-](?:att(?:achment)?|appendix|annex|附件)[_-]?\d*)$", re.I
)


def _source_family(doc_id: str, metadata: dict) -> str:
    """Derive a source-family key from portable metadata or the document id."""
    source = str(metadata.get("src") or "")
    stem = pathlib.PurePosixPath(source).stem if source else doc_id
    return _RELATED_SUFFIX_RE.sub("", stem)


def _append_related_sources(selected: list[str], candidates: list[str],
                            metadata: dict, max_docs: int) -> None:
    """Add structurally related main/attachment files while capacity remains."""
    candidate_set = set(candidates)
    families = {_source_family(doc_id, metadata[doc_id]) for doc_id in selected}
    for doc_id in candidates:
        if len(selected) >= max_docs:
            break
        if doc_id in selected or doc_id not in candidate_set:
            continue
        if _source_family(doc_id, metadata[doc_id]) in families:
            selected.append(doc_id)


def _append_identity_matches(question: dict, selected: list[str],
                             candidates: list[str], max_docs: int) -> None:
    """Add policies whose source-derived aliases are named in the query."""
    if question.get("domain") != "insurance" or len(selected) >= max_docs:
        return
    identity_path = PROCESSED_DIR / "insurance_titles.json"
    if not identity_path.exists():
        return
    identities = json.loads(identity_path.read_text(encoding="utf-8"))
    query = _question_text(question)
    candidate_set = set(candidates)
    for doc_id, identity in identities.items():
        if len(selected) >= max_docs:
            break
        if doc_id in selected or doc_id not in candidate_set:
            continue
        names = [identity.get("product", "")]
        names.extend(identity.get("alias", identity.get("aliases", [])) or [])
        if any(len(name) >= 2 and name in query for name in names):
            selected.append(doc_id)


def _finalize_picks(question: dict, selected: list[str],
                    candidates: list[str], max_docs: int) -> list[str]:
    """Apply source-derived lexical coverage fallbacks."""
    selected = list(dict.fromkeys(selected))[:max_docs]
    if not selected and candidates:
        selected.append(candidates[0])

    metadata = retrieval.docs_meta()
    _append_related_sources(selected, candidates, metadata, max_docs)
    _append_identity_matches(question, selected, candidates, max_docs)

    if len(selected) < max_docs and len(candidates) > 1:
        index = domain_doc_index(question["domain"])
        top = index.search(_question_text(question), k=1)
        if top:
            doc_id = top[0][0]["doc_id"]
            if doc_id not in selected:
                selected.append(doc_id)

    # Option-level retrieval broadens recall for multi-claim questions without
    # depending on a particular domain or option position.
    index = domain_doc_index(question["domain"])
    additions = 0
    for option in (question.get("options") or {}).values():
        if additions >= 2 or len(selected) >= max_docs:
            break
        top = index.search(str(option), k=1)
        if top:
            doc_id = top[0][0]["doc_id"]
            if doc_id not in selected:
                selected.append(doc_id)
                additions += 1

    if len(selected) == 1 and _requires_multiple_sources(question):
        runner_up = next((doc_id for doc_id in candidates
                          if doc_id not in selected), None)
        if runner_up is not None and len(selected) < max_docs:
            selected.append(runner_up)
    return selected[:max_docs]


def select_docs(question: dict, qid: str | None = None,
                model: str = DEFAULT_MODEL, k_coarse: int = 12,
                max_docs: int = 4) -> list[str]:
    """Return document ids needed to answer one question."""
    question_id = str(question.get("qid") or "").strip()
    if not question_id:
        raise ValueError("document selection requires the question qid")
    if qid is not None and str(qid).strip() != question_id:
        raise ValueError("document-selection qid differs from question qid")
    candidates = coarse_candidates(question, k=k_coarse)
    if not candidates:
        return []
    if len(candidates) <= 2:
        return candidates

    head_limit = 130
    metadata = retrieval.docs_meta()
    cards = []
    for doc_id in candidates:
        source_description = metadata[doc_id].get("source_description")
        head = (source_description or _content_head(doc_id, head_limit))[:head_limit]
        cards.append(f"[{doc_id}] {_doc_card(doc_id)} | {head}")

    options = "\n".join(
        f"{letter}. {value}"
        for letter, value in (question.get("options") or {}).items()
    )
    example = json.dumps(candidates[:min(2, len(candidates))], ensure_ascii=False)
    prompt = (
        "问题:\n" + question.get("question", "") + "\n选项:\n" + options +
        "\n\n候选文档（方括号内为文档ID）:\n" + "\n".join(cards) +
        f"\n\n请选择回答问题所需的文档，最多{max_docs}份。涉及多个独立主体或"
        "需要比较时覆盖每个相关来源。只输出文档ID的JSON数组，文档ID必须与"
        f"候选列表完全一致，例如 {example}。"
    )
    content, _reasoning, _usage = chat(
        [{"role": "user", "content": prompt}],
        qid=question_id,
        model=model,
        thinking=False,
        max_tokens=220,
        tag="docsel",
    )
    selected = _parse_selected_ids(content, candidates, max_docs)
    return _finalize_picks(question, selected, candidates, max_docs)
