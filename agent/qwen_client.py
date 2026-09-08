"""Qwen API client with retries, request audit, and direct token accounting."""

import atexit
import json
import os
import re
import threading
import time
from pathlib import Path

from openai import OpenAI

from .paths import ENV_FILE


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"
_ALLOWED_MODEL_RE = re.compile(
    r"^qwen3\.(?:5|6|7)(?:[-._][a-z0-9][a-z0-9._-]*)?$",
)
_KEY_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_CREDENTIAL_HEADER_RE = re.compile(
    r"(?i)((?:authorization|x-api-key|api[_ -]?key)[\"']?\s*[:=]\s*"
    r"[\"']?)(?:bearer\s+)?([^\s,;}'\"]+)"
)


def is_allowed_model(model: str) -> bool:
    """Return whether ``model`` belongs to the supported Qwen generations."""

    return isinstance(model, str) and bool(_ALLOWED_MODEL_RE.fullmatch(model))


def _load_key() -> str:
    global _active_key
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("DASHSCOPE_API_KEY="):
                key = line.split("=", 1)[1].strip()
                break
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not found (env or .env)")
    _active_key = key
    return key


def _redact_error_message(value) -> str:
    """Remove credentials that an HTTP/proxy exception may echo."""

    message = str(value)
    known_values = {
        os.environ.get("DASHSCOPE_API_KEY", "").strip(),
        _active_key.strip(),
    }
    for known in known_values:
        if known:
            message = message.replace(known, "[REDACTED]")
    message = _KEY_TOKEN_RE.sub("[REDACTED]", message)
    message = _CREDENTIAL_HEADER_RE.sub(r"\1[REDACTED]", message)
    return message


def _token_count(value, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"invalid {label} returned by API: {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid {label} returned by API: {value!r}"
        ) from exc
    if result < 0 or str(value).strip() not in {str(result), f"+{result}"}:
        raise RuntimeError(f"invalid {label} returned by API: {value!r}")
    return result


def _usage_dict(response) -> dict:
    """Read and validate the unmodified usage object from one API response."""

    if response.usage is None:
        raise RuntimeError("successful API response has no usage object")
    usage = response.usage.model_dump()
    if not isinstance(usage, dict):
        raise RuntimeError("successful API response has invalid usage object")
    prompt = _token_count(usage.get("prompt_tokens"), "prompt_tokens")
    completion = _token_count(
        usage.get("completion_tokens"), "completion_tokens"
    )
    total = _token_count(usage.get("total_tokens"), "total_tokens")
    if total != prompt + completion:
        raise RuntimeError(
            "API usage total does not equal prompt plus completion tokens"
        )
    return usage


class TokenLedger:
    """Thread-safe ledger that assigns each raw API usage to one question."""

    def __init__(self):
        self._lock = threading.Lock()
        self.per_qid = {}
        self.calls = []

    def add(self, qid: str, model: str, usage: dict, tag: str = "") -> dict:
        qid = str(qid or "").strip()
        if not qid:
            raise RuntimeError("every Qwen call must name one nonempty qid")
        prompt = _token_count(usage.get("prompt_tokens"), "prompt_tokens")
        completion = _token_count(
            usage.get("completion_tokens"), "completion_tokens"
        )
        total = _token_count(usage.get("total_tokens"), "total_tokens")
        if total != prompt + completion:
            raise RuntimeError(
                "API usage total does not equal prompt plus completion tokens"
            )
        # A JSON round trip makes an immutable, serialisable copy while
        # preserving every field returned by the compatible API.
        raw_usage = json.loads(json.dumps(usage, ensure_ascii=False))
        call = {
            "qid": qid,
            "model": model,
            "tag": tag,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "usage": raw_usage,
            "ts": time.time(),
        }
        with self._lock:
            slot = self.per_qid.setdefault(qid, [0, 0])
            slot[0] += prompt
            slot[1] += completion
            self.calls.append(call)
        return call

    def totals(self):
        with self._lock:
            prompt = sum(value[0] for value in self.per_qid.values())
            completion = sum(value[1] for value in self.per_qid.values())
        return prompt, completion, prompt + completion

    def dump(self, path):
        with self._lock:
            payload = {
                "per_qid": dict(self.per_qid),
                "calls": list(self.calls),
            }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)


LEDGER = TokenLedger()
_client = None
_client_lock = threading.Lock()
_audit_lock = threading.Lock()
_audit_file = None
_active_key = ""


def configure_audit(path, *, append=False):
    """Write every API attempt to a full, key-free JSONL audit log."""

    global _audit_file
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _audit_lock:
        if _audit_file is not None:
            _audit_file.close()
        _audit_file = open(
            target, "a" if append else "w", encoding="utf-8"
        )


def close_audit():
    global _audit_file
    with _audit_lock:
        if _audit_file is not None:
            _audit_file.flush()
            _audit_file.close()
            _audit_file = None


atexit.register(close_audit)


def _write_audit(record):
    with _audit_lock:
        if _audit_file is None:
            return
        _audit_file.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )
        _audit_file.flush()


def client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                api_key=_load_key(), base_url=BASE_URL, timeout=300
            )
    return _client


def chat(
    messages,
    *,
    qid,
    model=DEFAULT_MODEL,
    thinking=False,
    thinking_budget=None,
    max_tokens=4096,
    temperature=None,
    tag="",
    max_retries=5,
):
    """Return ``(content, reasoning, usage)`` and record raw per-qid usage."""

    qid = str(qid or "").strip()
    if not qid:
        raise ValueError("qid is required for every Qwen call")
    if not is_allowed_model(model):
        raise ValueError(
            "model must belong to Qwen3.5, Qwen3.6, or Qwen3.7: "
            f"{model!r}"
        )
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be a positive integer")

    extra = {"enable_thinking": bool(thinking)}
    if thinking and thinking_budget:
        extra["thinking_budget"] = int(thinking_budget)
    if temperature is None:
        temperature = 0.6 if thinking else 0.1
    request = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": extra,
    }
    api_client = client()

    last_error = None
    response = None
    response_started = None
    response_attempt = None
    for attempt in range(1, max_retries + 1):
        started = time.time()
        try:
            response = api_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra,
            )
        except Exception as exc:  # network, timeout, rate limit, or API error
            last_error = exc
            _write_audit(
                {
                    "status": "request_error",
                    "qid": qid,
                    "tag": tag,
                    "model": model,
                    "attempt": attempt,
                    "started_at": started,
                    "finished_at": time.time(),
                    "request": request,
                    "error": {
                        "type": type(exc).__name__,
                        "message": _redact_error_message(exc),
                    },
                }
            )
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))
            continue
        response_started = started
        response_attempt = attempt
        break

    if response is None:
        raise RuntimeError(
            "chat failed after "
            f"{max_retries} attempts: {_redact_error_message(last_error)}"
        )

    # Everything below happens after the API returned a response. Failures in
    # usage extraction, accounting, or response parsing are terminal: retrying
    # here would spend again and obscure the already-returned call.
    usage = None
    usage_recorded = False
    try:
        usage = _usage_dict(response)
        LEDGER.add(qid, model, usage, tag)
        usage_recorded = True
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None) or ""
        content = (message.content or "").strip()
    except Exception as exc:
        _write_audit(
            {
                "status": "postprocess_error",
                "qid": qid,
                "tag": tag,
                "model": model,
                "attempt": response_attempt,
                "started_at": response_started,
                "finished_at": time.time(),
                "request": request,
                "usage": usage,
                "usage_recorded": usage_recorded,
                "error": {
                    "type": type(exc).__name__,
                    "message": _redact_error_message(exc),
                },
            }
        )
        raise RuntimeError(
            "Qwen response was received but could not be recorded and parsed"
        ) from exc

    _write_audit(
        {
            "status": "ok",
            "qid": qid,
            "tag": tag,
            "model": model,
            "attempt": response_attempt,
            "started_at": response_started,
            "finished_at": time.time(),
            "request": request,
            "response": {
                "content": content,
                "reasoning_content": reasoning,
            },
            "usage": usage,
        }
    )
    return content, reasoning, usage


__all__ = [
    "BASE_URL",
    "DEFAULT_MODEL",
    "LEDGER",
    "TokenLedger",
    "chat",
    "client",
    "close_audit",
    "configure_audit",
    "is_allowed_model",
]
