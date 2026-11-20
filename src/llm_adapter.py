"""Swappable model adapter: gemini | openai | anthropic | heuristic.

* Pure-stdlib HTTPS (urllib) so no vendor SDK is required; swap provider/model with one flag.
* `openai` speaks the OpenAI chat-completions protocol, so it also drives any OpenAI-compatible endpoint (OpenRouter,
  Together, vLLM, Ollama, a company gateway): set OPENAI_BASE_URL and pass the model id. OPENAI_JSON_MODE=0 turns off
  response_format for servers that do not support it.
* complete_json_cached(): every answer is stored under .llm_cache/<provider_model>/<sha256 of the request>.json, so a
  repeated request (a re-run, a live demo) is answered instantly with no network, and --cache_only replays them.
* Explicit provider + missing key -> MissingAPIKeyError (never a silent fallback to heuristic).
* Rate-limit resilience: HTTP 429 / RESOURCE_EXHAUSTED -> exponential backoff 15s -> 30s -> 60s -> 120s
  (4 retries, or the server's retryDelay if longer). Daily-quota or zero-quota errors fail fast with a
  clear message because waiting cannot fix them.
* Pacing: a fixed gap between consecutive calls (12 s default) keeps free-tier RPM limits happy.
* Every call is metered (prompt / output / thinking tokens, latency, USD at list price) and logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import config


class LLMError(RuntimeError):
    pass


class MissingAPIKeyError(LLMError):
    pass


class RateLimitError(LLMError):
    def __init__(self, msg: str, retry_after: float = 0.0, fatal: bool = False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.fatal = fatal          # daily / zero quota: retrying cannot help


class TransientError(LLMError):
    pass


@dataclass
class LLMResponse:
    text: str
    data: Dict[str, Any]
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_s: float
    provider: str
    model: str
    finish_reason: str = ""
    attempts: int = 1


@dataclass
class UsageMeter:
    calls: int = 0
    failures: int = 0
    rate_limit_retries: int = 0
    transient_retries: int = 0
    backoff_wait_s: float = 0.0
    pacing_wait_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cache_hits: int = 0
    errors: List[str] = field(default_factory=list)
    calls_log: List[Dict[str, Any]] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    return int(len(text) / config.CHARS_PER_TOKEN) + 1


def price_usd(model: str, tin: int, tout: int) -> float:
    """List-price cost; `tout` must already include thinking tokens (billed as output)."""
    pin, pout = config.MODEL_PRICES_PER_M.get(model, (0.0, 0.0))
    return tin / 1e6 * pin + tout / 1e6 * pout


def extract_json(text: str) -> Dict[str, Any]:
    """Tolerant JSON extraction (strips ``` fences, finds the outermost object)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {exc}") from exc
    raise LLMError("model did not return JSON")


def _parse_retry_delay(body: str) -> float:
    """Gemini puts e.g. {"@type": ".../RetryInfo", "retryDelay": "37s"} in 429 bodies."""
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    return float(m.group(1)) if m else 0.0


class LLMAdapter:
    SUPPORTED = ("heuristic", "gemini", "openai", "anthropic")

    def __init__(self, provider: str = "heuristic", model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: float = config.LLM_TIMEOUT_SEC, temperature: float = 0.0,
                 pacing_sec: float = config.LLM_PACING_SEC,
                 backoff_schedule: Sequence[float] = config.RATE_LIMIT_BACKOFF_SEC,
                 transient_retries: int = 2, log: Optional[Callable[[str], None]] = None,
                 cache_only: bool = False):
        provider = (provider or "heuristic").lower().strip()
        if provider not in self.SUPPORTED:
            raise ValueError(f"provider must be one of {self.SUPPORTED}, got {provider!r}")
        config.load_dotenv_if_present()
        self.provider = provider
        self.requested_provider = provider
        self.model = model or config.PROVIDER_DEFAULT_MODELS[provider]
        self.api_key = api_key or (os.environ.get(config.PROVIDER_ENV_KEYS[provider], "")
                                   if provider in config.PROVIDER_ENV_KEYS else "")
        self.fallback_reason = ""          # kept for backward compatibility; never set for explicit providers
        # cache_only: replay saved model answers without a key or network (development preview; new promos are
        # skipped and listed, so a cache-only run never passes for a real one)
        self.cache_only = bool(cache_only) and provider != "heuristic"
        if provider != "heuristic" and not self.api_key and not self.cache_only:
            env = config.PROVIDER_ENV_KEYS[provider]
            raise MissingAPIKeyError(
                f"--provider {provider} was requested but {env} is not set. Add {env}=... to .env (or the "
                f"environment, or pass --api_key). Refusing to fall back to the heuristic scorer silently.")
        self.timeout = timeout
        self.temperature = temperature
        self.pacing_sec = float(pacing_sec)
        self.backoff_schedule = list(backoff_schedule)
        self.transient_retries = transient_retries
        self.log = log or (lambda m: None)
        self.meter = UsageMeter()
        self._lock = threading.Lock()
        self._last_call_end: Optional[float] = None

    @property
    def is_live(self) -> bool:
        return self.provider != "heuristic"

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    # ------------------------------------------------------------------ public
    def cache_path(self, system: str, user: str, max_tokens: int) -> Path:
        key = hashlib.sha256(f"{self.label}\n{max_tokens}\n{system}\n{user}".encode("utf-8")).hexdigest()[:40]
        return config.LLM_CACHE_DIR / re.sub(r"[^A-Za-z0-9._-]+", "_", self.label) / f"{key}.json"

    def complete_json_cached(self, system: str, user: str, max_tokens: int = 2048, tag: str = "",
                             use_cache: bool = True) -> Tuple[Dict[str, Any], bool]:
        """complete_json() behind a content-addressed cache. Returns (data, came_from_cache)."""
        path = self.cache_path(system, user, max_tokens)
        if use_cache and path.exists():
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                self.meter.cache_hits += 1
                return rec["data"], True
            except (OSError, ValueError, KeyError):
                pass
        resp = self.complete_json(system, user, max_tokens, tag=tag)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"label": self.label, "tag": tag, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "data": resp.data}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return resp.data, False

    def complete_json(self, system: str, user: str, max_tokens: int = 2048, tag: str = "") -> LLMResponse:
        if getattr(self, "cache_only", False):
            raise LLMError("cache-only mode: no saved answer for this request")
        if not self.is_live:
            raise LLMError("heuristic provider has no remote model")
        # pacing between consecutive calls (batches)
        if self._last_call_end is not None and self.pacing_sec > 0:
            wait = self.pacing_sec - (time.time() - self._last_call_end)
            if wait > 0:
                self.log(f"    pacing: waiting {wait:.1f}s before {tag or 'next call'}")
                time.sleep(wait)
                self.meter.pacing_wait_s += wait
        rl_attempt = 0
        tr_attempt = 0
        attempts = 0
        while True:
            attempts += 1
            t0 = time.time()
            try:
                text, tin, tout, tthink, finish = getattr(self, f"_call_{self.provider}")(system, user, max_tokens)
                lat = time.time() - t0
                if finish and finish.upper() in ("MAX_TOKENS", "LENGTH", "MAX_OUTPUT_TOKENS"):
                    raise LLMError(f"response truncated (finish_reason={finish}); "
                                   f"{tout} output + {tthink} thinking tokens")
                data = extract_json(text)
                tin = tin or estimate_tokens(system + user)
                tout = tout or estimate_tokens(text)
                cost = price_usd(self.model, tin, tout + tthink)
                with self._lock:
                    self.meter.calls += 1
                    self.meter.input_tokens += tin
                    self.meter.output_tokens += tout
                    self.meter.thinking_tokens += tthink
                    self.meter.latency_s += lat
                    self.meter.cost_usd += cost
                    self.meter.calls_log.append({"tag": tag, "input_tokens": tin, "output_tokens": tout,
                                                 "thinking_tokens": tthink, "latency_s": round(lat, 2),
                                                 "attempts": attempts, "cost_usd": round(cost, 6)})
                self._last_call_end = time.time()
                return LLMResponse(text, data, tin, tout, tthink, lat, self.provider, self.model, finish, attempts)
            except RateLimitError as exc:
                self._last_call_end = time.time()
                if exc.fatal or rl_attempt >= len(self.backoff_schedule):
                    self._fail(exc)
                    raise LLMError(f"{self.label} rate limit not recovered after {rl_attempt} retries: {exc}") from exc
                wait = max(self.backoff_schedule[rl_attempt], exc.retry_after)
                rl_attempt += 1
                self.meter.rate_limit_retries += 1
                self.meter.backoff_wait_s += wait
                self.log(f"    rate-limited (429/RESOURCE_EXHAUSTED) on {tag or 'call'}: retry "
                         f"{rl_attempt}/{len(self.backoff_schedule)} in {wait:.0f}s")
                time.sleep(wait)
            except TransientError as exc:
                self._last_call_end = time.time()
                if tr_attempt >= self.transient_retries:
                    self._fail(exc)
                    raise LLMError(f"{self.label} failed after {tr_attempt} transient retries: {exc}") from exc
                wait = self.backoff_schedule[min(tr_attempt, len(self.backoff_schedule) - 1)] if self.backoff_schedule else 1
                tr_attempt += 1
                self.meter.transient_retries += 1
                self.meter.backoff_wait_s += wait
                self.log(f"    transient error on {tag or 'call'} ({exc}); retry {tr_attempt}/"
                         f"{self.transient_retries} in {wait:.0f}s")
                time.sleep(wait)
            except LLMError as exc:
                self._last_call_end = time.time()
                self._fail(exc)
                raise

    def _fail(self, exc: Exception) -> None:
        with self._lock:
            self.meter.failures += 1
            self.meter.errors.append(repr(exc)[:400])

    # ------------------------------------------------------------------ transport
    def _post(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if e.code == 429 or "RESOURCE_EXHAUSTED" in body or "ResourceExhausted" in body:
                fatal = bool(re.search(r"limit:\s*0\b", body)) or "PerDay" in body or "per day" in body.lower()
                hint = (" This looks like a daily or zero quota for this model on this key's project; waiting "
                        "will not help - enable billing on the project or pick another model.") if fatal else ""
                raise RateLimitError(f"HTTP {e.code}: {body[:300]}{hint}", _parse_retry_delay(body), fatal) from e
            if e.code in (500, 502, 503, 504):
                raise TransientError(f"HTTP {e.code}: {body[:200]}") from e
            raise LLMError(f"HTTP {e.code}: {body[:400]}") from e
        except urllib.error.URLError as e:
            reason = str(e.reason)
            if "Tunnel connection failed" in reason or "403" in reason:
                raise LLMError(f"network blocked: could not reach {url.split('/')[2]} ({reason}). Run from a "
                               "machine/network that allows this API host.") from e
            raise TransientError(f"network error: {reason}") from e
        except (TimeoutError, ConnectionError) as e:
            raise TransientError(f"network error: {e!r}") from e

    def _call_gemini(self, system: str, user: str, max_tokens: int):
        base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
        url = f"{base}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        gen_cfg: Dict[str, Any] = {"temperature": self.temperature, "maxOutputTokens": max_tokens,
                                   "responseMimeType": "application/json"}
        if "2.5" in self.model and config.GEMINI_THINKING_BUDGET is not None:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": config.GEMINI_THINKING_BUDGET}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen_cfg,
        }
        data = self._post(url, payload, {})
        cands = data.get("candidates", [])
        if not cands:
            fb = data.get("promptFeedback", {})
            raise LLMError(f"gemini returned no candidates (promptFeedback={fb})")
        text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", [])
                       if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        return (text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0),
                usage.get("thoughtsTokenCount", 0), cands[0].get("finishReason", ""))

    def _call_openai(self, system: str, user: str, max_tokens: int):
        payload = {
            "model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if os.environ.get("OPENAI_JSON_MODE", "1") != "0":
            payload["response_format"] = {"type": "json_object"}
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        data = self._post(f"{base}/chat/completions", payload, {"Authorization": f"Bearer {self.api_key}"})
        ch = data["choices"][0]
        usage = data.get("usage", {})
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        return (ch["message"]["content"], usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0) - reasoning, reasoning, ch.get("finish_reason", ""))

    def _call_anthropic(self, system: str, user: str, max_tokens: int):
        payload = {
            "model": self.model, "max_tokens": max_tokens, "temperature": self.temperature, "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        data = self._post(f"{base}/v1/messages", payload,
                          {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0), 0, data.get("stop_reason", "")
