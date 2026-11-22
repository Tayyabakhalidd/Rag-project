"""Guardrails: input checks, output checks and the rate limiter.

Where each guardrail lives:

* Empty or too-long questions -> rejected by the Pydantic schema (422).
* Prompt injection / jailbreak / secret requests in the question ->
  ``check_question`` (400 ``guardrail_blocked``).
* Off-topic questions -> similarity floor + LLM topic check in the QA service.
* Instructions hidden in uploaded documents -> chunks are wrapped in
  ``<document>`` tags and the system prompt says to treat them as data
  (see ``generation.py``); ``neutralize_tags`` stops a chunk closing its tag.
* Invented citations -> ``generation.validate_citations``.
* Leaked system prompt or API keys -> ``check_output``.
* Too many requests -> ``RateLimiter`` (429 with ``Retry-After``).

Pattern matching is fast and explainable but can be bypassed by creative
wording. That is why it is only one layer: the prompt design and the output
checks still apply to questions that slip through.
"""

import math
import re
import threading
import time
import unicodedata
from dataclasses import dataclass

from app.logging_setup import KEY_LIKE_PATTERN

# Zero-width and other invisible characters that can hide trigger words.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))


def normalize_for_matching(text: str) -> str:
    """Lower-case, Unicode-normalised text with punctuation collapsed to spaces.

    "IGNORE   all *previous* instructions!" becomes "ignore all previous instructions".
    Used only for pattern matching; the original question is what gets answered.
    """
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).casefold()
    text = re.sub(r"[^\w]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Each rule: (reason, compiled pattern over normalised text).
#
# The corpus itself is about these attacks, so questions such as "What is
# system prompt leakage?" or "What is a jailbreak?" must still be answered.
# The patterns therefore target commands aimed at the assistant ("ignore your
# instructions", "show me your system prompt"), not mentions of the topic.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "prompt_injection",
        re.compile(
            r"\b(ignore|disregard|forget|override)\s(all\s)?(of\s)?(the\s|your\s|any\s|these\s)?"
            r"(previous|prior|above|earlier|preceding|original|initial|system|existing)\s(\w+\s)?"
            r"(instructions?|rules|prompts?|directions|guidelines|messages?)\b"
        ),
    ),
    (
        "prompt_injection",
        re.compile(r"\b(ignore|disregard|forget|override)\s(all\s)?your\s(\w+\s)?(instructions|rules|guidelines|programming)\b"),
    ),
    (
        "prompt_injection",
        re.compile(
            r"\b(reveal|show|print|repeat|output|display|leak|dump|share|tell me|give me|write out)\s(me\s|us\s)?"
            r"(your|the|this)\s(\w+\s){0,2}(system|hidden|initial|original|developer|secret)\s"
            r"(prompt|instructions?|message)\b"
        ),
    ),
    ("prompt_injection", re.compile(r"\bwhat (is|are|was|were) your (\w+ )?(system prompt|instructions|rules)\b")),
    ("prompt_injection", re.compile(r"\b(repeat|print|output)\s(everything|all|the text|the words)\s(above|before)\b")),
    (
        "jailbreak",
        re.compile(r"\b(developer|dan|god|unfiltered|unrestricted|jailbreak|jailbroken) mode\b|\bdo anything now\b"),
    ),
    (
        "jailbreak",
        re.compile(
            r"\b(you are now|from now on you are|act as|pretend (to be|you are)|roleplay as)\b(\s\w+){0,6}\s"
            r"(dan|unrestricted|uncensored|unfiltered|jailbroken|evil|without (any )?(rules|restrictions|limits|filters))\b"
        ),
    ),
    (
        "jailbreak",
        re.compile(
            r"\b(ignore|bypass|disable|turn off|remove)\s(all\s)?your\s(\w+\s){0,2}"
            r"(safety|guardrails?|filters?|restrictions|limits|ethics|ethical guidelines|content policy)\b"
        ),
    ),
    (
        "secret_request",
        re.compile(
            r"\b(reveal|show|print|give|tell|leak|dump|output|send)\s(me\s|us\s)?(your|the)\s(\w+\s){0,2}"
            r"(api keys?|secret keys?|access tokens?|credentials|passwords?|env file|environment variables)\b"
        ),
    ),
]

# Rules on the raw text, where layout matters: fake role headers and tags.
_RAW_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("prompt_injection", re.compile(r"(?im)^\s*(system|developer|assistant)\s*:")),
    ("prompt_injection", re.compile(r"(?i)<\s*/?\s*(system|instructions?|document)\b[^>]*>")),
]


@dataclass(frozen=True)
class GuardrailVerdict:
    blocked: bool
    reason: str | None = None


def check_question(question: str) -> GuardrailVerdict:
    """Look for prompt-injection, jailbreak and secret-extraction patterns."""
    for reason, pattern in _RAW_RULES:
        if pattern.search(question):
            return GuardrailVerdict(True, reason)
    normalized = normalize_for_matching(question)
    for reason, pattern in _RULES:
        if pattern.search(normalized):
            return GuardrailVerdict(True, reason)
    return GuardrailVerdict(False)


def neutralize_tags(text: str) -> str:
    """Stop document text from opening or closing the prompt's ``<document>`` tags."""
    return re.sub(r"<\s*(/?)\s*document", r"< \1document", text, flags=re.IGNORECASE)


def check_output(texts: list[str | None], secrets: list[str], canary: str, prompt_fingerprints: list[str]) -> str | None:
    """Return a reason if any model output text must be withheld, else ``None``.

    Blocks API keys (configured values and key-like strings), the random canary
    placed in the system prompt, and verbatim sentences of the system prompt.
    """
    for text in texts:
        if not text:
            continue
        if any(secret and secret in text for secret in secrets) or KEY_LIKE_PATTERN.search(text):
            return "secret_leak"
        if canary and canary in text:
            return "system_prompt_leak"
        lowered = text.casefold()
        if any(fingerprint in lowered for fingerprint in prompt_fingerprints):
            return "system_prompt_leak"
    return None


class RateLimiter:
    """In-memory token bucket per client key (for example the client IP).

    Each client has ``per_minute`` tokens that refill continuously. A request
    takes one token; with none left it gets 429 and a ``Retry-After`` value.
    Limitation: state lives in one process, so several workers would each
    allow ``per_minute``.
    """

    MAX_CLIENTS = 10_000  # bound memory: forget the oldest clients beyond this

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.refill_per_second = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last update time)
        self._lock = threading.Lock()

    def acquire(self, key: str) -> int | None:
        """Take a token. Returns ``None`` if allowed, else seconds to wait."""
        now = time.monotonic()
        with self._lock:
            tokens, updated = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - updated) * self.refill_per_second)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                if len(self._buckets) > self.MAX_CLIENTS:
                    self._buckets.pop(next(iter(self._buckets)))
                return None
            self._buckets[key] = (tokens, now)
            return max(1, math.ceil((1.0 - tokens) / self.refill_per_second))
