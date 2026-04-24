"""
explain.py — Human-readable explanation layer over the rule features.

Given a raw SMS message, produces:
  * `highlights` — character spans for the frontend to color-code
                   (URLs, short-links, phones, money, urgency words, action
                   words, suspicious keywords, leet, embedded emails)
  * `reasons`    — short human-readable bullet points
                   (e.g. "suspicious URL detected", "urgency keyword detected")
  * `rule_flags` — the 9 binary flags from features.rule_features as a dict
  * `heuristic_score` — Jayaprakash weighted score in [0, 1]

This layer is intentionally separate from the ML model: it is purely
deterministic and explains the rule branch of the hybrid feature pipeline.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any

import numpy as np

from .preprocess import URL_RE, PHONE_RE, MONEY_RE
from .features import (
    rule_features,
    heuristic_score,
    SUSPICIOUS,
    _MATH_RE,
    _REPLY_RE,
    _LEET_RE,
    _EMAIL_RE,
)

# ---------------------------------------------------------------------------
# Lexicons split by tone (urgency vs call-to-action) for finer highlighting
# ---------------------------------------------------------------------------

URGENCY_WORDS = {
    "urgent", "urgently", "immediately", "now", "asap", "alert",
    "important", "expire", "expires", "expired", "expiring",
    "suspended", "blocked", "locked", "limited", "hurry",
    "warning", "final", "last", "deadline",
}

ACTION_WORDS = {
    "verify", "confirm", "click", "claim", "update", "activate",
    "reactivate", "approve", "reply", "call", "text", "visit",
    "login", "log-in", "sign-in", "submit", "register",
}

# Common URL shortener domains — short-link detection
SHORT_LINK_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly",
    "is.gd", "soo.gd", "t.ly", "rebrand.ly", "lnkd.in", "fb.me",
    "shorturl.at", "cutt.ly", "rb.gy", "tiny.cc", "shorte.st",
    "bl.ink", "snip.ly", "qr.ae", "v.gd", "y2u.be", "x.co",
}

# Mapping from rule flag index → label and the reason string we surface.
# Order must match `rule_features` in features.py.
_RULE_REASONS = [
    ("url_present",        "suspicious URL detected"),
    ("math_operators",     "math/operator symbols present"),
    ("currency_symbol",    "currency symbol detected"),
    ("phone_number",       "phone number embedded"),
    ("suspicious_keyword", "suspicious keyword detected"),
    ("long_message",       "unusually long message (>150 chars)"),
    ("self_call_to_action","self-answering call-to-action (reply / call / text)"),
    ("leet_speak",         "leet-speak / obfuscated characters"),
    ("embedded_email",     "embedded email address"),
]


# ---------------------------------------------------------------------------
# Highlight extraction
# ---------------------------------------------------------------------------

def _add(spans: List[Dict[str, Any]], type_: str, m: re.Match) -> None:
    spans.append({
        "type":  type_,
        "start": m.start(),
        "end":   m.end(),
        "text":  m.group(0),
    })


def _is_short_link(url: str) -> bool:
    lower = url.lower()
    for dom in SHORT_LINK_DOMAINS:
        if dom in lower:
            return True
    return False


def _extract_word_spans(msg: str, vocab: set, type_: str) -> List[Dict[str, Any]]:
    spans = []
    for m in re.finditer(r"\b\w+\b", msg):
        if m.group(0).lower() in vocab:
            spans.append({
                "type":  type_,
                "start": m.start(),
                "end":   m.end(),
                "text":  m.group(0),
            })
    return spans


def extract_highlights(msg: str) -> List[Dict[str, Any]]:
    """Return character spans annotated with a `type` for frontend coloring.

    Span types
    ----------
    url, short_link, phone, money, urgency, action, keyword, email, leet
    """
    spans: List[Dict[str, Any]] = []

    for m in URL_RE.finditer(msg):
        url = m.group(0)
        spans.append({
            "type":  "short_link" if _is_short_link(url) else "url",
            "start": m.start(),
            "end":   m.end(),
            "text":  url,
        })

    for m in PHONE_RE.finditer(msg):
        _add(spans, "phone", m)

    for m in MONEY_RE.finditer(msg):
        _add(spans, "money", m)

    for m in _EMAIL_RE.finditer(msg):
        _add(spans, "email", m)

    for m in _LEET_RE.finditer(msg):
        _add(spans, "leet", m)

    spans.extend(_extract_word_spans(msg, URGENCY_WORDS, "urgency"))
    spans.extend(_extract_word_spans(msg, ACTION_WORDS,  "action"))

    # Generic suspicious keywords from the Jain & Gupta lexicon — only those
    # not already covered by urgency/action (avoid double-tagging).
    leftover = SUSPICIOUS - URGENCY_WORDS - ACTION_WORDS
    spans.extend(_extract_word_spans(msg, leftover, "keyword"))

    # Resolve overlaps: prefer the longer / higher-priority span.
    spans = _dedupe_spans(spans)
    spans.sort(key=lambda s: s["start"])
    return spans


# Priority used to break ties when two spans overlap. Higher wins.
_TYPE_PRIORITY = {
    "short_link": 9,
    "url":        8,
    "money":      7,
    "phone":      7,
    "email":      6,
    "urgency":    5,
    "action":     4,
    "keyword":    3,
    "leet":       2,
}


def _dedupe_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop spans fully contained in a higher-priority span."""
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    keep: List[Dict[str, Any]] = []
    for s in spans:
        dropped = False
        for k in keep:
            # Full containment
            if s["start"] >= k["start"] and s["end"] <= k["end"]:
                if _TYPE_PRIORITY.get(s["type"], 0) <= _TYPE_PRIORITY.get(k["type"], 0):
                    dropped = True
                    break
        if not dropped:
            keep.append(s)
    return keep


# ---------------------------------------------------------------------------
# Reason synthesis
# ---------------------------------------------------------------------------

def build_reasons(msg: str, highlights: List[Dict[str, Any]]) -> List[str]:
    """Produce a deduplicated list of human-readable reasons.

    Pulled from a mix of: highlight types present in the message and the
    9 rule flags. Order roughly mirrors how a human analyst would describe
    the message.
    """
    types_present = {h["type"] for h in highlights}
    reasons: List[str] = []

    if "short_link" in types_present:
        reasons.append("short-link pattern detected")
    if "url" in types_present:
        reasons.append("suspicious URL detected")
    if "urgency" in types_present:
        reasons.append("urgency keyword detected")
    if "action" in types_present:
        reasons.append("call-to-action keyword (verify / claim / click)")
    if "money" in types_present:
        reasons.append("currency symbol detected")
    if "phone" in types_present:
        reasons.append("phone number embedded")
    if "email" in types_present:
        reasons.append("embedded email address")
    if "leet" in types_present:
        reasons.append("leet-speak / obfuscated characters")
    if "keyword" in types_present and "suspicious keyword detected" not in reasons:
        reasons.append("suspicious keyword detected")

    if len(msg) > 150:
        reasons.append("unusually long message (>150 chars)")

    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def explain(msg: str) -> Dict[str, Any]:
    """Compute the full explanation payload for a single SMS message.

    Returns
    -------
    dict with keys:
        highlights       : list[{type, start, end, text}]
        reasons          : list[str]
        rule_flags       : dict[str, bool]   (9 entries)
        heuristic_score  : float in [0, 1]
    """
    rule_vec  = rule_features(msg)
    score     = heuristic_score(rule_vec)
    highlights = extract_highlights(msg)
    reasons    = build_reasons(msg, highlights)

    rule_flags = {
        name: bool(rule_vec[i])
        for i, (name, _label) in enumerate(_RULE_REASONS)
    }

    return {
        "highlights":      highlights,
        "reasons":         reasons,
        "rule_flags":      rule_flags,
        "heuristic_score": float(score),
    }
