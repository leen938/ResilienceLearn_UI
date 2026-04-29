"""
Rule-based emotional support chat (no LLM, not therapy).

Safe defaults: validate feelings, suggest coping and study-life balance, escalate on crisis keywords.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


def _pick_variant(text: str, n: int) -> int:
    """Stable, process-independent index in ``0..n-1`` for template rotation."""
    if not n:
        return 0
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


DISCLAIMER = (
    "I am not a therapist or doctor and cannot diagnose or treat anything. "
    "If you are in immediate danger, contact local emergency services or someone you trust right away."
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _context_snippet(ctx: dict[str, Any] | None) -> str:
    if not ctx:
        return ""
    parts = []
    if ctx.get("status_label"):
        parts.append(f"Your last check-in model status was **{ctx['status_label']}** (informational only).")
    if ctx.get("uncertainty_message"):
        parts.append(str(ctx["uncertainty_message"]))
    if ctx.get("top_factors") and isinstance(ctx["top_factors"], list):
        labels = []
        for tf in ctx["top_factors"][:2]:
            if isinstance(tf, dict) and tf.get("label"):
                labels.append(str(tf["label"]))
        if labels:
            parts.append("Themes that showed up in that snapshot: " + ", ".join(labels) + ".")
    return " ".join(parts)


def _crisis(text: str) -> bool:
    crisis = (
        r"\b(kill myself|suicid|end my life|want to die|hurt myself|self[- ]harm|"
        r"no reason to live)\b"
    )
    return bool(re.search(crisis, text))


# Substrings after normalization (lowercase, whitespace collapsed).
_DISTRESS_MARKERS = (
    "not feeling well",
    "dont feel well",
    "don't feel well",
    "not feel well",
    "unwell",
    "feel sick",
    "feeling sick",
    "feel awful",
    "feeling awful",
    "feel horrible",
    "feeling horrible",
    "feel terrible",
    "feeling terrible",
    "feel bad",
    "feeling bad",
    "feel rough",
    "feeling rough",
    "feel rubbish",
    "so bad",
    "really bad",
    "very bad",
    "how bad",
    "cant express",
    "can't express",
    "cannot express",
    "hard to express",
    "at a loss for words",
    "dont know how to explain",
    "don't know how to explain",
    "feeling low",
    "feel low",
    "feel down",
    "feeling down",
    "im down",
    "i'm down",
    "emotionally drained",
    "feel empty",
    "feeling empty",
    "feel numb",
    "feeling numb",
    "numb inside",
    "crying",
    "been crying",
    "cant stop crying",
    "can't stop crying",
    "so sad",
    "very sad",
    "im sad",
    "i'm sad",
    "hopeless",
    "worthless",
    "worth nothing",
    "feel useless",
    "feeling useless",
    "pathetic",
    "miserable",
    "in a dark place",
    "hurting",
    "hurts so much",
    "heart hurts",
    "not okay",
    "not ok",
    "not alright",
    "im not okay",
    "i'm not okay",
    "struggling",
    "struggle right now",
    "depressed",
    "depression",
    "want to disappear",
)

_EMOTIONAL_DISTRESS_BODIES = (
    (
        "When your mind or body says you are **not okay**, that signal is real — even if the reasons are messy or hard to name. "
        "You do not have to minimize it to deserve kindness, and heavy feelings can sit next to small steps forward."
    ),
    (
        "What you described sounds **painful**, and it makes sense that words might feel too small for it. "
        "Naming even a corner of the feeling can still count; you are allowed to move slowly."
    ),
    (
        "Sometimes \"feeling bad\" is your whole system asking for rest, safety, or connection — not a report card on your character. "
        "You deserve gentleness while you figure out the next practical step."
    ),
)

_GENERAL_OPENERS = (
    (
        "Whatever you are holding today, noticing it and putting it into words already takes courage. "
        "There is no need to be polished here."
    ),
    (
        "It sounds like things feel heavy right now, and you are allowed to say that without earning permission first. "
        "You are not \"too much\" for being honest."
    ),
    (
        "Sometimes the hardest part is describing the feeling — if words are thin, your experience can still be valid. "
        "We can go gently, one piece at a time."
    ),
)


def generate_support_reply(
    message: str,
    context: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = message.strip()
    if not raw:
        return {
            "reply": "I'm here. In a few words, how has your day felt so far?",
            "mood_detected": None,
            "supportive_actions": ["Take one slow breath", "Name one small thing that went okay today"],
            "check_in_prompt": "What is taking up the most mental space right now — study, home, or something else?",
            "safety_note": DISCLAIMER,
        }

    t = _normalize(raw)
    ctx_line = _context_snippet(context)

    if _crisis(t):
        return {
            "reply": (
                "I'm really glad you typed this here. What you're describing can be serious, and you deserve real-time help from a person, not an app.\n\n"
                "Please reach out now to **local emergency services**, a **crisis hotline** in your area, or someone you trust who can stay with you. "
                "You are not a burden for asking for help."
            ),
            "mood_detected": "crisis_language",
            "supportive_actions": [
                "If you can, move to a safer place with another person nearby",
                "Contact a crisis line or emergency services for your country",
                "Tell someone you trust what you are feeling",
            ],
            "check_in_prompt": None,
            "safety_note": DISCLAIMER,
        }

    mood: str | None = None
    lead = "Thank you for sharing that. "
    body = ""

    if any(w in t for w in ("overwhelm", "too much", "can't cope", "breaking down")):
        mood = "overwhelmed"
        body = (
            "It makes sense that everything can feel stacked on top of you, especially when studies and life outside class do not pause for each other. "
            "You are not failing for finding this hard."
        )
    elif any(w in t for w in ("lonely", "alone", "no one understands")):
        mood = "lonely"
        body = (
            "Feeling alone with stress is heavy, and it does not mean you are unlovable or behind. "
            "Connection can start very small — one message, one person, even a quiet study partner."
        )
    elif any(w in t for w in ("can't sleep", "insomnia", "exhausted", "tired")):
        mood = "fatigue_sleep"
        body = (
            "Sleep and stress often twist together. Even one tiny wind-down step (same bedtime, dim lights, less scrolling) can be a kindness to future-you, not a lecture."
        )
    elif any(w in t for w in ("anxious", "anxiety", "panic", "heart racing")):
        mood = "anxious"
        body = (
            "Anxiety can show up in your body faster than words. Grounding can help: notice **5 things you see**, **4 you can touch**, **3 you hear**, **2 you smell**, **1 you taste** — no rush."
        )
    elif any(w in t for w in ("exam", "fail", "grades", "gpa", "behind")):
        mood = "academic_pressure"
        body = (
            "Academic pressure can feel like a verdict on who you are. It is really about **what you are carrying** this term — context, access to power and internet, and stress — not your worth as a person."
        )
    elif any(w in t for w in ("money", "financial", "broke", "can't afford")):
        mood = "financial_stress"
        body = (
            "Money stress is not a personal moral failure; it is structural for many students. "
            "If you can, explore **campus aid**, student groups, or NGOs — asking is a practical step, not weakness."
        )
    elif any(w in t for w in _DISTRESS_MARKERS):
        mood = "emotional_distress"
        matched = next(w for w in _DISTRESS_MARKERS if w in t)
        variant_key = f"{raw}\0{matched}"
        body = _EMOTIONAL_DISTRESS_BODIES[
            _pick_variant(variant_key, len(_EMOTIONAL_DISTRESS_BODIES))
        ]
    else:
        mood = "general"
        # Same mood bucket gets variety so short check-ins do not all read identically.
        body = _GENERAL_OPENERS[_pick_variant(raw, len(_GENERAL_OPENERS))]

    acts = [
        "Try **5–10 minutes** of the smallest possible study step (open slides, one problem, one paragraph).",
        "Drink **water** and step outside or stretch once if you can.",
        "Text or sit with **one trusted person**, even without a full explanation.",
    ]
    if mood == "fatigue_sleep":
        acts[0] = "Pick a **wind-down time** 30 minutes before bed; screens down if that feels doable."
    if mood == "anxious":
        acts[0] = "Try **box breathing**: inhale 4, hold 4, exhale 4, hold 4 — two rounds only."
    if mood == "emotional_distress":
        acts[0] = "If it feels okay, place a **hand on your chest** and breathe a little slower for three cycles — no performance."
        acts[1] = "Pick **one** tiny comfort: water, washing your face, fresh air, or lying down for ten minutes."
    if mood == "lonely":
        acts[1] = "Send one low-pressure message — even a thumbs-up emoji can reopen a thread when shame is loud."

    prompts_by_mood = {
        "general": "If you have one sentence in you: what colour would this week be — grey, red, foggy, something else?",
        "emotional_distress": "What needs the softest response right now — your body, your mind, or something practical like food or rest?",
        "overwhelmed": "What would feel **slightly** easier tonight: rest, one study sprint, or talking to someone?",
        "lonely": "Who is one person you could sit near quietly, even without a big conversation?",
        "fatigue_sleep": "What is one wind-down step you could try tonight for ten minutes only?",
        "anxious": "Name one place that feels **slightly** safer than here (room, bench, voice note) — no fixing, just noticing.",
        "academic_pressure": "If grades were only data — not your whole story — what would you want your week to prove to *you*?",
        "financial_stress": "What is one practical question you could ask a trusted office or student group without judging yourself for needing it?",
    }
    prompt = prompts_by_mood.get(
        mood,
        "What would feel **slightly** easier tonight: rest, one study sprint, or talking to someone?",
    )
    closing = f"\n\n{DISCLAIMER}"

    personalization = ""
    if ctx_line:
        personalization = f"\n\nIf it helps, {ctx_line} This chat is still just support — not a medical or academic verdict."

    return {
        "reply": lead + body + personalization + closing,
        "mood_detected": mood,
        "supportive_actions": acts,
        "check_in_prompt": prompt,
        "safety_note": DISCLAIMER,
    }
