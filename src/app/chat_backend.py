"""Free-form assistant for the web app, backed by an LLM.

Given the current pause's recommendation context (the ranked candidate actions and
their rationales, the tracked players), answer the coach's question and, when they
ask to see a different play, tell the UI which candidate to display.

Provider is chosen by env var: GROQ_API_KEY (fast, OpenAI-compatible, JSON mode) or
ANTHROPIC_API_KEY (Claude, forced tool call). Keys come from the environment and are
never stored. If neither is set the caller falls back to the offline rule-based
assistant.
"""
from __future__ import annotations

import os

MODEL = "claude-opus-4-8"

_TOOL = {
    "name": "respond",
    "description": "Reply to the coach and optionally switch the play drawn on the frame.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string",
                      "description": "1-3 sentence answer in plain English. No coordinates. "
                                     "Only cite numbers (EPV, %, feet) that appear in the context."},
            "select_action_id": {"type": ["string", "null"],
                                  "description": "id of the candidate to draw, or null to leave it."},
        },
        "required": ["reply"],
    },
}


def _system_prompt(rec: dict) -> str:
    lines = [
        "You are a basketball assistant sitting next to a coach who has paused an NBA possession.",
        "An expected-points (EPV) model has scored every legal action for the ball-handler.",
        "Explain and adjust the recommendation. Be concise and concrete.",
        "Displaying options: whenever the coach asks to SEE or SHOW any option, set "
        "select_action_id to that candidate's id and draw it — ALWAYS honor the request, "
        "even for a low-rated or clearly-suboptimal option (they're exploring). Match loosely: "
        "'the drive', 'pass to #7', 'the screen', 'the worst option', 'the second option' all map "
        "to a candidate below. Only leave select_action_id null if the coach asked a pure question "
        "with no request to change the drawing. State the option's EPV so they see the trade-off.",
        "Rules: never invent numbers — only echo EPV/percent/feet values given below. "
        "Never output coordinates. Refer to players by their # when known, else 'a teammate'/'the defender'.",
        "",
        f"Confidence in the tracking: {int(round(rec.get('confidence', 0) * 100))}%. "
        f"Players tracked: {rec.get('n_players', 'some')}.",
        "Candidate actions, best expected-points first:",
    ]
    for i, a in enumerate(rec.get("actions", [])):
        r = a.get("rationale") or {}
        tag = " (currently shown)" if i == rec.get("_sel", 0) else ""
        lines.append(f"- id={a['id']} · {a['label']}{tag}"
                     + (f" — {r.get('headline')}" if r.get("headline") else ""))
    jerseys = sorted({p["jersey"] for p in rec.get("players", []) if p.get("jersey") is not None})
    if jerseys:
        lines.append("Readable jersey numbers on court: " + ", ".join(f"#{j}" for j in jerseys) + ".")
    return "\n".join(lines)


def answer(message: str, rec: dict) -> dict:
    """Return {"reply": str, "select": action_id|None}. Raises if no provider key.

    Prefers Groq (fast, OpenAI-compatible) if GROQ_API_KEY is set, else Anthropic.
    """
    if os.environ.get("GROQ_API_KEY"):
        return _answer_groq(message, rec)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _answer_anthropic(message, rec)
    raise RuntimeError("set GROQ_API_KEY or ANTHROPIC_API_KEY to enable the assistant")


def _clean_select(sel):
    return None if sel in (None, "", "null", "None", "none") else sel


def _answer_groq(message: str, rec: dict) -> dict:
    import json
    import requests
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    system = _system_prompt(rec) + (
        '\n\nRespond ONLY with a JSON object of the form '
        '{"reply": "<1-3 sentence answer>", "select_action_id": "<candidate id to '
        'display, e.g. a1, or null to leave it>"}.')
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
        json={"model": model, "temperature": 0.3, "max_tokens": 600,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": message}]},
        timeout=60)
    r.raise_for_status()
    d = json.loads(r.json()["choices"][0]["message"]["content"])
    return {"reply": d.get("reply", ""), "select": _clean_select(d.get("select_action_id"))}


def _answer_anthropic(message: str, rec: dict) -> dict:
    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=MODEL, max_tokens=600, system=_system_prompt(rec),
        tools=[_TOOL], tool_choice={"type": "tool", "name": "respond"},
        messages=[{"role": "user", "content": message}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "respond":
            return {"reply": block.input.get("reply", ""),
                    "select": _clean_select(block.input.get("select_action_id"))}
    return {"reply": "(no response)", "select": None}
