"""Free-form assistant for the web app, backed by Claude.

Given the current pause's recommendation context (the ranked candidate actions and
their rationales, the tracked players), answer the coach's question and, when they
ask to see a different play, tell the UI which candidate to display. Structured via
a forced tool call so the reply and the action-selection are always well-formed.

The API key comes from ``$ANTHROPIC_API_KEY`` — never stored. If the key or the
SDK is missing the caller falls back to the offline rule-based assistant.
"""
from __future__ import annotations

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
        "Rules: never invent numbers — only echo EPV/percent/feet values given below. "
        "Never output coordinates. Refer to players by their # when known, else 'a teammate'/'the defender'. "
        "If the coach wants to see a specific action, set select_action_id to that candidate's id.",
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
    """Return {"reply": str, "select": action_id|None}. Raises if the SDK/key is absent."""
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
                    "select": block.input.get("select_action_id")}
    return {"reply": "(no response)", "select": None}
