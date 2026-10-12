"""Where a first-run onboarding run lives: one ``onboarding_run`` key in the session's
``sessions.model_config`` (no schema change; a compression rotation carries it, see
``hermes_state_compression._CONVERSATION_MODEL_CONFIG_KEYS``).

``advance`` is the one writer after ``start``: load, ``onboarding_flow.apply``, save, and when the
run ends write the install-level flag in the root config so the next launch does not reopen it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from agent.onboarding_flow import (
    Applied, Event, Facts, Step, applies_to, apply, expire, is_active, new_record,
)

logger = logging.getLogger(__name__)

KEY = "onboarding_run"


def _read(db, session_id: str) -> dict | None:
    record = db.get_session_model_config_value(session_id, KEY) if db is not None and session_id else None
    return record if isinstance(record, dict) else None


def load(db, session_id: str) -> dict | None:
    """The run record, with the quiet-spell fallback applied (and saved) when it is due."""
    record = _read(db, session_id)
    if record is not None and expire(record, now=time.time()) is not record:
        step = _transition(db, session_id, None)
        return step.record if step is not None else None
    return record


def _transition(db, session_id: str, event: Event | None) -> Step | None:
    """Expire, apply *event* and write the result in one database transaction, so Skip setup pressed
    while a card answer is being applied is never overwritten by the older copy. An ended run then
    closes the install flag (a Skip or Done retried after a failed config write repairs it), unless
    another run is still open: the flag belongs to that run, and an old chat must not end it."""
    holder: dict[str, Step] = {}

    def update(current: Any) -> Any:
        holder.pop("step", None)
        if not isinstance(current, dict):
            return current
        now = time.time()
        record = expire(current, now=now)
        holder["step"] = (apply(record, event, Facts.from_stored(record["facts"]), now=now)
                          if event is not None else Step(record, {}))
        return holder["step"].record

    db.update_session_model_config_value(session_id, KEY, update)
    step = holder.get("step")
    if step is not None and not is_active(step.record) and active_session_id(db) is None:
        _close_install_flag()
    return step


def _close_install_flag() -> None:
    from hermes_cli import config as config_mod
    from hermes_cli.onboarding_run import set_run
    from hermes_constants import get_default_hermes_root

    from agent.onboarding import PROFILE_BUILD_FLAG

    section = config_mod.read_user_config_raw(get_default_hermes_root() / "config.yaml").get("onboarding")
    section = section if isinstance(section, dict) else {}
    seen = section.get("seen") if isinstance(section.get("seen"), dict) else {}
    if section.get("run") is not False or not seen.get(PROFILE_BUILD_FLAG):
        set_run(False, mark_profile_offered=True)


def start(db, session_id: str, facts: Facts) -> dict:
    """Write a fresh run record (the row must exist) and mark the install as mid-onboarding, so a
    crash before the end reopens this chat on the next launch."""
    from hermes_cli.onboarding_run import set_run

    record = new_record(facts, now=time.time())
    db.patch_session_model_config(session_id, {KEY: record})
    set_run(True)
    return record


def advance(db, session_id: str, event: Event) -> Step:
    """Apply one event to the stored run; LookupError when this session has no run."""
    step = _transition(db, session_id, event) if db is not None and session_id else None
    if step is None:
        raise LookupError(f"session {session_id} has no onboarding run")
    return step


def active_session_id(db) -> str | None:
    """The newest chat whose run is still in progress. Rotation leaves the active copy on the
    compression parent, so each id is read at its compression tip."""
    tips: set[str] = set()
    for sid in db.session_ids_with_model_config_key(KEY):
        tip = db.resolve_resume_session_id(sid) or sid
        if tip not in tips:
            tips.add(tip)
            if is_active(load(db, tip)):
                return tip
    return None


def observe_tool_result(agent, name: str, args: Mapping[str, Any], result: Any) -> None:
    """Settle a beat from a general tool's result (``set_theme``, ``manage_connections`` display,
    ``gui_tour``). Runs for every tool result; only an onboarding chat (one whose agent carries
    ``setup_beat``) reads its record."""
    if not applies_to(name) or "setup_beat" not in (getattr(agent, "valid_tool_names", None) or ()):
        return
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", "") or ""
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and load(db, session_id) is not None:
            advance(db, session_id, Applied(name, dict(args or {}), parsed))
    except Exception:
        logger.warning("onboarding: %s result did not update the run record", name, exc_info=True)
