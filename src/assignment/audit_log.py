"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.

The log stores what the user actually received (after the output guardrail),
so a leaked secret that was redacted in the reply is never written back to
disk through the audit trail.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_CHARS = 500


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    return str(REPO_ROOT / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._open_inputs: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id (generated if absent)."""
        request_id = request_id or uuid.uuid4().hex[:12]
        self._open[request_id] = time.perf_counter()
        self._open_inputs[request_id] = {
            "user_id": user_id,
            "input": (text or "")[:PREVIEW_CHARS],
            "received_at": utc_now_iso(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Store output, layer decision, latency; append to self.logs."""
        started = self._open.pop(request_id, None) if request_id else None
        opened = self._open_inputs.pop(request_id, {}) if request_id else {}
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 1) if started is not None else None
        )
        entry = {
            "request_id": request_id,
            "user_id": user_id,
            "received_at": opened.get("received_at"),
            "responded_at": utc_now_iso(),
            "input": opened.get("input"),
            "output": (text or "")[:PREVIEW_CHARS],
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | Path | None = None) -> Path:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default.

        A relative ``filepath`` also resolves from the repo root, never the cwd.
        """
        path = Path(filepath or default_audit_log_path())
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
