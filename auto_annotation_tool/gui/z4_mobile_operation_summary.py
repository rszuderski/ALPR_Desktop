#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB007: read-only, operation-centric presentation of mobile export events.

The append-only JSONL log remains the sole evidence. This module merely
produces UI rows; it does not rewrite records, infer conversion from files,
or certify any Android execution/model precision.
"""
from __future__ import annotations

from collections import defaultdict


TERMINAL_OUTCOMES = frozenset({"SUCCEEDED", "FAILED", "UNCONFIRMED"})


def summarize_operations(events: list[dict]) -> list[dict]:
    """Collapse STARTED/final events into one operation row for browsing.

    Records without a usable operation ID remain separate, rather than being
    incorrectly grouped by filename/date. The raw records are attached to
    each aggregate for the detailed audit panel. Conflicting terminal states
    are never silently reported as successes.
    """
    if not isinstance(events, list):
        raise TypeError("Oczekiwano listy zdarzeń")
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for index, original in enumerate(events):
        if not isinstance(original, dict):
            continue
        event = dict(original)
        operation = str(event.get("operation") or "").strip()
        operation_id = str(event.get("operation_id") or "").strip()
        # A missing ID must never collapse unrelated historical events.
        key = (operation, operation_id) if operation and operation_id else ("_unique_event", str(index))
        groups[key].append((index, event))

    result = []
    for records in groups.values():
        # UTC timestamps are recorded only to whole seconds. Within the same
        # second the append-only JSONL order is authoritative: a subsequent
        # SUCCEEDED/FAILED must never be sorted before its STARTED event.
        # Across different timestamps we retain the existing chronological
        # presentation for historical/reordered input.
        ordered = [entry for _, entry in sorted(
            records, key=lambda p: (str(p[1].get("observed_utc") or ""), p[0])
        )]
        latest = ordered[-1]
        outcomes = [str(e.get("outcome") or "").upper() for e in ordered]
        terminal = {value for value in outcomes if value in TERMINAL_OUTCOMES}
        succeeded_sha = {str(e.get("artifact_sha256") or "").lower()
                         for e in ordered if e.get("outcome") == "SUCCEEDED"}
        # Multiple terminal verdicts, divergent successful artifact digests,
        # or a new STARTED after a terminal status are ambiguous; fail closed.
        inconsistent = (len(terminal) > 1 or len(succeeded_sha) > 1 or
                        (terminal and outcomes[-1] == "STARTED"))
        if inconsistent:
            outcome = "INCONSISTENT_HISTORY"
        elif outcomes[-1] == "STARTED":
            outcome = "STARTED_NO_RESULT"
        else:
            outcome = outcomes[-1] or "UNKNOWN"

        display = dict(latest)
        if not display.get("artifact_path"):
            display["artifact_path"] = next(
                (str(e["artifact_path"]) for e in reversed(ordered) if e.get("artifact_path")), ""
            )
        # A completed artifact SHA is evidence only if this group's latest
        # status is the successful one; do not inherit an old hash after error.
        if outcome != "SUCCEEDED":
            display["artifact_sha256"] = None
        display.update({
            "outcome": outcome,
            "_operation_summary": True,
            "_event_count": len(ordered),
            "_source_index": min(index for index, _ in records),
            "_raw_events": ordered,
            "_authority": "RECORDED_EVENTS_ONLY_NOT_GRAPH_OR_ANDROID_VERIFICATION",
        })
        result.append(display)

    return sorted(
        result, key=lambda event: (
            str(event.get("observed_utc") or ""),
            str(event.get("operation_id") or ""),
            str(event.get("event_id") or ""),
        ), reverse=True
    )
