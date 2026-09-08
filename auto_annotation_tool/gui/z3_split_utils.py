#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split allocation helpers shared by Z3 dataset export UI."""

import random


def allocate_split_counts(total_entries: int, ratios: dict[str, float] | None = None) -> dict[str, int]:
    ordered_ratios = list((ratios or {}).items())
    if not ordered_ratios:
        ordered_ratios = [("train", 1.0), ("val", 0.0), ("test", 0.0)]

    total_entries = max(0, int(total_entries or 0))
    normalized_ratios: list[tuple[str, float]] = []
    ratio_sum = 0.0
    for split_name, raw_ratio in ordered_ratios:
        ratio_value = max(0.0, float(raw_ratio or 0.0))
        normalized_ratios.append((str(split_name), ratio_value))
        ratio_sum += ratio_value

    if ratio_sum <= 0.0:
        normalized_ratios = [
            (name, 1.0 if idx == 0 else 0.0)
            for idx, (name, _value) in enumerate(normalized_ratios)
        ]
        ratio_sum = 1.0

    counts = {name: 0 for name, _value in normalized_ratios}
    if total_entries <= 0:
        return counts

    weighted_rows: list[tuple[str, int, float, float]] = []
    assigned = 0
    for idx, (split_name, ratio_value) in enumerate(normalized_ratios):
        raw_target = float(total_entries) * (ratio_value / ratio_sum)
        base_count = int(raw_target)
        fractional = raw_target - float(base_count)
        counts[split_name] = base_count
        assigned += base_count
        weighted_rows.append((split_name, idx, fractional, ratio_value))

    remaining = max(0, total_entries - assigned)
    if remaining > 0:
        weighted_rows.sort(
            key=lambda item: (
                item[2],
                item[3],
                -item[1],
            ),
            reverse=True,
        )
        row_count = len(weighted_rows)
        for offset in range(remaining):
            split_name = weighted_rows[offset % row_count][0]
            counts[split_name] = int(counts.get(split_name, 0) or 0) + 1

    return {key: max(0, int(value or 0)) for key, value in counts.items()}


def build_split_entries(
    entries: list,
    ratios: dict[str, float],
    shuffle_seed: int = 42,
) -> tuple[dict[str, list], dict[str, int]]:
    ratios = dict(ratios or {})
    split_names = list(ratios.keys()) or ["train", "val", "test"]
    working_entries = list(entries or [])
    random.Random(int(shuffle_seed)).shuffle(working_entries)

    counts = allocate_split_counts(len(working_entries), ratios)
    split_entries = {name: [] for name in split_names}

    start_idx = 0
    for split_name in split_names:
        take_count = max(0, int(counts.get(split_name, 0) or 0))
        split_entries[split_name] = working_entries[start_idx:start_idx + take_count]
        start_idx += take_count

    return split_entries, {
        name: int(len(split_entries.get(name, [])) or 0)
        for name in split_names
    }
