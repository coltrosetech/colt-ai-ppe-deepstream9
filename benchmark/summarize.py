#!/usr/bin/env python3
"""Summarize DeepStream PERF logs and per-run nvidia-smi CSV samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PERF_PAIR = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)")
PERF_HEADER_STREAM = re.compile(r"FPS\s+(\d+)\s*\(Avg\)")
GPU_SAMPLE_INTERVAL_SECONDS = 1.0
INITIAL_IDLE_MAX_UTILIZATION_PERCENT = 10.0
INITIAL_BUSY_MIN_UTILIZATION_PERCENT = 20.0


def rounded(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean": rounded(statistics.fmean(values)) if values else None,
        "p05": rounded(percentile(values, 0.05)),
        "p50": rounded(percentile(values, 0.50)),
        "p95": rounded(percentile(values, 0.95)),
        "min": rounded(min(values)) if values else None,
        "max": rounded(max(values)) if values else None,
    }


def parse_perf(
    path: Path,
    expected_streams: int,
    duration_seconds: int,
    perf_interval_seconds: float,
    *,
    active_row_start: int = 0,
    active_row_end: int | None = None,
) -> dict[str, object]:
    if active_row_start < 0:
        raise ValueError("active_row_start must be non-negative")
    if active_row_end is not None and active_row_end < active_row_start:
        raise ValueError("active_row_end must be >= active_row_start")
    requested_intervals = math.ceil(duration_seconds / perf_interval_seconds)
    if not path.exists():
        return {
            "status": "missing_log",
            "log": str(path),
            "requested_perf_intervals": requested_intervals,
            "perf_intervals_raw_active": 0,
            "perf_intervals_analyzed": 0,
        }

    rows: list[tuple[list[float], list[float]]] = []
    header_streams: int | None = None
    header_stream_ids: list[int] | None = None
    malformed_perf_lines = 0

    for line in path.read_text(errors="replace").splitlines():
        if "**PERF:" not in line:
            continue

        header_ids = [int(value) for value in PERF_HEADER_STREAM.findall(line)]
        if header_ids:
            header_stream_ids = header_ids
            header_streams = max(header_ids) + 1
            continue

        pairs = PERF_PAIR.findall(line.partition("**PERF:")[2])
        if not pairs:
            malformed_perf_lines += 1
            continue
        if len(pairs) != expected_streams:
            malformed_perf_lines += 1
            continue

        current = [float(pair[0]) for pair in pairs]
        running_average = [float(pair[1]) for pair in pairs]
        rows.append((current, running_average))

    zero_rows = sum(1 for current, _ in rows if not any(current))
    raw_active_rows = [(current, average) for current, average in rows if any(current)]
    bounded_active_rows = raw_active_rows[active_row_start:active_row_end]
    analysis_rows = (
        bounded_active_rows
        if active_row_end is not None
        else bounded_active_rows[:requested_intervals]
    )
    aggregate_current = [sum(current) for current, _ in analysis_rows]
    flat_current = [value for current, _ in analysis_rows for value in current]

    inactive_stream_ids = [
        stream_id
        for stream_id in range(expected_streams)
        if not any(current[stream_id] > 0 for current, _ in analysis_rows)
    ]
    nonpositive_stream_sample_counts = {
        str(stream_id): sum(
            1 for current, _ in analysis_rows if current[stream_id] <= 0
        )
        for stream_id in range(expected_streams)
    }
    streams_with_nonpositive_fps = [
        int(stream_id)
        for stream_id, count in nonpositive_stream_sample_counts.items()
        if count
    ]

    per_stream = []
    for stream_id in range(expected_streams):
        current_values = [current[stream_id] for current, _ in analysis_rows]
        final_running_average = analysis_rows[-1][1][stream_id] if analysis_rows else None
        per_stream.append(
            {
                "stream_id": stream_id,
                "current_fps": stats(current_values),
                "final_deepstream_running_average_fps": rounded(final_running_average),
            }
        )

    if not bounded_active_rows:
        status = "no_active_perf_samples"
    elif len(bounded_active_rows) < requested_intervals:
        status = "insufficient_perf_window"
    elif inactive_stream_ids:
        status = "inactive_streams"
    elif streams_with_nonpositive_fps:
        status = "nonpositive_stream_samples"
    else:
        status = "ok"
    if header_stream_ids is not None and header_stream_ids != list(range(expected_streams)):
        status = "stream_count_mismatch"

    return {
        "status": status,
        "log": str(path),
        "expected_streams": expected_streams,
        "header_streams": header_streams,
        "header_stream_ids": header_stream_ids,
        "requested_duration_seconds": duration_seconds,
        "perf_interval_seconds": perf_interval_seconds,
        "requested_perf_intervals": requested_intervals,
        "perf_rows_total": len(rows),
        "perf_intervals_raw_active": len(raw_active_rows),
        "measurement_active_row_start": active_row_start,
        "measurement_active_row_end": active_row_end,
        "perf_intervals_in_measurement_bounds": len(bounded_active_rows),
        "perf_intervals_analyzed": len(analysis_rows),
        "perf_intervals_discarded_after_requested_window": max(
            0, len(bounded_active_rows) - len(analysis_rows)
        ),
        "analyzed_window_coverage_seconds": min(
            duration_seconds, len(analysis_rows) * perf_interval_seconds
        ),
        "discarded_all_zero_intervals": zero_rows,
        "malformed_perf_lines": malformed_perf_lines,
        "active_streams": expected_streams - len(inactive_stream_ids),
        "inactive_stream_ids": inactive_stream_ids,
        "streams_with_nonpositive_fps": streams_with_nonpositive_fps,
        "nonpositive_stream_sample_counts": nonpositive_stream_sample_counts,
        # `current_fps` is the value before parentheses on a DeepStream PERF
        # line. The value in parentheses is DeepStream's cumulative average.
        "per_stream_current_fps_analyzed_samples": stats(flat_current),
        "aggregate_current_fps": stats(aggregate_current),
        "final_aggregate_deepstream_running_average_fps": rounded(
            sum(analysis_rows[-1][1]) if analysis_rows else None
        ),
        "per_stream": per_stream,
    }


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def active_flag(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"active", "yes", "true", "1"}


def parse_gpu_metrics(
    path: Path,
    duration_seconds: int,
    *,
    sample_start: int | None = None,
    sample_end: int | None = None,
) -> dict[str, object]:
    if sample_start is not None and sample_start < 0:
        raise ValueError("sample_start must be non-negative")
    if sample_end is not None and sample_start is not None and sample_end < sample_start:
        raise ValueError("sample_end must be >= sample_start")
    requested_samples = math.ceil(duration_seconds / GPU_SAMPLE_INTERVAL_SECONDS)
    if not path.exists():
        return {
            "status": "missing_metrics",
            "csv": str(path),
            "requested_samples": requested_samples,
            "samples_raw": 0,
            "samples_analyzed": 0,
        }

    numeric_columns = {
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
        "temperature_c",
        "power_draw_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_requested_limit_w",
        "power_current_limit_w",
        "power_default_limit_w",
    }
    clock_event_columns = {
        "clock_event_sw_power_cap",
        "clock_event_sw_thermal_slowdown",
        "clock_event_hw_slowdown",
        "clock_event_hw_thermal_slowdown",
        "clock_event_hw_power_brake_slowdown",
    }
    raw_rows: list[dict[str, str]] = []
    gpu_names: set[str] = set()
    csv_columns: list[str] = []

    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        csv_columns = list(reader.fieldnames or [])
        for row in reader:
            raw_rows.append(row)
            if row.get("gpu_name"):
                gpu_names.add(row["gpu_name"].strip())

    # nvidia-smi monitoru Docker'dan hemen once baslatilir. Yalnizca fazladan
    # ornek varsa, ilk ornek idle (<=10%) ve ikinci ornek belirgin bicimde
    # yuklu (>=20%) ise bu tek pre-run ornegini atla. Ilk ornek zaten yukluyse
    # veya veri belirsizse ornek 0 analiz penceresinde kalir.
    analysis_start = 0 if sample_start is None else sample_start
    if sample_start is None and len(raw_rows) > requested_samples and len(raw_rows) >= 2:
        first_utilization = parse_number(raw_rows[0].get("gpu_utilization_percent"))
        second_utilization = parse_number(raw_rows[1].get("gpu_utilization_percent"))
        if (
            first_utilization is not None
            and second_utilization is not None
            and first_utilization <= INITIAL_IDLE_MAX_UTILIZATION_PERCENT
            and second_utilization >= INITIAL_BUSY_MIN_UTILIZATION_PERCENT
        ):
            analysis_start = 1

    bounded_rows = raw_rows[analysis_start:sample_end]
    analysis_rows = (
        bounded_rows if sample_end is not None else bounded_rows[:requested_samples]
    )
    values: dict[str, list[float]] = {column: [] for column in numeric_columns}
    for row in analysis_rows:
        for column in numeric_columns:
            number = parse_number(row.get(column, ""))
            if number is not None:
                values[column].append(number)

    raw_timestamps = [
        row["timestamp"].strip() for row in raw_rows if row.get("timestamp")
    ]
    analysis_timestamps = [
        row["timestamp"].strip() for row in analysis_rows if row.get("timestamp")
    ]

    required_numeric_columns = {
        "gpu_utilization_percent",
        "memory_used_mib",
        "temperature_c",
        "power_draw_w",
    }
    missing_numeric_samples = {
        column: len(analysis_rows) - len(values[column])
        for column in sorted(required_numeric_columns)
        if len(values[column]) != len(analysis_rows)
    }
    pstate_counts = dict(
        sorted(
            Counter(
                row["pstate"].strip()
                for row in analysis_rows
                if row.get("pstate") and row["pstate"].strip()
            ).items()
        )
    )
    active_clock_event_samples = {
        column: sum(1 for row in analysis_rows if active_flag(row.get(column)))
        for column in sorted(clock_event_columns)
        if column in csv_columns
    }
    active_mask_counts = dict(
        sorted(
            Counter(
                row["clock_event_reasons_active_mask"].strip()
                for row in analysis_rows
                if row.get("clock_event_reasons_active_mask")
                and row["clock_event_reasons_active_mask"].strip()
            ).items()
        )
    )

    if not raw_rows:
        status = "no_samples"
    elif len(analysis_rows) < requested_samples:
        status = "insufficient_gpu_window"
    elif missing_numeric_samples:
        status = "incomplete_gpu_metrics"
    else:
        status = "ok"

    return {
        "status": status,
        "csv": str(path),
        "sample_interval_seconds": GPU_SAMPLE_INTERVAL_SECONDS,
        "requested_samples": requested_samples,
        "samples_raw": len(raw_rows),
        "measurement_sample_start": analysis_start,
        "measurement_sample_end": sample_end,
        "samples_in_measurement_bounds": len(bounded_rows),
        "samples_analyzed": len(analysis_rows),
        "samples_discarded_before_measurement": analysis_start,
        "samples_discarded_initial_idle": analysis_start if sample_start is None else 0,
        "samples_discarded_after_requested_window": max(
            0, len(bounded_rows) - len(analysis_rows)
        ),
        "initial_idle_rule": (
            "skip at most one leading sample only when extra samples exist, "
            "sample 0 GPU utilization is <=10%, and sample 1 is >=20%"
        ),
        "raw_first_timestamp": raw_timestamps[0] if raw_timestamps else None,
        "raw_last_timestamp": raw_timestamps[-1] if raw_timestamps else None,
        "analysis_first_timestamp": (
            analysis_timestamps[0] if analysis_timestamps else None
        ),
        "analysis_last_timestamp": (
            analysis_timestamps[-1] if analysis_timestamps else None
        ),
        "gpu_names": sorted(gpu_names),
        "csv_columns": csv_columns,
        "power_throttle_telemetry_available": {
            "power_limits": all(
                column in csv_columns
                for column in (
                    "power_current_limit_w",
                    "power_default_limit_w",
                )
            ),
            "pstate": "pstate" in csv_columns,
            "clock_event_reasons": any(
                column in csv_columns for column in clock_event_columns
            ),
        },
        "pstate_counts": pstate_counts,
        "clock_event_reasons_active_mask_counts": active_mask_counts,
        "active_clock_event_samples": active_clock_event_samples,
        "required_numeric_columns": sorted(required_numeric_columns),
        "missing_numeric_samples": missing_numeric_samples,
        "metrics": {column: stats(values[column]) for column in sorted(numeric_columns)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepStream **PERF loglarini ve nvidia-smi CSV metriklerini ozetler."
    )
    parser.add_argument("root", nargs="?", default="benchmark/results", type=Path)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument(
        "--perf-interval",
        type=float,
        default=5.0,
        help="DeepStream PERF olcum araligi, saniye (varsayilan: 5)",
    )
    parser.add_argument("--streams", type=int, default=12)
    parser.add_argument("--sizes", nargs="+", type=int, default=[640, 960])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.streams <= 0 or args.perf_interval <= 0:
        raise SystemExit("--duration, --streams ve --perf-interval pozitif olmali")

    args.root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_duration_seconds": args.duration,
        "perf_interval_seconds": args.perf_interval,
        "gpu_sample_interval_seconds": GPU_SAMPLE_INTERVAL_SECONDS,
        "streams": args.streams,
        "runs": {},
    }

    runs = report["runs"]
    assert isinstance(runs, dict)
    for size in args.sizes:
        run = parse_perf(
            args.root / f"deepstream-12x-{size}.log",
            args.streams,
            args.duration,
            args.perf_interval,
        )
        run["gpu"] = parse_gpu_metrics(
            args.root / f"gpu-12x-{size}.csv", args.duration
        )
        runs[str(size)] = run

    destination = args.root / "summary.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
