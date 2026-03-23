#!/usr/bin/env python
"""Infer winners for replays by simulating to endTick.

Runs the JS engine to fast-forward replays and determine the winner from
defeat/resignation/score state. Updates the replay_index.db with results.

Usage:
    # Infer winners for replays in a file list
    python infer_winners.py --input selected_pinch_point_soviet_1k.txt

    # Infer winners for a query result
    python infer_winners.py --input selected_replays.txt --workers 4 --batch-size 20
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("d:/workspace/ra2-headless-mix")
DEFAULT_JS_SCRIPT = PROJECT_ROOT / "packages" / "py-chronodivide" / "infer_replay_winners.mjs"
PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer replay winners via simulation.")
    parser.add_argument("--input", type=Path, required=True, help="File with replay paths (one per line)")
    parser.add_argument("--db", type=Path, default=PACKAGE_ROOT / "replay_index.db", help="SQLite database to update")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="RA2 data directory")
    parser.add_argument("--js-script", type=Path, default=DEFAULT_JS_SCRIPT, help="Path to infer_replay_winners.mjs")
    parser.add_argument("--batch-size", type=int, default=25, help="Replays per Node invocation")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Node processes")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Also write raw JSONL results to this file")
    parser.add_argument("--resume", action="store_true", help="Skip replays already with winner data")
    return parser.parse_args(argv)


def ensure_winner_columns(conn: sqlite3.Connection) -> None:
    """Add winner columns to the replays table if they don't exist."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(replays)").fetchall()}
    new_columns = {
        "winner_name": "TEXT",
        "loser_name": "TEXT",
        "winner_reason": "TEXT",
        "winner_inferred": "INTEGER DEFAULT 0",
    }
    for col, col_type in new_columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE replays ADD COLUMN {col} {col_type}")
    conn.commit()


def get_inferred_paths(conn: sqlite3.Connection) -> set[str]:
    """Get paths that already have winner inference results."""
    try:
        cursor = conn.execute("SELECT path FROM replays WHERE winner_inferred = 1")
        return {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        return set()


def run_batch(
    js_script: Path,
    data_dir: Path,
    replay_paths: list[str],
) -> list[dict[str, Any]]:
    """Run the JS winner inference on a batch of replays."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in replay_paths:
            f.write(p + "\n")
        input_path = f.name

    try:
        result = subprocess.run(
            ["node", str(js_script), "--data-dir", str(data_dir), "--input", input_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=1800,  # 30 min per batch (full simulation is slow)
            encoding="utf-8",
            errors="replace",
        )
    finally:
        os.unlink(input_path)

    stdout = result.stdout or ""
    records = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def update_db(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> tuple[int, int]:
    """Update the database with winner inference results. Returns (updated, errors)."""
    updated = 0
    errors = 0
    for record in records:
        rpath = record.get("path", "")
        if "error" in record:
            errors += 1
            continue

        winner = record.get("winner")
        loser = record.get("loser")
        reason = record.get("reason", "unknown")

        try:
            conn.execute(
                """UPDATE replays
                   SET winner_name = ?, loser_name = ?, winner_reason = ?, winner_inferred = 1
                   WHERE path = ?""",
                (winner, loser, reason, rpath),
            )
            updated += 1
        except sqlite3.Error:
            errors += 1

    conn.commit()
    return updated, errors


def run_batch_worker(args: tuple[Path, Path, list[str]]) -> list[dict[str, Any]]:
    """Worker function for parallel batch processing."""
    js_script, data_dir, replay_paths = args
    return run_batch(js_script, data_dir, replay_paths)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    replay_paths = [
        line.strip()
        for line in args.input.read_text().strip().split("\n")
        if line.strip()
    ]
    print(f"Loaded {len(replay_paths):,} replay paths from {args.input}")

    conn = None
    if args.db.exists():
        conn = sqlite3.connect(str(args.db))
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_winner_columns(conn)

        if args.resume:
            inferred = get_inferred_paths(conn)
            before = len(replay_paths)
            replay_paths = [p for p in replay_paths if p not in inferred]
            print(f"Resuming: {before - len(replay_paths):,} already inferred, {len(replay_paths):,} remaining")

    if not replay_paths:
        print("No replays to process.")
        return 0

    # Open JSONL output if requested
    jsonl_file = open(args.output_jsonl, "a", encoding="utf-8") if args.output_jsonl else None

    # Split into batches
    batches: list[list[str]] = []
    for i in range(0, len(replay_paths), args.batch_size):
        batches.append(replay_paths[i : i + args.batch_size])

    total_updated = 0
    total_errors = 0
    all_records: list[dict[str, Any]] = []
    start_time = time.time()

    if args.workers > 1:
        import concurrent.futures

        work_items = [(args.js_script, args.data_dir, batch) for batch in batches]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_batch_worker, item): idx for idx, item in enumerate(work_items)}
            for future in concurrent.futures.as_completed(futures):
                batch_idx = futures[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                    if jsonl_file:
                        for rec in records:
                            jsonl_file.write(json.dumps(rec) + "\n")
                        jsonl_file.flush()
                    if conn:
                        u, e = update_db(conn, records)
                        total_updated += u
                        total_errors += e
                    else:
                        total_updated += sum(1 for r in records if "error" not in r)
                        total_errors += sum(1 for r in records if "error" in r)
                except Exception as exc:
                    print(f"\nBatch {batch_idx} failed: {exc}", file=sys.stderr)
                    total_errors += len(batches[batch_idx])

                done = total_updated + total_errors
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(replay_paths) - done) / rate if rate > 0 else 0
                print(
                    f"\r  [{done:,}/{len(replay_paths):,}] "
                    f"{total_updated:,} ok, {total_errors:,} err, "
                    f"{rate:.1f} replays/s, ~{remaining:.0f}s remaining",
                    end="",
                    flush=True,
                )
    else:
        for batch_idx, batch in enumerate(batches):
            try:
                records = run_batch(args.js_script, args.data_dir, batch)
                all_records.extend(records)
                if jsonl_file:
                    for rec in records:
                        jsonl_file.write(json.dumps(rec) + "\n")
                    jsonl_file.flush()
                if conn:
                    u, e = update_db(conn, records)
                    total_updated += u
                    total_errors += e
                else:
                    total_updated += sum(1 for r in records if "error" not in r)
                    total_errors += sum(1 for r in records if "error" in r)
            except Exception as exc:
                print(f"\nBatch {batch_idx} failed: {exc}", file=sys.stderr)
                total_errors += len(batch)

            done = total_updated + total_errors
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(replay_paths) - done) / rate if rate > 0 else 0
            print(
                f"\r  [{done:,}/{len(replay_paths):,}] "
                f"{total_updated:,} ok, {total_errors:,} err, "
                f"{rate:.1f} replays/s, ~{remaining:.0f}s remaining",
                end="",
                flush=True,
            )

    if jsonl_file:
        jsonl_file.close()

    elapsed = time.time() - start_time
    print(f"\n\nCompleted in {elapsed:.1f}s ({len(replay_paths) / elapsed:.1f} replays/s)")

    # Print summary
    reason_counts: dict[str, int] = {}
    winner_determined = 0
    for rec in all_records:
        if "error" in rec:
            continue
        reason = rec.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if rec.get("winner"):
            winner_determined += 1

    print(f"\nWinner inference results:")
    print(f"  Winner determined: {winner_determined:,}")
    print(f"  Errors: {total_errors:,}")
    print(f"\n  By reason:")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {cnt:,}")

    if conn:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
