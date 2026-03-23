#!/usr/bin/env python
"""Index a large replay database into a SQLite database.

Scans all .rpl files in a directory tree, extracts replay headers (player names,
countries, sides, map, game length) via the JS engine, and stores results in a
queryable SQLite database.

Usage:
    python index_replay_database.py --replay-dir D:\\Downloads\\RA2REPLAY --output replay_index.db

The JS extractor processes replays in batches to amortize Node startup cost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("d:/workspace/ra2-headless-mix")
DEFAULT_JS_SCRIPT = PROJECT_ROOT / "packages" / "py-chronodivide" / "extract_replay_headers.mjs"
PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a replay database into SQLite.")
    parser.add_argument("--replay-dir", type=Path, required=True, help="Root directory containing .rpl files")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "replay_index.db", help="Output SQLite database path")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="RA2 data directory for map resolution")
    parser.add_argument("--js-script", type=Path, default=DEFAULT_JS_SCRIPT, help="Path to extract_replay_headers.mjs")
    parser.add_argument("--batch-size", type=int, default=500, help="Replays per Node invocation")
    parser.add_argument("--workers", type=int, default=1, help="Parallel Node processes")
    parser.add_argument("--resume", action="store_true", help="Skip replays already in the database")
    parser.add_argument("--limit", type=int, default=None, help="Max replays to process (for testing)")
    return parser.parse_args(argv)


def discover_replays(replay_dir: Path) -> list[Path]:
    """Find all .rpl files recursively."""
    replays = []
    for root, _dirs, files in os.walk(replay_dir):
        for fname in files:
            if fname.endswith(".rpl"):
                replays.append(Path(root) / fname)
    return sorted(replays)


_MAP_RE = re.compile(rb'([a-zA-Z0-9_\- ]+\.(?:map|mpr))')


def extract_map_name_from_header(replay_path: Path) -> str | None:
    """Extract map name from replay header without engine. Reads first 2KB as bytes for speed."""
    try:
        with open(replay_path, "rb") as f:
            chunk = f.read(2048)
        # Map name appears on line 3 as a comma-separated field ending in .map or .mpr
        lines = chunk.split(b"\n", 4)
        if len(lines) >= 3:
            line3 = lines[2]
            m = _MAP_RE.search(line3)
            if m:
                return m.group(1).decode("ascii", errors="replace")
    except Exception:
        pass
    return None


def discover_available_maps(data_dir: Path) -> set[str]:
    """Find all map files in the data dir."""
    maps: set[str] = set()
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".map") or fname.endswith(".mpr"):
                maps.add(fname)
    return maps


def _check_one(args: tuple[str, frozenset[str]]) -> tuple[str, str | None]:
    """Worker for parallel pre-filter. Returns (path_str, map_name_or_None)."""
    path_str, available = args
    try:
        with open(path_str, "rb") as f:
            chunk = f.read(2048)
        lines = chunk.split(b"\n", 4)
        if len(lines) >= 3:
            m = _MAP_RE.search(lines[2])
            if m:
                name = m.group(1).decode("ascii", errors="replace")
                if name in available:
                    return path_str, name
    except Exception:
        pass
    return path_str, None


def pre_filter_replays(
    replays: list[Path], available_maps: set[str], workers: int = 6,
) -> tuple[list[Path], int]:
    """Filter replays to only those using maps we can parse. Uses multiprocessing."""
    import multiprocessing

    frozen = frozenset(available_maps)
    work = [(str(rp), frozen) for rp in replays]

    filtered = []
    skipped = 0
    with multiprocessing.Pool(workers) as pool:
        for path_str, map_name in pool.imap_unordered(_check_one, work, chunksize=2000):
            if map_name is not None:
                filtered.append(Path(path_str))
            else:
                skipped += 1
    return sorted(filtered), skipped


def infer_server_and_date(replay_path: Path, replay_dir: Path) -> tuple[str | None, str | None]:
    """Infer server name and date from directory structure: replay_dir/month/day/server/uuid.rpl"""
    try:
        rel = replay_path.relative_to(replay_dir)
        parts = rel.parts
        if len(parts) >= 4:
            # month/day/server/file.rpl
            return str(parts[2]), str(parts[1])
        if len(parts) >= 3:
            return str(parts[1]), str(parts[0])
    except ValueError:
        pass
    return None, None


def init_database(db_path: Path) -> sqlite3.Connection:
    """Create the SQLite database and tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS replays (
            game_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            map_name TEXT,
            game_mode INTEGER,
            end_tick INTEGER,
            player_count INTEGER,
            file_size_bytes INTEGER,
            server TEXT,
            date TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS players (
            game_id TEXT NOT NULL,
            player_index INTEGER NOT NULL,
            name TEXT NOT NULL,
            country_id INTEGER,
            country_name TEXT,
            side_id INTEGER,
            color_id INTEGER,
            start_pos INTEGER,
            team_id INTEGER,
            PRIMARY KEY (game_id, player_index),
            FOREIGN KEY (game_id) REFERENCES replays(game_id)
        );

        CREATE INDEX IF NOT EXISTS idx_replays_map ON replays(map_name);
        CREATE INDEX IF NOT EXISTS idx_replays_player_count ON replays(player_count);
        CREATE INDEX IF NOT EXISTS idx_replays_end_tick ON replays(end_tick);
        CREATE INDEX IF NOT EXISTS idx_players_side_id ON players(side_id);
        CREATE INDEX IF NOT EXISTS idx_players_country_name ON players(country_name);
        CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
    """)
    conn.commit()
    return conn


def get_indexed_paths(conn: sqlite3.Connection) -> set[str]:
    """Get set of replay paths already in the database."""
    cursor = conn.execute("SELECT path FROM replays")
    return {row[0] for row in cursor.fetchall()}


def run_batch(
    js_script: Path,
    data_dir: Path,
    replay_paths: list[Path],
) -> list[dict[str, Any]]:
    """Run the JS header extractor on a batch of replays."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in replay_paths:
            f.write(str(p.resolve()) + "\n")
        input_path = f.name

    try:
        result = subprocess.run(
            ["node", str(js_script), "--data-dir", str(data_dir), "--input", input_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=600,
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


def insert_records(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
    replay_dir: Path,
    path_to_size: dict[str, int],
) -> tuple[int, int]:
    """Insert parsed records into the database. Returns (success_count, error_count)."""
    success = 0
    errors = 0
    for record in records:
        rpath = record.get("path", "")
        game_id = record.get("gameId")
        error = record.get("error")

        if error and not game_id:
            # Use path as a pseudo game_id for error records
            game_id = f"error:{Path(rpath).stem}"

        server, date = infer_server_and_date(Path(rpath), replay_dir)
        file_size = path_to_size.get(rpath, 0)

        try:
            conn.execute(
                """INSERT OR REPLACE INTO replays
                   (game_id, path, map_name, game_mode, end_tick, player_count, file_size_bytes, server, date, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    game_id,
                    rpath,
                    record.get("mapName"),
                    record.get("gameMode"),
                    record.get("endTick"),
                    record.get("playerCount"),
                    file_size,
                    server,
                    date,
                    error,
                ),
            )

            if not error:
                for idx, player in enumerate(record.get("players", [])):
                    conn.execute(
                        """INSERT OR REPLACE INTO players
                           (game_id, player_index, name, country_id, country_name, side_id, color_id, start_pos, team_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            game_id,
                            idx,
                            player["name"],
                            player.get("countryId"),
                            player.get("countryName"),
                            player.get("sideId"),
                            player.get("colorId"),
                            player.get("startPos"),
                            player.get("teamId"),
                        ),
                    )
                success += 1
            else:
                errors += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    return success, errors


def run_batch_worker(args: tuple[Path, Path, list[Path]]) -> list[dict[str, Any]]:
    """Worker function for parallel batch processing."""
    js_script, data_dir, replay_paths = args
    return run_batch(js_script, data_dir, replay_paths)


def print_summary(conn: sqlite3.Connection) -> None:
    """Print a summary of the indexed database."""
    total = conn.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
    success = conn.execute("SELECT COUNT(*) FROM replays WHERE error IS NULL").fetchone()[0]
    errors = conn.execute("SELECT COUNT(*) FROM replays WHERE error IS NOT NULL").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"Database Summary")
    print(f"{'='*60}")
    print(f"Total replays indexed: {total:,}")
    print(f"  Successfully parsed: {success:,}")
    print(f"  Errors (missing maps etc): {errors:,}")

    if success == 0:
        return

    # Player count distribution
    print(f"\nPlayer count distribution:")
    for row in conn.execute(
        "SELECT player_count, COUNT(*) as cnt FROM replays WHERE error IS NULL GROUP BY player_count ORDER BY cnt DESC LIMIT 10"
    ):
        print(f"  {row[0]} players: {row[1]:,}")

    # Side distribution
    print(f"\nSide distribution (all players):")
    side_names = {0: "Allied", 1: "Soviet", 2: "Yuri"}
    for row in conn.execute(
        "SELECT side_id, COUNT(*) as cnt FROM players GROUP BY side_id ORDER BY cnt DESC"
    ):
        name = side_names.get(row[0], f"Side {row[0]}")
        print(f"  {name}: {row[1]:,}")

    # Country distribution
    print(f"\nTop 15 countries:")
    for row in conn.execute(
        "SELECT country_name, COUNT(*) as cnt FROM players GROUP BY country_name ORDER BY cnt DESC LIMIT 15"
    ):
        print(f"  {row[0]}: {row[1]:,}")

    # Map distribution
    print(f"\nTop 20 maps:")
    for row in conn.execute(
        "SELECT map_name, COUNT(*) as cnt FROM replays WHERE error IS NULL GROUP BY map_name ORDER BY cnt DESC LIMIT 20"
    ):
        print(f"  {row[0]}: {row[1]:,}")

    # Game length distribution (1v1 only)
    print(f"\n1v1 game length distribution (ticks, ~30 tps):")
    for label, lo, hi in [
        ("<1 min", 0, 1800),
        ("1-3 min", 1800, 5400),
        ("3-5 min", 5400, 9000),
        ("5-10 min", 9000, 18000),
        ("10-15 min", 18000, 27000),
        ("15-20 min", 27000, 36000),
        ("20-30 min", 36000, 54000),
        (">30 min", 54000, 999999999),
    ]:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM replays WHERE error IS NULL AND player_count=2 AND end_tick >= ? AND end_tick < ?",
            (lo, hi),
        ).fetchone()[0]
        print(f"  {label}: {cnt:,}")

    # Soviet 1v1 stats
    soviet_1v1 = conn.execute("""
        SELECT COUNT(DISTINCT r.game_id)
        FROM replays r
        JOIN players p ON r.game_id = p.game_id
        WHERE r.error IS NULL AND r.player_count = 2 AND p.side_id = 1
    """).fetchone()[0]
    print(f"\n1v1 games with at least one Soviet player: {soviet_1v1:,}")

    # Server distribution
    print(f"\nServer distribution:")
    for row in conn.execute(
        "SELECT server, COUNT(*) as cnt FROM replays WHERE error IS NULL AND server IS NOT NULL GROUP BY server ORDER BY cnt DESC"
    ):
        print(f"  {row[0]}: {row[1]:,}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.replay_dir.exists():
        print(f"Replay directory not found: {args.replay_dir}", file=sys.stderr)
        return 1

    print(f"Discovering replays in {args.replay_dir}...")
    all_replays = discover_replays(args.replay_dir)
    print(f"Found {len(all_replays):,} .rpl files")

    # Pre-filter by map availability (fast text scan, no engine needed)
    print(f"Scanning available maps in {args.data_dir}...")
    available_maps = discover_available_maps(args.data_dir)
    print(f"Found {len(available_maps)} maps in data dir")
    print(f"Pre-filtering replays by map availability...")
    pre_filter_start = time.time()
    replays, skipped = pre_filter_replays(all_replays, available_maps)
    pre_filter_elapsed = time.time() - pre_filter_start
    print(f"Pre-filter: {len(replays):,} matchable, {skipped:,} skipped ({pre_filter_elapsed:.1f}s)")

    conn = init_database(args.output)

    if args.resume:
        indexed = get_indexed_paths(conn)
        replays = [r for r in replays if str(r.resolve()) not in indexed]
        print(f"Resuming: {len(indexed):,} already indexed, {len(replays):,} remaining")

    if args.limit:
        replays = replays[: args.limit]
        print(f"Limited to {len(replays):,} replays")

    if not replays:
        print("No replays to process.")
        print_summary(conn)
        conn.close()
        return 0

    # Build path -> file size mapping for the batch
    path_to_size: dict[str, int] = {}
    for rp in replays:
        try:
            path_to_size[str(rp.resolve())] = rp.stat().st_size
        except OSError:
            pass

    # Split into batches
    batches: list[list[Path]] = []
    for i in range(0, len(replays), args.batch_size):
        batches.append(replays[i : i + args.batch_size])

    total_success = 0
    total_errors = 0
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
                    s, e = insert_records(conn, records, args.replay_dir, path_to_size)
                    total_success += s
                    total_errors += e
                except Exception as exc:
                    print(f"Batch {batch_idx} failed: {exc}", file=sys.stderr)
                    total_errors += len(batches[batch_idx])

                done = total_success + total_errors
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(replays) - done) / rate if rate > 0 else 0
                print(
                    f"\r  [{done:,}/{len(replays):,}] "
                    f"{total_success:,} ok, {total_errors:,} err, "
                    f"{rate:.0f} replays/s, ~{remaining:.0f}s remaining",
                    end="",
                    flush=True,
                )
    else:
        for batch_idx, batch in enumerate(batches):
            try:
                records = run_batch(args.js_script, args.data_dir, batch)
                s, e = insert_records(conn, records, args.replay_dir, path_to_size)
                total_success += s
                total_errors += e
            except Exception as exc:
                print(f"\nBatch {batch_idx} failed: {exc}", file=sys.stderr)
                total_errors += len(batch)

            done = total_success + total_errors
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(replays) - done) / rate if rate > 0 else 0
            print(
                f"\r  [{done:,}/{len(replays):,}] "
                f"{total_success:,} ok, {total_errors:,} err, "
                f"{rate:.0f} replays/s, ~{remaining:.0f}s remaining",
                end="",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\n\nCompleted in {elapsed:.1f}s ({len(replays) / elapsed:.0f} replays/s)")

    print_summary(conn)
    conn.close()
    print(f"\nDatabase written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
