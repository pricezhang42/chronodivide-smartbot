#!/usr/bin/env python
"""Fast replay indexer — pure Python, no game engine required.

Parses replay headers directly from text format and builds SQLite index.
Uses hardcoded country/side mappings instead of engine initialization.

Usage:
    python index_replay_database_fast.py --replay-dir D:\\Downloads\\RA2REPLAY --output replay_index.db
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

# RA2 country mapping: countryId -> (countryName, sideId)
# sideId: 0=Allied, 1=Soviet
COUNTRY_MAP = {
    0: ("Americans", 0),
    1: ("Alliance", 0),
    2: ("French", 0),
    3: ("Germans", 0),
    4: ("British", 0),
    5: ("Africans", 1),
    6: ("Arabs", 1),
    7: ("Confederation", 1),
    8: ("Russians", 1),
    # Yuri's Revenge countries
    9: ("Yuri", 2),
}

_MAP_RE = re.compile(rb'([a-zA-Z0-9_\- ]+\.(?:map|mpr))')


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast-index a replay database into SQLite.")
    parser.add_argument("--replay-dir", type=Path, required=True, help="Root directory containing .rpl files")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "replay_index.db", help="Output SQLite database path")
    parser.add_argument("--data-dir", type=Path, default=Path("d:/workspace/ra2-headless-mix"), help="RA2 data directory (for map filtering)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers for file scanning")
    parser.add_argument("--limit", type=int, default=None, help="Max replays to process")
    parser.add_argument("--resume", action="store_true", help="Skip replays already in database")
    parser.add_argument("--no-map-filter", action="store_true", help="Don't filter by available maps")
    return parser.parse_args(argv)


def discover_replays(replay_dir: Path) -> list[Path]:
    replays = []
    for root, _dirs, files in os.walk(replay_dir):
        for fname in files:
            if fname.endswith(".rpl"):
                replays.append(Path(root) / fname)
    return sorted(replays)


def discover_available_maps(data_dir: Path) -> set[str]:
    maps: set[str] = set()
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".map") or fname.endswith(".mpr"):
                maps.add(fname.lower())
    return maps


def parse_replay_header(path_str: str) -> dict | None:
    """Parse a replay file header. Returns dict or None on failure."""
    try:
        with open(path_str, "rb") as f:
            chunk = f.read(4096)

        lines = chunk.split(b"\n", 4)
        if len(lines) < 3:
            return None

        # Line 1: RA2TSREPL_v6
        # Line 2: ENGINE version hash
        # Line 3: gameId timestamp field0,field1,...,mapName,...,playerBlock
        line3 = lines[2].decode("utf-8", errors="replace").strip()

        # Split: "gameId timestamp rest"
        space_parts = line3.split(" ", 2)
        if len(space_parts) < 3:
            return None

        game_id = space_parts[0]
        timestamp = space_parts[1]
        rest = space_parts[2]

        # The rest is comma-separated with player blocks separated by ':'
        # Format: opts...,mapName,hash,more,...:player1Name,countryId,...:player2Name,...:@:...
        colon_parts = rest.split(":")

        # First section has game options including map name
        opts_section = colon_parts[0]
        opts_fields = opts_section.split(",")

        # Find map name
        map_name = None
        game_mode = None
        for field in opts_fields:
            f = field.strip()
            if f.endswith(".map") or f.endswith(".mpr"):
                map_name = f
                break

        # Game mode is typically the first numeric field after timestamp
        # Actually field index varies; let's find it from opts
        # The format is: field0,field1,...,gameMode(int),...,mapName,...
        # Try to extract game_mode from early fields
        for f in opts_fields[:5]:
            f = f.strip()
            try:
                v = int(f)
                if v in (0, 1, 2, 3):
                    game_mode = v
            except ValueError:
                pass

        # Parse players from the player block (colon_parts[1])
        # All players in one block: name,countryId,colorId,startPos,teamId,?,?,?,name2,...
        # Each player has exactly 8 fields
        FIELDS_PER_PLAYER = 8
        players = []
        if len(colon_parts) > 1:
            player_block = colon_parts[1].strip()
            if player_block and player_block != "@":
                fields = player_block.split(",")
                num_players = len(fields) // FIELDS_PER_PLAYER
                for p_idx in range(num_players):
                    base = p_idx * FIELDS_PER_PLAYER
                    if base + 4 >= len(fields):
                        break
                    name = fields[base]
                    try:
                        country_id = int(fields[base + 1])
                    except (ValueError, IndexError):
                        country_id = -1
                    try:
                        color_id = int(fields[base + 2])
                    except (ValueError, IndexError):
                        color_id = -1
                    try:
                        start_pos = int(fields[base + 3])
                    except (ValueError, IndexError):
                        start_pos = -1
                    try:
                        team_id = int(fields[base + 4])
                    except (ValueError, IndexError):
                        team_id = -1

                    country_name, side_id = COUNTRY_MAP.get(country_id, (f"Unknown_{country_id}", -1))
                    players.append({
                        "name": name,
                        "country_id": country_id,
                        "country_name": country_name,
                        "side_id": side_id,
                        "color_id": color_id,
                        "start_pos": start_pos,
                        "team_id": team_id,
                    })

        # Parse end_tick from last data line (approximate — scan for highest tick)
        # The replay body has lines like "tick=playerId|base64data"
        # We can estimate end_tick from the last few lines
        end_tick = 0
        all_lines = chunk.split(b"\n")
        for data_line in reversed(all_lines):
            data_line = data_line.strip()
            if b"=" in data_line:
                try:
                    tick_str = data_line.split(b"=", 1)[0]
                    tick = int(tick_str)
                    if tick > end_tick:
                        end_tick = tick
                    break
                except (ValueError, IndexError):
                    continue

        # For accurate end_tick, read the tail of the file
        file_size = os.path.getsize(path_str)
        if file_size > 4096:
            try:
                with open(path_str, "rb") as f:
                    f.seek(max(0, file_size - 2048))
                    tail = f.read()
                tail_lines = tail.split(b"\n")
                for data_line in reversed(tail_lines):
                    data_line = data_line.strip()
                    if b"=" in data_line:
                        try:
                            tick_str = data_line.split(b"=", 1)[0]
                            tick = int(tick_str)
                            if tick > end_tick:
                                end_tick = tick
                            break
                        except (ValueError, IndexError):
                            continue
            except Exception:
                pass

        return {
            "path": path_str,
            "gameId": game_id,
            "mapName": map_name,
            "gameMode": game_mode,
            "endTick": end_tick,
            "playerCount": len(players),
            "players": players,
            "fileSize": file_size,
        }

    except Exception as e:
        return {"path": path_str, "error": str(e)}


def _parse_worker(path_str: str) -> dict | None:
    return parse_replay_header(path_str)


def infer_server_and_date(replay_path: str, replay_dir: str) -> tuple[str | None, str | None]:
    try:
        rel = os.path.relpath(replay_path, replay_dir)
        parts = Path(rel).parts
        if len(parts) >= 4:
            return str(parts[2]), str(parts[1])
        if len(parts) >= 3:
            return str(parts[1]), str(parts[0])
    except ValueError:
        pass
    return None, None


def init_database(db_path: Path) -> sqlite3.Connection:
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
        CREATE INDEX IF NOT EXISTS idx_replays_error ON replays(error);
        CREATE INDEX IF NOT EXISTS idx_players_side_id ON players(side_id);
        CREATE INDEX IF NOT EXISTS idx_players_country_name ON players(country_name);
        CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
    """)
    conn.commit()
    return conn


def get_indexed_game_ids(conn: sqlite3.Connection) -> set[str]:
    cursor = conn.execute("SELECT game_id FROM replays")
    return {row[0] for row in cursor.fetchall()}


def insert_record(conn: sqlite3.Connection, record: dict, replay_dir: str) -> bool:
    game_id = record.get("gameId")
    error = record.get("error")
    rpath = record.get("path", "")

    if error and not game_id:
        game_id = f"error:{Path(rpath).stem}"

    if not game_id:
        return False

    server, date = infer_server_and_date(rpath, replay_dir)

    try:
        conn.execute(
            """INSERT OR IGNORE INTO replays
               (game_id, path, map_name, game_mode, end_tick, player_count, file_size_bytes, server, date, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id,
                rpath,
                record.get("mapName"),
                record.get("gameMode"),
                record.get("endTick"),
                record.get("playerCount"),
                record.get("fileSize", 0),
                server,
                date,
                error,
            ),
        )

        if not error:
            for idx, player in enumerate(record.get("players", [])):
                conn.execute(
                    """INSERT OR IGNORE INTO players
                       (game_id, player_index, name, country_id, country_name, side_id, color_id, start_pos, team_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        game_id,
                        idx,
                        player["name"],
                        player.get("country_id"),
                        player.get("country_name"),
                        player.get("side_id"),
                        player.get("color_id"),
                        player.get("start_pos"),
                        player.get("team_id"),
                    ),
                )
            return True
    except sqlite3.IntegrityError:
        pass
    return False


def print_summary(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
    success = conn.execute("SELECT COUNT(*) FROM replays WHERE error IS NULL").fetchone()[0]
    errors = conn.execute("SELECT COUNT(*) FROM replays WHERE error IS NOT NULL").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"Database Summary")
    print(f"{'='*60}")
    print(f"Total replays indexed: {total:,}")
    print(f"  Successfully parsed: {success:,}")
    print(f"  Errors: {errors:,}")

    if success == 0:
        return

    print(f"\nPlayer count distribution:")
    for row in conn.execute(
        "SELECT player_count, COUNT(*) as cnt FROM replays WHERE error IS NULL GROUP BY player_count ORDER BY cnt DESC LIMIT 10"
    ):
        print(f"  {row[0]} players: {row[1]:,}")

    side_names = {0: "Allied", 1: "Soviet", 2: "Yuri", -1: "Unknown"}
    print(f"\nSide distribution (all players):")
    for row in conn.execute(
        "SELECT side_id, COUNT(*) as cnt FROM players GROUP BY side_id ORDER BY cnt DESC"
    ):
        print(f"  {side_names.get(row[0], f'Side {row[0]}')}: {row[1]:,}")

    print(f"\nTop 15 countries:")
    for row in conn.execute(
        "SELECT country_name, COUNT(*) as cnt FROM players GROUP BY country_name ORDER BY cnt DESC LIMIT 15"
    ):
        print(f"  {row[0]}: {row[1]:,}")

    print(f"\nTop 20 maps:")
    for row in conn.execute(
        "SELECT map_name, COUNT(*) as cnt FROM replays WHERE error IS NULL GROUP BY map_name ORDER BY cnt DESC LIMIT 20"
    ):
        print(f"  {row[0]}: {row[1]:,}")

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

    soviet_1v1 = conn.execute("""
        SELECT COUNT(DISTINCT r.game_id)
        FROM replays r
        JOIN players p ON r.game_id = p.game_id
        WHERE r.error IS NULL AND r.player_count = 2 AND p.side_id = 1
    """).fetchone()[0]
    print(f"\n1v1 games with at least one Soviet player: {soviet_1v1:,}")

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
    t0 = time.time()
    all_replays = discover_replays(args.replay_dir)
    print(f"Found {len(all_replays):,} .rpl files ({time.time()-t0:.1f}s)")

    # Optional map pre-filter
    if not args.no_map_filter and args.data_dir.exists():
        available_maps = discover_available_maps(args.data_dir)
        print(f"Found {len(available_maps)} maps in data dir, pre-filtering...")
        t1 = time.time()
        # Quick single-pass filter
        filtered = []
        skipped = 0
        for rp in all_replays:
            try:
                with open(rp, "rb") as f:
                    chunk = f.read(2048)
                lines = chunk.split(b"\n", 4)
                if len(lines) >= 3:
                    m = _MAP_RE.search(lines[2])
                    if m:
                        name = m.group(1).decode("ascii", errors="replace")
                        if name.lower() in available_maps:
                            filtered.append(rp)
                            continue
            except Exception:
                pass
            skipped += 1
        all_replays = filtered
        print(f"Pre-filter: {len(all_replays):,} match, {skipped:,} skip ({time.time()-t1:.1f}s)")

    conn = init_database(args.output)

    if args.resume:
        indexed = get_indexed_game_ids(conn)
        before = len(all_replays)
        # We can't easily filter by game_id without parsing, so just track
        print(f"Resume mode: {len(indexed):,} already in DB")

    if args.limit:
        all_replays = all_replays[: args.limit]
        print(f"Limited to {len(all_replays):,} replays")

    if not all_replays:
        print("No replays to process.")
        print_summary(conn)
        conn.close()
        return 0

    # Parse all headers in parallel
    print(f"\nParsing {len(all_replays):,} replay headers with {args.workers} workers...")
    path_strs = [str(rp) for rp in all_replays]
    replay_dir_str = str(args.replay_dir)

    existing_ids = get_indexed_game_ids(conn) if args.resume else set()

    success = 0
    errors = 0
    skipped_existing = 0
    start = time.time()
    batch_records: list[dict] = []
    batch_size = 5000

    with multiprocessing.Pool(args.workers) as pool:
        for record in pool.imap_unordered(_parse_worker, path_strs, chunksize=500):
            if record is None:
                errors += 1
                continue

            if args.resume and record.get("gameId") in existing_ids:
                skipped_existing += 1
                continue

            batch_records.append(record)

            if len(batch_records) >= batch_size:
                for rec in batch_records:
                    if insert_record(conn, rec, replay_dir_str):
                        success += 1
                    else:
                        errors += 1
                conn.commit()
                batch_records.clear()

                done = success + errors + skipped_existing
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(path_strs) - done) / rate if rate > 0 else 0
                print(
                    f"\r  [{done:,}/{len(path_strs):,}] "
                    f"{success:,} ok, {errors:,} err, "
                    f"{rate:,.0f}/s, ~{remaining:.0f}s left",
                    end="",
                    flush=True,
                )

    # Flush remaining
    for rec in batch_records:
        if insert_record(conn, rec, replay_dir_str):
            success += 1
        else:
            errors += 1
    conn.commit()

    elapsed = time.time() - start
    print(f"\n\nCompleted in {elapsed:.1f}s ({len(path_strs) / elapsed:,.0f} replays/s)")
    print(f"  Success: {success:,}, Errors: {errors:,}, Skipped (existing): {skipped_existing:,}")

    print_summary(conn)
    conn.close()
    print(f"\nDatabase written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
