#!/usr/bin/env python
"""Query the replay index database and output replay paths for the training pipeline.

Common queries:
    # 1v1 Soviet-winning games on pinch_point, 5-20 min, limit 1000
    python query_replay_index.py --db replay_index.db \
        --1v1 --soviet-player --min-ticks 9000 --max-ticks 36000 \
        --map-contains pinch_point --limit 1000

    # All 1v1 games with at least one Soviet, any map
    python query_replay_index.py --db replay_index.db --1v1 --soviet-player

    # Summary stats only
    python query_replay_index.py --db replay_index.db --summary

    # Export replay paths to a file
    python query_replay_index.py --db replay_index.db --1v1 --soviet-player --output selected_replays.txt
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the replay index database.")
    parser.add_argument("--db", type=Path, default=PACKAGE_ROOT / "replay_index.db", help="SQLite database path")

    # Filters
    parser.add_argument("--1v1", dest="one_v_one", action="store_true", help="1v1 games only")
    parser.add_argument("--soviet-player", action="store_true", help="At least one Soviet player")
    parser.add_argument("--both-soviet", action="store_true", help="Both players are Soviet (1v1)")
    parser.add_argument("--side", type=int, help="Filter for games with a player of this side (0=Allied, 1=Soviet)")
    parser.add_argument("--country", type=str, help="Filter for games with a player of this country name")
    parser.add_argument("--map-contains", type=str, help="Map name contains this substring")
    parser.add_argument("--map-exact", type=str, help="Exact map name")
    parser.add_argument("--min-ticks", type=int, help="Minimum game length in ticks")
    parser.add_argument("--max-ticks", type=int, help="Maximum game length in ticks")
    parser.add_argument("--min-minutes", type=float, help="Minimum game length in minutes (at 30 tps)")
    parser.add_argument("--max-minutes", type=float, help="Maximum game length in minutes (at 30 tps)")
    parser.add_argument("--server", type=str, help="Server name")

    # Output
    parser.add_argument("--limit", type=int, help="Max results")
    parser.add_argument("--random", action="store_true", help="Randomize result order (instead of longest-first)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (use with --random)")
    parser.add_argument("--output", type=Path, help="Write paths to file")
    parser.add_argument("--summary", action="store_true", help="Print summary of matching results")
    parser.add_argument("--format", choices=["paths", "tsv", "json"], default="paths", help="Output format")
    parser.add_argument("--player-name", type=str, help="Output paths with this player name (for --format tsv)")

    return parser.parse_args(argv)


def build_query(args: argparse.Namespace) -> tuple[str, list]:
    """Build SQL query from args. Returns (sql, params)."""
    conditions = ["r.error IS NULL"]
    params: list = []
    joins = []

    if args.one_v_one:
        conditions.append("r.player_count = 2")

    if args.soviet_player:
        joins.append("JOIN players ps ON r.game_id = ps.game_id AND ps.side_id = 1")

    if args.both_soviet:
        joins.append("""
            JOIN players ps1 ON r.game_id = ps1.game_id AND ps1.side_id = 1 AND ps1.player_index = 0
            JOIN players ps2 ON r.game_id = ps2.game_id AND ps2.side_id = 1 AND ps2.player_index = 1
        """)

    if args.side is not None:
        joins.append(f"JOIN players pside ON r.game_id = pside.game_id AND pside.side_id = ?")
        params.append(args.side)

    if args.country:
        joins.append(f"JOIN players pcountry ON r.game_id = pcountry.game_id AND pcountry.country_name = ?")
        params.append(args.country)

    if args.map_contains:
        conditions.append("r.map_name LIKE ?")
        params.append(f"%{args.map_contains}%")

    if args.map_exact:
        conditions.append("r.map_name = ?")
        params.append(args.map_exact)

    min_ticks = args.min_ticks
    max_ticks = args.max_ticks
    if args.min_minutes is not None:
        min_ticks = int(args.min_minutes * 60 * 30)
    if args.max_minutes is not None:
        max_ticks = int(args.max_minutes * 60 * 30)

    if min_ticks is not None:
        conditions.append("r.end_tick >= ?")
        params.append(min_ticks)
    if max_ticks is not None:
        conditions.append("r.end_tick <= ?")
        params.append(max_ticks)

    if args.server:
        conditions.append("r.server = ?")
        params.append(args.server)

    join_clause = "\n".join(joins)
    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT DISTINCT r.game_id, r.path, r.map_name, r.end_tick, r.player_count
        FROM replays r
        {join_clause}
        WHERE {where_clause}
        ORDER BY {"RANDOM()" if args.random and args.seed is None else "r.end_tick DESC"}
    """

    # When using --random without --seed, let SQLite handle it with LIMIT
    # When using --random with --seed, fetch all and shuffle in Python
    if args.limit and not (args.random and args.seed is not None):
        sql += " LIMIT ?"
        params.append(args.limit)

    return sql, params


def print_match_summary(conn: sqlite3.Connection, game_ids: list[str]) -> None:
    """Print summary statistics of matched replays."""
    if not game_ids:
        print("No matching replays.")
        return

    placeholders = ",".join("?" for _ in game_ids)

    count = len(game_ids)
    print(f"\nMatched {count:,} replays")

    # Map distribution
    rows = conn.execute(
        f"SELECT map_name, COUNT(*) as cnt FROM replays WHERE game_id IN ({placeholders}) GROUP BY map_name ORDER BY cnt DESC LIMIT 15",
        game_ids,
    ).fetchall()
    print(f"\nTop maps:")
    for name, cnt in rows:
        print(f"  {cnt:6,} {name}")

    # Side distribution
    rows = conn.execute(
        f"SELECT side_id, COUNT(*) as cnt FROM players WHERE game_id IN ({placeholders}) GROUP BY side_id ORDER BY cnt DESC",
        game_ids,
    ).fetchall()
    side_names = {0: "Allied", 1: "Soviet", 2: "Yuri"}
    print(f"\nSide distribution:")
    for sid, cnt in rows:
        print(f"  {side_names.get(sid, f'Side {sid}')}: {cnt:,}")

    # Country distribution
    rows = conn.execute(
        f"SELECT country_name, COUNT(*) as cnt FROM players WHERE game_id IN ({placeholders}) GROUP BY country_name ORDER BY cnt DESC LIMIT 10",
        game_ids,
    ).fetchall()
    print(f"\nTop countries:")
    for name, cnt in rows:
        print(f"  {cnt:6,} {name}")

    # Game length stats
    rows = conn.execute(
        f"SELECT MIN(end_tick), AVG(end_tick), MAX(end_tick) FROM replays WHERE game_id IN ({placeholders})",
        game_ids,
    ).fetchone()
    if rows and rows[0] is not None:
        print(f"\nGame length (ticks @ 30 tps):")
        print(f"  Min: {rows[0]:,} ({rows[0]/30/60:.1f} min)")
        print(f"  Avg: {rows[1]:,.0f} ({rows[1]/30/60:.1f} min)")
        print(f"  Max: {rows[2]:,} ({rows[2]/30/60:.1f} min)")


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(args.db))

    sql, params = build_query(args)
    rows = conn.execute(sql, params).fetchall()

    # Seeded random shuffle in Python (SQLite RANDOM() isn't seedable)
    if args.random and args.seed is not None:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        if args.limit:
            rows = rows[: args.limit]

    game_ids = [row[0] for row in rows]
    paths = [row[1] for row in rows]

    if args.summary or (not args.output and not paths):
        print_match_summary(conn, game_ids)
    elif args.output:
        with open(args.output, "w") as f:
            for p in paths:
                f.write(p + "\n")
        print(f"Wrote {len(paths):,} replay paths to {args.output}")
        print_match_summary(conn, game_ids)
    else:
        if args.format == "paths":
            for p in paths:
                print(p)
        elif args.format == "tsv":
            print("game_id\tpath\tmap_name\tend_tick\tplayer_count")
            for row in rows:
                print("\t".join(str(x) for x in row))

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
