"""Rank MLE CLI — port of katrain/tools/rank_mle_cli.py to the stripped-down
KataGoClient backend. No katrain dependency.

Usage:
    uv run python rank_mle.py FILE.sgf [FILE2.sgf ...]
        --katago /opt/homebrew/bin/katago
        --human-model ~/.katrain/b18c384nbt-humanv0.bin.gz
        --config /path/to/analysis_config.cfg
        [--model PATH]   # defaults to --human-model (same-file-twice)
        [--no-cache] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time

from katago_client import (
    HUMAN_RANKS,
    KataGoClient,
    KataGoConfig,
    build_rank_policy_queries,
)
from sgf_loader import LoadedGame, gtp_to_index, load_sgf

CACHE_VERSION = 2
CACHE_COMPAT_VERSIONS = {1, CACHE_VERSION}
CACHE_DIRNAME = ".rank_mle_cache"
EPS = 1e-7


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(sgf_path: str, sha: str) -> str:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(sgf_path)), CACHE_DIRNAME)
    return os.path.join(cache_dir, f"{os.path.basename(sgf_path)}.{sha}.json")


def _load_cache(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if data.get("version") in CACHE_COMPAT_VERSIONS else None
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)


def _empty_stats(n_ranks: int) -> dict:
    return {
        player: {"n_moves": 0, "loglik_sums": [0.0] * n_ranks}
        for player in ("B", "W")
    }


def _stats_from_legacy_moves(data: dict) -> dict:
    ranks = data["ranks"]
    stats = _empty_stats(len(ranks))
    for move in data.get("moves", []):
        player = move.get("player")
        if player not in stats:
            continue
        stats[player]["n_moves"] += 1
        for i, ll in enumerate(move.get("logliks", [])[: len(ranks)]):
            stats[player]["loglik_sums"][i] += ll
    return stats


def _cache_needs_compaction(data: dict) -> bool:
    return (
        data.get("version") != CACHE_VERSION
        or data.get("stats") is None
        or "moves" in data
    )


def _compact_cache(data: dict) -> dict:
    if not _cache_needs_compaction(data):
        return data

    ranks = data["ranks"]
    moves = data.get("moves", [])
    return {
        "version": CACHE_VERSION,
        "ranks": ranks,
        "stats": data.get("stats") or _stats_from_legacy_moves(data),
        "total_moves": data.get("total_moves", len(moves)),
        "players": data.get("players", {}),
    }


def _analyze_game(
    game: LoadedGame,
    client: KataGoClient,
    progress_cb=None,
) -> dict:
    n_moves = len(game.moves)
    n_ranks = len(HUMAN_RANKS)

    if n_moves == 0:
        return {
            "version": CACHE_VERSION,
            "ranks": HUMAN_RANKS,
            "stats": _empty_stats(n_ranks),
            "total_moves": 0,
            "players": game.players,
        }

    move_players = [m[0] for m in game.moves]
    fallback_log_p = math.log(EPS)
    stats = _empty_stats(n_ranks)
    for player in move_players:
        stats[player]["n_moves"] += 1
        stats[player]["loglik_sums"] = [
            s + fallback_log_p for s in stats[player]["loglik_sums"]
        ]

    lock = threading.Lock()
    done_count = [0]
    error_count = [0]
    first_error: list[str | None] = [None]
    expected = n_moves * n_ranks
    done_event = threading.Event()
    last_log = [time.time()]
    rank_indices = {rank: i for i, rank in enumerate(HUMAN_RANKS)}

    def make_cb(move_idx: int, rank: str, played_idx: int):
        def cb(msg: dict):
            policy = msg.get("humanPolicy") or []
            prob = float(policy[played_idx]) if 0 <= played_idx < len(policy) else 0.0
            log_p = math.log(max(prob, EPS))
            with lock:
                done_count[0] += 1
                rank_idx = rank_indices[rank]
                player = move_players[move_idx]
                stats[player]["loglik_sums"][rank_idx] += log_p - fallback_log_p
                now = time.time()
                if now - last_log[0] > 2.0:
                    print(f"  {done_count[0]}/{expected} queries done", file=sys.stderr)
                    last_log[0] = now
                if progress_cb is not None:
                    progress_cb(done_count[0], expected)
                if done_count[0] >= expected:
                    done_event.set()

        return cb

    def err_cb(msg: dict):
        err = msg.get("error", "unknown error")
        print(f"  katago error: {err}", file=sys.stderr)
        with lock:
            error_count[0] += 1
            if first_error[0] is None:
                first_error[0] = err
            done_count[0] += 1
            if done_count[0] >= expected:
                done_event.set()

    # For move i, query the position with moves[:i] played, asking for policy
    # at the position where it's `move_players[i]` to play.
    for i, (player, gtp) in enumerate(game.moves):
        prefix = game.moves[:i]
        played_idx = gtp_to_index(gtp, game.board_size)
        queries = build_rank_policy_queries(
            prefix,
            initial_stones=game.initial_stones,
            board_size=game.board_size,
            komi=game.komi,
            rules=game.rules,
            initial_player=game.initial_player,
        )
        for rank, q in zip(HUMAN_RANKS, queries):
            client.send_query(q, make_cb(i, rank, played_idx), err_cb)

    while not done_event.is_set():
        if not client.is_alive():
            raise RuntimeError("KataGo process died during analysis")
        done_event.wait(timeout=0.5)

    if error_count[0] == expected:
        raise RuntimeError(
            f"KataGo rejected all {expected} queries — {first_error[0]!r}"
        )

    return {
        "version": CACHE_VERSION,
        "ranks": HUMAN_RANKS,
        "stats": stats,
        "total_moves": n_moves,
        "players": game.players,
    }


def analyze_path(
    client: KataGoClient,
    sgf_path: str,
    *,
    use_cache: bool = True,
    progress_cb=None,
) -> dict:
    """Analyze an SGF, returning the cache-shaped data dict.

    Hits/writes the sidecar cache unless use_cache is False. progress_cb is
    called as (done, total) during analysis.
    """
    sha = _sha_file(sgf_path)
    cp = _cache_path(sgf_path, sha)
    if use_cache:
        cached = _load_cache(cp)
        if cached is not None:
            if _cache_needs_compaction(cached):
                cached = _compact_cache(cached)
                _save_cache(cp, cached)
            if progress_cb is not None:
                progress_cb(1, 1)
            return cached
    game = load_sgf(sgf_path)
    data = _analyze_game(game, client, progress_cb=progress_cb)
    _save_cache(cp, data)
    return data


def _predict_per_player(data: dict) -> dict:
    ranks = data["ranks"]
    n_ranks = len(ranks)
    stats = data.get("stats") or _stats_from_legacy_moves(data)
    out = {}
    for player in ("B", "W"):
        player_stats = stats.get(player, {})
        n_moves = player_stats.get("n_moves", 0)
        if not n_moves:
            out[player] = {"rank": None, "n_moves": 0}
            continue
        sums = player_stats.get("loglik_sums", [])[:n_ranks]
        sums = sums + [math.log(EPS) * n_moves] * (n_ranks - len(sums))
        means = [s / n_moves for s in sums]
        best = max(range(n_ranks), key=lambda i: means[i])
        out[player] = {
            "rank": ranks[best].replace("rank_", ""),
            "n_moves": n_moves,
            "mean_loglik": means[best],
        }
    return out


def _format_text(result: dict) -> str:
    pred = result["prediction"]
    players = result.get("players", {})

    def fmt(p: str) -> str:
        info = pred[p]
        pi = players.get(p, {})
        name = pi.get("name") or "?"
        rated = f" [{pi['rating']}]" if pi.get("rating") else ""
        if info["rank"] is None:
            return f"{name}{rated} -"
        return f"{name}{rated} predicted {info['rank']} ({info['n_moves']} moves)"

    return f"{result['sgf']}\n  Black: {fmt('B')}\n  White: {fmt('W')}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sgf_files", nargs="+")
    p.add_argument("--katago", default="/opt/homebrew/bin/katago")
    p.add_argument("--human-model", required=True)
    p.add_argument("--model", default=None, help="defaults to --human-model")
    p.add_argument("--config", required=True)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    model = args.model or args.human_model

    targets = []
    for sgf in args.sgf_files:
        if not os.path.isfile(sgf):
            print(f"warning: {sgf} not found, skipping", file=sys.stderr)
            continue
        sha = _sha_file(sgf)
        cp = _cache_path(sgf, sha)
        cached = None if args.no_cache else _load_cache(cp)
        targets.append((sgf, cp, cached))

    need_engine = any(c is None for _, _, c in targets)

    client = None
    if need_engine:
        cfg = KataGoConfig(
            katago=args.katago, model=model, human_model=args.human_model, config=args.config
        )
        client = KataGoClient(cfg)
        client.start()

    results = []
    try:
        for sgf, cp, cached in targets:
            if cached is None:
                print(f"analyzing {sgf} ...", file=sys.stderr)
                t0 = time.time()
                game = load_sgf(sgf)
                data = _analyze_game(game, client)
                print(
                    f"  done in {time.time() - t0:.1f}s, {data['total_moves']} moves",
                    file=sys.stderr,
                )
                _save_cache(cp, data)
            else:
                if _cache_needs_compaction(cached):
                    cached = _compact_cache(cached)
                    _save_cache(cp, cached)
                data = cached
            pred = _predict_per_player(data)
            result = {
                "sgf": sgf,
                "players": data.get("players", {}),
                "prediction": pred,
            }
            results.append(result)
            if not args.json:
                print(_format_text(result), flush=True)
    finally:
        if client is not None:
            client.shutdown()

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
