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
    build_score_query,
)
from sgf_loader import LoadedGame, gtp_to_index, index_to_gtp, load_sgf

CACHE_VERSION = 2
CACHE_COMPAT_VERSIONS = {1, CACHE_VERSION}
CACHE_DIRNAME = ".rank_mle_cache"
EPS = 1e-7
IMPROVEMENTS_VERSION = 5
DEFAULT_IMPROVEMENT_VISITS = 64
DEFAULT_IMPROVEMENT_TOP_POLICY = 5
DEFAULT_IMPROVEMENT_MIN_POLICY = 0.01
DEFAULT_ALTERNATIVE_MIN_POLICY = 0.05
DEFAULT_IMPROVEMENT_TOP_N = 10
DEFAULT_IMPROVEMENT_POINT_THRESHOLD = 1.0


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
        "improvements": data.get("improvements"),
    }


def _improvement_cache_key(
    *,
    visits: int,
    top_policy: int,
    min_policy: float,
    top_n: int,
    point_threshold: float,
) -> str:
    return (
        f"v{IMPROVEMENTS_VERSION}:visits={visits}:top_policy={top_policy}:"
        f"min_policy={min_policy:g}:top_n={top_n}:"
        f"point_threshold={point_threshold:g}"
    )


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
    include_improvements: bool = False,
    improvement_visits: int = DEFAULT_IMPROVEMENT_VISITS,
    improvement_top_policy: int = DEFAULT_IMPROVEMENT_TOP_POLICY,
    improvement_top_n: int = DEFAULT_IMPROVEMENT_TOP_N,
    improvement_point_threshold: float = DEFAULT_IMPROVEMENT_POINT_THRESHOLD,
    progress_cb=None,
) -> dict:
    """Analyze an SGF, returning the cache-shaped data dict.

    Hits/writes the sidecar cache unless use_cache is False. progress_cb is
    called as (done, total) during analysis.
    """
    sha = _sha_file(sgf_path)
    cp = _cache_path(sgf_path, sha)
    improvement_key = _improvement_cache_key(
        visits=improvement_visits,
        top_policy=improvement_top_policy,
        min_policy=DEFAULT_IMPROVEMENT_MIN_POLICY,
        top_n=improvement_top_n,
        point_threshold=improvement_point_threshold,
    )
    if use_cache:
        cached = _load_cache(cp)
        if cached is not None:
            if _cache_needs_compaction(cached):
                cached = _compact_cache(cached)
                _save_cache(cp, cached)
            cached_improvements = cached.get("improvements") or {}
            if (
                include_improvements
                and cached_improvements.get("cache_key") != improvement_key
            ):
                game = load_sgf(sgf_path)
                pred = _predict_per_player(cached)
                cached["improvements"] = {
                    "cache_key": improvement_key,
                    **_compute_improvements(
                        game,
                        client,
                        pred,
                        visits=improvement_visits,
                        top_policy=improvement_top_policy,
                        top_n=improvement_top_n,
                        point_threshold=improvement_point_threshold,
                        progress_cb=progress_cb,
                    ),
                }
                _save_cache(cp, cached)
            elif progress_cb is not None:
                progress_cb(1, 1)
            return cached
    game = load_sgf(sgf_path)
    data = _analyze_game(game, client, progress_cb=progress_cb)
    if include_improvements:
        pred = _predict_per_player(data)
        data["improvements"] = {
            "cache_key": improvement_key,
            **_compute_improvements(
                game,
                client,
                pred,
                visits=improvement_visits,
                top_policy=improvement_top_policy,
                top_n=improvement_top_n,
                point_threshold=improvement_point_threshold,
                progress_cb=progress_cb,
            ),
        }
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


def _rank_name_to_index(rank: str | None) -> int | None:
    if rank is None:
        return None
    full = rank if rank.startswith("rank_") else f"rank_{rank}"
    try:
        return HUMAN_RANKS.index(full)
    except ValueError:
        return None


def _response_move_scores(msg: dict) -> dict[str, float]:
    scores = {}
    for info in msg.get("moveInfos") or []:
        move = info.get("move")
        score = info.get("scoreLead")
        if move is not None and score is not None:
            scores[move] = float(score)
    return scores


def _top_policy_moves(
    policy: list,
    board_size: tuple[int, int],
    limit: int,
    min_probability: float = DEFAULT_IMPROVEMENT_MIN_POLICY,
) -> list[str]:
    moves = []
    for idx, _prob in sorted(
        ((i, float(p)) for i, p in enumerate(policy) if p and p > 0),
        key=lambda t: t[1],
        reverse=True,
    ):
        if _prob < min_probability:
            break
        try:
            moves.append(index_to_gtp(idx, board_size))
        except ValueError:
            continue
        if len(moves) >= limit:
            break
    return moves


def _ordered_unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _score_improvements(
    game: LoadedGame,
    prediction: dict,
    policies_by_move: dict[int, dict[str, list]],
    scores_by_move: dict[int, dict[str, float]],
    *,
    top_policy: int = DEFAULT_IMPROVEMENT_TOP_POLICY,
    min_policy: float = DEFAULT_IMPROVEMENT_MIN_POLICY,
    top_n: int = DEFAULT_IMPROVEMENT_TOP_N,
    point_threshold: float = DEFAULT_IMPROVEMENT_POINT_THRESHOLD,
) -> dict:
    """Score per-move alternatives from temporary policy arrays and search scores."""
    n_ranks = len(HUMAN_RANKS)
    out = {}
    for player, label in (("B", "Black"), ("W", "White")):
        rank_idx = _rank_name_to_index((prediction.get(player) or {}).get("rank"))
        if rank_idx is None:
            out[player] = {
                "player": label,
                "rank": None,
                "target_rank": None,
                "scored_moves": 0,
                "total_moves": 0,
                "moves": [],
            }
            continue
        target_idx = rank_idx + 2
        if target_idx >= n_ranks:
            out[player] = {
                "player": label,
                "rank": HUMAN_RANKS[rank_idx].replace("rank_", ""),
                "target_rank": None,
                "scored_moves": 0,
                "total_moves": 0,
                "moves": [],
            }
            continue

        rank = HUMAN_RANKS[rank_idx]
        target_rank = HUMAN_RANKS[target_idx]
        sign = 1 if player == "B" else -1
        total_moves = 0
        scored_moves = 0
        mistakes = []

        for move_idx, (move_player, played_gtp) in enumerate(game.moves):
            if move_player != player:
                continue
            total_moves += 1
            policies = policies_by_move.get(move_idx, {})
            pol_r = policies.get(rank)
            pol_r2 = policies.get(target_rank)
            move_scores = scores_by_move.get(move_idx, {})
            played_score = move_scores.get(played_gtp)
            if not pol_r or not pol_r2 or played_score is None:
                continue
            scored_moves += 1

            played_idx = gtp_to_index(played_gtp, game.board_size)
            played_p_r = float(pol_r[played_idx]) if played_idx < len(pol_r) else 0.0
            played_p_r2 = (
                float(pol_r2[played_idx]) if played_idx < len(pol_r2) else 0.0
            )
            played_gain_pp = (played_p_r2 - played_p_r) * 100.0

            top_alts = sorted(
                (
                    (i, float(p))
                    for i, p in enumerate(pol_r2)
                    if p and p >= min_policy
                ),
                key=lambda t: t[1],
                reverse=True,
            )[:top_policy]

            alternatives = []
            for alt_idx, alt_p_r2 in top_alts:
                try:
                    alt_gtp = index_to_gtp(alt_idx, game.board_size)
                except ValueError:
                    continue
                if alt_gtp == played_gtp:
                    continue
                if alt_p_r2 <= DEFAULT_ALTERNATIVE_MIN_POLICY:
                    continue
                alt_score = move_scores.get(alt_gtp)
                if alt_score is None:
                    continue
                point_gain = sign * (alt_score - played_score)
                if point_gain < point_threshold:
                    continue
                alt_p_r = float(pol_r[alt_idx]) if alt_idx < len(pol_r) else 0.0
                alt_gain_pp = (alt_p_r2 - alt_p_r) * 100.0
                alternatives.append(
                    {
                        "move": alt_gtp,
                        "kind": "candidate",
                        "point_gain": point_gain,
                        "score": alt_score,
                        "p_rank": alt_p_r * 100.0,
                        "p_target": alt_p_r2 * 100.0,
                        "gain_pp": alt_gain_pp,
                    }
                )

            if alternatives:
                most_likely = max(alternatives, key=lambda alt: alt["p_target"])
                best_gain = max(alternatives, key=lambda alt: alt["gain_pp"])
                selected = []
                for kind, alt in (
                    ("most_human_likely", most_likely),
                    ("biggest_human_gain", best_gain),
                ):
                    if any(existing["move"] == alt["move"] for existing in selected):
                        continue
                    selected.append({**alt, "kind": kind})

                first = selected[0]
                mistakes.append(
                    {
                        "move_num": move_idx + 1,
                        "player": player,
                        "played": played_gtp,
                        "alternative": first["move"],
                        "rank": rank.replace("rank_", ""),
                        "target_rank": target_rank.replace("rank_", ""),
                        "ranking_pp": -played_gain_pp,
                        "played_gain_pp": played_gain_pp,
                        "alternatives": selected,
                        "played_score": played_score,
                        "played_p_rank": played_p_r * 100.0,
                        "played_p_target": played_p_r2 * 100.0,
                    }
                )

        mistakes.sort(key=lambda m: m["ranking_pp"], reverse=True)
        out[player] = {
            "player": label,
            "rank": rank.replace("rank_", ""),
            "target_rank": target_rank.replace("rank_", ""),
            "scored_moves": scored_moves,
            "total_moves": total_moves,
            "moves": mistakes[:top_n],
        }
    return out


def _compute_improvements(
    game: LoadedGame,
    client: KataGoClient,
    prediction: dict,
    *,
    visits: int = DEFAULT_IMPROVEMENT_VISITS,
    top_policy: int = DEFAULT_IMPROVEMENT_TOP_POLICY,
    min_policy: float = DEFAULT_IMPROVEMENT_MIN_POLICY,
    top_n: int = DEFAULT_IMPROVEMENT_TOP_N,
    point_threshold: float = DEFAULT_IMPROVEMENT_POINT_THRESHOLD,
    progress_cb=None,
) -> dict:
    rank_indices = {}
    for player in ("B", "W"):
        rank_idx = _rank_name_to_index((prediction.get(player) or {}).get("rank"))
        if rank_idx is not None and rank_idx + 2 < len(HUMAN_RANKS):
            rank_indices[player] = (rank_idx, rank_idx + 2)

    policy_requests = []
    for move_idx, (player, _gtp) in enumerate(game.moves):
        if player not in rank_indices:
            continue
        r1, r2 = rank_indices[player]
        policy_requests.append((move_idx, HUMAN_RANKS[r1]))
        policy_requests.append((move_idx, HUMAN_RANKS[r2]))

    total_expected = len(policy_requests)
    if total_expected == 0:
        return {
            "version": IMPROVEMENTS_VERSION,
            "visits": visits,
            "top_policy": top_policy,
            "min_policy": min_policy,
            "top_n": top_n,
            "point_threshold": point_threshold,
            "players": _score_improvements(
                game,
                prediction,
                {},
                {},
                top_policy=top_policy,
                min_policy=min_policy,
                top_n=top_n,
                point_threshold=point_threshold,
            ),
        }

    lock = threading.Lock()
    done_count = [0]
    error_count = [0]
    done_event = threading.Event()
    first_error: list[str | None] = [None]
    policies_by_move: dict[int, dict[str, list]] = {}
    scores_by_move: dict[int, dict[str, float]] = {}

    def mark_done() -> None:
        with lock:
            done_count[0] += 1
            done = done_count[0]
            if progress_cb is not None:
                progress_cb(done, total_expected)
            if done >= total_expected:
                done_event.set()

    def err_cb(msg: dict):
        err = msg.get("error", "unknown error")
        print(f"  katago improvement error: {err}", file=sys.stderr)
        with lock:
            error_count[0] += 1
            if first_error[0] is None:
                first_error[0] = err
        mark_done()

    def make_policy_cb(move_idx: int, rank: str):
        def cb(msg: dict):
            policy = list(msg.get("humanPolicy") or [])
            with lock:
                policies_by_move.setdefault(move_idx, {})[rank] = policy
            mark_done()

        return cb

    for move_idx, rank in policy_requests:
        prefix = game.moves[:move_idx]
        query = build_rank_policy_queries(
            prefix,
            initial_stones=game.initial_stones,
            board_size=game.board_size,
            komi=game.komi,
            rules=game.rules,
            initial_player=game.initial_player,
            ranks=[rank],
        )[0]
        client.send_query(query, make_policy_cb(move_idx, rank), err_cb)

    while not done_event.is_set():
        if not client.is_alive():
            raise RuntimeError("KataGo process died during improvement analysis")
        done_event.wait(timeout=0.5)

    if error_count[0] == total_expected:
        raise RuntimeError(
            f"KataGo rejected all {total_expected} improvement policy queries — {first_error[0]!r}"
        )

    score_requests = []
    with lock:
        for move_idx, (player, played_gtp) in enumerate(game.moves):
            if player not in rank_indices:
                continue
            _r1, r2 = rank_indices[player]
            target_rank = HUMAN_RANKS[r2]
            target_policy = policies_by_move.get(move_idx, {}).get(target_rank)
            if not target_policy:
                continue
            forced_moves = _ordered_unique(
                [played_gtp]
                + _top_policy_moves(
                    target_policy,
                    game.board_size,
                    top_policy,
                    min_probability=min_policy,
                )
            )
            score_requests.append((move_idx, player, forced_moves))

        done_count[0] = 0
        error_count[0] = 0
        first_error[0] = None
        total_expected = len(score_requests)
        done_event.clear()

    if total_expected == 0:
        players = _score_improvements(
            game,
            prediction,
            policies_by_move,
            scores_by_move,
            top_policy=top_policy,
            min_policy=min_policy,
            top_n=top_n,
            point_threshold=point_threshold,
        )
        return {
            "version": IMPROVEMENTS_VERSION,
            "visits": visits,
            "top_policy": top_policy,
            "min_policy": min_policy,
            "top_n": top_n,
            "point_threshold": point_threshold,
            "players": players,
        }

    def make_score_cb(move_idx: int):
        def cb(msg: dict):
            with lock:
                scores_by_move[move_idx] = _response_move_scores(msg)
            mark_done()

        return cb

    for move_idx, player, forced_moves in score_requests:
        prefix = game.moves[:move_idx]
        query = build_score_query(
            prefix,
            initial_stones=game.initial_stones,
            board_size=game.board_size,
            komi=game.komi,
            rules=game.rules,
            initial_player=game.initial_player,
            max_visits=visits,
            allowed_player=player,
            allowed_moves=forced_moves,
        )
        client.send_query(query, make_score_cb(move_idx), err_cb)

    while not done_event.is_set():
        if not client.is_alive():
            raise RuntimeError("KataGo process died during improvement analysis")
        done_event.wait(timeout=0.5)

    if error_count[0] == total_expected:
        raise RuntimeError(
            f"KataGo rejected all {total_expected} improvement score queries — {first_error[0]!r}"
        )

    return {
        "version": IMPROVEMENTS_VERSION,
        "visits": visits,
        "top_policy": top_policy,
        "min_policy": min_policy,
        "top_n": top_n,
        "point_threshold": point_threshold,
        "players": _score_improvements(
            game,
            prediction,
            policies_by_move,
            scores_by_move,
            top_policy=top_policy,
            min_policy=min_policy,
            top_n=top_n,
            point_threshold=point_threshold,
        ),
    }


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

    lines = [f"{result['sgf']}", f"  Black: {fmt('B')}", f"  White: {fmt('W')}"]
    improvements = result.get("improvements")
    if improvements:
        for color, label in (("B", "Black"), ("W", "White")):
            info = (improvements.get("players") or {}).get(color) or {}
            moves = info.get("moves") or []
            target = info.get("target_rank")
            if not target:
                continue
            lines.append(
                f"  {label} moves to review ({info.get('rank')} -> {target}, "
                f"{info.get('scored_moves', 0)}/{info.get('total_moves', 0)} scored):"
            )
            for m in moves[:5]:
                alts = m.get("alternatives") or [{"move": m["alternative"]}]
                alt_text = ", ".join(alt["move"] for alt in alts)
                lines.append(
                    f"    move {m['move_num']}: {m['played']} ({m.get('played_gain_pp', 0.0):+.1f}pp) "
                    f"-> {alt_text}"
                )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sgf_files", nargs="+")
    p.add_argument("--katago", default="/opt/homebrew/bin/katago")
    p.add_argument("--human-model", required=True)
    p.add_argument("--model", default=None, help="defaults to --human-model")
    p.add_argument("--config", required=True)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--improvements",
        action="store_true",
        help="also find stronger-rank alternative moves; slower and not run by default",
    )
    p.add_argument(
        "--improvement-visits",
        type=int,
        default=DEFAULT_IMPROVEMENT_VISITS,
        help=f"KataGo visits for alternative-move scoring (default {DEFAULT_IMPROVEMENT_VISITS})",
    )
    args = p.parse_args(argv)

    model = args.model or args.human_model
    if args.improvements and args.model is None:
        print(
            "warning: --improvements uses KataGo scoreLead; pass --model with a regular KataGo model for useful scoring",
            file=sys.stderr,
        )

    targets = []
    for sgf in args.sgf_files:
        if not os.path.isfile(sgf):
            print(f"warning: {sgf} not found, skipping", file=sys.stderr)
            continue
        sha = _sha_file(sgf)
        cp = _cache_path(sgf, sha)
        cached = None if args.no_cache else _load_cache(cp)
        targets.append((sgf, cp, cached))

    improvement_key = _improvement_cache_key(
        visits=args.improvement_visits,
        top_policy=DEFAULT_IMPROVEMENT_TOP_POLICY,
        min_policy=DEFAULT_IMPROVEMENT_MIN_POLICY,
        top_n=DEFAULT_IMPROVEMENT_TOP_N,
        point_threshold=DEFAULT_IMPROVEMENT_POINT_THRESHOLD,
    )
    need_engine = any(
        c is None
        or (
            args.improvements
            and (c.get("improvements") or {}).get("cache_key") != improvement_key
        )
        for _, _, c in targets
    )

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
                if args.improvements:
                    pred = _predict_per_player(data)
                    print("  finding moves to review ...", file=sys.stderr)
                    data["improvements"] = {
                        "cache_key": improvement_key,
                        **_compute_improvements(
                            game,
                            client,
                            pred,
                            visits=args.improvement_visits,
                        ),
                    }
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
                if args.improvements and (
                    (data.get("improvements") or {}).get("cache_key") != improvement_key
                ):
                    print(f"finding moves to review for {sgf} ...", file=sys.stderr)
                    game = load_sgf(sgf)
                    pred = _predict_per_player(data)
                    data["improvements"] = {
                        "cache_key": improvement_key,
                        **_compute_improvements(
                            game,
                            client,
                            pred,
                            visits=args.improvement_visits,
                        ),
                    }
                    _save_cache(cp, data)
            pred = _predict_per_player(data)
            result = {
                "sgf": sgf,
                "players": data.get("players", {}),
                "prediction": pred,
                "improvements": data.get("improvements") if args.improvements else None,
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
