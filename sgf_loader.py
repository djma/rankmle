"""SGF parsing using sgfmill — extract metadata and the move sequence as
(player, gtp) tuples for KataGo's analysis API.
"""

from __future__ import annotations

from dataclasses import dataclass

from sgfmill import sgf

_GTP_COLS = "ABCDEFGHJKLMNOPQRST"

# Map common SGF RU abbreviations to the strings KataGo accepts.
_RULES_MAP = {
    "jp": "japanese",
    "cn": "chinese",
    "tt": "tromp-taylor",
    "kr": "korean",
    "nz": "tromp-taylor",
    "aga": "aga",
    "stone-scoring": "stone-scoring",
}


def _normalize_rules(rules: str) -> str:
    return _RULES_MAP.get(rules.lower().strip(), rules.lower().strip())


def _coord_to_gtp(coord: tuple[int, int] | None, board_size: int) -> str:
    if coord is None:
        return "pass"
    row, col = coord
    return f"{_GTP_COLS[col]}{row + 1}"


def gtp_to_index(gtp: str, board_size: tuple[int, int]) -> int:
    """Index into the policy array for a GTP coord. Last index is pass."""
    bx, by = board_size
    if gtp == "pass":
        return bx * by
    col = _GTP_COLS.index(gtp[0])
    row = int(gtp[1:]) - 1
    return (by - 1 - row) * bx + col


@dataclass
class LoadedGame:
    sgf_path: str
    board_size: tuple[int, int]
    komi: float
    rules: str
    initial_player: str
    initial_stones: list[tuple[str, str]]
    moves: list[tuple[str, str]]
    players: dict


def load_sgf_bytes(body: bytes) -> sgf.Sgf_game:
    """Parse SGF bytes, treating missing CA as UTF-8 when possible.

    SGF defaults a missing CA property to ISO-8859-1, but many modern servers
    emit UTF-8 SGFs without CA. Prefer the declared encoding when present; for
    undeclared files, use UTF-8 if it parses cleanly and otherwise keep
    sgfmill's standards-compliant default.
    """
    game = sgf.Sgf_game.from_bytes(body)
    try:
        game.get_root().get_raw("CA")
        return game
    except KeyError:
        pass

    try:
        return sgf.Sgf_game.from_bytes(body, override_encoding="UTF-8")
    except Exception:
        return game


def load_sgf(path: str) -> LoadedGame:
    with open(path, "rb") as f:
        game = load_sgf_bytes(f.read())

    bsz = game.get_size()
    root = game.get_root()
    komi = game.get_komi()
    try:
        rules = root.get("RU")
    except KeyError:
        rules = "japanese"

    setup_b, setup_w, _ = root.get_setup_stones()
    initial_stones: list[tuple[str, str]] = []
    for r, c in sorted(setup_b):
        initial_stones.append(("B", _coord_to_gtp((r, c), bsz)))
    for r, c in sorted(setup_w):
        initial_stones.append(("W", _coord_to_gtp((r, c), bsz)))

    initial_player = "B"
    try:
        pl = root.get("PL")
        if pl in ("B", "W"):
            initial_player = pl
    except KeyError:
        pass

    moves: list[tuple[str, str]] = []
    for node in game.get_main_sequence()[1:]:
        colour, coord = node.get_move()
        if colour is None:
            continue
        moves.append((colour.upper(), _coord_to_gtp(coord, bsz)))

    def _safe(prop: str) -> str:
        try:
            return root.get(prop) or ""
        except KeyError:
            return ""

    players = {
        "B": {"name": _safe("PB"), "rating": _safe("BR")},
        "W": {"name": _safe("PW"), "rating": _safe("WR")},
    }

    return LoadedGame(
        sgf_path=path,
        board_size=(bsz, bsz),
        komi=float(komi) if komi is not None and abs(float(komi)) <= 50 else 6.5,
        rules=_normalize_rules(rules) if isinstance(rules, str) else "japanese",
        initial_player=initial_player,
        initial_stones=initial_stones,
        moves=moves,
        players=players,
    )
