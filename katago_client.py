"""Minimal KataGo analysis-engine client.

Speaks the JSONL protocol of `katago analysis` over stdin/stdout. No katrain
deps, no Kivy, no GameNode. Threadsafe send_query with id-based callback
dispatch. Designed for human SL policy queries (maxVisits=1) but works for
any analysis query.

Usage:
    cfg = KataGoConfig(
        katago="/path/to/katago",
        model="/path/to/any-katago-net.bin.gz",
        human_model="/path/to/b18c384nbt-humanv0.bin.gz",
        config="/path/to/analysis_example.cfg",
    )
    with KataGoClient(cfg) as client:
        done = threading.Event()
        result = {}
        def cb(msg):
            result.update(msg)
            done.set()
        client.send_query({
            "rules": "japanese",
            "boardXSize": 19, "boardYSize": 19, "komi": 6.5,
            "initialStones": [], "moves": [["B", "Q16"], ["W", "D4"]],
            "analyzeTurns": [2], "maxVisits": 1,
            "includePolicy": True,
            "overrideSettings": {"humanSLProfile": "rank_3d"},
        }, cb)
        done.wait(timeout=30)
        print(result["humanPolicy"][:5])
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class KataGoConfig:
    katago: str
    model: str  # regular/full KataGo net for the analysis -model flag
    human_model: str
    config: str
    extra_args: list[str] = field(default_factory=list)


CallbackT = Callable[[dict], None]


class KataGoClient:
    """One subprocess, three threads (stdin writer, stdout reader, stderr reader)."""

    def __init__(
        self,
        cfg: KataGoConfig,
        on_stderr: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = cfg
        self.on_stderr = on_stderr or (
            lambda line: print(f"[katago] {line}", file=sys.stderr)
        )
        self._proc: Optional[subprocess.Popen] = None
        self._queries: dict[str, tuple[CallbackT, Optional[CallbackT], float]] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._writeq: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._shutdown = False

    def start(self) -> None:
        cmd = [
            self.cfg.katago,
            "analysis",
            "-model",
            self.cfg.model,
            "-human-model",
            self.cfg.human_model,
            "-config",
            self.cfg.config,
            *self.cfg.extra_args,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for target in (self._read_stdout, self._read_stderr, self._write_stdin):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def shutdown(self) -> None:
        self._shutdown = True
        proc = self._proc
        if proc:
            self._proc = None
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        for t in self._threads:
            t.join(timeout=1)

    def send_query(
        self,
        query: dict,
        callback: CallbackT,
        error_callback: Optional[CallbackT] = None,
    ) -> str:
        with self._lock:
            self._counter += 1
            qid = f"q{self._counter}"
            query = {**query, "id": qid}
            self._queries[qid] = (callback, error_callback, time.time())
        self._writeq.put(query)
        return qid

    def pending(self) -> int:
        with self._lock:
            return len(self._queries)

    def _write_stdin(self) -> None:
        while not self._shutdown:
            try:
                q = self._writeq.get(timeout=0.1)
            except queue.Empty:
                continue
            proc = self._proc
            if proc is None or proc.stdin is None:
                return
            try:
                proc.stdin.write((json.dumps(q) + "\n").encode())
                proc.stdin.flush()
            except (OSError, ValueError):
                return

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._shutdown:
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.on_stderr(f"non-json from katago: {line!r}")
                continue
            qid = msg.get("id")
            with self._lock:
                entry = self._queries.get(qid)
            if entry is None:
                continue
            cb, ecb, _t0 = entry
            if "error" in msg:
                with self._lock:
                    self._queries.pop(qid, None)
                if ecb:
                    ecb(msg)
                else:
                    self.on_stderr(f"katago error for {qid}: {msg.get('error')}")
                continue
            if "warning" in msg:
                self.on_stderr(f"katago warning for {qid}: {msg.get('warning')}")
                continue
            if msg.get("isDuringSearch"):
                continue
            with self._lock:
                self._queries.pop(qid, None)
            try:
                cb(msg)
            except Exception as e:
                self.on_stderr(f"callback error for {qid}: {e!r}")

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            if self._shutdown:
                return
            self.on_stderr(line.decode(errors="ignore").rstrip())

    def __enter__(self) -> "KataGoClient":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.shutdown()


HUMAN_RANKS = [f"rank_{k}k" for k in range(20, 0, -1)] + [
    f"rank_{d}d" for d in range(1, 7)
]


def build_rank_policy_queries(
    moves: list[tuple[str, str]],
    *,
    initial_stones: list[tuple[str, str]] | None = None,
    board_size: tuple[int, int] = (19, 19),
    komi: float = 6.5,
    rules: str = "japanese",
    initial_player: str = "B",
    ranks: list[str] = HUMAN_RANKS,
) -> list[dict]:
    """Build one query per rank for the position *after* `moves`.

    `moves` and `initial_stones` are (player, gtp) pairs; player is "B" or "W".
    Returns query dicts; the caller adds `id` via send_query and a callback.
    """
    base = {
        "rules": rules,
        "boardXSize": board_size[0],
        "boardYSize": board_size[1],
        "komi": komi,
        "initialStones": [list(s) for s in (initial_stones or [])],
        "initialPlayer": initial_player,
        "moves": [list(m) for m in moves],
        "analyzeTurns": [len(moves)],
        "maxVisits": 1,
        "includePolicy": True,
        "includeOwnership": False,
        "includeMovesOwnership": False,
    }
    return [
        {
            **base,
            "overrideSettings": {
                "humanSLProfile": rank,
                "ignorePreRootHistory": False,
                "nnRandomize": False,
            },
        }
        for rank in ranks
    ]


def build_score_query(
    moves: list[tuple[str, str]],
    *,
    initial_stones: list[tuple[str, str]] | None = None,
    board_size: tuple[int, int] = (19, 19),
    komi: float = 6.5,
    rules: str = "japanese",
    initial_player: str = "B",
    max_visits: int = 64,
    allowed_player: str | None = None,
    allowed_moves: list[str] | None = None,
) -> dict:
    """Build a KataGo search query for the position after `moves`.

    The response's moveInfos provide scoreLead for candidate moves. The score is
    reported from Black's perspective when the config uses reportAnalysisWinratesAs
    BLACK, which the bundled server config does.
    """
    query = {
        "rules": rules,
        "boardXSize": board_size[0],
        "boardYSize": board_size[1],
        "komi": komi,
        "initialStones": [list(s) for s in (initial_stones or [])],
        "initialPlayer": initial_player,
        "moves": [list(m) for m in moves],
        "analyzeTurns": [len(moves)],
        "maxVisits": max_visits,
        "includePolicy": False,
        "includeOwnership": False,
        "includeMovesOwnership": False,
    }
    if allowed_player is not None and allowed_moves:
        query["allowMoves"] = [
            {"player": allowed_player, "moves": allowed_moves, "untilDepth": 1}
        ]
    return query
