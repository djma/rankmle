"""Smoke test: launch katago and fire one rank-policy query.

Env vars:
    KATAGO_BIN          path to katago binary (default /opt/homebrew/bin/katago)
    KATAGO_MODEL        path to regular/full KataGo .bin.gz (required)
    KATAGO_HUMAN_MODEL  path to human SL .bin.gz (required)
    KATAGO_CONFIG       path to analysis config (optional; a minimal one is written to a tempfile if unset)
"""

import os
import tempfile
import threading

from katago_client import KataGoClient, KataGoConfig, build_rank_policy_queries

_MINIMAL_CONFIG = "numSearchThreads = 1\n"

human_model = os.environ.get("KATAGO_HUMAN_MODEL")
if not human_model:
    raise SystemExit("KATAGO_HUMAN_MODEL env var is required")
model = os.environ.get("KATAGO_MODEL")
if not model:
    raise SystemExit("KATAGO_MODEL env var is required")

config_path = os.environ.get("KATAGO_CONFIG")
_tmp = None
if not config_path:
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False)
    _tmp.write(_MINIMAL_CONFIG)
    _tmp.close()
    config_path = _tmp.name

try:
    cfg = KataGoConfig(
        katago=os.environ.get("KATAGO_BIN", "/opt/homebrew/bin/katago"),
        model=model,
        human_model=human_model,
        config=config_path,
    )

    with KataGoClient(cfg) as client:
        moves = [("B", "Q16"), ("W", "D4"), ("B", "Q4"), ("W", "D16")]
        queries = build_rank_policy_queries(moves, ranks=["rank_3d"])
        done = threading.Event()
        result = {}

        def cb(msg):
            result.update(msg)
            done.set()

        def err(msg):
            print(f"ERROR: {msg}")
            done.set()

        print(f"sending query: ranks=[rank_3d], {len(moves)} moves played")
        client.send_query(queries[0], cb, err)
        if not done.wait(timeout=60):
            print("TIMEOUT after 60s")
            raise SystemExit(1)

        if "error" in result:
            print(f"katago returned error: {result['error']}")
            raise SystemExit(1)

        policy = result.get("humanPolicy")
        if not policy:
            print(f"no humanPolicy in response. keys={list(result.keys())}")
            raise SystemExit(1)

        indexed = sorted(enumerate(policy), key=lambda kv: -kv[1])[:5]
        bsz = 19
        for idx, prob in indexed:
            if idx == len(policy) - 1:
                coord = "pass"
            else:
                x = idx % bsz
                y = bsz - 1 - (idx // bsz)
                col = "ABCDEFGHJKLMNOPQRST"[x]
                coord = f"{col}{y + 1}"
            print(f"  {coord:>5}  {prob:.4f}")

        print("OK")
finally:
    if _tmp is not None:
        os.unlink(_tmp.name)
