# rankmle

Predict a Go player's rank from an SGF, using KataGo's human SL model.

For each move in the game, queries KataGo 26 times (one per rank profile
`rank_20k`...`rank_6d`) and records the policy probability of the move that
was actually played. The argmax mean log-likelihood across ranks is the
predicted rank for that player.

Two interfaces:

- `python rank_mle.py FILE.sgf [...]` — CLI, with sidecar JSON cache
- `uvicorn server:app` — FastAPI web app with progress bar, queue tracking,
  SGF validation, and warnings for too-short or non-medium-time-control games

## Requirements

You provide:

1. A `katago` binary — [release downloads](https://github.com/lightvector/KataGo/releases),
   or build from source. The Eigen (CPU) build works but is ~10× slower than
   GPU; OpenCL/CUDA strongly recommended.
2. A KataGo human SL model — typically `b18c384nbt-humanv0.bin.gz`. Download
   from the [KataGo human SL models page](https://katagotraining.org/networks).
3. A regular/full KataGo model for the `-model` flag, such as a current `kata1`
   network from the [KataGo networks page](https://katagotraining.org/networks).
4. An analysis config — KataGo's `configs/analysis_example.cfg` works as-is.

KataGo now requires `-model` to point at a regular/full model. Do not pass the
human SL model for both `-model` and `-human-model`.

## Local quickstart

```bash
uv sync

# Create .env once, then omit the repeated KataGo flags.
cat > .env <<'EOF'
KATAGO_BIN=/opt/homebrew/bin/katago
KATAGO_MODEL=~/models/kata1.bin.gz
KATAGO_HUMAN_MODEL=~/models/b18c384nbt-humanv0.bin.gz
KATAGO_CONFIG=~/configs/analysis_example.cfg
KGS_RANK=3k
EOF

# CLI
uv run python rank_mle.py path/to/game.sgf

# Optional: also find moves to review. This is slower.
uv run python rank_mle.py path/to/game.sgf --improvements

# Review one game. --kgs-rank can also be omitted when KGS_RANK is set.
uv run python review-game.py path/to/game.sgf --csv path/to/review.csv

# Web app
uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Paste an SGF, paste an
[online-go.com](https://online-go.com/) game link (e.g.
`https://online-go.com/game/86429302`), or upload a file. Analysis takes
roughly 1–2 minutes per 250-move game on a warm GPU.

The web app can optionally find moves to review. That path compares each
player's predicted rank policy to the policy two ranks stronger, asks KataGo to
score the played move plus up to 5 stronger-rank policy moves with at least 1%
policy probability. Moves are ranked by how much the played move's policy drops
at the stronger rank. For each reviewed move, it shows the most likely
stronger-rank human move and the move with the biggest human-policy gain,
limited to alternatives above 5% policy that score at least one point better.
Annotated SGFs add those suggestions as variations at the original move. Only
the compact scored results are cached; temporary policy arrays are discarded.

## Environment Variables

| Var                  | Default                            | Notes                           |
| -------------------- | ---------------------------------- | ------------------------------- |
| `KATAGO_BIN`         | `/opt/homebrew/bin/katago`         | Path to katago binary           |
| `KATAGO_HUMAN_MODEL` | _(required)_                       | Human SL `.bin.gz`              |
| `KATAGO_MODEL`       | _(required)_                       | Regular/full KataGo net         |
| `KATAGO_CONFIG`      | _(required for CLI; built-in server default)_ | Analysis config      |
| `KGS_RANK`           | _(required for `review-game.py`)_  | Default for `--kgs-rank`        |
| `UPLOAD_DIR`         | `./uploads`                        | Where submitted SGFs are stored |

The CLI and server load these from `.env` automatically. Command-line flags
still override `.env` values.

## API

```
POST /analyze
  multipart: sgf_file=@game.sgf
        OR: sgf_text=(SGF body | online-go.com game URL | OGS game ID)
  → 200 {job_id, sgf_sha, n_moves, warnings, queue: {position, length}}
  → 400 {detail: "..."}    on validation failure

  When sgf_text is an OGS link or numeric game ID, the server fetches
  https://online-go.com/api/v1/games/{id}/sgf and analyzes the result.

GET /jobs/{job_id}
  → {status, progress: {done, total},  // in moves
     result: {players, prediction: {B: {rank, n_moves, mean_loglik}, W: ...}},
     error, warnings, n_moves, queue}

GET /healthz
  → {alive: bool}
```

`status` is one of `queued`, `running`, `done`, `error`. `queue.position` is
0 for the running job, 1+ for jobs in line, -1 once the job has finished.

## Deployment

Pick the option that matches where you're hosting.

### Native — Apple Silicon Mac (recommended for self-hosting)

Hosting on your own Mac? Run natively. **Don't use Docker** — Docker Desktop on
Mac runs Linux containers in a VM with no Metal / OpenCL access, so KataGo
would fall back to CPU and lose ~5–10× throughput. Native macOS gets full
GPU acceleration through OpenCL.

```bash
cd rankmle
uv sync
KATAGO_BIN=/opt/homebrew/bin/katago \
KATAGO_HUMAN_MODEL=~/.katrain/b18c384nbt-humanv0.bin.gz \
KATAGO_CONFIG=~/path/to/analysis_example.cfg \
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

To run as a background service that survives reboots, use **launchd**: drop a
plist at `~/Library/LaunchAgents/com.yourname.rankmle.plist` invoking the
above command. `man launchd.plist` covers the format.

### Native — Linux box with GPU

Same idea — install KataGo built against your GPU backend (CUDA / OpenCL /
TensorRT), then `uv sync && uv run uvicorn ...`. Use systemd for process
management.

### Docker — Linux + NVIDIA GPU

For a real cloud deployment with CUDA, the Dockerfile needs two changes:

1. Use a CUDA base image for the runtime stage:
   ```dockerfile
   FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04 AS runtime
   # then reinstall python3.12 + uv
   ```
2. Override `KATAGO_URL` at build time with the matching release:
   ```bash
   docker build -t rankmle \
     --build-arg KATAGO_URL=https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda12.5-cudnn9.2.1-linux-x64.zip \
     .
   docker run --rm --gpus all -p 8000:8000 \
     -v ~/models:/models:ro -v ~/configs:/config:ro -v rankmle-data:/data \
     rankmle
   ```

The supplied Dockerfile is a CPU starting point — wire up the CUDA bits when
you actually have a GPU host.

## Layout

```
katago_client.py   subprocess + JSONL protocol, ~210 lines
sgf_loader.py      sgfmill-based SGF → moves
ogs.py             parse online-go.com URLs and fetch SGFs from the OGS API
rank_mle.py        CLI; analyze_path() is the importable entry point
server.py          FastAPI app + inline browser UI
test_smoke.py      one-query sanity test
```

## Development

```bash
uv sync
uv run ruff check .
uv run python test_smoke.py   # one-shot KataGo round-trip
```
