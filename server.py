"""FastAPI server for rank MLE analysis.

One long-lived KataGoClient. Jobs run serially in a single worker thread —
KataGo's internal numAnalysisThreads already saturates the GPU, so submitting
multiple games in parallel from Python wouldn't speed anything up.

Run:
    uv run uvicorn server:app --host 0.0.0.0 --port 8000

Env:
    KATAGO_BIN          path to katago (default /opt/homebrew/bin/katago)
    KATAGO_HUMAN_MODEL  path to human SL .bin.gz (required)
    KATAGO_MODEL        path to regular/full KataGo model .bin.gz (required)
    KATAGO_CONFIG       path to analysis config (optional; a sensible default is used if unset)
    UPLOAD_DIR          where uploaded SGFs are stored (default ./uploads)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from env_config import env_path
from katago_client import KataGoClient, KataGoConfig
from ogs import fetch_ogs_sgf, parse_ogs_url
from rank_mle import (
    DEFAULT_IMPROVEMENT_VISITS,
    _predict_per_player,
    analyze_path,
)
from sgf_loader import gtp_to_sgf_coord, load_sgf_bytes, normalize_rules

MIN_MOVES = 20
MAX_MOVES = 400
RECOMMENDED_MIN_MOVES = 150
MIN_MAIN_TIME_SEC = 600  # below: blitz
MAX_MAIN_TIME_SEC = 4 * 3600  # above: correspondence
MAX_ACTIVE_JOBS_PER_IP = 5


def _validate_sgf(body: bytes) -> tuple[int, list[str]]:
    """Parse SGF, enforce hard limits, return (n_moves, warnings).

    Raises HTTPException(400) on parse failure or out-of-range move count.
    """
    try:
        game = load_sgf_bytes(body)
    except Exception as e:
        raise HTTPException(400, f"could not parse SGF: {e}") from e

    n_moves = sum(
        1 for node in game.get_main_sequence()[1:] if node.get_move()[0] is not None
    )
    if n_moves < MIN_MOVES:
        raise HTTPException(
            400,
            f"game has only {n_moves} moves; need at least {MIN_MOVES} for a meaningful prediction",
        )
    if n_moves > MAX_MOVES:
        raise HTTPException(
            400,
            f"game has {n_moves} moves; cap is {MAX_MOVES}. Trim and resubmit.",
        )

    warnings: list[str] = []
    if n_moves < RECOMMENDED_MIN_MOVES:
        warnings.append(
            f"only {n_moves} moves — at least {RECOMMENDED_MIN_MOVES} recommended for a stable prediction"
        )

    root = game.get_root()
    main_secs: Optional[float] = None
    try:
        tm_raw = root.get_raw("TM").decode("utf-8", "ignore").strip()
        if tm_raw:
            main_secs = float(tm_raw)
    except (KeyError, ValueError):
        pass

    if main_secs is not None:
        if main_secs < MIN_MAIN_TIME_SEC:
            warnings.append(
                f"fast time setting (main time {main_secs:g}s) — rank predictions are calibrated on medium-length games"
            )
        elif main_secs > MAX_MAIN_TIME_SEC:
            warnings.append(
                f"long time setting (main time {main_secs:g}s) — rank predictions are calibrated on medium-length games"
            )

    try:
        ru_raw = root.get_raw("RU").decode("utf-8", "ignore").strip()
    except KeyError:
        ru_raw = ""
    if ru_raw:
        _, defaulted = normalize_rules(ru_raw)
        if defaulted:
            warnings.append(
                f"unrecognized ruleset {ru_raw!r} — analyzing as Japanese rules"
            )

    return n_moves, warnings


@dataclass
class Job:
    job_id: str
    sgf_path: str
    sgf_sha: str
    status: str = "queued"  # queued | running | done | error
    progress_done: int = 0
    progress_total: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    n_moves: int = 0
    seq: int = 0  # submission order, for queue-position computation
    client_ip: str = ""
    improvement_visits: int = DEFAULT_IMPROVEMENT_VISITS
    improvement_status: str = "idle"  # idle | queued | running | done | error
    improvement_progress_done: int = 0
    improvement_progress_total: int = 0
    improvement_error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
JOB_SEQ = [0]  # monotonic counter under JOBS_LOCK


def _queue_snapshot(target: Optional[Job] = None) -> tuple[int, int]:
    """Return (position, length).

    `length` is the number of jobs queued or running. `position` is the target's
    place in line (0 = running, 1 = next up, ...) or -1 if the target is None
    or not active. Done/error jobs do not count.
    """
    with JOBS_LOCK:
        active = sorted(
            (j for j in JOBS.values() if j.status in ("queued", "running")),
            key=lambda j: j.seq,
        )
    length = len(active)
    if target is None:
        return -1, length
    for i, j in enumerate(active):
        if j.job_id == target.job_id:
            return i, length
    return -1, length


CLIENT: Optional[KataGoClient] = None
EXECUTOR: Optional[ThreadPoolExecutor] = None
UPLOAD_DIR = env_path("UPLOAD_DIR", os.path.abspath("./uploads"))
_CONFIG_TMPFILE: Optional[tempfile.NamedTemporaryFile] = None

_DEFAULT_CONFIG = """\
reportAnalysisWinratesAs = BLACK
conservativePass = true
maxVisits = 1
numAnalysisThreads = 12
numSearchThreads = 8
nnMaxBatchSize = 96
nnCacheSizePowerOfTwo = 20
nnMutexPoolSizePowerOfTwo = 16
nnRandomize = true
"""


def _build_client() -> tuple[KataGoClient, Optional[tempfile.NamedTemporaryFile]]:
    human = env_path("KATAGO_HUMAN_MODEL")
    if not human:
        raise RuntimeError("KATAGO_HUMAN_MODEL env var is required")
    model = env_path("KATAGO_MODEL")
    if not model:
        raise RuntimeError("KATAGO_MODEL env var is required and must point to a regular KataGo model")
    cfg_path = env_path("KATAGO_CONFIG")
    tmp = None
    if not cfg_path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False)
        tmp.write(_DEFAULT_CONFIG)
        tmp.close()
        cfg_path = tmp.name
    return KataGoClient(
        KataGoConfig(
            katago=env_path("KATAGO_BIN", "/opt/homebrew/bin/katago"),
            model=model,
            human_model=human,
            config=cfg_path,
        )
    ), tmp


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return
    with job.lock:
        job.status = "running"

    def progress(done: int, total: int) -> None:
        # Translate query progress to move progress for the UI.
        moves_done = (done * job.n_moves) // total if total else 0
        with job.lock:
            job.progress_done = max(job.progress_done, moves_done)
            job.progress_total = job.n_moves

    try:
        data = analyze_path(
            CLIENT,
            job.sgf_path,
            improvement_visits=job.improvement_visits,
            progress_cb=progress,
        )
        pred = _predict_per_player(data)
        with job.lock:
            job.status = "done"
            job.result = {
                "players": data.get("players", {}),
                "prediction": pred,
                "improvements": None,
            }
    except Exception as e:
        traceback.print_exc()
        with job.lock:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"


def _run_improvement_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return
    with job.lock:
        if job.status != "done" or job.result is None:
            job.improvement_status = "error"
            job.improvement_error = "rank analysis is not complete"
            return
        if job.result.get("improvements") is not None:
            job.improvement_status = "done"
            job.improvement_progress_done = 1
            job.improvement_progress_total = 1
            return
        job.improvement_status = "running"
        job.improvement_error = None

    def progress(done: int, total: int) -> None:
        with job.lock:
            job.improvement_progress_done = done
            job.improvement_progress_total = total

    try:
        data = analyze_path(
            CLIENT,
            job.sgf_path,
            include_improvements=True,
            improvement_visits=job.improvement_visits,
            progress_cb=progress,
        )
        with job.lock:
            if job.result is None:
                job.result = {
                    "players": data.get("players", {}),
                    "prediction": _predict_per_player(data),
                }
            job.result["improvements"] = data.get("improvements")
            job.improvement_status = "done"
            if job.improvement_progress_total == 0:
                job.improvement_progress_done = 1
                job.improvement_progress_total = 1
    except Exception as e:
        traceback.print_exc()
        with job.lock:
            job.improvement_status = "error"
            job.improvement_error = f"{type(e).__name__}: {e}"


def _prediction_comment(result: dict) -> str:
    pred = result.get("prediction", {})
    players = result.get("players", {})
    lines = ["Predicted KGS ranks:"]
    for color, label in (("B", "Black"), ("W", "White")):
        meta = players.get(color) or {}
        info = pred.get(color) or {}
        name = meta.get("name") or "?"
        rank = info.get("rank") or "unknown"
        lines.append(f"{label} ({name}): {rank}")
    return "\n".join(lines)


def _alternative_label(kind: str) -> str:
    if kind == "most_human_likely":
        return "most human-likely"
    if kind == "biggest_human_gain":
        return "biggest human-policy gain"
    return "suggested"


def _alternative_comment(move: dict, alt: dict) -> str:
    return (
        "Rank MLE move to review\n"
        f"Played: {move['played']}\n"
        f"Suggested: {alt['move']} ({_alternative_label(alt.get('kind', ''))})\n"
        f"Policy shift {move['rank']}->{move['target_rank']}: "
        f"played {move['played_p_rank']:.1f}%->{move['played_p_target']:.1f}%, "
        f"suggested {alt['p_rank']:.1f}%->{alt['p_target']:.1f}%\n"
        f"Played move policy gain: {move.get('played_gain_pp', 0.0):+.1f} percentage points\n"
        f"Alternative policy gain: {alt['gain_pp']:+.1f} percentage points\n"
        f"KataGo score gain: +{alt['point_gain']:.1f} points"
    )


def _review_options_comment(move: dict) -> str:
    lines = [
        f"Rank MLE options before move {move['move_num']} ({move['player']})",
        f"Played: {move['played']}  policy {move['played_p_rank']:.1f}%->{move['played_p_target']:.1f}% ({move.get('played_gain_pp', 0.0):+.1f}pp)",
    ]
    for alt in move.get("alternatives") or []:
        lines.append(
            f"{_alternative_label(alt.get('kind', ''))}: {alt['move']}  "
            f"policy {alt['p_rank']:.1f}%->{alt['p_target']:.1f}% "
            f"({alt['gain_pp']:+.1f}pp), score +{alt['point_gain']:.1f}pt"
        )
    return "\n".join(lines)


def _append_node_comment(node, comment: str) -> None:
    try:
        existing = node.get("C")
    except KeyError:
        existing = ""
    node.set("C", f"{existing}\n\n{comment}" if existing else comment)


def _add_improvement_variations(game, result: dict) -> None:
    improvements = result.get("improvements") or {}
    by_player = improvements.get("players") or {}
    review_moves = []
    for color in ("B", "W"):
        review_moves.extend((by_player.get(color) or {}).get("moves") or [])
    if not review_moves:
        return

    main_sequence = game.get_main_sequence()
    for move in sorted(review_moves, key=lambda m: (m["move_num"], m["player"])):
        move_num = int(move.get("move_num", 0))
        if move_num <= 0 or move_num >= len(main_sequence):
            continue
        parent = main_sequence[move_num - 1]
        _append_node_comment(parent, _review_options_comment(move))
        for alt in move.get("alternatives") or []:
            branch = parent.new_child()
            branch.set_move(move["player"].lower(), gtp_to_sgf_coord(alt["move"]))
            branch.set("C", _alternative_comment(move, alt))


def _annotated_sgf_bytes(job: Job) -> bytes:
    if job.result is None:
        raise HTTPException(409, "job is not complete")
    with open(job.sgf_path, "rb") as f:
        game = load_sgf_bytes(f.read())
    root = game.get_root()
    comment = _prediction_comment(job.result)
    try:
        existing = root.get("C")
    except KeyError:
        existing = ""
    root.set("C", f"{comment}\n\n{existing}" if existing else comment)
    _add_improvement_variations(game, job.result)
    return game.serialise()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CLIENT, EXECUTOR, _CONFIG_TMPFILE
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    CLIENT, _CONFIG_TMPFILE = _build_client()
    CLIENT.start()
    EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rank-mle")
    try:
        yield
    finally:
        if EXECUTOR is not None:
            EXECUTOR.shutdown(wait=False, cancel_futures=True)
        if CLIENT is not None:
            CLIENT.shutdown()
        if _CONFIG_TMPFILE is not None:
            os.unlink(_CONFIG_TMPFILE.name)


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"alive": CLIENT is not None and CLIENT.is_alive()}


def _client_ip(request: Request) -> str:
    # Trust X-Forwarded-For only if you've put a proxy in front. For now,
    # prefer the direct peer; deployments behind a real proxy can swap this.
    return request.client.host if request.client else "unknown"


def _count_active_for_ip(ip: str) -> int:
    with JOBS_LOCK:
        return sum(
            1
            for j in JOBS.values()
            if j.client_ip == ip and j.status in ("queued", "running")
        )


@app.post("/analyze")
async def analyze(
    request: Request,
    sgf_file: Optional[UploadFile] = File(default=None),
    sgf_text: Optional[str] = Form(default=None),
):
    if sgf_file is None and not sgf_text:
        raise HTTPException(400, "provide sgf_file or sgf_text")

    ip = _client_ip(request)
    active = _count_active_for_ip(ip)
    if active >= MAX_ACTIVE_JOBS_PER_IP:
        raise HTTPException(
            429,
            f"too many active jobs from your address ({active}); cap is {MAX_ACTIVE_JOBS_PER_IP}. Wait for one to finish.",
        )

    if sgf_file is not None:
        body = await sgf_file.read()
    else:
        ogs_game_id = parse_ogs_url(sgf_text)
        if ogs_game_id is not None:
            try:
                body = await asyncio.to_thread(fetch_ogs_sgf, ogs_game_id)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        else:
            body = sgf_text.encode("utf-8")
    n_moves, warnings = _validate_sgf(body)

    sha = hashlib.sha256(body).hexdigest()[:16]
    sgf_path = os.path.join(UPLOAD_DIR, f"{sha}.sgf")
    if not os.path.isfile(sgf_path):
        with open(sgf_path, "wb") as f:
            f.write(body)

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOB_SEQ[0] += 1
        seq = JOB_SEQ[0]
    job = Job(
        job_id=job_id,
        sgf_path=sgf_path,
        sgf_sha=sha,
        warnings=warnings,
        n_moves=n_moves,
        seq=seq,
        client_ip=ip,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    loop = asyncio.get_running_loop()
    loop.run_in_executor(EXECUTOR, _run_job, job_id)
    position, length = _queue_snapshot(job)
    return {
        "job_id": job_id,
        "sgf_sha": sha,
        "n_moves": n_moves,
        "warnings": warnings,
        "queue": {"position": position, "length": length},
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    position, length = _queue_snapshot(job)
    with job.lock:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": {"done": job.progress_done, "total": job.progress_total},
            "improvement_status": job.improvement_status,
            "improvement_progress": {
                "done": job.improvement_progress_done,
                "total": job.improvement_progress_total,
            },
            "result": job.result,
            "error": job.error,
            "improvement_error": job.improvement_error,
            "warnings": job.warnings,
            "n_moves": job.n_moves,
            "queue": {"position": position, "length": length},
        }


@app.post("/jobs/{job_id}/improvements")
async def start_improvements(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")

    with job.lock:
        if job.status == "error":
            raise HTTPException(409, "rank analysis failed")
        if job.status != "done" or job.result is None:
            raise HTTPException(409, "rank analysis is not complete")
        if job.result.get("improvements") is not None:
            job.improvement_status = "done"
            return {
                "job_id": job.job_id,
                "improvement_status": job.improvement_status,
                "result": job.result,
            }
        if job.improvement_status in ("queued", "running"):
            return {
                "job_id": job.job_id,
                "improvement_status": job.improvement_status,
                "result": job.result,
            }
        job.improvement_status = "queued"
        job.improvement_progress_done = 0
        job.improvement_progress_total = 0
        job.improvement_error = None

    loop = asyncio.get_running_loop()
    loop.run_in_executor(EXECUTOR, _run_improvement_job, job_id)
    return {
        "job_id": job.job_id,
        "improvement_status": "queued",
        "result": job.result,
    }


@app.get("/jobs/{job_id}/annotated.sgf")
def get_annotated_sgf(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    with job.lock:
        if job.status == "error":
            raise HTTPException(409, "job failed")
        if job.status != "done":
            raise HTTPException(409, "job is not complete")
        body = _annotated_sgf_bytes(job)
        filename = f"{job.sgf_sha}-rankmle.sgf"
    return Response(
        body,
        media_type="application/x-go-sgf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


INDEX_HTML = """<!doctype html>
<html><head><meta charset=utf-8><title>Rank MLE</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:2em auto;padding:0 1em}
textarea{box-sizing:border-box;width:100%;height:8em;font-family:monospace;font-size:12px}
button{padding:.5em 1em;font-size:14px}
label{font-size:13px}
input[type=number]{width:5em}
.dropzone{border:2px dashed #bbb;border-radius:8px;padding:1em;margin:.8em 0;background:#fafafa}
.dropzone.dragging{border-color:#4a90e2;background:#eef6ff}
.bar{height:8px;background:#eee;border-radius:4px;overflow:hidden;margin:.5em 0}
.bar > div{height:100%;background:#4a90e2;transition:width .3s}
.row{display:flex;justify-content:space-between;padding:.4em 0;border-bottom:1px solid #eee}
.job{border-top:1px solid #ddd;margin-top:1em;padding-top:1em}
.job h2{font-size:16px;margin:.2em 0}
.muted{color:#888;font-size:12px}
.warn{background:#fff7e6;border-left:3px solid #f5a623;padding:.5em .8em;margin:.5em 0;list-style:none}
.warn li{font-size:13px;color:#7a5300}
details.method{margin:.5em 0}
details.method summary{cursor:pointer;font-size:12px;color:#888;user-select:none}
details.method p{margin:.4em 0 0;font-size:12px;color:#888;line-height:1.5}
.notice{background:#fff7e6;border-left:3px solid #f5a623;padding:.5em .8em;margin:.8em 0;font-size:13px;color:#7a5300}
.errmsg{color:#b00;font-size:13px}
.download{display:inline-block;margin-top:.8em;font-size:13px}
.review{margin-top:.8em}
.review h3{font-size:14px;margin:.7em 0 .3em}
.review ol{margin:.3em 0 .3em 1.4em;padding:0;font-size:13px}
.review li{margin:.25em 0}
.reviewControls{margin-top:.8em;font-size:13px}
.reviewControls button{margin-right:.6em}
</style></head><body>
<h1>Rank MLE</h1>
<p class=muted>Paste an SGF or an <a href="https://online-go.com/" target=_blank rel=noopener>online-go.com</a> game link, upload files, or drag SGFs onto the page. Analysis takes ~1-2 minutes per game. Up to 5 games queued at a time.</p>
<details class=method>
<summary>How it works</summary>
<p>Uses KataGo's humanSL policy network, which assigns a probability to each move given the board position and move history (move history matters because humans are influenced by it).</p>
<p>The predicted rank is the maximum-likelihood estimate over 26 KGS rank profiles (20k–6d) at medium time settings — so predictions assume your game comes from a similar distribution.</p>
<p>Capped at 20k–6d amateur; data outside that range is sparse.</p>
<p>In practice, per-game estimates vary roughly ±5 ranks in the DDK range to ±2 in the high SDK range even for the same player.</p>
</details>
<form id=f>
  <div id=dropzone class=dropzone>
    <textarea id=sgf placeholder="Paste SGF or online-go.com game link here..."></textarea>
    <p><input type=file id=file accept=".sgf" multiple> <button type=submit>Analyze</button></p>
    <p class=muted id=dropHint>Drop one or more .sgf files here.</p>
  </div>
</form>
<div id=out></div>
<script>
const out = document.getElementById('out');
const f = document.getElementById('f');
const fileInput = document.getElementById('file');
const dropzone = document.getElementById('dropzone');
const jobs = new Map();
let dragDepth = 0;

const OGS_URL_RE = /^(?:https?:\/\/)?(?:www\.)?online-go\.com\/game\/(?:view\/)?(\d+)(?:[/?#].*)?$/i;
function ogsLabel(text) {
  const trimmed = text.trim();
  const m = OGS_URL_RE.exec(trimmed);
  if (m) return `OGS game ${m[1]}`;
  if (/^\d+$/.test(trimmed)) return `OGS game ${trimmed}`;
  return null;
}

f.onsubmit = async (e) => {
  e.preventDefault();
  const files = Array.from(fileInput.files || []);
  const text = document.getElementById('sgf').value.trim();
  if (files.length) submitFiles(files);
  else if (text) submitText(text);
};

function sgfFiles(fileList) {
  return Array.from(fileList || []).filter(file =>
    file.name.toLowerCase().endsWith('.sgf') || file.type === 'application/x-go-sgf'
  );
}

document.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth += 1;
  dropzone.classList.add('dragging');
});
document.addEventListener('dragover', (e) => {
  e.preventDefault();
});
document.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropzone.classList.remove('dragging');
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropzone.classList.remove('dragging');
  const files = sgfFiles(e.dataTransfer.files);
  if (files.length) submitFiles(files);
});

async function submitText(text) {
  const fd = new FormData();
  fd.append('sgf_text', text);
  await submitOne(fd, ogsLabel(text) || 'Pasted SGF');
}

async function submitFiles(files) {
  for (const file of files) {
    const fd = new FormData();
    fd.append('sgf_file', file);
    submitOne(fd, file.name);
  }
}

async function submitOne(fd, label) {
  const localId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random());
  jobs.set(localId, {label, status: 'submitting'});
  renderAll();
  const r = await fetch('/analyze', {method:'POST', body: fd});
  if (!r.ok) {
    let msg;
    try { msg = (await r.json()).detail; } catch { msg = await r.text(); }
    if (r.status === 429) {
      jobs.delete(localId);
      renderAll();
      showNotice('Queue full — wait for a game to finish, then resubmit.');
    } else {
      jobs.set(localId, {label, status: 'error', error: msg || r.statusText});
      renderAll();
    }
    return;
  }
  const {job_id} = await r.json();
  jobs.delete(localId);
  jobs.set(job_id, {label, status: 'queued', progress: {done: 0, total: 0}});
  renderAll();
  poll(job_id);
}

async function poll(job_id) {
  while (true) {
    const r = await fetch('/jobs/' + job_id);
    const j = await r.json();
    const existing = jobs.get(job_id) || {};
    jobs.set(job_id, {...existing, ...j});
    renderAll();
    if (
      (j.status === 'done' || j.status === 'error') &&
      j.improvement_status !== 'queued' &&
      j.improvement_status !== 'running'
    ) return;
    await new Promise(r => setTimeout(r, 1000));
  }
}
out.addEventListener('click', async (e) => {
  const button = e.target.closest('button[data-find-moves]');
  if (!button) return;
  const jobId = button.dataset.jobId;
  const existing = jobs.get(jobId) || {};
  jobs.set(jobId, {...existing, improvement_status: 'queued', improvement_progress: {done: 0, total: 0}});
  renderAll();
  const r = await fetch(`/jobs/${encodeURIComponent(jobId)}/improvements`, {method:'POST'});
  if (!r.ok) {
    let msg;
    try { msg = (await r.json()).detail; } catch { msg = await r.text(); }
    jobs.set(jobId, {...(jobs.get(jobId) || {}), improvement_status: 'error', improvement_error: msg || r.statusText});
    renderAll();
    return;
  }
  const j = await r.json();
  jobs.set(jobId, {...(jobs.get(jobId) || {}), ...j});
  renderAll();
  poll(jobId);
});
let noticeTimer;
function showNotice(msg) {
  let notice = document.getElementById('notice');
  if (!notice) {
    notice = document.createElement('p');
    notice.id = 'notice';
    notice.className = 'notice';
    out.parentNode.insertBefore(notice, out);
  }
  notice.textContent = msg;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => notice.remove(), 6000);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function renderAll() {
  out.innerHTML = Array.from(jobs.values()).map(renderJob).join('');
}
function renderJob(j) {
  const title = `<h2>${escapeHtml(j.label || j.job_id || 'SGF')}</h2>`;
  if (j.status === 'submitting') return `<div class=job>${title}<p class=muted>submitting...</p></div>`;
  if (j.status === 'error') return `<div class=job>${title}<p class=errmsg>Error: ${escapeHtml(j.error)}</p></div>`;
  const pct = j.progress.total ? Math.floor(100 * j.progress.done / j.progress.total) : 0;
  let queueLine = '';
  if (j.queue && j.queue.length > 0) {
    if (j.status === 'queued' && j.queue.position > 0) {
      const ahead = j.queue.position;
      queueLine = ` — waiting, ${ahead} game${ahead===1?'':'s'} ahead (queue: ${j.queue.length})`;
    } else if (j.status === 'running') {
      queueLine = ` — queue: ${j.queue.length}`;
    }
  }
  let html = `<div class=job>${title}<p class=muted>${j.status}${queueLine} — ${j.progress.done}/${j.progress.total} moves</p>
    <div class=bar><div style="width:${pct}%"></div></div>`;
  if (j.warnings && j.warnings.length) {
    html += '<ul class=warn>' + j.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('') + '</ul>';
  }
  if (j.result) {
    const p = j.result.prediction, pl = j.result.players;
    for (const c of ['B','W']) {
      const info = p[c]; const meta = pl[c] || {};
      const name = escapeHtml(meta.name || '?'); const rated = meta.rating ? ` [${escapeHtml(meta.rating)}]` : '';
      const pred = info.rank ? `predicted KGS <b>${info.rank}</b>` : '-';
      html += `<div class=row><span>${c==='B'?'⚫':'⚪'} ${name}${rated}</span><span>${pred}</span></div>`;
    }
    html += renderImprovementSection(j);
    if (j.result.improvements) {
      html += `<a class=download href="/jobs/${encodeURIComponent(j.job_id)}/annotated.sgf">Download annotated SGF</a>`;
    }
  }
  return html + '</div>';
}
function renderImprovementSection(j) {
  const improvements = j.result && j.result.improvements;
  if (improvements) return renderImprovements(improvements);
  if (j.improvement_status === 'queued' || j.improvement_status === 'running') {
    const progress = j.improvement_progress || {done: 0, total: 0};
    const total = progress.total || 0;
    const pct = total ? Math.floor(100 * progress.done / total) : 0;
    return `<div class=reviewControls><p class=muted>finding moves to review... ${progress.done}/${total}</p><div class=bar><div style="width:${pct}%"></div></div></div>`;
  }
  if (j.improvement_status === 'error') {
    return `<div class=reviewControls><p class=errmsg>Move review error: ${escapeHtml(j.improvement_error || 'failed')}</p></div>`;
  }
  return `<div class=reviewControls><button type=button data-find-moves data-job-id="${escapeHtml(j.job_id)}">Find moves to review</button></div>`;
}
function renderImprovements(improvements) {
  if (!improvements || !improvements.players) return '';
  let html = '<div class=review>';
  for (const c of ['B','W']) {
    const info = improvements.players[c];
    if (!info || !info.target_rank) continue;
    const moves = info.moves || [];
    html += `<h3>${c==='B'?'Black':'White'} review moves (${escapeHtml(info.rank)} → ${escapeHtml(info.target_rank)}, ${info.scored_moves}/${info.total_moves} scored)</h3>`;
    if (!moves.length) {
      html += '<p class=muted>No candidates found.</p>';
      continue;
    }
    html += '<ol>';
    for (const m of moves.slice(0, 5)) {
      const alts = m.alternatives || [];
      const altText = alts.map(alt => {
        const gain = alt.gain_pp === null || alt.gain_pp === undefined ? '' : `, ${alt.gain_pp >= 0 ? '+' : ''}${alt.gain_pp.toFixed(1)}pp`;
        return `<b>${escapeHtml(alt.move)}</b><span class=muted>${gain}, +${alt.point_gain.toFixed(1)}pt</span>`;
      }).join('; ');
      const playedGain = m.played_gain_pp || 0;
      html += `<li>move ${m.move_num}: ${escapeHtml(m.played)} <span class=muted>${playedGain >= 0 ? '+' : ''}${playedGain.toFixed(1)}pp</span> → ${altText}</li>`;
    }
    html += '</ol>';
  }
  return html + '</div>';
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
