"""Detect Online-Go.com (OGS) game links and fetch their SGFs.

OGS exposes a public, unauthenticated endpoint:

    https://online-go.com/api/v1/games/{id}/sgf

which returns the SGF body as text. We accept the canonical web URL
(`https://online-go.com/game/{id}`) or a bare numeric game ID.
"""

from __future__ import annotations

import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OGS_SGF_URL = "https://online-go.com/api/v1/games/{game_id}/sgf"
DEFAULT_TIMEOUT_SEC = 30.0
MAX_SGF_BYTES = 4 * 1024 * 1024

_OGS_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?online-go\.com/game/(?:view/)?(\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)


def parse_ogs_url(value: str) -> int | None:
    """Return the OGS game ID if `value` looks like an OGS game link, else None.

    Accepts the canonical `https://online-go.com/game/<id>` form (with or
    without scheme/`www.`/trailing slash/query/fragment), and a bare numeric
    ID. Anything else returns None so the caller can fall back to treating
    the input as raw SGF text.
    """
    s = value.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    m = _OGS_URL_RE.match(s)
    if m:
        return int(m.group(1))
    return None


def fetch_ogs_sgf(game_id: int, timeout: float = DEFAULT_TIMEOUT_SEC) -> bytes:
    """Fetch the SGF for an OGS game ID. Raises ValueError on any failure."""
    url = OGS_SGF_URL.format(game_id=game_id)
    req = Request(url, headers={"User-Agent": "rankmle/0.1"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read(MAX_SGF_BYTES + 1)
    except HTTPError as e:
        if e.code == 404:
            raise ValueError(f"OGS game {game_id} not found") from e
        raise ValueError(f"OGS returned HTTP {e.code} for game {game_id}") from e
    except URLError as e:
        raise ValueError(f"could not reach OGS: {e.reason}") from e
    except TimeoutError as e:
        raise ValueError(f"OGS request timed out after {timeout:g}s") from e

    if len(body) > MAX_SGF_BYTES:
        raise ValueError(
            f"OGS response exceeds {MAX_SGF_BYTES} bytes; refusing to load"
        )
    if not body.lstrip().startswith(b"("):
        raise ValueError(f"OGS did not return an SGF for game {game_id}")
    return body
