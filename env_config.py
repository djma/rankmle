"""Shared environment-backed defaults for KataGo entry points."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv


load_dotenv()


def env_path(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if not value:
        return value
    return os.path.abspath(os.path.expanduser(value))


def add_katago_args(parser: argparse.ArgumentParser) -> None:
    katago = env_path("KATAGO_BIN", "/opt/homebrew/bin/katago")
    human_model = env_path("KATAGO_HUMAN_MODEL")
    model = env_path("KATAGO_MODEL")
    config = env_path("KATAGO_CONFIG")

    parser.add_argument(
        "--katago",
        default=katago,
        help="KataGo binary path (default: KATAGO_BIN or /opt/homebrew/bin/katago)",
    )
    parser.add_argument(
        "--human-model",
        default=human_model,
        required=human_model is None,
        help="human SL model path (default: KATAGO_HUMAN_MODEL)",
    )
    parser.add_argument(
        "--model",
        default=model,
        required=model is None,
        help="regular/full KataGo model (default: KATAGO_MODEL)",
    )
    parser.add_argument(
        "--config",
        default=config,
        required=config is None,
        help="analysis config path (default: KATAGO_CONFIG)",
    )
