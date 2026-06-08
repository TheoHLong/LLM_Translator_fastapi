from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    translation_model: str
    summary_model: str
    auto_pull: bool
    request_timeout: int
    pull_timeout: int
    config_path: Path


@dataclass(frozen=True)
class AppConfig:
    ollama: OllamaConfig


def load_config(path: Optional[Path] = None) -> AppConfig:
    config_path = Path(os.getenv("TRANSLATOR_CONFIG_PATH") or path or DEFAULT_CONFIG_PATH)
    data = _load_json_config(config_path)
    ollama_data = data.get("ollama", {}) if isinstance(data.get("ollama", {}), dict) else {}

    translation_model = _string_setting(
        os.getenv("OLLAMA_TRANSLATION_MODEL"),
        ollama_data.get("translation_model"),
        "translategemma",
    )
    summary_model = _string_setting(
        os.getenv("OLLAMA_SUMMARY_MODEL"),
        ollama_data.get("summary_model"),
        translation_model,
    )

    return AppConfig(
        ollama=OllamaConfig(
            base_url=_string_setting(
                os.getenv("OLLAMA_BASE_URL"),
                ollama_data.get("base_url"),
                "http://localhost:11434/api/generate",
            ),
            translation_model=translation_model,
            summary_model=summary_model,
            auto_pull=_bool_setting(
                os.getenv("OLLAMA_AUTO_PULL"),
                ollama_data.get("auto_pull"),
                True,
            ),
            request_timeout=_int_setting(
                os.getenv("OLLAMA_REQUEST_TIMEOUT"),
                ollama_data.get("request_timeout"),
                300,
            ),
            pull_timeout=_int_setting(
                os.getenv("OLLAMA_PULL_TIMEOUT"),
                ollama_data.get("pull_timeout"),
                1800,
            ),
            config_path=config_path,
        )
    )


def _load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _string_setting(env_value: Optional[str], file_value: Any, default: str) -> str:
    for value in (env_value, file_value):
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _bool_setting(env_value: Optional[str], file_value: Any, default: bool) -> bool:
    for value in (env_value, file_value):
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _int_setting(env_value: Optional[str], file_value: Any, default: int) -> int:
    for value in (env_value, file_value):
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return default
