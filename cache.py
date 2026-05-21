from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TranslationCache:
    """Small persistent JSON cache keyed by translation-affecting inputs."""

    def __init__(self, path: Path, max_entries: int = 5000):
        self.path = path
        self.max_entries = max_entries
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"entries": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            with self.path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        if isinstance(loaded, dict) and isinstance(loaded.get("entries"), dict):
            self._data = loaded

    def _save(self) -> None:
        temp_path = self.path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.path)

    @staticmethod
    def hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def make_key(
        cls,
        *,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str,
        mode: str,
        prompt_version: str,
    ) -> str:
        payload = {
            "text_hash": cls.hash_text(text),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": model,
            "mode": mode,
            "prompt_version": prompt_version,
        }
        return cls.hash_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._data["entries"].get(key)
            if not isinstance(entry, dict):
                return None
            value = entry.get("translation")
            if not isinstance(value, str):
                return None
            entry["last_used_at"] = time.time()
            return value

    def set(self, key: str, translation: str, metadata: Dict[str, Any]) -> None:
        with self._lock:
            self._data["entries"][key] = {
                "translation": translation,
                "metadata": metadata,
                "created_at": time.time(),
                "last_used_at": time.time(),
            }
            self._prune()
            self._save()

    def _prune(self) -> None:
        entries = self._data["entries"]
        if len(entries) <= self.max_entries:
            return

        excess_count = len(entries) - self.max_entries
        oldest_keys = sorted(
            entries,
            key=lambda item_key: entries[item_key].get(
                "last_used_at",
                entries[item_key].get("created_at", 0),
            ),
        )[:excess_count]
        for item_key in oldest_keys:
            entries.pop(item_key, None)
