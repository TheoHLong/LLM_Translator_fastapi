from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests


PROMPT_VERSION = "fastapi-core-v1"


class OllamaTranslationError(RuntimeError):
    pass


@dataclass
class ProtectedText:
    text: str
    placeholders: Dict[str, str]


@dataclass
class RefinementContextEntry:
    label: str
    source: str
    quick_translation: Optional[str] = None


@dataclass
class SummaryContextEntry:
    label: str
    summary: str


class TextProtector:
    """Protect LaTeX math and code from translation edits."""

    PATTERNS: List[Tuple[str, str]] = [
        (r"```[\s\S]*?```", "CODE"),
        (r"`[^`\n]+`", "CODE"),
        (r"\$\$[\s\S]+?\$\$", "MATH"),
        (r"\\\[[\s\S]+?\\\]", "MATH"),
        (r"\\\([\s\S]+?\\\)", "MATH"),
        (r"\\begin\{([A-Za-z*]+)\}[\s\S]+?\\end\{\1\}", "MATH"),
        (r"(?<!\\)\$(?!\$)(?:\\.|[^$\n]){1,500}?(?<!\\)\$", "MATH"),
    ]

    def protect(self, text: str) -> ProtectedText:
        placeholders: Dict[str, str] = {}
        index = 0

        def protect_pattern(pattern: str, label: str, current_text: str) -> str:
            nonlocal index

            def replace(match: re.Match) -> str:
                nonlocal index
                placeholder = f"ZXQ_{label}_{index:04d}_ZXQ"
                index += 1
                placeholders[placeholder] = match.group(0)
                return placeholder

            return re.sub(pattern, replace, current_text, flags=re.DOTALL)

        protected = text
        for pattern, label in self.PATTERNS:
            protected = protect_pattern(pattern, label, protected)

        return ProtectedText(text=protected, placeholders=placeholders)

    @staticmethod
    def restore(text: str, placeholders: Dict[str, str]) -> str:
        restored = text
        for placeholder, original in placeholders.items():
            restored = restored.replace(placeholder, original)
        return restored


class OllamaTranslator:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        timeout: int = 300,
        auto_pull: bool = True,
        pull_timeout: int = 1800,
        config_path: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
        # "gemma4:e2b-mlx" "translategemma"
        self.model = model or os.getenv("OLLAMA_TRANSLATION_MODEL", "translategemma")
        self.summary_model = summary_model or os.getenv("OLLAMA_SUMMARY_MODEL", self.model)
        self.timeout = timeout
        self.auto_pull = auto_pull
        self.pull_timeout = pull_timeout
        self.config_path = str(config_path) if config_path is not None else None
        self.protector = TextProtector()
        self._model_lock = threading.Lock()
        self._ensured_models: Set[str] = set()

    @staticmethod
    def detect_language(text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return "English"
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
        latin_count = len(re.findall(r"[A-Za-z]", compact))
        total_signal = cjk_count + latin_count
        if total_signal == 0:
            return "English"
        return "Chinese" if cjk_count / total_signal >= 0.2 else "English"

    @staticmethod
    def normalize_direction(direction: str, combined_text: str) -> Tuple[str, str]:
        if direction == "zh-en":
            return "Chinese", "English"
        if direction == "en-zh":
            return "English", "Chinese"

        source_lang = OllamaTranslator.detect_language(combined_text)
        return source_lang, "English" if source_lang == "Chinese" else "Chinese"

    @staticmethod
    def should_skip_translation(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if stripped.startswith("```") and stripped.endswith("```"):
            return True
        if re.fullmatch(r"\$\$[\s\S]*\$\$", stripped):
            return True
        return False

    def summarize(
        self,
        text: str,
        source_lang: str,
        quality: str = "quick",
        *,
        initial: Optional[str] = None,
        context: Optional[List[SummaryContextEntry]] = None,
    ) -> str:
        final = ""
        for partial in self.summarize_stream(
            text,
            source_lang,
            quality,
            initial=initial,
            context=context,
        ):
            final = partial
        return final or text

    def summarize_stream(
        self,
        text: str,
        source_lang: str,
        quality: str = "quick",
        *,
        initial: Optional[str] = None,
        context: Optional[List[SummaryContextEntry]] = None,
    ) -> Iterator[str]:
        protected = self.protector.protect(text)

        if quality == "refine":
            context_block = self._summary_context_block(context or [], source_lang)
            prompt = self._summary_improvement_prompt(
                protected.text,
                source_lang,
                initial or protected.text,
                context_block,
            )
            yield from self._completion_stream(
                prompt,
                f"You are an expert at editing {source_lang} summaries.",
                model=self.summary_model,
                placeholders=protected.placeholders,
            )
            return

        prompt, system_message = self._summary_prompt(protected.text, source_lang)
        yield from self._completion_stream(
            prompt,
            system_message,
            model=self.summary_model,
            placeholders=protected.placeholders,
        )

    def translate(self, text: str, source_lang: str, target_lang: str, quality: str) -> str:
        final = ""
        for partial in self.translate_stream(text, source_lang, target_lang, quality):
            final = partial
        return final

    def translate_stream(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        quality: str = "refine",
        initial: Optional[str] = None,
        context: Optional[List[RefinementContextEntry]] = None,
    ) -> Iterator[str]:
        protected = self.protector.protect(text)

        if quality == "refine":
            quick = initial or self._translate_quick(protected, source_lang, target_lang)
            context_block = self._refinement_context_block(context or [], source_lang, target_lang)
            notes = self._reflection_notes(protected, source_lang, target_lang, quick, context_block)
            prompt = self._improvement_prompt(protected.text, target_lang, quick, notes, context_block)
            yield from self._completion_stream(
                prompt,
                "You are an expert translation editor.",
                placeholders=protected.placeholders,
            )
            return

        prompt, system_message = self._quick_prompt(protected.text, source_lang, target_lang)
        yield from self._completion_stream(
            prompt,
            system_message,
            placeholders=protected.placeholders,
        )

    def _translate_quick(
        self,
        protected: ProtectedText,
        source_lang: str,
        target_lang: str,
    ) -> str:
        prompt, system_message = self._quick_prompt(protected.text, source_lang, target_lang)
        return self._completion(prompt, system_message, placeholders=protected.placeholders)

    @staticmethod
    def _summary_prompt(text: str, source_lang: str) -> Tuple[str, str]:
        system_message = f"You are an expert at summarizing {source_lang} text."
        prompt = f"""Summarize the following {source_lang} passage in one clear paragraph.
Keep essential facts, names, technical terms, and important equations.
Output only the summary.

Text:
{text}
"""
        return prompt, system_message

    def _quick_prompt(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str]:
        system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}."
        prompt = f"""This is a {source_lang} to {target_lang} translation.
Translate natural-language prose only.
Preserve Markdown syntax, headings, lists, links, code fences, inline code, URLs, and LaTeX math.
Do not provide explanations or text apart from the translation.

{source_lang}: {text}

{target_lang}:
"""
        return prompt, system_message

    def _reflection_notes(
        self,
        protected: ProtectedText,
        source_lang: str,
        target_lang: str,
        initial: str,
        context_block: str = "",
    ) -> str:
        context_section = f"{context_block}\n\n" if context_block else ""
        prompt = f"""Review this {source_lang} to {target_lang} translation for the CURRENT segment.
Use the surrounding segments only as context for terminology, pronouns, transitions, and discourse flow.
Do not rewrite or translate the surrounding segments.

{context_section}Current source:
{protected.text}

Current draft translation:
{initial}

Give concise, specific improvement notes for the CURRENT segment only.
"""
        return self._completion(
            prompt,
            "You are an expert translation reviewer.",
            placeholders=protected.placeholders,
        )

    @staticmethod
    def _improvement_prompt(
        text: str,
        target_lang: str,
        initial: str,
        notes: str,
        context_block: str = "",
    ) -> str:
        context_section = f"{context_block}\n\n" if context_block else ""
        return f"""Improve only the CURRENT segment translation using the notes and context.
Use surrounding segments only to keep terminology, references, pronouns, tense, tone, and sentence transitions consistent.
Do not include labels, notes, explanations, or any surrounding segment content.

{context_section}Current source:
{text}

Current draft translation:
{initial}

Notes:
{notes}

Output only the improved {target_lang} translation for the CURRENT segment.
"""

    @staticmethod
    def _refinement_context_block(
        context: List[RefinementContextEntry],
        source_lang: str,
        target_lang: str,
    ) -> str:
        if not context:
            return ""

        blocks = ["Surrounding context segments (reference only; do not output these):"]
        for entry in context:
            blocks.append(f"[{entry.label}]")
            blocks.append(f"{source_lang} source:")
            blocks.append(entry.source)
            if entry.quick_translation:
                blocks.append(f"Quick {target_lang} translation:")
                blocks.append(entry.quick_translation)
            else:
                blocks.append(f"Quick {target_lang} translation: unavailable")
            blocks.append("")
        return "\n".join(blocks).strip()

    @staticmethod
    def _summary_context_block(
        context: List[SummaryContextEntry],
        source_lang: str,
    ) -> str:
        if not context:
            return ""

        blocks = ["Surrounding first-pass summaries (reference only; do not output these):"]
        for entry in context:
            blocks.append(f"[{entry.label}]")
            blocks.append(f"{source_lang} summary:")
            blocks.append(entry.summary)
            blocks.append("")
        return "\n".join(blocks).strip()

    @staticmethod
    def _summary_improvement_prompt(
        text: str,
        source_lang: str,
        initial: str,
        context_block: str = "",
    ) -> str:
        context_section = f"{context_block}\n\n" if context_block else ""
        return f"""Improve only the CURRENT draft summary using the CURRENT source and surrounding first-pass summaries.
Use the CURRENT source as the factual authority for the improved summary.
Use surrounding summaries only to understand references, continuity, duplicated details, terminology, and what facts matter.
Do not output the surrounding summaries.
Do not introduce facts that are absent from the CURRENT source.
Do not include labels, notes, or explanations.

{context_section}Current source:
{text}

Current draft summary:
{initial}

Output only the improved {source_lang} summary for the CURRENT segment in one clear paragraph.
"""

    def _completion(
        self,
        prompt: str,
        system_message: str,
        *,
        model: Optional[str] = None,
        placeholders: Optional[Dict[str, str]] = None,
    ) -> str:
        active_model = model or self.model
        self.ensure_model_available(active_model)
        response = requests.post(
            self.base_url,
            json=self._payload(prompt, system_message, stream=False, model=active_model, placeholders=placeholders),
            timeout=self.timeout,
        )
        self._raise_for_response(response)
        text = response.json().get("response", "")
        text = self._strip_thinking(str(text))
        return self.protector.restore(text, placeholders or {})

    def _completion_stream(
        self,
        prompt: str,
        system_message: str,
        *,
        model: Optional[str] = None,
        placeholders: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        raw_chunks = self._raw_completion_stream(
            prompt,
            system_message,
            model=model,
            placeholders=placeholders,
        )
        visible_chunks = self._hide_thinking_chunks(raw_chunks)
        raw_visible = ""
        last_visible = ""

        for chunk in visible_chunks:
            raw_visible += chunk
            restored = self.protector.restore(raw_visible, placeholders or {}).strip()
            if restored != last_visible:
                last_visible = restored
                yield restored

    def _raw_completion_stream(
        self,
        prompt: str,
        system_message: str,
        *,
        model: Optional[str] = None,
        placeholders: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        active_model = model or self.model
        self.ensure_model_available(active_model)
        try:
            with requests.post(
                self.base_url,
                json=self._payload(
                    prompt,
                    system_message,
                    stream=True,
                    model=active_model,
                    placeholders=placeholders,
                ),
                timeout=self.timeout,
                stream=True,
            ) as response:
                self._raise_for_response(response)
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = str(data.get("response", ""))
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            raise OllamaTranslationError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

    def _payload(
        self,
        prompt: str,
        system_message: str,
        *,
        stream: bool,
        model: Optional[str] = None,
        placeholders: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return {
            "model": model or self.model,
            "prompt": f"{system_message}{self._preservation_instruction(placeholders or {})}\n\n{prompt}",
            "stream": stream,
        }

    @staticmethod
    def _preservation_instruction(placeholders: Dict[str, str]) -> str:
        if not placeholders:
            return ""
        tokens = ", ".join(placeholders)
        return (
            "\n\nPreserve every protected placeholder token exactly as written. "
            "Do not translate, split, reorder, add spaces inside, or remove these tokens: "
            f"{tokens}"
        )

    @staticmethod
    def _raise_for_response(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaTranslationError(f"Ollama request failed: {response.text}") from exc

    def configured_models(self) -> List[str]:
        return self._unique_models([self.model, self.summary_model])

    def required_models_for_mode(self, processing_mode: str) -> List[str]:
        if processing_mode == "summarize":
            return self._unique_models([self.summary_model, self.model])
        return [self.model]

    def ensure_model_available(self, model: str) -> None:
        for _event in self.ensure_models_available_events([model]):
            pass

    def ensure_models_available_events(self, models: Iterable[str]) -> Iterator[Dict[str, Any]]:
        for model in self._unique_models(models):
            yield from self._ensure_model_available_events(model)

    def _ensure_model_available_events(self, model: str) -> Iterator[Dict[str, Any]]:
        with self._model_lock:
            if model in self._ensured_models:
                yield self._model_status_event("ready", model, f"Ollama model ready: {model}")
                return

            yield self._model_status_event("checking", model, f"Checking Ollama model: {model}")
            model_names = self._installed_model_names(timeout=min(float(self.timeout), 10.0))
            if self._model_ready(model, model_names, timeout=min(float(self.timeout), 10.0)):
                self._ensured_models.add(model)
                yield self._model_status_event("ready", model, f"Ollama model ready: {model}")
                return

            if not self.auto_pull:
                raise OllamaTranslationError(
                    f"Model {model} is not available to Ollama. Install or repair it with: ollama pull {model}"
                )

            yield self._model_status_event("pulling", model, f"Pulling Ollama model: {model}")
            yield from self._pull_model_events(model)

            model_names = self._installed_model_names(timeout=min(float(self.timeout), 10.0))
            if not self._model_ready(model, model_names, timeout=min(float(self.timeout), 10.0)):
                raise OllamaTranslationError(f"Model {model} was pulled but Ollama still cannot load it.")

            self._ensured_models.add(model)
            yield self._model_status_event("ready", model, f"Ollama model ready: {model}")

    def _pull_model_events(self, model: str) -> Iterator[Dict[str, Any]]:
        pull_url = self._ollama_api_url("pull")
        last_status = ""
        last_percent = -1

        try:
            with requests.post(
                pull_url,
                json={"model": model, "stream": True},
                timeout=self.pull_timeout,
                stream=True,
            ) as response:
                self._raise_for_response(response)
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaTranslationError(f"Ollama pull failed for {model}: {data.get('error')}")

                    status = str(data.get("status") or "pulling")
                    percent = self._pull_percent(data)
                    should_emit = status != last_status
                    if percent is not None and percent != last_percent and (
                        percent == 100 or percent % 5 == 0
                    ):
                        should_emit = True

                    if should_emit:
                        last_status = status
                        last_percent = percent if percent is not None else last_percent
                        yield self._model_status_event(
                            "pulling",
                            model,
                            self._pull_message(model, status, percent),
                            ollama_status=status,
                            progress_percent=percent,
                        )
        except requests.RequestException as exc:
            raise OllamaTranslationError(f"Cannot pull Ollama model {model} from {pull_url}: {exc}") from exc

    def _installed_model_names(self, timeout: Optional[float] = None) -> List[str]:
        tags_url = self._ollama_api_url("tags")
        try:
            response = requests.get(tags_url, timeout=timeout or self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise OllamaTranslationError(f"Cannot reach Ollama at {tags_url}: {exc}") from exc

        return [
            str(item.get("name", ""))
            for item in data.get("models", [])
            if isinstance(item, dict) and item.get("name")
        ]

    def _ollama_api_url(self, endpoint: str) -> str:
        endpoint = endpoint.strip("/")
        parsed = urlsplit(self.base_url)
        path = parsed.path.rstrip("/")

        if "/api/" in path:
            prefix = path.split("/api/", 1)[0]
        elif path.endswith("/api"):
            prefix = path[:-4]
        else:
            prefix = path

        api_path = f"{prefix}/api/{endpoint}" if prefix else f"/api/{endpoint}"
        return urlunsplit((parsed.scheme, parsed.netloc, api_path, "", ""))

    def _model_status_event(
        self,
        status: str,
        model: str,
        message: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "type": "model_status",
            "status": status,
            "model": model,
            "translation_model": self.model,
            "summary_model": self.summary_model,
            "ok": status == "ready",
            "ready": status == "ready",
            "auto_pull": self.auto_pull,
            "message": message,
        }
        event.update(extra)
        return event

    @staticmethod
    def _pull_percent(data: Dict[str, Any]) -> Optional[int]:
        completed = data.get("completed")
        total = data.get("total")
        if not isinstance(completed, (int, float)) or not isinstance(total, (int, float)) or total <= 0:
            return None
        return max(0, min(100, int(completed * 100 / total)))

    @staticmethod
    def _pull_message(model: str, status: str, percent: Optional[int]) -> str:
        if percent is None:
            return f"Pulling {model}: {status}"
        return f"Pulling {model}: {status} ({percent}%)"

    @staticmethod
    def _unique_models(models: Iterable[str]) -> List[str]:
        unique: List[str] = []
        seen: Set[str] = set()
        for model in models:
            name = str(model or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            unique.append(name)
        return unique

    @staticmethod
    def _model_available(model: str, model_names: Iterable[str]) -> bool:
        for name in model_names:
            if name == model:
                return True
            if ":" not in model and name.split(":", 1)[0] == model:
                return True
        return False

    def _model_ready(self, model: str, model_names: Iterable[str], timeout: Optional[float] = None) -> bool:
        if not self._model_available(model, model_names):
            return False
        return self._model_resolves(model, timeout=timeout)

    def _model_resolves(self, model: str, timeout: Optional[float] = None) -> bool:
        show_url = self._ollama_api_url("show")
        try:
            response = requests.post(show_url, json={"model": model}, timeout=timeout or self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return False
        return True

    @classmethod
    def _hide_thinking_chunks(cls, chunks: Iterator[str]) -> Iterator[str]:
        in_think = False
        carry = ""
        for chunk in chunks:
            text = carry + chunk
            carry = ""
            while text:
                tag = "</think>" if in_think else "<think>"
                index = text.find(tag)
                if index >= 0:
                    if in_think:
                        text = text[index + len(tag):]
                        in_think = False
                    else:
                        visible = text[:index]
                        if visible:
                            yield visible
                        text = text[index + len(tag):]
                        in_think = True
                    continue

                if in_think:
                    carry = cls._partial_tag_suffix(text, "</think>")
                else:
                    carry = cls._partial_tag_suffix(text, "<think>")
                    visible = text[:len(text) - len(carry)] if carry else text
                    if visible:
                        yield visible
                break

        if carry and not in_think:
            yield carry

    @staticmethod
    def _partial_tag_suffix(text: str, tag: str) -> str:
        max_length = min(len(tag) - 1, len(text))
        for length in range(max_length, 0, -1):
            suffix = text[-length:]
            if tag.startswith(suffix):
                return suffix
        return ""

    @staticmethod
    def _strip_thinking(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        if "<think>" in text:
            return text.split("<think>", 1)[0].strip()
        return text.strip()

    def health_check(self, timeout: float = 2.0) -> Dict[str, Any]:
        try:
            model_names = self._installed_model_names(timeout=timeout)
        except OllamaTranslationError as exc:
            return {
                "ok": False,
                "ready": False,
                "online": False,
                "model": self.model,
                "translation_model": self.model,
                "summary_model": self.summary_model,
                "model_available": False,
                "summary_model_available": False,
                "auto_pull": self.auto_pull,
                "config_path": self.config_path,
                "error": str(exc),
            }

        model_available = self._model_ready(self.model, model_names, timeout=timeout)
        summary_model_available = self._model_ready(self.summary_model, model_names, timeout=timeout)
        ready = model_available and summary_model_available
        missing_models = [
            model
            for model, available in (
                (self.model, model_available),
                (self.summary_model, summary_model_available),
            )
            if not available
        ]
        missing_models = self._unique_models(missing_models)
        return {
            "ok": ready,
            "ready": ready,
            "online": True,
            "model": self.model,
            "translation_model": self.model,
            "summary_model": self.summary_model,
            "model_available": model_available,
            "summary_model_available": summary_model_available,
            "auto_pull": self.auto_pull,
            "config_path": self.config_path,
            "models": model_names,
            "missing_models": missing_models,
            "error": "" if ready else self._missing_models_message(missing_models),
        }

    def _missing_models_message(self, missing_models: List[str]) -> str:
        if not missing_models:
            return ""
        model_list = ", ".join(missing_models)
        if self.auto_pull:
            return f"Model not available locally: {model_list}. It will be pulled on first use."
        return f"Model not available locally: {model_list}."
