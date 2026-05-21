from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests


PROMPT_VERSION = "fastapi-core-v1"


class OllamaTranslationError(RuntimeError):
    pass


@dataclass
class ProtectedText:
    text: str
    placeholders: Dict[str, str]


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
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
        self.model = model or os.getenv("OLLAMA_TRANSLATION_MODEL", "translategemma")
        self.summary_model = summary_model or os.getenv("OLLAMA_SUMMARY_MODEL", self.model)
        self.timeout = timeout
        self.protector = TextProtector()

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

    def summarize(self, text: str, source_lang: str) -> str:
        protected = self.protector.protect(text)
        prompt = f"""Summarize the following {source_lang} passage in one clear paragraph.
Keep essential facts, names, technical terms, and important equations.
Output only the summary.

Text:
{protected.text}
"""
        result = self._completion(
            prompt,
            f"You are an expert at summarizing {source_lang} text.",
            model=self.summary_model,
            placeholders=protected.placeholders,
        )
        return result or text

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
    ) -> Iterator[str]:
        protected = self.protector.protect(text)

        if quality == "refine":
            quick = initial or self._translate_quick(protected, source_lang, target_lang)
            notes = self._reflection_notes(protected, source_lang, target_lang, quick)
            prompt = self._improvement_prompt(protected.text, target_lang, quick, notes)
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
    ) -> str:
        prompt = f"""Review this {source_lang} to {target_lang} translation.

Source:
{protected.text}

Translation:
{initial}

Give concise, specific improvement notes only.
"""
        return self._completion(
            prompt,
            "You are an expert translation reviewer.",
            placeholders=protected.placeholders,
        )

    @staticmethod
    def _improvement_prompt(text: str, target_lang: str, initial: str, notes: str) -> str:
        return f"""Improve the translation using the notes.

Source:
{text}

Current translation:
{initial}

Notes:
{notes}

Output only the improved {target_lang} translation.
"""

    def _completion(
        self,
        prompt: str,
        system_message: str,
        *,
        model: Optional[str] = None,
        placeholders: Optional[Dict[str, str]] = None,
    ) -> str:
        response = requests.post(
            self.base_url,
            json=self._payload(prompt, system_message, stream=False, model=model, placeholders=placeholders),
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
        placeholders: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        raw_chunks = self._raw_completion_stream(prompt, system_message, placeholders=placeholders)
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
        placeholders: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        try:
            with requests.post(
                self.base_url,
                json=self._payload(prompt, system_message, stream=True, placeholders=placeholders),
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
        tags_url = self.base_url.replace("/api/generate", "/api/tags")
        try:
            response = requests.get(tags_url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return {
                "ok": False,
                "online": False,
                "model": self.model,
                "model_available": False,
                "error": str(exc),
            }

        model_names = [
            str(item.get("name", ""))
            for item in data.get("models", [])
            if isinstance(item, dict)
        ]
        model_available = any(
            name == self.model or name.split(":", 1)[0] == self.model
            for name in model_names
        )
        return {
            "ok": model_available,
            "online": True,
            "model": self.model,
            "model_available": model_available,
            "models": model_names,
            "error": "" if model_available else f"Model {self.model} is not installed.",
        }
