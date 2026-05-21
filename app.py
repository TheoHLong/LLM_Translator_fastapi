from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cache import TranslationCache
from document_parser import Segment, parse_upload
from translator import OllamaTranslationError, OllamaTranslator, PROMPT_VERSION


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
CACHE_PATH = ROOT / ".cache" / "translation_cache.json"

translator = OllamaTranslator()
cache = TranslationCache(CACHE_PATH)
api = FastAPI(title="FastAPI LLM Translator")
api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app = api


@api.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@api.get("/index.html")
def index_html() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@api.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse(translator.health_check())


@api.post("/api/translate/stream")
async def translate_file_stream(
    file: UploadFile = File(...),
    processing_mode: str = Form("translate"),
    quality: str = Form("quick"),
    direction: str = Form("en-zh"),
    max_segment_chars: int = Form(1600),
) -> StreamingResponse:
    content = await file.read()
    filename = file.filename or "uploaded"
    return StreamingResponse(
        stream_translation_events(
            filename=filename,
            content=content,
            processing_mode=processing_mode,
            quality=quality,
            direction=direction,
            max_segment_chars=max_segment_chars,
        ),
        media_type="application/x-ndjson",
    )


def stream_translation_events(
    *,
    filename: str,
    content: bytes,
    processing_mode: str,
    quality: str,
    direction: str,
    max_segment_chars: int,
) -> Iterator[str]:
    try:
        safe_quality = quality if quality in {"quick", "refine"} else "quick"
        safe_mode = processing_mode if processing_mode in {"translate", "extract", "summarize"} else "translate"
        if safe_mode == "extract":
            safe_mode = "summarize"
        parsed = parse_upload(filename, content, max_segment_chars=max_segment_chars)
        combined_text = "\n\n".join(segment.text for segment in parsed.segments)
        source_lang, target_lang = translator.normalize_direction(direction, combined_text)
        stats = {"cached": 0, "translated": 0, "skipped": 0, "summarized": 0, "summary_cached": 0}

        yield ndjson_event(
            {
                "type": "meta",
                "filename": parsed.filename,
                "file_type": parsed.file_type,
                "segment_count": len(parsed.segments),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model": translator.model,
                "mode": safe_mode,
                "quality": safe_quality,
            }
        )

        for segment in parsed.segments:
            yield from process_segment(
                segment=segment,
                processing_mode=safe_mode,
                quality=safe_quality,
                source_lang=source_lang,
                target_lang=target_lang,
                stats=stats,
            )

        yield ndjson_event({"type": "done", "stats": stats})

    except OllamaTranslationError as exc:
        yield ndjson_event({"type": "error", "message": str(exc)})
    except Exception as exc:
        yield ndjson_event({"type": "error", "message": str(exc)})


def process_segment(
    *,
    segment: Segment,
    processing_mode: str,
    quality: str,
    source_lang: str,
    target_lang: str,
    stats: Dict[str, int],
) -> Iterator[str]:
    source_text = segment.text
    yield ndjson_event({"type": "segment_start", "id": segment.id, "source": source_text})

    if translator.should_skip_translation(source_text):
        stats["skipped"] += 1
        yield ndjson_event(
            {
                "type": "segment_done",
                "id": segment.id,
                "translation": source_text,
                "cached": False,
                "skipped": True,
            }
        )
        return

    working_text = source_text
    if processing_mode == "summarize" and len(source_text) >= 240:
        summary_key = make_summary_cache_key(text=source_text, source_lang=source_lang)
        cached_summary = cache.get(summary_key)
        if cached_summary is not None:
            working_text = cached_summary
            stats["summary_cached"] += 1
            yield ndjson_event({"type": "segment_status", "id": segment.id, "status": "summary_cached"})
        else:
            yield ndjson_event({"type": "segment_status", "id": segment.id, "status": "summarizing"})
            working_text = translator.summarize(source_text, source_lang)
            cache.set(
                summary_key,
                working_text,
                {
                    "source_lang": source_lang,
                    "target_lang": source_lang,
                    "model": translator.summary_model,
                    "mode": "summarize:source",
                    "prompt_version": PROMPT_VERSION,
                },
            )
            stats["summarized"] += 1
        yield ndjson_event({"type": "segment_source_update", "id": segment.id, "source": working_text})

    cache_key = make_cache_key(
        text=working_text,
        source_lang=source_lang,
        target_lang=target_lang,
        mode=f"{processing_mode}:{quality}",
    )
    cached_value = cache.get(cache_key)
    if cached_value is not None:
        stats["cached"] += 1
        yield ndjson_event(
            {
                "type": "segment_done",
                "id": segment.id,
                "translation": cached_value,
                "cached": True,
                "skipped": False,
            }
        )
        return

    quick_initial = None
    if quality == "refine":
        quick_key = make_cache_key(
            text=working_text,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=f"{processing_mode}:quick",
        )
        quick_initial = cache.get(quick_key)

    yield ndjson_event(
        {
            "type": "segment_status",
            "id": segment.id,
            "status": "refining" if quality == "refine" else "translating",
        }
    )

    final_translation = ""
    for partial in translator.translate_stream(
        working_text,
        source_lang,
        target_lang,
        quality,
        initial=quick_initial,
    ):
        final_translation = partial
        yield ndjson_event({"type": "chunk", "id": segment.id, "text": partial})

    cache.set(
        cache_key,
        final_translation,
        {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": translator.model,
            "mode": f"{processing_mode}:{quality}",
            "prompt_version": PROMPT_VERSION,
        },
    )
    stats["translated"] += 1
    yield ndjson_event(
        {
            "type": "segment_done",
            "id": segment.id,
            "translation": final_translation,
            "cached": False,
            "skipped": False,
        }
    )


def make_cache_key(*, text: str, source_lang: str, target_lang: str, mode: str) -> str:
    return cache.make_key(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        model=translator.model,
        mode=mode,
        prompt_version=PROMPT_VERSION,
    )


def make_summary_cache_key(*, text: str, source_lang: str) -> str:
    return cache.make_key(
        text=text,
        source_lang=source_lang,
        target_lang=source_lang,
        model=translator.summary_model,
        mode="summarize:source",
        prompt_version=PROMPT_VERSION,
    )


def ndjson_event(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastAPI LLM Translator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("app:api", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
