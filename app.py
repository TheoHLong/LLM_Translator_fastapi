from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cache import TranslationCache
from config import load_config
from document_parser import Segment, parse_upload
from translator import (
    OllamaTranslationError,
    OllamaTranslator,
    PROMPT_VERSION,
    RefinementContextEntry,
    SummaryContextEntry,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
CACHE_PATH = ROOT / ".cache" / "translation_cache.json"

app_config = load_config()
translator = OllamaTranslator(
    base_url=app_config.ollama.base_url,
    model=app_config.ollama.translation_model,
    summary_model=app_config.ollama.summary_model,
    timeout=app_config.ollama.request_timeout,
    auto_pull=app_config.ollama.auto_pull,
    pull_timeout=app_config.ollama.pull_timeout,
    config_path=str(app_config.ollama.config_path),
)
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
    refine_context_neighbors: int = Form(1),
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
            refine_context_neighbors=refine_context_neighbors,
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
    refine_context_neighbors: int,
) -> Iterator[str]:
    try:
        safe_quality = quality if quality in {"quick", "refine"} else "quick"
        safe_mode = processing_mode if processing_mode in {"translate", "extract", "summarize"} else "translate"
        safe_context_neighbors = normalize_refine_context_neighbors(refine_context_neighbors)
        if safe_mode == "extract":
            safe_mode = "summarize"

        for event in translator.ensure_models_available_events(translator.required_models_for_mode(safe_mode)):
            yield ndjson_event(event)

        parsed = parse_upload(filename, content, max_segment_chars=max_segment_chars)
        combined_text = "\n\n".join(segment.text for segment in parsed.segments)
        source_lang, target_lang = translator.normalize_direction(direction, combined_text)
        stats = {
            "cached": 0,
            "translated": 0,
            "skipped": 0,
            "summarized": 0,
            "summary_cached": 0,
            "summary_refined": 0,
            "summary_refine_cached": 0,
        }

        yield ndjson_event(
            {
                "type": "meta",
                "filename": parsed.filename,
                "file_type": parsed.file_type,
                "segment_count": len(parsed.segments),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model": translator.model,
                "translation_model": translator.model,
                "summary_model": translator.summary_model,
                "auto_pull": translator.auto_pull,
                "mode": safe_mode,
                "quality": safe_quality,
                "refine_context_neighbors": safe_context_neighbors,
            }
        )

        if should_process_by_stage(safe_mode):
            yield from process_segments_by_stage(
                segments=parsed.segments,
                quality=safe_quality,
                source_lang=source_lang,
                target_lang=target_lang,
                refine_context_neighbors=safe_context_neighbors,
                stats=stats,
            )
        else:
            for segment_index, segment in enumerate(parsed.segments):
                yield from process_segment(
                    segment=segment,
                    all_segments=parsed.segments,
                    segment_index=segment_index,
                    processing_mode=safe_mode,
                    quality=safe_quality,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    refine_context_neighbors=safe_context_neighbors,
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
    all_segments: List[Segment],
    segment_index: int,
    processing_mode: str,
    quality: str,
    source_lang: str,
    target_lang: str,
    refine_context_neighbors: int,
    stats: Dict[str, int],
    working_text_override: Optional[str] = None,
    emit_start: bool = True,
) -> Iterator[str]:
    source_text = segment.text
    if emit_start:
        yield segment_start_event(segment)

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

    working_text = working_text_override if working_text_override is not None else source_text
    if working_text_override is None and processing_mode == "summarize" and len(source_text) >= 240:
        working_text = yield from summarize_segment_text(
            segment=segment,
            all_segments=all_segments,
            segment_index=segment_index,
            quality=quality,
            source_lang=source_lang,
            refine_context_neighbors=refine_context_neighbors,
            stats=stats,
        )

    quick_initial = None
    context_entries: List[RefinementContextEntry] = []
    cache_mode = f"{processing_mode}:{quality}"
    cache_text = working_text

    if quality == "refine":
        quick_initial = get_cached_quick_translation(
            text=working_text,
            source_lang=source_lang,
            target_lang=target_lang,
            processing_mode=processing_mode,
        )
        context_entries = build_refinement_context(
            segments=all_segments,
            current_index=segment_index,
            neighbor_count=refine_context_neighbors,
            processing_mode=processing_mode,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        cache_mode = f"{processing_mode}:refine:ctx{refine_context_neighbors}"
        cache_text = make_refine_cache_text(
            current_text=working_text,
            current_quick=quick_initial,
            context_entries=context_entries,
            neighbor_count=refine_context_neighbors,
        )

    cache_key = make_cache_key(
        text=cache_text,
        source_lang=source_lang,
        target_lang=target_lang,
        mode=cache_mode,
    )
    cached_value = cache.get(cache_key)
    if cached_value is not None:
        if quality == "refine":
            quick_initial = cached_value
        else:
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
        context=context_entries,
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
            "mode": cache_mode,
            "prompt_version": PROMPT_VERSION,
            "refine_context_neighbors": refine_context_neighbors if quality == "refine" else None,
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


def should_process_by_stage(processing_mode: str) -> bool:
    return processing_mode == "summarize" and translator.summary_model != translator.model


def process_segments_by_stage(
    *,
    segments: List[Segment],
    quality: str,
    source_lang: str,
    target_lang: str,
    refine_context_neighbors: int,
    stats: Dict[str, int],
) -> Iterator[str]:
    yield ndjson_event(
        {
            "type": "pipeline_stage",
            "stage": "summarizing",
            "message": f"Summarizing with {translator.summary_model}",
        }
    )

    working_text_by_id: Dict[str, str] = {}
    for segment_index, segment in enumerate(segments):
        yield segment_start_event(segment)
        source_text = segment.text

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
            continue

        working_text = source_text
        if len(source_text) >= 240:
            working_text = yield from summarize_segment_text(
                segment=segment,
                all_segments=segments,
                segment_index=segment_index,
                quality=quality,
                source_lang=source_lang,
                refine_context_neighbors=refine_context_neighbors,
                stats=stats,
            )
        working_text_by_id[segment.id] = working_text

    yield ndjson_event(
        {
            "type": "pipeline_stage",
            "stage": "translating",
            "message": f"Translating with {translator.model}",
        }
    )

    for segment_index, segment in enumerate(segments):
        working_text = working_text_by_id.get(segment.id)
        if working_text is None:
            continue
        yield from process_segment(
            segment=segment,
            all_segments=segments,
            segment_index=segment_index,
            processing_mode="summarize",
            quality=quality,
            source_lang=source_lang,
            target_lang=target_lang,
            refine_context_neighbors=refine_context_neighbors,
            stats=stats,
            working_text_override=working_text,
            emit_start=False,
        )


def segment_start_event(segment: Segment) -> str:
    return ndjson_event(
        {
            "type": "segment_start",
            "id": segment.id,
            "source": segment.text,
            "kind": segment.kind,
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


def make_summary_cache_key(*, text: str, source_lang: str, mode: str = "summarize:source") -> str:
    return cache.make_key(
        text=text,
        source_lang=source_lang,
        target_lang=source_lang,
        model=translator.summary_model,
        mode=mode,
        prompt_version=PROMPT_VERSION,
    )


def normalize_refine_context_neighbors(value: int) -> int:
    return value if value in {1, 2, 3} else 1


def get_cached_quick_translation(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    processing_mode: str,
) -> Optional[str]:
    quick_key = make_cache_key(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        mode=f"{processing_mode}:quick",
    )
    return cache.get(quick_key)


def summarize_segment_text(
    *,
    segment: Segment,
    all_segments: List[Segment],
    segment_index: int,
    quality: str,
    source_lang: str,
    refine_context_neighbors: int,
    stats: Dict[str, int],
) -> Generator[str, None, str]:
    if quality == "refine":
        return (yield from summarize_segment_text_refined(
            segment=segment,
            all_segments=all_segments,
            segment_index=segment_index,
            source_lang=source_lang,
            refine_context_neighbors=refine_context_neighbors,
            stats=stats,
        ))

    return (yield from summarize_segment_text_quick(
        segment=segment,
        source_lang=source_lang,
        stats=stats,
    ))


def summarize_segment_text_quick(
    *,
    segment: Segment,
    source_lang: str,
    stats: Dict[str, int],
) -> Generator[str, None, str]:
    summary_key = make_summary_cache_key(text=segment.text, source_lang=source_lang)
    cached_summary = cache.get(summary_key)
    if cached_summary is not None:
        stats["summary_cached"] += 1
        yield ndjson_event({"type": "segment_status", "id": segment.id, "status": "summary_cached"})
        yield ndjson_event(
            {
                "type": "segment_source_update",
                "id": segment.id,
                "source": cached_summary,
                "streaming": False,
            }
        )
        return cached_summary

    yield ndjson_event({"type": "segment_status", "id": segment.id, "status": "summarizing"})
    final_summary = ""
    for partial in translator.summarize_stream(segment.text, source_lang):
        final_summary = partial
        yield ndjson_event(
            {
                "type": "segment_source_update",
                "id": segment.id,
                "source": partial,
                "streaming": True,
            }
        )

    final_summary = final_summary or segment.text
    cache.set(
        summary_key,
        final_summary,
        {
            "source_lang": source_lang,
            "target_lang": source_lang,
            "model": translator.summary_model,
            "mode": "summarize:source",
            "prompt_version": PROMPT_VERSION,
        },
    )
    stats["summarized"] += 1
    yield ndjson_event(
        {
            "type": "segment_source_update",
            "id": segment.id,
            "source": final_summary,
            "streaming": False,
        }
    )
    return final_summary


def summarize_segment_text_refined(
    *,
    segment: Segment,
    all_segments: List[Segment],
    segment_index: int,
    source_lang: str,
    refine_context_neighbors: int,
    stats: Dict[str, int],
) -> Generator[str, None, str]:
    quick_summary = yield from summarize_segment_text_quick(
        segment=segment,
        source_lang=source_lang,
        stats=stats,
    )
    yield ndjson_event({"type": "segment_status", "id": segment.id, "status": "summarizing_context"})
    context_entries = build_summary_refinement_context(
        segments=all_segments,
        current_index=segment_index,
        neighbor_count=refine_context_neighbors,
        source_lang=source_lang,
        stats=stats,
    )
    cache_mode = f"summarize:source:refine:ctx{refine_context_neighbors}"
    cache_text = make_summary_refine_cache_text(
        current_text=segment.text,
        current_quick=quick_summary,
        context_entries=context_entries,
        neighbor_count=refine_context_neighbors,
    )
    summary_key = make_summary_cache_key(
        text=cache_text,
        source_lang=source_lang,
        mode=cache_mode,
    )
    cached_summary = cache.get(summary_key)
    summary_draft = quick_summary
    if cached_summary is not None:
        stats["summary_refine_cached"] += 1
        yield ndjson_event({"type": "segment_status", "id": segment.id, "status": "summary_cached"})
        yield ndjson_event(
            {
                "type": "segment_source_update",
                "id": segment.id,
                "source": cached_summary,
                "streaming": False,
            }
        )
        summary_draft = cached_summary

    yield ndjson_event({"type": "segment_status", "id": segment.id, "status": "refining_summary"})
    final_summary = ""
    for partial in translator.summarize_stream(
        segment.text,
        source_lang,
        "refine",
        initial=summary_draft,
        context=context_entries,
    ):
        final_summary = partial
        yield ndjson_event(
            {
                "type": "segment_source_update",
                "id": segment.id,
                "source": partial,
                "streaming": True,
            }
        )

    final_summary = final_summary or summary_draft or segment.text
    cache.set(
        summary_key,
        final_summary,
        {
            "source_lang": source_lang,
            "target_lang": source_lang,
            "model": translator.summary_model,
            "mode": cache_mode,
            "prompt_version": PROMPT_VERSION,
            "refine_context_neighbors": refine_context_neighbors,
        },
    )
    stats["summary_refined"] += 1
    yield ndjson_event(
        {
            "type": "segment_source_update",
            "id": segment.id,
            "source": final_summary,
            "streaming": False,
        }
    )
    return final_summary


def build_refinement_context(
    *,
    segments: List[Segment],
    current_index: int,
    neighbor_count: int,
    processing_mode: str,
    source_lang: str,
    target_lang: str,
) -> List[RefinementContextEntry]:
    context: List[RefinementContextEntry] = []
    start_index = max(0, current_index - neighbor_count)
    end_index = min(len(segments), current_index + neighbor_count + 1)

    for index in range(start_index, end_index):
        if index == current_index:
            continue
        source_text = context_source_text(
            segment=segments[index],
            processing_mode=processing_mode,
            source_lang=source_lang,
        )
        quick_translation = None
        if not translator.should_skip_translation(source_text):
            quick_translation = get_cached_quick_translation(
                text=source_text,
                source_lang=source_lang,
                target_lang=target_lang,
                processing_mode=processing_mode,
            )
        context.append(
            RefinementContextEntry(
                label=refinement_context_label(index - current_index),
                source=source_text,
                quick_translation=quick_translation,
            )
        )

    return context


def build_summary_refinement_context(
    *,
    segments: List[Segment],
    current_index: int,
    neighbor_count: int,
    source_lang: str,
    stats: Dict[str, int],
) -> List[SummaryContextEntry]:
    context: List[SummaryContextEntry] = []
    start_index = max(0, current_index - neighbor_count)
    end_index = min(len(segments), current_index + neighbor_count + 1)

    for index in range(start_index, end_index):
        if index == current_index:
            continue
        context.append(
            SummaryContextEntry(
                label=refinement_context_label(index - current_index),
                summary=get_quick_summary_for_context(
                    segment=segments[index],
                    source_lang=source_lang,
                    stats=stats,
                ),
            )
        )

    return context


def get_quick_summary_for_context(
    *,
    segment: Segment,
    source_lang: str,
    stats: Dict[str, int],
) -> str:
    if translator.should_skip_translation(segment.text) or len(segment.text) < 240:
        return segment.text

    summary_key = make_summary_cache_key(text=segment.text, source_lang=source_lang)
    cached_summary = cache.get(summary_key)
    if cached_summary is not None:
        stats["summary_cached"] += 1
        return cached_summary

    summary = translator.summarize(segment.text, source_lang)
    cache.set(
        summary_key,
        summary,
        {
            "source_lang": source_lang,
            "target_lang": source_lang,
            "model": translator.summary_model,
            "mode": "summarize:source",
            "prompt_version": PROMPT_VERSION,
        },
    )
    stats["summarized"] += 1
    return summary


def context_source_text(*, segment: Segment, processing_mode: str, source_lang: str) -> str:
    if processing_mode == "summarize" and len(segment.text) >= 240:
        cached_summary = cache.get(make_summary_cache_key(text=segment.text, source_lang=source_lang))
        if cached_summary is not None:
            return cached_summary
    return segment.text


def refinement_context_label(offset: int) -> str:
    if offset < 0:
        return f"previous segment {abs(offset)}"
    return f"next segment {offset}"


def make_refine_cache_text(
    *,
    current_text: str,
    current_quick: Optional[str],
    context_entries: List[RefinementContextEntry],
    neighbor_count: int,
) -> str:
    payload = {
        "current_text": current_text,
        "current_quick": current_quick or "",
        "neighbor_count": neighbor_count,
        "context": [
            {
                "label": entry.label,
                "source": entry.source,
                "quick_translation": entry.quick_translation or "",
            }
            for entry in context_entries
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def make_summary_refine_cache_text(
    *,
    current_text: str,
    current_quick: Optional[str],
    context_entries: List[SummaryContextEntry],
    neighbor_count: int,
) -> str:
    payload = {
        "current_text": current_text,
        "current_quick": current_quick or "",
        "neighbor_count": neighbor_count,
        "context": [
            {
                "label": entry.label,
                "summary": entry.summary,
            }
            for entry in context_entries
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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
