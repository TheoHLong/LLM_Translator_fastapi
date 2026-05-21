from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mammoth
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdfminer_extract_text
from PyPDF2 import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".vtt", ".srt"}


@dataclass
class Segment:
    id: str
    text: str


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    segments: List[Segment]


def parse_upload(filename: str, content: bytes, max_segment_chars: int = 1600) -> ParsedDocument:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}")

    if extension == ".pdf":
        text = extract_pdf_text(content)
    elif extension == ".docx":
        text = extract_docx_text(content)
    elif extension == ".html":
        text = html_to_text(decode_text(content))
    elif extension in {".vtt", ".srt"}:
        text = extract_subtitle_text(content)
    else:
        text = decode_text(content)

    segments = split_segments(text, max_segment_chars=max_segment_chars)
    return ParsedDocument(filename=filename, file_type=extension.lstrip("."), segments=segments)


def decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def extract_pdf_text(content: bytes) -> str:
    candidates: List[str] = []

    for extractor in (
        extract_pdf_text_with_pdftotext_raw,
        extract_pdf_text_with_pdftotext_layout,
        extract_pdf_text_with_pdfminer,
        extract_pdf_text_with_pypdf2,
    ):
        try:
            text = clean_extracted_pdf_text(extractor(content))
        except Exception:
            continue
        if not text.strip():
            continue
        if is_usable_pdf_text(text):
            return text
        candidates.append(text)

    try:
        ocr_text = clean_extracted_pdf_text(extract_pdf_text_with_ocr(content))
    except Exception:
        ocr_text = ""
    if ocr_text.strip():
        if is_usable_pdf_text(ocr_text):
            return ocr_text
        candidates.append(ocr_text)

    return max(candidates, key=pdf_text_quality_score, default="")


def extract_pdf_text_with_pypdf2(content: bytes) -> str:
    with io.BytesIO(content) as pdf_file:
        reader = PdfReader(pdf_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(page.strip() for page in pages if page.strip())


def extract_pdf_text_with_pdfminer(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(content)
        tmp.flush()
        return pdfminer_extract_text(tmp.name) or ""


def extract_pdf_text_with_pdftotext_raw(content: bytes) -> str:
    return extract_pdf_text_with_pdftotext(content, "-raw")


def extract_pdf_text_with_pdftotext_layout(content: bytes) -> str:
    return extract_pdf_text_with_pdftotext(content, "-layout")


def extract_pdf_text_with_pdftotext(content: bytes, mode: str) -> str:
    executable = shutil.which("pdftotext")
    if not executable:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(content)
        tmp.flush()
        result = subprocess.run(
            [executable, "-q", mode, "-enc", "UTF-8", tmp.name, "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )

    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", errors="replace")


def extract_pdf_text_with_ocr(content: bytes) -> str:
    pdftoppm = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if not pdftoppm or not tesseract:
        return ""

    lang = os.getenv("PDF_OCR_LANG", "eng+chi_sim+chi_tra")
    dpi = os.getenv("PDF_OCR_DPI", "200")
    max_pages = os.getenv("PDF_OCR_MAX_PAGES", "60")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        pdf_path = tmpdir_path / "input.pdf"
        image_prefix = tmpdir_path / "page"
        pdf_path.write_bytes(content)

        render_args = [pdftoppm, "-q", "-r", dpi, "-png"]
        if max_pages and max_pages != "0":
            render_args.extend(["-f", "1", "-l", max_pages])
        render_args.extend([str(pdf_path), str(image_prefix)])
        render_result = subprocess.run(
            render_args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if render_result.returncode != 0:
            return ""

        texts = []
        for image_path in sorted(tmpdir_path.glob("page-*.png")):
            result = subprocess.run(
                [tesseract, str(image_path), "stdout", "-l", lang, "--psm", "1"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )
            if result.returncode == 0:
                text = result.stdout.decode("utf-8", errors="replace").strip()
                if text:
                    texts.append(text)

    return "\n\n".join(texts)


def clean_extracted_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    pages = text.split("\x0c")
    cleaned_lines: List[str] = []

    for page in pages:
        lines = [normalize_pdf_line(line) for line in page.splitlines()]
        lines = remove_pdf_noise_lines(lines)
        if lines:
            cleaned_lines.extend(lines)

    cleaned = reflow_pdf_lines(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_pdf_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line.strip())
    line = re.sub(r"(?<=[a-z]),(?=[A-Z])", ", ", line)
    line = re.sub(r"(?<=[A-Za-z])ID(?=[*.,;\s]|$)", "", line)
    return line


PDF_METADATA_PREFIXES = (
    "citation:",
    "copyright:",
    "editor:",
    "published:",
    "funding:",
    "competing interests:",
    "data availability:",
    "received:",
    "accepted:",
    "correspondence:",
)


def remove_pdf_noise_lines(lines: List[str]) -> List[str]:
    cleaned = []
    discard_rest_of_page = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue

        if discard_rest_of_page:
            continue

        if is_pdf_noise_line(stripped):
            if stripped.upper() == "OPEN ACCESS":
                discard_rest_of_page = True
            continue

        if cleaned and re.match(r"^(Table|Figure|Fig\.)\s+\d+\b", stripped):
            discard_rest_of_page = True
            continue

        lowered = stripped.lower()
        if lowered.startswith(PDF_METADATA_PREFIXES):
            discard_rest_of_page = True
            continue

        cleaned.append(stripped)

    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def is_pdf_noise_line(line: str) -> bool:
    if re.fullmatch(r"a1{5,}", line):
        return True
    if line in {"PLOS COMPUTATIONAL BIOLOGY", "OPEN ACCESS", "EDITORIAL"}:
        return True
    if re.match(r"^PLOS Computational Biology\s+\|", line):
        return True
    if re.fullmatch(r"https?://doi\.org/\S+", line):
        return True
    if re.search(r"\b\d+\s*/\s*\d+\s*$", line) and "doi.org" in line:
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", line):
        return True
    return False


def reflow_pdf_lines(lines: List[str]) -> str:
    blocks: List[str] = []
    current: List[str] = []
    seen_body = False

    def flush_current() -> None:
        if current:
            blocks.append(fix_pdf_spacing(" ".join(current)))
            current.clear()

    for line in lines:
        if not line:
            flush_current()
            continue

        if not seen_body:
            if is_pdf_body_start(line):
                seen_body = True
                flush_current()
                blocks.append(line)
            else:
                blocks.append(fix_pdf_spacing(line))
            continue

        if is_pdf_heading_line(line):
            flush_current()
            blocks.append(line)
            continue

        if current and should_start_new_pdf_paragraph(current[-1], line):
            flush_current()
        current.append(line)

    flush_current()
    return "\n\n".join(block for block in blocks if block.strip())


def is_pdf_body_start(line: str) -> bool:
    return bool(
        re.fullmatch(r"(Abstract|Introduction|Background|Summary|Overview)", line, flags=re.IGNORECASE)
        or re.match(r"^Rule\s+\d+\b", line)
    )


def is_pdf_heading_line(line: str) -> bool:
    if is_pdf_body_start(line):
        return True
    if re.match(r"^(Rule|Table|Figure|Fig\.)\s+\d+\b", line):
        return True
    if len(line) <= 72 and not re.search(r"[.!?。！？,;:]$", line):
        words = re.findall(r"[A-Za-z]+", line)
        if 1 <= len(words) <= 8:
            return True
    return False


def should_start_new_pdf_paragraph(previous_line: str, line: str) -> bool:
    if re.match(r"^(Rule|Table|Figure|Fig\.)\s+\d+\b", line):
        return True
    if not re.search(r"[.!?。！？)”’\"'\]]$", previous_line):
        return False
    return bool(re.match(r"^[A-Z“\"'(\[]", line))


def fix_pdf_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:!?])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


COMMON_ENGLISH_WORDS = {
    "the",
    "and",
    "that",
    "for",
    "with",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "you",
    "your",
    "paper",
    "research",
    "science",
    "scientific",
    "reading",
}


def is_usable_pdf_text(text: str) -> bool:
    return pdf_text_quality_score(text) >= 0.55


def pdf_text_quality_score(text: str) -> float:
    sample = text.strip()[:12000]
    if len(sample) < 80:
        return 0.0

    nonspace = re.sub(r"\s+", "", sample)
    if not nonspace:
        return 0.0

    replacement_ratio = sample.count("\ufffd") / len(nonspace)
    odd_ratio = len(re.findall(r"[�€{}|~\\]", sample)) / len(nonspace)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", sample))
    latin_chars = re.findall(r"[A-Za-z]", sample)

    score = 1.0
    score -= min(replacement_ratio * 25, 0.8)
    score -= min(odd_ratio * 8, 0.8)

    if latin_chars:
        vowel_count = len(re.findall(r"[AEIOUaeiou]", "".join(latin_chars)))
        vowel_ratio = vowel_count / len(latin_chars)
        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", sample)
        common_hits = sum(1 for word in words if word.lower().strip("'") in COMMON_ENGLISH_WORDS)
        common_ratio = common_hits / max(len(words), 1)

        if vowel_ratio < 0.2:
            score -= 0.25
        if len(words) >= 30 and common_ratio < 0.015:
            score -= 0.25
        score += min(common_ratio * 3, 0.25)

    if cjk_count >= 20:
        score += 0.2

    return max(0.0, min(score, 1.0))


def extract_docx_text(content: bytes) -> str:
    result = mammoth.convert_to_html(io.BytesIO(content))
    return html_to_text(result.value)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "meta", "link"]):
        element.decompose()

    blocks = []
    for element in soup.find_all(["h1", "h2", "h3", "p", "li", "figcaption"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if text:
            blocks.append(text)

    if blocks:
        return "\n\n".join(blocks)

    return soup.get_text("\n", strip=True)


SUBTITLE_TIMESTAMP_PATTERN = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*"
    r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
)


def extract_subtitle_text(content: bytes) -> str:
    lines = decode_text(content).splitlines()
    text_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if stripped.upper() == "WEBVTT":
            continue
        if stripped.startswith(("NOTE", "STYLE", "REGION")):
            continue
        if "-->" in stripped and SUBTITLE_TIMESTAMP_PATTERN.match(stripped):
            continue
        text_lines.append(stripped)

    return "\n".join(text_lines)


def split_segments(text: str, max_segment_chars: int = 1600) -> List[Segment]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]

    line_based = len(raw_blocks) <= 1
    if line_based:
        raw_blocks = [line.strip() for line in normalized.splitlines() if line.strip()]

    split_blocks: List[str] = []
    for block in raw_blocks:
        if len(block) <= max_segment_chars:
            split_blocks.append(block)
        else:
            split_blocks.extend(split_long_block(block, max_segment_chars))

    separator = "\n" if line_based else "\n\n"
    segments = coalesce_blocks(split_blocks, max_segment_chars, separator)
    return [Segment(id=str(index), text=value) for index, value in enumerate(segments) if value.strip()]


def coalesce_blocks(blocks: List[str], max_segment_chars: int, separator: str = "\n\n") -> List[str]:
    """Merge short adjacent blocks so subtitles and line-based text summarize coherently."""
    segments: List[str] = []
    current = ""

    for block in blocks:
        if not block.strip():
            continue

        candidate = f"{current}{separator}{block}".strip() if current else block
        if current and len(candidate) > max_segment_chars:
            segments.append(current.strip())
            current = block
        else:
            current = candidate

    if current:
        segments.append(current.strip())

    return segments


def split_long_block(block: str, max_segment_chars: int) -> List[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", block)
    chunks: List[str] = []
    current = ""

    for part in parts:
        if not part:
            continue
        if current and len(current) + 1 + len(part) > max_segment_chars:
            chunks.append(current.strip())
            current = part
        else:
            current = f"{current} {part}".strip()

    if current:
        chunks.append(current.strip())

    if chunks:
        return chunks

    return [block[index:index + max_segment_chars] for index in range(0, len(block), max_segment_chars)]
