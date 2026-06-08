# FastAPI LLM Translator

A local FastAPI version of the LLM document translator. It keeps the current
project's core workflow while moving streaming display into browser-managed DOM:

- upload PDF, DOCX, TXT, HTML, VTT, or SRT files
- split extracted content into readable segments
- choose document translation or summarize mode
- stream Ollama translation output per segment
- use quick translation by default, with a separate Refine action
- refine with selectable neighboring segment context windows
- reuse cached summaries and quick translations when refining summarize output
- preserve LaTeX math and code with placeholders
- render final math with MathJax
- cache completed segment translations and summaries locally
- prefer Poppler `pdftotext` for PDFs, with OCR fallback when text extraction is unusable

## Run

Start Ollama. The app checks the configured model on each run and can pull it
from Ollama automatically when it is missing locally.

Install dependencies and start the app:

```bash
cd /Users/longtenghai/code/LLM_Translator_fastapi
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 app.py
```

For best PDF extraction, install Poppler. OCR fallback also requires Tesseract:

```bash
brew install poppler tesseract tesseract-lang
```

Open:

```text
http://127.0.0.1:8766
```

## Configuration

Edit `config.json` to switch models:

```json
{
  "ollama": {
    "base_url": "http://localhost:11434/api/generate",
    "translation_model": "translategemma",
    "summary_model": "",
    "auto_pull": true,
    "request_timeout": 300,
    "pull_timeout": 1800
  }
}
```

- `translation_model`: model used for translation and refinement
- `summary_model`: model used for summaries; leave empty to reuse `translation_model`
- `auto_pull`: when `true`, missing models are pulled through Ollama before processing starts

Environment variables still override `config.json`:

- `OLLAMA_BASE_URL`: defaults to `http://localhost:11434/api/generate`
- `OLLAMA_TRANSLATION_MODEL`: defaults to `translategemma`
- `OLLAMA_SUMMARY_MODEL`: defaults to the translation model
- `OLLAMA_AUTO_PULL`: defaults to `true`
- `OLLAMA_REQUEST_TIMEOUT`: defaults to `300`
- `OLLAMA_PULL_TIMEOUT`: defaults to `1800`
- `TRANSLATOR_CONFIG_PATH`: optional path to a different JSON config file

## Notes

The PDF path prefers Poppler `pdftotext`, then falls back to `pdfminer.six`,
`PyPDF2`, and finally OCR through Poppler + Tesseract when extracted text looks
garbled. This avoids browser automation for routine uploads while still handling
PDFs with broken embedded font mappings.
