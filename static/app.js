(function () {
  const els = {
    form: document.getElementById("translateForm"),
    fileInput: document.getElementById("fileInput"),
    processingMode: document.getElementById("processingMode"),
    direction: document.getElementById("direction"),
    maxSegmentChars: document.getElementById("maxSegmentChars"),
    refineContextNeighbors: document.getElementById("refineContextNeighbors"),
    startButton: document.getElementById("startButton"),
    refineButton: document.getElementById("refineButton"),
    cancelButton: document.getElementById("cancelButton"),
    status: document.getElementById("status"),
    modelLine: document.getElementById("modelLine"),
    documentLabel: document.getElementById("documentLabel"),
    stats: document.getElementById("stats"),
    translationFontSize: document.getElementById("translationFontSize"),
    translationFontSizeValue: document.getElementById("translationFontSizeValue"),
    progress: document.getElementById("progress"),
    segments: document.getElementById("segments")
  };

  const translationFontSizeKey = "llmTranslator.translationFontSize";
  const defaultTranslationFontSize = 15;
  const minTranslationFontSize = 13;
  const maxTranslationFontSize = 18;

  let currentAbort = null;
  let segmentCount = 0;
  let completedCount = 0;

  function normalizeTranslationFontSize(value) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return defaultTranslationFontSize;
    return Math.min(maxTranslationFontSize, Math.max(minTranslationFontSize, parsed));
  }

  function applyTranslationFontSize(value, persist) {
    const size = normalizeTranslationFontSize(value);
    document.documentElement.style.setProperty("--translation-font-size", `${size}px`);
    els.translationFontSize.value = String(size);
    els.translationFontSizeValue.textContent = `${size}px`;
    if (persist) {
      try {
        window.localStorage.setItem(translationFontSizeKey, String(size));
      } catch (_error) {
        // Local storage can be unavailable in restricted browser contexts.
      }
    }
  }

  function restoreTranslationFontSize() {
    let storedSize = "";
    try {
      storedSize = window.localStorage.getItem(translationFontSizeKey) || "";
    } catch (_error) {
      storedSize = "";
    }
    applyTranslationFontSize(
      storedSize || defaultTranslationFontSize,
      false
    );
  }

  function setStatus(message, tone) {
    els.status.textContent = message;
    els.status.classList.toggle("is-busy", tone === "busy");
    els.status.classList.toggle("is-error", tone === "error");
  }

  function updateModelLine(data) {
    const translationModel = data.translation_model || data.model || "unknown";
    const summaryModel = data.summary_model || translationModel;
    const modelText = summaryModel && summaryModel !== translationModel
      ? `Ollama models: ${translationModel} / summary ${summaryModel}`
      : `Ollama model: ${translationModel}`;

    if (data.online === false) {
      els.modelLine.textContent = `${modelText} / offline`;
      return;
    }
    if (data.ready === false || data.ok === false) {
      els.modelLine.textContent = data.auto_pull
        ? `${modelText} / auto-pull enabled`
        : `${modelText} / missing`;
      return;
    }
    els.modelLine.textContent = `${modelText} / ready`;
  }

  function setRunning(isRunning) {
    els.startButton.disabled = isRunning;
    els.refineButton.disabled = isRunning;
    els.cancelButton.disabled = !isRunning;
  }

  function resetOutput() {
    segmentCount = 0;
    completedCount = 0;
    els.progress.value = 0;
    els.progress.max = 1;
    els.stats.textContent = "";
    els.segments.innerHTML = "";
    els.segments.classList.add("empty");
  }

  function applySegmentKind(segment, kind) {
    const safeKind = ["title", "heading", "front-matter", "paragraph"].includes(kind)
      ? kind
      : "paragraph";
    segment.dataset.segmentKind = safeKind;
    segment.classList.remove(
      "segment--title",
      "segment--heading",
      "segment--front-matter",
      "segment--paragraph"
    );
    segment.classList.add(`segment--${safeKind}`);
  }

  function ensureSegment(id, sourceText, kind) {
    let segment = document.querySelector(`[data-segment-id="${cssEscape(id)}"]`);
    if (segment) {
      if (kind) applySegmentKind(segment, kind);
      return segment;
    }

    els.segments.classList.remove("empty");
    segment = document.createElement("section");
    segment.className = "segment";
    segment.dataset.segmentId = id;
    applySegmentKind(segment, kind);
    segment.innerHTML = [
      `<div class="cell source-cell">`,
      `<pre class="text-block source"></pre>`,
      `</div>`,
      `<div class="cell translation-cell">`,
      `<span class="segment-state"></span>`,
      `<div class="text-block translation"></div>`,
      `</div>`
    ].join("");
    segment.querySelector(".source").textContent = sourceText || "";
    els.segments.appendChild(segment);
    return segment;
  }

  function updateSegmentSource(id, sourceText, streaming) {
    const segment = ensureSegment(id, sourceText);
    const source = segment.querySelector(".source");
    source.textContent = sourceText || "";
    source.classList.toggle("is-streaming", Boolean(streaming));
  }

  function updateSegmentStatus(id, status) {
    const segment = ensureSegment(id, "");
    segment.querySelector(".segment-state").textContent = statusLabel(status);
  }

  function updateSegmentTranslation(id, text, streaming) {
    const segment = ensureSegment(id, "");
    const target = segment.querySelector(".translation");
    target.textContent = text || "";
    target.classList.toggle("is-streaming", Boolean(streaming));
  }

  function finishSegment(id, text, event) {
    const segment = ensureSegment(id, "");
    updateSegmentTranslation(id, text || "", false);
    segment.querySelector(".segment-state").textContent = "";
    typesetMath(segment.querySelector(".translation"));
    completedCount += 1;
    els.progress.value = completedCount;
  }

  function statusLabel(status) {
    const labels = {
      summarizing: "Summarizing",
      summarizing_context: "Summarizing context",
      summary_cached: "Using summary",
      refining_summary: "Refining summary",
      translating: "Translating",
      refining: "Refining"
    };
    return labels[status] || "Working";
  }

  function typesetMath(node) {
    if (!node || !window.MathJax || !window.MathJax.typesetPromise) return;
    if (window.MathJax.typesetClear) {
      window.MathJax.typesetClear([node]);
    }
    window.MathJax.typesetPromise([node]).catch(() => {});
  }

  async function startTranslation(event, quality) {
    if (event) {
      event.preventDefault();
    }
    if (!els.fileInput.files || els.fileInput.files.length === 0) {
      setStatus("Choose a file first", "error");
      return;
    }

    if (currentAbort) {
      currentAbort.abort();
    }
    currentAbort = new AbortController();
    resetOutput();
    setRunning(true);
    setStatus("Uploading", "busy");

    const formData = new FormData();
    formData.append("file", els.fileInput.files[0]);
    formData.append("processing_mode", els.processingMode.value);
    formData.append("quality", quality);
    formData.append("direction", els.direction.value);
    formData.append("max_segment_chars", els.maxSegmentChars.value);
    formData.append("refine_context_neighbors", els.refineContextNeighbors.value);

    try {
      const response = await fetch("/api/translate/stream", {
        method: "POST",
        body: formData,
        signal: currentAbort.signal
      });

      if (!response.ok) {
        throw new Error("Translation request failed.");
      }
      if (!response.body) {
        throw new Error("Streaming response is not available.");
      }

      const activeMode = els.processingMode.value === "summarize" ? "Summarizing" : "Translating";
      setStatus(activeMode, "busy");
      await consumeStream(response.body);
      setStatus("Complete", null);
    } catch (error) {
      if (error.name !== "AbortError") {
        setStatus(error.message, "error");
        checkHealth();
      } else {
        setStatus("Canceled", null);
      }
    } finally {
      setRunning(false);
      currentAbort = null;
    }
  }

  async function consumeStream(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        handleEvent(JSON.parse(line));
      }

      if (done) break;
    }

    if (buffer.trim()) {
      handleEvent(JSON.parse(buffer));
    }
  }

  function handleEvent(event) {
    if (event.type === "model_status") {
      updateModelLine(event);
      setStatus(event.message || "Checking Ollama model", "busy");
      return;
    }
    if (event.type === "pipeline_stage") {
      setStatus(event.message || "Processing", "busy");
      return;
    }
    if (event.type === "meta") {
      segmentCount = event.segment_count || 0;
      els.progress.max = Math.max(segmentCount, 1);
      els.progress.value = 0;
      els.documentLabel.textContent = `${event.filename} (${event.file_type})`;
      updateModelLine(event);
      const contextLabel = event.quality === "refine"
        ? ` / context ±${event.refine_context_neighbors || 1}`
        : "";
      els.stats.textContent = `${event.source_lang} to ${event.target_lang} / ${event.mode} / ${event.quality}${contextLabel}`;
      setStatus(event.mode === "summarize" ? "Summarizing" : "Translating", "busy");
      return;
    }
    if (event.type === "segment_start") {
      ensureSegment(event.id, event.source || "", event.kind);
      updateSegmentStatus(event.id, "translating");
      return;
    }
    if (event.type === "segment_source_update") {
      updateSegmentSource(event.id, event.source || "", event.streaming);
      return;
    }
    if (event.type === "segment_status") {
      updateSegmentStatus(event.id, event.status);
      return;
    }
    if (event.type === "chunk") {
      updateSegmentTranslation(event.id, event.text || "", true);
      return;
    }
    if (event.type === "segment_done") {
      finishSegment(event.id, event.translation || "", event);
      return;
    }
    if (event.type === "done") {
      const stats = event.stats || {};
      const parts = [
        `${stats.translated || 0} translated`,
        `${stats.cached || 0} cached`
      ];
      if ((stats.summarized || 0) > 0 || (stats.summary_cached || 0) > 0) {
        parts.push(`${stats.summarized || 0} summarized`);
        parts.push(`${stats.summary_cached || 0} summaries reused`);
      }
      if ((stats.summary_refined || 0) > 0 || (stats.summary_refine_cached || 0) > 0) {
        parts.push(`${stats.summary_refined || 0} summary refinements`);
        parts.push(`${stats.summary_refine_cached || 0} refined summaries reused`);
      }
      els.stats.textContent = parts.join(", ");
      return;
    }
    if (event.type === "error") {
      throw new Error(event.message || "Translation failed.");
    }
  }

  async function checkHealth() {
    try {
      const response = await fetch("/api/health");
      const data = await response.json();
      updateModelLine(data);
      if (data.online === false) {
        setStatus(data.error || "Ollama unavailable", "error");
      } else if (!data.ok) {
        setStatus(data.error || "Ollama model missing", data.auto_pull ? "busy" : "error");
      }
    } catch (_error) {
      setStatus("Health check failed", "error");
    }
  }

  function cancelTranslation() {
    if (currentAbort) {
      currentAbort.abort();
    }
  }

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === "function") {
      return window.CSS.escape(String(value));
    }
    return String(value).replace(/"/g, '\\"');
  }

  els.form.addEventListener("submit", (event) => startTranslation(event, "quick"));
  els.refineButton.addEventListener("click", () => startTranslation(null, "refine"));
  els.cancelButton.addEventListener("click", cancelTranslation);
  els.translationFontSize.addEventListener("input", () => {
    applyTranslationFontSize(els.translationFontSize.value, true);
  });
  restoreTranslationFontSize();
  checkHealth();
})();
