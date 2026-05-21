(function () {
  const els = {
    form: document.getElementById("translateForm"),
    fileInput: document.getElementById("fileInput"),
    processingMode: document.getElementById("processingMode"),
    direction: document.getElementById("direction"),
    maxSegmentChars: document.getElementById("maxSegmentChars"),
    startButton: document.getElementById("startButton"),
    refineButton: document.getElementById("refineButton"),
    cancelButton: document.getElementById("cancelButton"),
    status: document.getElementById("status"),
    modelLine: document.getElementById("modelLine"),
    documentLabel: document.getElementById("documentLabel"),
    stats: document.getElementById("stats"),
    progress: document.getElementById("progress"),
    segments: document.getElementById("segments")
  };

  let currentAbort = null;
  let segmentCount = 0;
  let completedCount = 0;

  function setStatus(message, tone) {
    els.status.textContent = message;
    els.status.classList.toggle("is-busy", tone === "busy");
    els.status.classList.toggle("is-error", tone === "error");
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

  function ensureSegment(id, sourceText) {
    let segment = document.querySelector(`[data-segment-id="${cssEscape(id)}"]`);
    if (segment) return segment;

    els.segments.classList.remove("empty");
    segment = document.createElement("section");
    segment.className = "segment";
    segment.dataset.segmentId = id;
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

  function updateSegmentSource(id, sourceText) {
    const segment = ensureSegment(id, sourceText);
    segment.querySelector(".source").textContent = sourceText || "";
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
      summary_cached: "Using summary",
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

      setStatus("Translating", "busy");
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
    if (event.type === "meta") {
      segmentCount = event.segment_count || 0;
      els.progress.max = Math.max(segmentCount, 1);
      els.progress.value = 0;
      els.documentLabel.textContent = `${event.filename} (${event.file_type})`;
      els.modelLine.textContent = `Ollama model: ${event.model}`;
      els.stats.textContent = `${event.source_lang} to ${event.target_lang} / ${event.mode} / ${event.quality}`;
      return;
    }
    if (event.type === "segment_start") {
      ensureSegment(event.id, event.source || "");
      updateSegmentStatus(event.id, "translating");
      return;
    }
    if (event.type === "segment_source_update") {
      updateSegmentSource(event.id, event.source || "");
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
      els.modelLine.textContent = `Ollama model: ${data.model}`;
      if (!data.ok) {
        setStatus(data.error || "Ollama unavailable", "error");
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
  checkHealth();
})();
