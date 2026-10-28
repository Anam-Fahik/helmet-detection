/* HelmetScan — uploader, preview and verdict rendering. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    dropzone: $("dropzone"),
    browseBtn: $("browse-btn"),
    fileInput: $("file-input"),
    stage: $("stage"),
    stageImage: $("stage-image"),
    stageBusy: $("stage-busy"),
    stageMeta: $("stage-meta"),
    stageDims: $("stage-dims"),
    stageActions: $("stage-actions"),
    runBtn: $("run-btn"),
    chooseBtn: $("choose-btn"),
    resultPanel: $("result-panel"),
    resultIdle: $("result-idle"),
    resultBody: $("result-body"),
    resultError: $("result-error"),
    errorTitle: $("error-title"),
    errorCopy: $("error-copy"),
    verdict: $("verdict"),
    verdictLabel: $("verdict-label"),
    verdictDetail: $("verdict-detail"),
    statPersons: $("stat-persons"),
    statWith: $("stat-with"),
    statWithout: $("stat-without"),
    statPlates: $("stat-plates"),
    detectionsList: $("detections-list"),
    detectionsCount: $("detections-count"),
    platesBlock: $("plates-block"),
    platesList: $("plates-list"),
    resultTiming: $("result-timing"),
    modelChip: $("model-chip"),
    modelChipText: $("model-chip-text"),
    modelNotice: $("model-notice"),
    modelNoticeCopy: $("model-notice-copy"),
    themeToggle: $("theme-toggle"),
    footConf: $("foot-conf"),
  };

  const state = {
    file: null,
    originalUrl: null,
    annotatedUrl: null,
    busy: false,
    maxBytes: 12 * 1024 * 1024,
  };

  const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];

  /* ---------------------------------------------------------------- theme */

  const applyTheme = (theme) => {
    document.documentElement.setAttribute("data-theme", theme);
    els.themeToggle.setAttribute("aria-pressed", String(theme === "light"));
    try {
      localStorage.setItem("helmetscan-theme", theme);
    } catch (_) {
      /* storage unavailable — theme stays for this session only */
    }
  };

  const initTheme = () => {
    let stored = null;
    try {
      stored = localStorage.getItem("helmetscan-theme");
    } catch (_) {
      stored = null;
    }
    const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
    applyTheme(stored || (prefersLight ? "light" : "dark"));
    els.themeToggle.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      applyTheme(next);
    });
  };

  /* --------------------------------------------------------------- health */

  const loadHealth = async () => {
    try {
      const response = await fetch("/api/health", { headers: { Accept: "application/json" } });
      const health = await response.json();
      if (typeof health.max_upload_mb === "number") {
        state.maxBytes = health.max_upload_mb * 1024 * 1024;
      }
      if (typeof health.confidence === "number") {
        els.footConf.textContent = health.confidence.toFixed(2);
      }
      const model = health.model || {};
      const ready = Boolean(health.weights_available);
      els.modelChip.classList.toggle("is-ok", ready);
      els.modelChip.classList.toggle("is-error", !ready);
      els.modelChipText.textContent = ready
        ? model.weights
          ? `model · ${model.weights.split(/[\\/]/).pop()}`
          : "model ready"
        : "no trained weights";
      els.modelChip.title = ready
        ? `Loaded weights: ${model.weights || "pending"}`
        : "Train the model with: python helmet_detection.py train --data <dataset>.yaml";

      if (model.note) {
        els.modelNoticeCopy.textContent = model.note;
        els.modelNotice.hidden = false;
      } else if (health.error) {
        els.modelNoticeCopy.textContent = health.error;
        els.modelNotice.hidden = false;
      }
    } catch (_) {
      els.modelChip.classList.add("is-error");
      els.modelChipText.textContent = "backend offline";
    }
  };

  /* ---------------------------------------------------------------- files */

  const releaseUrls = () => {
    if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);
    state.originalUrl = null;
    state.annotatedUrl = null;
  };

  const isAcceptable = (file) => {
    if (!file) return false;
    if (!ACCEPTED.includes(file.type)) return "Unsupported file type. Upload a JPG, PNG, WEBP or BMP image.";
    if (file.size > state.maxBytes) {
      const limit = Math.round(state.maxBytes / (1024 * 1024));
      return `Image is too large. The limit is ${limit} MB.`;
    }
    return null;
  };

  const showError = (title, copy) => {
    els.errorTitle.textContent = title;
    els.errorCopy.textContent = copy || "";
    els.resultError.hidden = false;
    els.resultIdle.hidden = true;
    els.resultBody.hidden = true;
  };

  const clearError = () => {
    els.resultError.hidden = true;
  };

  const setBusy = (busy) => {
    state.busy = busy;
    els.stageBusy.hidden = !busy;
    els.resultPanel.setAttribute("aria-busy", String(busy));
    els.runBtn.disabled = busy;
    els.runBtn.querySelector(".btn__label").textContent = busy ? "Detecting…" : "Run detection";
  };

  const handleFile = (file) => {
    const problem = isAcceptable(file);
    if (problem) {
      showError("Cannot use that file", problem);
      return;
    }
    clearError();
    releaseUrls();
    state.file = file;
    state.originalUrl = URL.createObjectURL(file);
    state.annotatedUrl = null;

    els.dropzone.hidden = true;
    els.stage.hidden = false;
    els.stageActions.hidden = false;
    els.stageImage.src = state.originalUrl;
    els.stageImage.alt = `Uploaded photo: ${file.name}`;
    els.stageMeta.textContent = file.name;
    els.stageDims.textContent = "";
    setView("original");
    runDetection();
  };

  /* ------------------------------------------------------------ detection */

  const runDetection = async () => {
    if (!state.file || state.busy) return;
    setBusy(true);
    els.resultIdle.hidden = true;
    els.resultBody.hidden = true;
    clearError();

    const body = new FormData();
    body.append("image", state.file, state.file.name);

    try {
      const response = await fetch("/api/detect", { method: "POST", body });
      const payload = await response.json().catch(() => ({
        ok: false,
        error: `Server returned ${response.status}.`,
      }));

      if (!response.ok || payload.ok === false) {
        showError(
          payload.code === "no_weights" ? "Model unavailable" : "Detection failed",
          payload.error || `Server returned ${response.status}.`
        );
        return;
      }
      renderResult(payload);
    } catch (error) {
      showError("Network error", "Could not reach the detection service. Is the server running?");
    } finally {
      setBusy(false);
    }
  };

  const renderResult = (payload) => {
    const summary = payload.summary || {};
    els.verdict.setAttribute("data-severity", payload.severity || "neutral");
    els.verdictLabel.textContent = payload.verdict_label || "Undetermined";
    els.verdictDetail.textContent = payload.verdict_detail || "";

    els.statPersons.textContent = summary.persons ?? 0;
    els.statWith.textContent = summary.with_helmet ?? 0;
    els.statWithout.textContent = summary.without_helmet ?? 0;
    els.statPlates.textContent = summary.plates ?? 0;

    const detections = payload.detections || [];
    els.detectionsCount.textContent = detections.length;
    els.detectionsList.innerHTML = "";

    if (!detections.length) {
      const empty = document.createElement("li");
      empty.className = "det__bbox";
      empty.textContent = "No objects were detected in this frame.";
      els.detectionsList.appendChild(empty);
    }

    detections.forEach((detection, index) => {
      const item = document.createElement("li");
      item.dataset.category = detection.category || "other";
      item.style.animation = `fade-rise 300ms var(--ease-out) ${index * 45}ms both`;

      const label = document.createElement("span");
      label.className = "det__label";
      label.textContent = detection.label;

      const confidence = document.createElement("span");
      confidence.className = "det__conf";
      confidence.textContent = `${Math.round((detection.confidence || 0) * 100)}%`;

      const bar = document.createElement("span");
      bar.className = "det__bar";
      const fill = document.createElement("i");
      fill.style.setProperty("--w", `${Math.round((detection.confidence || 0) * 100)}%`);
      bar.appendChild(fill);

      const bbox = document.createElement("span");
      bbox.className = "det__bbox";
      const [x1, y1, x2, y2] = detection.bbox || [];
      bbox.textContent = `x ${x1} · y ${y1} · ${detection.width || 0}×${detection.height || 0} px`;

      item.append(label, confidence, bar, bbox);
      els.detectionsList.appendChild(item);
    });

    const plates = (payload.plates || []).filter((plate) => plate.text);
    els.platesBlock.hidden = plates.length === 0;
    els.platesList.innerHTML = "";
    plates.forEach((plate) => {
      const item = document.createElement("li");
      item.textContent = plate.text;
      els.platesList.appendChild(item);
    });

    if (payload.image_size) {
      els.stageDims.textContent = `${payload.image_size.width} × ${payload.image_size.height} px`;
    }
    els.resultTiming.textContent = `${payload.elapsed_ms ?? 0} ms`;

    if (payload.annotated_image) {
      state.annotatedUrl = payload.annotated_image;
      setView("annotated");
    }

    const model = payload.model || {};
    if (model.note) {
      els.modelNoticeCopy.textContent = model.note;
      els.modelNotice.hidden = false;
    }

    els.resultIdle.hidden = true;
    els.resultError.hidden = true;
    els.resultBody.hidden = false;
  };

  /* ----------------------------------------------------------------- view */

  const setView = (view) => {
    document.querySelectorAll(".seg").forEach((button) => {
      const active = button.dataset.view === view;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    if (view === "annotated" && state.annotatedUrl) {
      els.stageImage.src = state.annotatedUrl;
    } else if (state.originalUrl) {
      els.stageImage.src = state.originalUrl;
    }
  };

  const reset = () => {
    releaseUrls();
    state.file = null;
    els.stage.hidden = true;
    els.stageActions.hidden = true;
    els.dropzone.hidden = false;
    els.resultIdle.hidden = false;
    els.resultBody.hidden = true;
    els.resultError.hidden = true;
    els.resultTiming.textContent = "";
    els.fileInput.value = "";
  };

  /* --------------------------------------------------------------- events */

  const initEvents = () => {
    els.dropzone.addEventListener("click", () => els.fileInput.click());
    els.dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        els.fileInput.click();
      }
    });

    els.browseBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      els.fileInput.click();
    });

    els.fileInput.addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) handleFile(file);
    });

    ["dragenter", "dragover"].forEach((type) => {
      els.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        els.dropzone.classList.add("is-dragging");
      });
    });

    ["dragleave", "dragend"].forEach((type) => {
      els.dropzone.addEventListener(type, () => els.dropzone.classList.remove("is-dragging"));
    });

    els.dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("is-dragging");
      const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    document.addEventListener("paste", (event) => {
      const items = event.clipboardData && event.clipboardData.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            handleFile(file);
            break;
          }
        }
      }
    });

    document.querySelectorAll(".seg").forEach((button) => {
      button.addEventListener("click", () => setView(button.dataset.view));
    });

    els.runBtn.addEventListener("click", () => runDetection());
    els.chooseBtn.addEventListener("click", () => {
      reset();
      els.fileInput.click();
    });

    window.addEventListener("beforeunload", releaseUrls);
  };

  initTheme();
  initEvents();
  loadHealth();
})();
