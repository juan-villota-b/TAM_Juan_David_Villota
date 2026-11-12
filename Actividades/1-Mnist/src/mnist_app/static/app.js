(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file");
  const statusLine = document.getElementById("status");
  const errorBox = document.getElementById("error");
  const result = document.getElementById("result");
  const originalEl = document.getElementById("original");
  const processedEl = document.getElementById("processed");
  const resetButton = document.getElementById("reset");
  const tabs = Array.from(document.querySelectorAll("[role='tab']"));

  let busy = false;
  let objectUrl = null;

  function createRows(container) {
    return Array.from({ length: 10 }, (_, digit) => {
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `
        <span class="k">${digit}</span>
        <div class="track"><div class="fill"></div></div>
        <span class="pct">0 %</span>
      `;
      container.append(row);
      return { row, fill: row.querySelector(".fill"), pct: row.querySelector(".pct") };
    });
  }

  const views = {};
  document.querySelectorAll(".model-panel").forEach((panel) => {
    views[panel.dataset.model] = {
      panel,
      digit: panel.querySelector("[data-role='digit']"),
      conf: panel.querySelector("[data-role='conf']"),
      rows: createRows(panel.querySelector("[data-role='dist']")),
    };
  });

  function activate(model, { focus = false } = {}) {
    tabs.forEach((tab) => {
      const selected = tab.dataset.model === model;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focus) tab.focus();
    });
    Object.entries(views).forEach(([key, view]) => {
      view.panel.hidden = key !== model;
    });
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activate(tab.dataset.model));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(tabs.indexOf(tab) + step + tabs.length) % tabs.length];
      activate(next.dataset.model, { focus: true });
    });
  });

  function setError(message) {
    errorBox.textContent = message || "";
    errorBox.hidden = !message;
  }

  function setStatus(message) {
    statusLine.textContent = message || "";
  }

  function setBusy(value) {
    busy = value;
    dropzone.classList.toggle("is-busy", value);
    dropzone.setAttribute("aria-busy", value ? "true" : "false");
  }

  function renderModel(view, prediction) {
    view.digit.textContent = String(prediction.digit);
    view.conf.textContent = `${(prediction.confidence * 100).toFixed(1)} %`;

    prediction.probabilities.forEach((probability, digit) => {
      const percentage = probability * 100;
      const { row, fill, pct } = view.rows[digit];
      row.classList.toggle("is-top", digit === prediction.digit);
      fill.style.width = `${Math.max(percentage, 0.4)}%`;
      pct.textContent = `${percentage.toFixed(1)} %`;
    });
  }

  function render(payload, file) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    originalEl.src = objectUrl;
    processedEl.src = payload.processed_png;

    Object.entries(views).forEach(([model, view]) => {
      if (payload[model]) renderModel(view, payload[model]);
    });

    dropzone.hidden = true;
    result.hidden = false;
  }

  async function submit(file) {
    if (busy || !file) return;

    if (file.type && !file.type.startsWith("image/")) {
      setError("El archivo no es una imagen. Usá PNG, JPG o WEBP.");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setError("La imagen supera los 8 MB.");
      return;
    }

    setError("");
    setBusy(true);
    setStatus("Procesando imagen…");

    const body = new FormData();
    body.append("image", file);

    try {
      const response = await fetch("/predict", { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `Error del servidor (${response.status}).`);
      }
      render(payload, file);
    } catch (error) {
      setError(error.message || "No se pudo procesar la imagen.");
    } finally {
      setBusy(false);
      setStatus("");
      fileInput.value = "";
    }
  }

  function openPicker() {
    if (!busy) fileInput.click();
  }

  dropzone.addEventListener("click", openPicker);
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) submit(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!busy) dropzone.classList.add("is-drag");
    });
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove("is-drag"));
  });

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-drag");
    const file = event.dataTransfer?.files?.[0];
    if (file) submit(file);
  });

  window.addEventListener("paste", (event) => {
    const file = Array.from(event.clipboardData?.files || [])[0];
    if (file) submit(file);
  });

  resetButton.addEventListener("click", () => {
    setError("");
    result.hidden = true;
    dropzone.hidden = false;
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    originalEl.removeAttribute("src");
    processedEl.removeAttribute("src");
    Object.values(views).forEach((view) => {
      view.digit.textContent = "–";
      view.conf.textContent = "–";
      view.rows.forEach(({ row, fill, pct }) => {
        row.classList.remove("is-top");
        fill.style.width = "0%";
        pct.textContent = "0 %";
      });
    });
    dropzone.focus();
  });

  activate("dense");
})();
