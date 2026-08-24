/* Live, advisory GWAS Catalog lookup. The server re-resolves on submit. */
(function () {
  "use strict";
  const GCST = /^GCST\d{3,}$/i;
  const states = {1: {seq: 0, auto: {}}, 2: {seq: 0, auto: {}}};
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => (
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

  function fmtN(meta) {
    if (meta.n_cases && meta.n_controls) {
      return Number(meta.n_cases).toLocaleString() + " cases / " +
             Number(meta.n_controls).toLocaleString() + " controls";
    }
    if (meta.n_eff) return "N ≈ " + Number(meta.n_eff).toLocaleString();
    return null;
  }

  function mayReplace(element, state, key, defaultValue) {
    if (!element) return false;
    const value = element.value.trim();
    return !value || value === defaultValue || value === String(state.auto[key] || "");
  }

  function setAuto(element, state, key, value) {
    if (!element) return;
    element.value = value == null ? "" : value;
    state.auto[key] = element.value;
  }

  async function lookup(input, info, trait) {
    const state = states[trait];
    const seq = ++state.seq;
    const accession = input.value.trim().toUpperCase();
    const form = input.form;
    const label = document.getElementById("label" + trait);
    const nEff = document.getElementById("n_eff" + trait);
    const cases = form.querySelector('[name="n_cases' + trait + '"]');
    const controls = form.querySelector('[name="n_controls' + trait + '"]');
    const autoN = document.getElementById("catalog_auto_n" + trait);
    const autoLabel = document.getElementById("catalog_auto_label" + trait);
    if (!accession) {
      info.innerHTML = "";
      if (autoLabel && autoLabel.value === "1") {
        setAuto(label, state, "label", "Trait " + trait);
        autoLabel.value = "";
      }
      if (autoN && autoN.value === "1") {
        setAuto(nEff, state, "nEff", "");
        setAuto(cases, state, "cases", "");
        setAuto(controls, state, "controls", "");
        autoN.value = "";
      }
      return;
    }
    if (!GCST.test(accession)) {
      info.innerHTML = '<span class="warn">Expected a GCST accession like ' +
                       "GCST90446168.</span>";
      return;
    }
    info.innerHTML = '<span class="muted">Looking up ' + esc(accession) +
                     "…</span>";
    let meta;
    try {
      const response = await fetch("/catalog/lookup?accession=" +
                                   encodeURIComponent(accession));
      meta = await response.json();
      if (!response.ok) throw new Error(meta.error || "lookup failed");
    } catch (err) {
      if (seq === state.seq) {
        info.innerHTML = '<span class="warn">' + esc(err.message) + "</span>";
      }
      return;
    }
    // A slower response for a previous accession must never populate this form.
    if (seq !== state.seq || input.value.trim().toUpperCase() !== accession) return;

    let html = '<div class="chips"><span class="chip"><b>' +
               esc(meta.accession) + "</b>→" + esc(meta.trait) +
               "</span></div>";
    const n = fmtN(meta);
    if (n) html += '<span class="muted">' + esc(n) + ".</span> ";
    if (meta.remote_bytes) {
      html += '<span class="muted">Harmonised file: ' +
              Math.round(meta.remote_bytes / 1048576) +
              " MB; streamed and filtered to the LD reference.</span>";
    }
    info.innerHTML = html;

    if (mayReplace(label, state, "label", "Trait " + trait)) {
      setAuto(label, state, "label", meta.trait);
      if (autoLabel) autoLabel.value = "1";
    }
    if (meta.n_cases && meta.n_controls &&
        mayReplace(cases, state, "cases", "") &&
        mayReplace(controls, state, "controls", "") &&
        mayReplace(nEff, state, "nEff", "")) {
      setAuto(nEff, state, "nEff", "");
      setAuto(cases, state, "cases", meta.n_cases);
      setAuto(controls, state, "controls", meta.n_controls);
      if (autoN) autoN.value = "1";
    } else if (meta.n_eff && mayReplace(nEff, state, "nEff", "") &&
               mayReplace(cases, state, "cases", "") &&
               mayReplace(controls, state, "controls", "")) {
      setAuto(cases, state, "cases", "");
      setAuto(controls, state, "controls", "");
      setAuto(nEff, state, "nEff", meta.n_eff);
      if (autoN) autoN.value = "1";
    }
  }

  for (const trait of [1, 2]) {
    const input = document.getElementById("gcst" + trait);
    const info = document.getElementById("catalog" + trait);
    if (!input || !info) continue;
    const state = states[trait];
    const autoN = document.getElementById("catalog_auto_n" + trait);
    const autoLabel = document.getElementById("catalog_auto_label" + trait);
    for (const [key, element] of [
      ["label", document.getElementById("label" + trait)],
      ["nEff", document.getElementById("n_eff" + trait)],
      ["cases", input.form.querySelector('[name="n_cases' + trait + '"]')],
      ["controls", input.form.querySelector('[name="n_controls' + trait + '"]')]
    ]) {
      if (element) {
        if ((key === "label" && autoLabel && autoLabel.value === "1") ||
            (key !== "label" && autoN && autoN.value === "1")) {
          state.auto[key] = element.value;
        }
        element.addEventListener("input", function () {
          if (element.value !== String(state.auto[key] || "")) {
            state.auto[key] = null;
            if (key === "label" && autoLabel) autoLabel.value = "";
            if (key !== "label" && autoN) autoN.value = "";
          }
        });
      }
    }
    let timer = null;
    input.addEventListener("input", function () {
      clearTimeout(timer);
      ++state.seq;                      // invalidate any in-flight response now
      timer = setTimeout(() => lookup(input, info, trait), 500);
    });
    if (input.value.trim()) lookup(input, info, trait);
  }
})();
