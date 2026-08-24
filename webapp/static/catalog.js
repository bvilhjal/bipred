/* Live GWAS Catalog lookup for the accession inputs: resolve the accession,
 * show what will be downloaded, and prefill label / sample size when the
 * user has not entered them. Advisory only — the server re-resolves at
 * submit time. */
(function () {
  "use strict";
  const GCST = /^GCST\d{3,}$/i;

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

  async function lookup(input, info, trait) {
    const accession = input.value.trim().toUpperCase();
    if (!accession) { info.innerHTML = ""; return; }
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
      info.innerHTML = '<span class="warn">' + esc(err.message) + "</span>";
      return;
    }
    let html = '<div class="chips"><span class="chip"><b>' +
               esc(meta.accession) + "</b>&rarr;" + esc(meta.trait) +
               "</span></div>";
    const n = fmtN(meta);
    if (n) html += '<span class="muted">' + esc(n) + ".</span> ";
    if (meta.remote_bytes) {
      html += '<span class="muted">Harmonised file: ' +
              Math.round(meta.remote_bytes / 1048576) +
              " MB (filtered to the LD reference on download).</span>";
    }
    info.innerHTML = html;

    // Prefill what the user has not supplied; never overwrite an entry.
    const label = document.getElementById("label" + trait);
    if (label && (!label.value.trim() ||
                  label.value.trim() === "Trait " + trait)) {
      label.value = meta.trait;
    }
    const nEff = document.getElementById("n_eff" + trait);
    const form = input.form;
    const cases = form.querySelector('[name="n_cases' + trait + '"]');
    const controls = form.querySelector('[name="n_controls' + trait + '"]');
    const nEmpty = (!nEff || !nEff.value.trim()) &&
                   (!cases || !cases.value.trim()) &&
                   (!controls || !controls.value.trim());
    if (nEmpty) {
      if (meta.n_cases && meta.n_controls && cases && controls) {
        cases.value = meta.n_cases;
        controls.value = meta.n_controls;
      } else if (meta.n_eff && nEff) {
        nEff.value = meta.n_eff;
      }
    }
  }

  for (const trait of [1, 2]) {
    const input = document.getElementById("gcst" + trait);
    const info = document.getElementById("catalog" + trait);
    if (!input || !info) continue;
    let timer = null;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => lookup(input, info, trait), 500);
    });
    if (input.value.trim()) lookup(input, info, trait);   // re-rendered form
  }
})();
