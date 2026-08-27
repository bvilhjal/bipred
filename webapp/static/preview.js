/* Client-side header preview for the upload form. Reads only a small slice
 * of each selected file (never uploads it), maps the header against the same
 * alias table ldpred3 uses server-side, and reports what was recognized.
 * Advisory only — the runner's detect_columns remains the authority. */
(function () {
  "use strict";
  const ALIASES = window.BIPRED_ALIASES || {};
  const MAX_BYTES = (window.BIPRED_MAX_MB || 0) * 1024 * 1024;
  const SLICE = 256 * 1024;
  /* Required to fit: an id, both alleles, an effect size (beta or OR), SE. */
  const REQUIRED = ["id", "ea", "oa", "se"];

  function sniffDelimiter(line) {
    if (line.indexOf("\t") !== -1) return "\t";
    if (line.indexOf(",") !== -1) return ",";
    return null;                            // whitespace-separated
  }

  async function readHeaderLine(file) {
    const gz = /\.(gz|bgz)$/i.test(file.name);
    if (gz && !("DecompressionStream" in window)) return null;
    if (gz) {
      const stream = file.slice(0, SLICE).stream()
        .pipeThrough(new DecompressionStream("gzip"));
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let text = "";
      try {
        while (text.indexOf("\n") === -1 && text.length < SLICE) {
          const {done, value} = await reader.read();
          if (done) break;
          text += decoder.decode(value, {stream: true});
        }
      } catch (err) {
        return null;                          // truncated/corrupt gzip slice
      }
      reader.cancel().catch(() => {});
      return text.split("\n")[0] || "";
    }
    const buf = await file.slice(0, SLICE).arrayBuffer();
    return (new TextDecoder().decode(buf)).split("\n")[0] || "";
  }

  /* Mirror ldpred3.sumstats._sniff_delimiter / _build_colmap. */
  function mapHeader(line) {
    const clean = line.replace(/^\uFEFF/, "").trim();
    if (!clean) return null;
    const d = sniffDelimiter(clean);
    const raw = d ? clean.split(d) : clean.split(/\s+/);
    const lower = raw.map((h) => h.trim().toLowerCase());
    const map = {};
    for (const field of Object.keys(ALIASES)) {
      for (const alias of ALIASES[field]) {
        const i = lower.indexOf(alias);
        if (i !== -1) { map[field] = i; break; }
      }
    }
    return {map, raw};
  }

  function render(el, file, parsed, headerUnavailable) {
    const nodes = [];
    function message(className, text) {
      const item = document.createElement("span");
      item.className = className;
      item.textContent = text;
      nodes.push(item);
    }
    if (MAX_BYTES && file.size > MAX_BYTES) {
      message("warn", file.name + " exceeds the " + window.BIPRED_MAX_MB +
              " MB combined upload limit on its own.");
    }
    if (headerUnavailable) {
      message("muted", "Header preview unavailable for this file in this " +
              "browser; columns will be detected when the job runs.");
    } else if (!parsed || !Object.keys(parsed.map).length) {
      message("warn", "No recognized columns in the header. Use the " +
              "Advanced column overrides below.");
    } else {
      const chips = document.createElement("div");
      chips.className = "chips";
      for (const field of Object.keys(parsed.map)) {
        const chip = document.createElement("span");
        chip.className = "chip";
        const key = document.createElement("b");
        key.textContent = field;
        chip.append(key, document.createTextNode("→" +
          parsed.raw[parsed.map[field]].trim()));
        chips.append(chip);
      }
      nodes.push(chips);
      const missing = REQUIRED.filter((f) => !(f in parsed.map));
      if (!("beta" in parsed.map) && !("or" in parsed.map)) {
        missing.push("beta (or or)");
      }
      if (missing.length) {
        message("warn", "Not recognized: " + missing.join(", ") +
                " — map them with the Advanced column overrides.");
      } else {
        message("muted", "All required columns recognized.");
      }
    }
    el.replaceChildren(...nodes);
  }

  async function preview(file, el, isCurrent) {
    if (!file) {
      if (isCurrent()) el.replaceChildren();
      return;
    }
    const line = await readHeaderLine(file);
    if (!isCurrent()) return;
    render(el, file, line === null ? null : mapHeader(line), line === null);
  }

  for (const [inputId, previewId] of [["sumstats1", "preview1"],
                                      ["sumstats2", "preview2"]]) {
    const input = document.getElementById(inputId);
    const el = document.getElementById(previewId);
    if (!input || !el) continue;
    let generation = 0;
    input.addEventListener("change", () => {
      const current = ++generation;
      const file = input.files[0];
      const isCurrent = () => current === generation && input.files[0] === file;
      preview(file, el, isCurrent).catch(() => {
        if (current !== generation || input.files[0] !== file) return;
        const message = document.createElement("span");
        message.className = "muted";
        message.textContent = "Header preview failed; columns will be " +
                              "detected when the job runs.";
        el.replaceChildren(message);
      });
    });
  }
})();
