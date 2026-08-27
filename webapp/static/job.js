/* Live job-status polling for the job page. No-JS fallback: the page still
 * renders server-side state, and a <noscript> block offers a reload link. */
(function () {
  "use strict";
  const cfg = window.BIPRED_JOB;
  if (!cfg || !["queued", "launching", "running"].includes(cfg.status)) return;

  const badge = document.getElementById("badge");
  let stageName = document.getElementById("stage-name");
  const stagesEl = document.getElementById("stages");
  const mungeEl = document.getElementById("munge");
  const failure = document.getElementById("failure");
  const failureMsg = document.getElementById("failure-msg");
  const note = document.getElementById("status-note");
  const progressLines = document.getElementById("progress-lines");
  let timer = null;
  let controller = null;
  let retryMs = 2000;
  let lastAnnouncement = "";
  const MAX_RETRY_MS = 30000;
  const HIDDEN_RETRY_MS = 15000;
  const REQUEST_TIMEOUT_MS = 10000;

  const NUM = new Intl.NumberFormat();
  const STAGE_BY_KEY = Object.fromEntries(
    cfg.stages.map(function (stage) { return [stage.key, stage]; })
  );

  function stageLabel(name) {
    return (STAGE_BY_KEY[name] && STAGE_BY_KEY[name].label) || name;
  }

  function announce(text) {
    if (!note || text === lastAnnouncement) return;
    lastAnnouncement = text;
    note.textContent = text;
  }

  function announceStage(label) {
    const text = "Current stage: " + label + ". This page updates live.";
    if (!note || text === lastAnnouncement) return;
    lastAnnouncement = text;
    const strong = document.createElement("strong");
    strong.id = "stage-name";
    strong.textContent = label;
    note.replaceChildren(document.createTextNode("Current stage: "), strong,
                         document.createTextNode(". This page updates live."));
    stageName = strong;
  }

  function schedule(delay) {
    window.clearTimeout(timer);
    timer = window.setTimeout(tick, document.hidden ?
      Math.max(delay, HIDDEN_RETRY_MS) : delay);
  }

  /* A library progress event: {step, done, total, unit, phase?}. `done`
   * counts finished units, except for a sequence of named steps, where it
   * counts the ones before the named one — so that reads as done + 1.
   * Concurrent Catalog events arrive wrapped as {traits: {trait1, trait2}}. */
  function renderStep(p) {
    /* Trait pipelines use ``phase`` only to group progress under the
     * overlapping prepare/screen stages. Their human-readable operation is
     * ``step``; fit events still use phase names such as burn-in/sampling. */
    const fitPhase = p.phase === "burn-in" || p.phase === "sampling";
    let text = fitPhase ? p.phase : (p.step || p.phase);
    if (!p.total) return text;
    const stepwise = p.unit === "step";
    const at = stepwise ? Math.min(p.done + 1, p.total) : p.done;
    /* Floored, not rounded: 399 of 400 rounds to "100%", which reads as
     * finished when a sweep is still to come. */
    const pct = Math.floor(Math.min(100, 100 * (stepwise ? at - 1 : at) /
                                    p.total));
    text += " — " + (p.unit ? p.unit + " " : "") + NUM.format(at) + " of " +
            NUM.format(p.total) + " (" + pct + "%)";
    return text;
  }

  function renderProgressEntry(p) {
    const where = p && p.accession ?
      p.accession + " (trait " + p.trait + ")" : "trait " + p.trait;
    let text = null;
    if (p && p.step) {
      text = (p.trait ? where + " — " : "") + renderStep(p);
    } else if (p && p.screen_waiting) {
      text = "Waiting for the safe LD-screen slot for " + where + " — " +
             (p.reason || "the loaded BLAS cannot run two eigensolvers safely");
    } else if (p && p.prepared_waiting) {
      text = "Waiting for another job to QC, harmonize, and screen " + where +
             " — " + p.prepared_waiting + " s";
    } else if (p && p.prepared_source) {
      if (p.prepared_source === "stored screened trait") {
        text = "Reusing fully QC'd, harmonized, and screened data for " + where;
      } else if (p.prepared_source === "screened and stored") {
        text = "QC, LD alignment, and mandatory screen complete for " + where +
               " — saved for future analyses";
      } else if (p.prepared_source === "stored copy") {
        text = "Reusing QC'd, LD-aligned summary statistics for " + where;
      } else {
        text = "QC and LD alignment complete for " + where +
               " — saved for future analyses";
      }
    } else if (p && p.waiting) {
      text = "Waiting for another job to finish the shared Catalog copy for " +
             where + " — " + p.waiting + " s";
    } else if (p && p.filtering) {
      if (p.source === "stored copy") {
        text = "Reusing stored summary statistics for " + where +
               " — filtering locally for this LD reference";
      } else if (p.source === "download") {
        text = "Download complete for " + where +
               " — filtering locally for this LD reference";
      } else {
        text = "Preparing " + where +
               " — filtering locally for this LD reference";
      }
    } else if (p && p.bytes) {
      text = "Downloading " + where + " — " +
             (p.bytes / 1048576).toFixed(0) + " MB read";
      if (p.total) {
        text += " of " + (p.total / 1048576).toFixed(0) + " MB (" +
                Math.min(100, 100 * p.bytes / p.total).toFixed(0) + "%)";
      }
      if (p.mb_s) text += ", " + p.mb_s + " MB/s";
    }
    return text;
  }

  function renderProgress(p) {
    if (!progressLines) return;
    let entries = [];
    if (p && p.traits && typeof p.traits === "object") {
      entries = Object.keys(p.traits).sort().map(function (key) {
        return p.traits[key];
      });
    } else if (p) {
      entries = [p];
    }
    const rows = entries.map(renderProgressEntry).filter(function (text) {
      return text !== null;
    });
    if (!rows.length) {
      progressLines.replaceChildren();
      progressLines.hidden = true;
      return;
    }
    const nodes = rows.map(function (text) {
      const row = document.createElement("p");
      row.textContent = text;
      return row;
    });
    progressLines.replaceChildren(...nodes);
    progressLines.hidden = false;
  }

  function renderStages(s) {
    if (!stagesEl) return;
    const details = s.stage_details || {};
    const activeStages = Array.isArray(s.active_stages) ? s.active_stages : [];
    const rows = cfg.stages.map(function (stage) {
      const name = stage.key;
      const detail = details[name] || {};
      const active = activeStages.length ? activeStages.includes(name) :
                                           name === s.stage;
      let cls = "pending", state = "pending";
      if (s.stages && Object.prototype.hasOwnProperty.call(s.stages, name)) {
        cls = "done";
        state = detail.skipped ? "skipped" :
                Number(s.stages[name]).toFixed(2) + " s";
      } else if (active) {
        cls = s.status === "failed" ? "failed" : "active";
        state = s.status === "failed" ? "failed" : "running";
      }
      const row = document.createElement("li");
      row.className = cls;
      const copy = document.createElement("span");
      copy.className = "stage-copy";
      const label = document.createElement("span");
      label.className = "stage-label";
      label.textContent = stage.label;
      const description = document.createElement("span");
      description.className = "stage-description";
      description.textContent = stage.description;
      const summary = document.createElement("span");
      summary.className = "stage-detail";
      summary.textContent = detail.summary || "";
      summary.hidden = !detail.summary;
      const stateEl = document.createElement("span");
      stateEl.className = "state";
      stateEl.textContent = state;
      copy.append(label, description, summary);
      row.append(copy, stateEl);
      return row;
    });
    stagesEl.replaceChildren(...rows);
  }

  function activeStageLabel(s) {
    const active = Array.isArray(s.active_stages) ? s.active_stages : [];
    if (!active.length) return stageLabel(s.stage || s.status);
    return active.map(stageLabel).join(" + ");
  }

  function renderMunge(m) {
    if (!m || !mungeEl) return;
    mungeEl.hidden = false;
    document.getElementById("m-n-cache").textContent = m.n_cache;
    document.getElementById("m-n-kept").textContent = m.n_kept;
    if (Number(cfg.stageSchema) >= 3) {
      ["trait1", "trait2"].forEach(function (key) {
        const screen = (m[key] && m[key].ld_consistency_screen) || {};
        ["input", "kept", "dropped"].forEach(function (field) {
          const cell = document.getElementById("m-screen-" + field + "-" + key);
          if (cell && screen["n_" + field] !== undefined) {
            cell.textContent = screen["n_" + field];
          }
        });
      });
    } else {
      document.getElementById("m-n-joint").textContent = m.n_joint;
      if (m.n_screen_drop) {
        document.getElementById("m-screen-row").hidden = false;
        document.getElementById("m-screen-drop").textContent = m.n_screen_drop;
      }
    }
  }

  async function tick() {
    if (document.hidden) {
      schedule(HIDDEN_RETRY_MS);
      return;
    }
    let s;
    controller = new AbortController();
    const requestTimeout = window.setTimeout(
      () => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch("/jobs/" + cfg.id + "/status",
                                   {signal: controller.signal});
      if (!response.ok) {
        if (response.status >= 400 && response.status < 500 &&
            ![408, 429].includes(response.status)) {
          announce("Status updates stopped: this job is no longer available " +
                   "(HTTP " + response.status + ").");
          return;
        }
        throw new Error("HTTP " + response.status);
      }
      s = await response.json();
      if (!s || !s.status) {
        throw new Error("invalid status response");
      }
      retryMs = 2000;
    } catch (err) {
      retryMs = Math.min(MAX_RETRY_MS, retryMs * 2);
      announce((err.name === "AbortError" ? "Status request timed out" :
                "Connection lost") + "; retrying in " +
               Math.round(retryMs / 1000) + " seconds.");
      schedule(retryMs);
      return;
    } finally {
      window.clearTimeout(requestTimeout);
      controller = null;
    }

    /* A queued job created by an older server is upgraded to the current
     * workflow when its runner starts. Reload so the server can send the
     * matching stage list and current-schema count table. The refreshed page
     * has the new schema, so this branch runs only once. */
    if (Number(s.stage_schema || 1) !== Number(cfg.stageSchema || 1)) {
      window.location.reload();
      return;
    }

    if (badge) {
      badge.textContent = s.status;
      badge.className = "badge badge-" + s.status;
    }
    const activeLabel = activeStageLabel(s);
    renderStages(s);
    renderMunge(s.munge);
    renderProgress(s.progress);
    if (note && !note.hidden) announceStage(activeLabel);

    if (s.status === "done") {
      window.location.href = "/jobs/" + cfg.id + "/results";
      return;
    }
    if (s.status === "failed") {
      if (note) note.hidden = true;
      if (failure) {
        failure.hidden = false;
        if (failureMsg && s.error) failureMsg.textContent = s.error;
      }
      return;
    }
    schedule(2000);
  }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      if (controller) controller.abort();
      schedule(HIDDEN_RETRY_MS);
    } else {
      schedule(0);
    }
  });
  schedule(2000);
})();
