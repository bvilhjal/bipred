/* Live job-status polling for the job page. No-JS fallback: the page still
 * renders server-side state, and a <noscript> block offers a reload link. */
(function () {
  "use strict";
  const cfg = window.BIPRED_JOB;
  if (!cfg || !["queued", "launching", "running"].includes(cfg.status)) return;

  const badge = document.getElementById("badge");
  const stageName = document.getElementById("stage-name");
  const stagesEl = document.getElementById("stages");
  const mungeEl = document.getElementById("munge");
  const failure = document.getElementById("failure");
  const failureMsg = document.getElementById("failure-msg");
  const note = document.getElementById("status-note");
  const progressLines = document.getElementById("progress-lines");

  const NUM = new Intl.NumberFormat();
  const STAGE_BY_KEY = Object.fromEntries(
    cfg.stages.map(function (stage) { return [stage.key, stage]; })
  );

  function stageLabel(name) {
    return (STAGE_BY_KEY[name] && STAGE_BY_KEY[name].label) || name;
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
    let s;
    try {
      const response = await fetch("/jobs/" + cfg.id + "/status");
      if (!response.ok) {
        if (note) note.textContent = "Connection lost; retrying…";
        setTimeout(tick, 2000);
        return;
      }
      s = await response.json();
    } catch (err) {
      if (note) note.textContent = "Connection lost; retrying…";
      setTimeout(tick, 2000);
      return;
    }
    if (!s || !s.status) { setTimeout(tick, 2000); return; }

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
    if (stageName) stageName.textContent = activeStageLabel(s);
    renderStages(s);
    renderMunge(s.munge);
    renderProgress(s.progress);
    if (note && !note.hidden) {
      note.innerHTML = "Current stage: <strong>" +
                       activeStageLabel(s) +
                       "</strong>. This page updates live.";
    }

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
    setTimeout(tick, 2000);
  }

  setTimeout(tick, 2000);
})();
