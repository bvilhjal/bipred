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
  const progressLine = document.getElementById("progress-line");

  const NUM = new Intl.NumberFormat();

  /* A library progress event: {step, done, total, unit, phase?}. `done`
   * counts finished units, except for a sequence of named steps, where it
   * counts the ones before the named one — so that reads as done + 1. */
  function renderStep(p) {
    let text = p.phase || p.step;
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

  function renderProgress(p) {
    if (!progressLine) return;
    const where = p ? p.accession + " (trait " + p.trait + ")" : "";
    let text = null;
    if (p && p.step) {
      text = renderStep(p);
    } else if (p && p.waiting) {
      text = "Waiting for another job's download of " + where + " — " +
             p.waiting + " s";
    } else if (p && p.filtering) {
      text = "Reusing the stored copy of " + where +
             " — filtering it to this LD reference";
    } else if (p && p.bytes) {
      text = "Downloading " + where + " — " +
             (p.bytes / 1048576).toFixed(0) + " MB read";
      if (p.total) {
        text += " of " + (p.total / 1048576).toFixed(0) + " MB (" +
                Math.min(100, 100 * p.bytes / p.total).toFixed(0) + "%)";
      }
      if (p.mb_s) text += ", " + p.mb_s + " MB/s";
    }
    if (text === null) { progressLine.hidden = true; return; }
    progressLine.textContent = text;
    progressLine.hidden = false;
  }

  function renderStages(s) {
    if (!stagesEl) return;
    stagesEl.innerHTML = cfg.stages.map(function (name) {
      let cls = "pending", state = "pending";
      if (s.stages && Object.prototype.hasOwnProperty.call(s.stages, name)) {
        cls = "done";
        state = Number(s.stages[name]).toFixed(2) + " s";
      } else if (name === s.stage) {
        cls = "active";
        state = "running";
      }
      return '<li class="' + cls + '"><span>' + name +
             '</span><span class="state">' + state + "</span></li>";
    }).join("");
  }

  function renderMunge(m) {
    if (!m || !mungeEl) return;
    mungeEl.hidden = false;
    document.getElementById("m-n-cache").textContent = m.n_cache;
    document.getElementById("m-n-joint").textContent = m.n_joint;
    document.getElementById("m-n-kept").textContent = m.n_kept;
    if (m.n_screen_drop) {
      document.getElementById("m-screen-row").hidden = false;
      document.getElementById("m-screen-drop").textContent = m.n_screen_drop;
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

    if (badge) {
      badge.textContent = s.status;
      badge.className = "badge badge-" + s.status;
    }
    if (stageName) stageName.textContent = s.stage || s.status;
    renderStages(s);
    renderMunge(s.munge);
    renderProgress(s.progress);
    if (note && !note.hidden) {
      note.innerHTML = "Current stage: <strong>" +
                       (s.stage || s.status) + "</strong>. This page updates live.";
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
