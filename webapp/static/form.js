/* Validate the mutually exclusive sources before a large multipart submit. */
(function () {
  "use strict";
  const form = document.getElementById("analysis-form");
  const button = document.getElementById("submit-button");
  const note = document.getElementById("submit-note");
  if (!form || !button) return;

  function validateSource(trait, focus) {
    const file = document.getElementById("sumstats" + trait);
    const accession = document.getElementById("gcst" + trait);
    const error = document.getElementById("source-error" + trait);
    const hasFile = Boolean(file && file.files && file.files.length);
    const hasAccession = Boolean(accession && accession.value.trim());
    const valid = hasFile !== hasAccession;
    const message = hasFile && hasAccession ?
      "Remove either the file or the GWAS Catalog accession." :
      "Choose a summary-statistics file or a GWAS Catalog accession.";
    if (file) file.setCustomValidity(valid ? "" : message);
    if (accession) accession.setCustomValidity(valid ? "" : message);
    if (error) {
      error.textContent = valid ? "" : message;
      error.hidden = valid;
    }
    if (!valid && focus) (hasAccession ? accession : file).focus();
    return valid;
  }

  for (const trait of [1, 2]) {
    const file = document.getElementById("sumstats" + trait);
    const accession = document.getElementById("gcst" + trait);
    if (file) file.addEventListener("change", () => validateSource(trait, false));
    if (accession) accession.addEventListener("input", () => validateSource(trait, false));
  }

  form.addEventListener("submit", function (event) {
    const firstValid = validateSource(1, true);
    const secondValid = validateSource(2, firstValid);
    if (!firstValid || !secondValid) {
      event.preventDefault();
      if (note) note.textContent = " Correct the source choices above.";
      return;
    }
    button.disabled = true;
    button.textContent = "Uploading…";
    if (note) note.textContent = " Keep this tab open until the job page appears.";
  });
})();
