/* Make a large multipart submission visibly intentional and non-repeatable. */
(function () {
  "use strict";
  const form = document.getElementById("analysis-form");
  const button = document.getElementById("submit-button");
  const note = document.getElementById("submit-note");
  if (!form || !button) return;
  form.addEventListener("submit", function () {
    button.disabled = true;
    button.textContent = "Uploading…";
    if (note) note.textContent = " Keep this tab open until the job page appears.";
  });
})();
