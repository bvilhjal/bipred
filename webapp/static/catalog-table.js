(function () {
  "use strict";
  const input = document.getElementById("catalog-search");
  const count = document.getElementById("catalog-count");
  const rows = Array.from(document.querySelectorAll("[data-catalog-row]"));
  if (!input) return;
  function filter() {
    const query = input.value.trim().toLowerCase();
    let shown = 0;
    for (const row of rows) {
      const match = !query || (row.dataset.search || "").toLowerCase().includes(query);
      row.hidden = !match;
      if (match) shown += 1;
    }
    if (count) count.textContent = shown + " of " + rows.length + " rows shown";
  }
  input.addEventListener("input", filter);
  filter();
})();
