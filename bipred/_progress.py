"""Optional progress reporting -- LDpred3's contract, re-exported.

The ``{"step", "done", "total", "unit"}`` event shape and the two helpers
that emit and validate it live in :mod:`ldpred3._progress` (they moved there
with the LD-consistency screen); :mod:`ldpred3.shim` publishes them as
``progress_report``/``progress_validate`` only from ldpred3 v0.7.23, above
the declared floor, so the seam binds the defining module. bipred's pairing
code reports through the same helpers so a caller's one ``progress``
callable sees one vocabulary whether the event came from preparation, the
screen, or the bivariate fit.
"""

from __future__ import annotations

from ._ldpred3_compat import report, validate

__all__ = ["report", "validate"]
