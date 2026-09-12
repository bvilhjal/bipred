"""Optional progress reporting -- LDpred3's contract, re-exported.

The ``{"step", "done", "total", "unit"}`` event shape and the two helpers
that emit and validate it live in LDpred3 and are published through
:mod:`ldpred3.shim` (they moved there with the LD-consistency screen).
bipred's pairing code reports through the
same helpers so a caller's one ``progress`` callable sees one vocabulary
whether the event came from preparation, the screen, or the bivariate fit.
"""

from __future__ import annotations

from ._ldpred3_compat import report, validate

__all__ = ["report", "validate"]
