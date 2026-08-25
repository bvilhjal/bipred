"""Optional progress reporting for the long-running steps.

Several steps run for minutes to hours on genome-scale data -- the LD
consistency screen and the Gibbs sampler above all -- and a caller driving
them from a user interface needs to know how far along they are. Each such
entry point takes an optional ``progress`` callable, invoked with one dict::

    {"step": "LD consistency screen, trait 1", "done": 12, "total": 1704}

``step`` names the work now running, ``done`` counts units of ``total``
already finished, and ``unit`` says what those units are -- ``"step"`` for a
sequence of named steps, ``"block"`` for the screen, ``"sweep"`` for the fit.
A step may add keys of its own (the fit adds ``phase``). For a sequence of
coarse steps, ``done`` is the number completed *before* the named one, so a
reader sees what is running rather than what has just ended.

Reporting is a side channel: it never changes a result, and a run that
reports nothing is numerically indistinguishable from one that does.

Callbacks are invoked from the calling thread only, never from a pool worker,
so an implementation needs no locking. Exceptions propagate rather than being
swallowed -- reporting that has silently stopped is worse than reporting that
fails loudly -- so a callback writing to a fallible destination should handle
its own errors.
"""

from __future__ import annotations

__all__ = ["report", "validate"]


def validate(progress, name="progress"):
    """Reject a non-callable ``progress`` at the public boundary."""
    if progress is not None and not callable(progress):
        raise TypeError(f"{name} must be callable or None, got "
                        f"{type(progress).__name__}")
    return progress


def report(progress, step, done, total, **extra):
    """Emit one progress event; a None ``progress`` costs one comparison."""
    if progress is None:
        return
    event = {"step": str(step), "done": int(done), "total": int(total)}
    event.update(extra)
    progress(event)
