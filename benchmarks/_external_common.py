"""Shared plumbing for the external-tool benchmarks (MiXeR / LDSC).

This module holds the pieces ``external_overlap.py`` and
``external_hdl_tg.py`` have in common, kept separate so the plumbing is unit
testable without the external tools installed:

* tool probes following the ``infer_vs_ldsc_sbayes.py`` pattern: an env var or
  default path is checked, and when the tool is absent the arm writes NaN rows
  with a ``backend``/``note`` trail instead of failing;
* parsers for the original tools' outputs (MiXeR ``fit2`` JSON, LDSC ``.log``);
* an honest provenance sidecar writer. ``real_data_inputs``'s writer hardcodes
  ``source_clean: True`` behind guards (clean-tree and the archived ldpred3
  0.4.5 pin) that do not apply to these runs, so the equivalent record is
  written here with the *actual* tree state.

Tool defaults (override with env vars):

* LDSC: ``LDSC_BIN`` or ``benchmarks/.venv-ldsc/bin`` holding the CBIIT PyPI
  port's console scripts (``ldsc.py``, ``munge_sumstats.py``).
* MiXeR: ``MIXER_PY`` (python of the source-build env), ``MIXER_SRC``
  (gsa-mixer checkout containing ``precimed/mixer.py``), ``MIXER_LIB``
  (built ``libbgmg`` shared object); the defaults point into
  ``benchmarks/.mixer/``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

LDSC_BIN = os.environ.get("LDSC_BIN", os.path.join(HERE, ".venv-ldsc", "bin"))
MIXER_PY = os.environ.get(
    "MIXER_PY", os.path.join(HERE, ".mixer", "env", "bin", "python"))
MIXER_SRC = os.environ.get("MIXER_SRC", os.path.join(HERE, ".mixer", "src"))
MIXER_LIB = os.environ.get("MIXER_LIB", "")


def ldsc_tools():
    """Return the ``(ldsc.py, munge_sumstats.py)`` console-script paths."""
    return (os.path.join(LDSC_BIN, "ldsc.py"),
            os.path.join(LDSC_BIN, "munge_sumstats.py"))


def probe_ldsc():
    """True when the LDSC console scripts exist and answer ``--help``."""
    ldsc, munge = ldsc_tools()
    if not (os.path.isfile(ldsc) and os.path.isfile(munge)):
        return False
    try:
        r = subprocess.run([ldsc, "--help"], capture_output=True,
                           timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def mixer_lib_path():
    """Best-effort ``libbgmg`` location: env override, else search the checkout
    and the default build directory (never a general parent walk — a bogus
    ``MIXER_SRC`` must fail fast, not scan the disk)."""
    if MIXER_LIB:
        return MIXER_LIB
    roots = [MIXER_SRC, os.path.join(HERE, ".mixer", "build"),
             os.path.join(HERE, ".mixer", "build-omp")]
    for root_dir in roots:
        if not os.path.isdir(root_dir):
            continue
        for root, _dirs, files in os.walk(root_dir):
            for name in files:
                if name.startswith("libbgmg") and os.path.splitext(name)[1] in (
                        ".so", ".dylib"):
                    return os.path.join(root, name)
    return None


def probe_mixer():
    """True when the MiXeR python, CLI and built library all resolve."""
    lib = mixer_lib_path()
    mixer_py = os.path.join(MIXER_SRC, "precimed", "mixer.py")
    if not (lib and os.path.isfile(lib) and os.path.isfile(mixer_py)
            and os.path.isfile(MIXER_PY)):
        return False
    try:
        r = subprocess.run(
            [MIXER_PY, mixer_py, "fit1", "--help"],
            capture_output=True, timeout=300,
            env={**os.environ, "BGMG_SHARED_LIBRARY": lib})
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def run_logged(cmd, *, timeout, log_path=None, env=None):
    """Run ``cmd``; stream combined output to ``log_path``; return seconds.

    Raises ``RuntimeError`` with the tail of the output on a non-zero exit, so
    the benchmark driver can turn a tool failure into a NaN row with context.
    """
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=env)
    dt = time.perf_counter() - started
    text = (proc.stdout or "") + (proc.stderr or "")
    if log_path is not None:
        with open(log_path, "w") as handle:
            handle.write("$ " + shlex.join(cmd) + "\n\n" + text)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {shlex.join(cmd)}\n"
            + text[-2000:])
    return dt


# --------------------------------------------------------------------------- #
#  Parsers                                                                     #
# --------------------------------------------------------------------------- #
def _ci_point(node):
    """Point estimate out of a MiXeR ``ci`` entry (scalar or dict form)."""
    if isinstance(node, dict):
        for key in ("point estimate", "point_estimate", "estimate", "mean"):
            if key in node:
                return float(node[key])
        return float("nan")
    return float(node)


def parse_mixer_fit2_json(path):
    """Extract the cross-trait MiXeR quantities from a ``fit2`` JSON.

    ``params.pi`` is ``(pi_unique1, pi_unique2, pi_shared)``; the per-trait
    polygenicities are the sums with the shared component. Derived point
    estimates (``rg``, ``h2_T1/2``...) sit in ``ci`` when present.
    """
    with open(path) as handle:
        doc = json.load(handle)
    params = doc.get("params", {})
    pi = [float(x) for x in params.get("pi", (float("nan"),) * 3)]
    while len(pi) < 3:
        pi.append(float("nan"))
    ci = doc.get("ci", {})

    def ci_get(*names):
        for name in names:
            if name in ci:
                return _ci_point(ci[name])
        return float("nan")

    return {
        "pi1": pi[0] + pi[2],
        "pi2": pi[1] + pi[2],
        "pi11": pi[2],
        "rho_beta": float(params.get("rho_beta", float("nan"))),
        "rho_zero": float(params.get("rho_zero", float("nan"))),
        "rg": ci_get("rg"),
        "h2_1": ci_get("h2_T1", "h2_t1"),
        "h2_2": ci_get("h2_T2", "h2_t2"),
    }


_LDSC_RG_PATTERN = re.compile(
    r"Genetic Correlation:\s*([-\d.eE+]+)\s*\(([-\d.eE+]+)\)")
_LDSC_GCOV_PATTERN = re.compile(
    r"Total Observed scale gencov:\s*([-\d.eE+]+)\s*\(([-\d.eE+]+)\)")
_LDSC_INTERCEPT_PATTERN = re.compile(
    r"Intercept:\s*([-\d.eE+]+)\s*\(([-\d.eE+]+)\)")
_LDSC_H2_PATTERN = re.compile(
    r"Total Observed scale h2:\s*([-\d.eE+]+)\s*\(([-\d.eE+]+)\)")
_LDSC_MEAN_CHI2_PATTERN = re.compile(r"Mean Chi\^2:\s*([-\d.eE+]+)")


def parse_ldsc_rg_log(path):
    """Extract rg/gencov/intercept/h2 estimates from an ``ldsc.py --rg`` log.

    Section-aware: ``Intercept:`` lines also appear in the per-trait h2 blocks,
    so the cross-trait intercept is read only after the "Genetic Covariance"
    header.
    """
    with open(path) as handle:
        text = handle.read()
    match = _LDSC_RG_PATTERN.search(text)
    if match is None:
        raise ValueError(f"LDSC log {path} lacks a Genetic Correlation line")
    out = {"rg": float(match.group(1)), "rg_se": float(match.group(2))}
    match = _LDSC_GCOV_PATTERN.search(text)
    if match is None:
        raise ValueError(f"LDSC log {path} lacks a gencov line")
    out["gcov"], out["gcov_se"] = float(match.group(1)), float(match.group(2))
    cov_section = text.split("Genetic Covariance", 1)[-1]
    match = _LDSC_INTERCEPT_PATTERN.search(cov_section)
    if match is None:
        raise ValueError(f"LDSC log {path} lacks a gencov intercept line")
    out["intercept"], out["intercept_se"] = (
        float(match.group(1)), float(match.group(2)))
    h2s = _LDSC_H2_PATTERN.findall(text)
    if len(h2s) < 2:
        raise ValueError(f"LDSC log {path} lacks two h2 blocks")
    out["h2_1"], out["h2_1_se"] = (float(x) for x in h2s[0])
    out["h2_2"], out["h2_2_se"] = (float(x) for x in h2s[1])
    chi2 = _LDSC_MEAN_CHI2_PATTERN.findall(text)
    if len(chi2) >= 2:
        out["mean_chi2_1"], out["mean_chi2_2"] = float(chi2[0]), float(chi2[1])
    return out


# --------------------------------------------------------------------------- #
#  Provenance                                                                  #
# --------------------------------------------------------------------------- #
def _git(repo_root, *args):
    proc = subprocess.run(["git", "-C", repo_root, *args], check=False,
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def write_external_provenance(csv_path, *, tools, inputs=None,
                              run_controls=None):
    """Write ``<stem>.provenance.json`` with the *actual* tree state.

    Unlike ``real_data_inputs.write_provenance_sidecar`` this is usable on a
    dirty tree: the recorded ``source_clean`` reflects ``git status`` instead
    of being a precondition.
    """
    import platform
    from datetime import datetime, timezone

    def _module_version(name):
        try:
            module = __import__(name)
            return str(getattr(module, "__version__", "unknown"))
        except ImportError:
            return "unavailable"

    revision = _git(REPO_ROOT, "rev-parse", "HEAD")
    dirty = _git(REPO_ROOT, "status", "--porcelain",
                 "--untracked-files=all")
    record = {
        "schema_version": 2,
        "artifact": os.path.basename(os.path.abspath(csv_path)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_clean": not bool(dirty),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: _module_version(name)
                     for name in ("bipred", "ldpred3", "numpy", "numba")},
        "dependency_sources": tools,
        "inputs_sha256": inputs or {},
        "run_controls": run_controls or {},
    }
    stem, _ = os.path.splitext(os.path.abspath(csv_path))
    sidecar = stem + ".provenance.json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return sidecar


def ldsc_version_label():
    """Version of the venv-installed LDSC, read from its dist-info (the ambient
    interpreter does not have ldsc installed, so importlib.metadata cannot see
    it)."""
    import glob
    hits = glob.glob(os.path.join(LDSC_BIN, os.pardir, "lib", "python*",
                                  "site-packages", "ldsc-*.dist-info"))
    if hits:
        name = os.path.basename(hits[0])
        version = name.split("-", 1)[1].rsplit(".dist-info", 1)[0]
        return f"ldsc {version} (CBIIT PyPI port)"
    return "ldsc (version unknown)"
