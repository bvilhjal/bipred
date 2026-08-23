#!/usr/bin/env bash
# Reproducible setup for the external-tool benchmark environments.
#
# Creates two isolated environments INSIDE the repo (both gitignored):
#
#   benchmarks/.venv-ldsc/   CBIIT/ldsc (PyPI `ldsc`, the maintained Python-3
#                            distribution of bulik/ldsc) plus two one-line
#                            compatibility patches described below.
#   benchmarks/.mixer/       gsa-mixer v2.2.1 source checkout + conda-prefix
#                            toolchain (python 3.10 / cmake / boost) and the
#                            compiled libbgmg. See .mixer/BUILD_LOG.txt for the
#                            exact steps used on this machine (macOS arm64).
#
# Usage:  bash benchmarks/external_setup.sh ldsc
#         # the MiXeR build is interactive and documented in BUILD_LOG.txt;
#         # this script intentionally does not attempt it unattended.
#
# The two LDSC venv patches fix py2-era bugs that are unfixed upstream
# (CBIIT/ldsc master, checked 2026-08) and only bite with modern
# bitarray/numpy:
#   1. ldscore/ldscore.py: bitarray.decode() has returned an iterator since
#      bitarray 1.x, so np.array(...) needs an explicit list().
#   2. bin/ldsc.py: the .l2.M / .l2.M_5_50 files are opened in binary mode but
#      written with print(str) -> TypeError. Open them in text mode.
#   3. bin/munge_sumstats.py: read_header() calls gzip.open in binary mode and
#      applies str.rstrip to the bytes; decode defensively instead.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "ldsc" ]; then
    python3 -m venv "$HERE/.venv-ldsc"
    "$HERE/.venv-ldsc/bin/pip" install --upgrade pip
    "$HERE/.venv-ldsc/bin/pip" install ldsc

    V="$HERE/.venv-ldsc"
    PY="$V/bin/python"
    LDSO=$(ls -d "$V"/lib/python*/site-packages/ldscore)
    BINLDSC="$V/bin/ldsc.py"

    "$PY" - "$LDSO/ldscore.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'np.array(slice.decode(self._bedcode), dtype="float64")'
new = 'np.array(list(slice.decode(self._bedcode)), dtype="float64")'
assert s.count(old) == 1, "ldscore.py upstream changed; re-review the patch"
open(p, "w").write(s.replace(old, new))
print("patched ldscore.py bitarray decode")
EOF
    "$PY" - "$BINLDSC" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
a = "fout_M = open(args.out + '.'+ file_suffix +'.M','wb')"
b = "fout_M_5_50 = open(args.out + '.'+ file_suffix +'.M_5_50','wb')"
assert s.count(a) == 1 and s.count(b) == 1, "ldsc.py upstream changed"
s = s.replace(a, a.replace("'wb'", "'w'")).replace(b, b.replace("'wb'", "'w'"))
open(p, "w").write(s)
print("patched ldsc.py .M file open modes")
EOF
    "$PY" - "$V/bin/munge_sumstats.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = ("    (openfunc, compression) = get_compression(fh)\n"
       "    return [x.rstrip('\\n') for x in openfunc(fh).readline().split()]")
new = ("    (openfunc, compression) = get_compression(fh)\n"
       "    first = openfunc(fh).readline()\n"
       "    if isinstance(first, bytes):\n"
       "        first = first.decode('utf-8')\n"
       "    return [x.rstrip('\\n') for x in first.split()]")
assert s.count(old) == 1, "munge_sumstats.py upstream changed"
open(p, "w").write(s.replace(old, new))
print("patched munge_sumstats.py read_header bytes handling")
EOF
    "$V/bin/ldsc.py" --help >/dev/null && "$V/bin/munge_sumstats.py" --help >/dev/null
    echo "LDSC env ready at $V"
else
    echo "usage: $0 ldsc" >&2
    echo "(MiXeR: see benchmarks/.mixer/BUILD_LOG.txt for the manual source build)" >&2
    exit 2
fi
