#!/usr/bin/env bash
# This script builds a substitute C++ compiler out of the Zig toolchain, for computers where no
# compiler can be installed. Most people never need it.
#
# More technical notes:
# Create .bin/zigcxx (a g++ stand-in using ziglang's clang) and fetch Python.h/pyconfig.h from the CPython source tree.
# ONLY for machines that lack g++/python3-dev and where you cannot install them (this is what the build box needed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p .bin
printf '#!/usr/bin/env bash\nexec "%s/.venv/bin/python" -m ziglang c++ "$@"\n' "$ROOT" > .bin/zigcxx; chmod +x .bin/zigcxx
PYV=$(python -c 'import platform;print(platform.python_version())')
if [ ! -f .bin/pyinclude/Python.h ]; then
  curl -sL "https://www.python.org/ftp/python/$PYV/Python-$PYV.tgz" -o /tmp/py.tgz && tar xzf /tmp/py.tgz -C /tmp
  (cd /tmp/Python-$PYV && ./configure -q >/dev/null 2>&1 && mkdir -p "$ROOT/.bin/pyinclude" && cp -r Include/* pyconfig.h "$ROOT/.bin/pyinclude/")
fi
cp "$(find .venv -name omp.h | head -1)" .bin/ompinc/omp.h 2>/dev/null || { mkdir -p .bin/ompinc; cp "$(find .venv -name omp.h | head -1)" .bin/ompinc/omp.h; }
echo ok
