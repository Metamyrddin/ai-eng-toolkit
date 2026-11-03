#!/usr/bin/env bash
# This script sets environment variables so that torch.compile works on a computer that has no C++
# compiler. Most people never need it.
#
# More technical notes:
# WORKAROUND used on the build box, which has gcc (C only) but no g++ / Python.h / root-installable packages.
# `source tools/env_inductor_cpu_nocompiler.sh` before running compile_bench.py.
# On a normal machine just install a C++ compiler (apt install g++ python3-dev  /  Visual Studio Build Tools) instead.
# Pieces: (1) ziglang's `zig c++` (clang 22) as CXX, (2) Python.h + pyconfig.h copied from the CPython 3.13.5 source
# tree into .bin/pyinclude, (3) omp.h copied from the intel-openmp pip package into .bin/ompinc.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/.bin/lib" "$ROOT/.bin/ompinc" "$ROOT/.bin/pyinclude"
IOMP=$(find "$ROOT/.venv" -name libiomp5.so | head -1)   # from: pip install intel-openmp ziglang
for n in libomp.so libgomp.so libiomp5.so; do ln -sf "$IOMP" "$ROOT/.bin/lib/$n"; done
export CXX="$ROOT/.bin/zigcxx"
export CPATH="$ROOT/.bin/pyinclude:$ROOT/.bin/ompinc"
# (4) pip `intel-openmp` provides libiomp5.so (exports the LLVM __kmpc_* OpenMP symbols clang needs); expose it as libomp.so
export LIBRARY_PATH="$ROOT/.bin/lib:/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$ROOT/.bin/lib:${LD_LIBRARY_PATH:-}"
