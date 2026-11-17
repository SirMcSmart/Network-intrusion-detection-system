#!/usr/bin/env bash
# XGBoost's macOS wheel needs the OpenMP runtime (libomp). The usual fix is
# `brew install libomp`. Where Homebrew can't provide it (e.g. Intel Macs on
# recent macOS, which Homebrew treats as unsupported), point XGBoost at the
# libomp.dylib that scikit-learn's wheel already bundles. Only files inside
# .venv are touched.
set -euo pipefail
SP=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SP/xgboost/lib/libxgboost.dylib"
test -f "$SP/sklearn/.dylibs/libomp.dylib" || { echo "scikit-learn's libomp not found" >&2; exit 1; }
if ! otool -l "$LIB" | grep -q "@loader_path/../../sklearn/.dylibs"; then
  install_name_tool -add_rpath @loader_path/../../sklearn/.dylibs "$LIB"
  codesign --force -s - "$LIB"
fi
.venv/bin/python -c "import xgboost; print('xgboost', xgboost.__version__, 'loads')"
