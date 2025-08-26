#!/usr/bin/env bash
set -euo pipefail

# configurable bits
TL_REF="${TL_REF:-v2.11.0}"  # TransformerLens tag/commit
TOT_DIR="${TOT_DIR:-tree-of-thought}"
CS_DIR="${CS_DIR:-circuit_stability}"
REQ_ENV_YML="${REQ_ENV_YML:-circuit_stability/notebooks-source/config/environment.yml}"

# 0) system deps for pygraphviz (Linux/mac)
OS="$(uname -s || true)"
if [ "$OS" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y graphviz libgraphviz-dev pkg-config build-essential python3-dev python3-venv
elif [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
  brew install graphviz || true
fi

# 1) upgrade pip + essentials
python -m pip install --upgrade pip
pip install pyyaml

# 2) core python deps (same pins you used on runpod)
pip install torch==2.4.1 transformers==4.44.2

# 2b) pygraphviz (handle mac headers if needed)
if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
  export GRAPHVIZ_DIR="$(brew --prefix graphviz)"
  export CFLAGS="${CFLAGS:-} -I$GRAPHVIZ_DIR/include"
  export LDFLAGS="${LDFLAGS:-} -L$GRAPHVIZ_DIR/lib"
fi
# try wheel first; if it fails, build from source
pip install pygraphviz==1.14 || pip install --no-binary=pygraphviz pygraphviz==1.14

# 3) tree-of-thought: requirements + editable install
if [ -d "$TOT_DIR" ]; then
  if [ -f "$TOT_DIR/requirements.txt" ]; then
    if [ "$OS" = "Darwin" ]; then
      pip install -r "$TOT_DIR/requirements.txt" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    else
      pip install -r "$TOT_DIR/requirements.txt"
    fi
  fi
  pip install -e "$TOT_DIR"
else
  echo "⚠️  '$TOT_DIR' not found; skipping ToT install."
fi

# 3b) circuit_stability: editable install
if [ -d "$CS_DIR" ]; then
  pip install -e "$CS_DIR"
else
  echo "⚠️  '$CS_DIR' not found; skipping circuit_stability install."
fi

# 4) TransformerLens from Git (pin to tag/commit)
pip install "git+https://github.com/TransformerLensOrg/TransformerLens@$TL_REF"

# 5) optional extras from environment.yml (inside circuit_stability)
if [ -f "$REQ_ENV_YML" ]; then
python - <<PY
import yaml, pathlib, os
env_file = pathlib.Path("$REQ_ENV_YML")
data = yaml.safe_load(open(env_file))
pip_pkgs = []
for dep in data.get("dependencies", []):
    if isinstance(dep, dict) and "pip" in dep:
        pip_pkgs.extend(dep["pip"])
if pip_pkgs:
    print("Installing pip deps from environment.yml:", *pip_pkgs, sep="\n  ")
    open("pip-extras.txt","w").write("\n".join(pip_pkgs))
    os.system("pip install -r pip-extras.txt")
else:
    print("No pip: section found in environment.yml")
PY
fi

# 6) verify
python - <<'PY'
import torch, sys
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
try:
    import transformer_lens as tl
    print("TransformerLens OK")
except Exception as e:
    print("TransformerLens import error:", e, file=sys.stderr)
try:
    import experiments
    print("experiments import OK ->", experiments.__file__)
except Exception as e:
    print("experiments import error:", e, file=sys.stderr)
try:
    import circuit_stability as cs
    print("circuit_stability import OK ->", cs.__file__)
except Exception as e:
    print("circuit_stability import error:", e, file=sys.stderr)
print("✅ Environment ready")
PY
