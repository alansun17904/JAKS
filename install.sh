#!/bin/bash

# --- 1) Install system dependencies ---
apt-get update
apt-get install -y graphviz libgraphviz-dev pkg-config build-essential python3-dev python3-venv

# --- 2) Create & activate virtual environment ---
pip install --upgrade pip pyyaml

# --- 3) Install core Python dependencies ---
pip install torch==2.4.1 transformers==4.44.2 transformer-lens==2.11.0 pygraphviz==1.14

# --- 4) Install any extra pip packages from environment.yml ---
python - << 'PY'
import yaml, pathlib, os
env_file = pathlib.Path("circuit-stability/notebooks-source/config/environment.yml")
if not env_file.exists():
    raise FileNotFoundError(env_file)
data = yaml.safe_load(open(env_file))
pip_pkgs = []
for dep in data.get("dependencies", []):
    if isinstance(dep, dict) and "pip" in dep:
        pip_pkgs.extend(dep["pip"])
if pip_pkgs:
    print("Installing pip deps from environment.yml:", *pip_pkgs, sep="\n  ")
    with open("pip-extras.txt", "w") as f:
        f.write("\n".join(pip_pkgs))
    os.system("pip install -r pip-extras.txt")
else:
    print("No pip: section found in environment.yml")
PY

# --- 5) Verify install ---
python -c "import torch, transformer_lens, pygraphviz; print('✅ Environment ready')"
