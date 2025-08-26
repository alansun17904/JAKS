import json
import glob
from scipy.stats import skew
from pathlib import Path


results_dir = Path(__file__).parent  # directory where analysis.py lives
for path in results_dir.glob("*.json"):
    with open(path, "r") as f:
        activation_matrix = json.load(f)

    # extract scores
    final_activation_matrix = {
        node: data["score"] for node, data in activation_matrix["edges"].items()
    }
    list_activation = list(final_activation_matrix.values())

    # compute skewness
    s = skew(list_activation)
    print(f"{path}: skewness = {s}")
