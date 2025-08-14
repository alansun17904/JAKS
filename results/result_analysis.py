import json
from scipy.stats import skew
import matplotlib.pyplot as plt

output_path = "/workspace/JAKS/test1.json"


with open(output_path, 'r') as f:
    activation_matrix = json.load(f)

importance_scores = activation_matrix["edges"]["input->a0.h0<q>"]
print(importance_scores)

print(activation_matrix["edges"]["input->a0.h0<q>"].items())

final_activation_matrix = {}

for node_name, activation_score in activation_matrix["edges"].items():
  #print(f"{node_name} :  {activation_score['score']}")
  final_activation_matrix[node_name] = activation_score['score']

list_activation = list(final_activation_matrix.values())
print(skew(list_activation))


plt.figure(figsize=(8,5))
plt.hist(list_activation, bins=5, edgecolor='black', log=True)
plt.xlabel("Importance Score")
plt.ylabel("Number of Edges")
plt.title("Distribution of Edge Importance Scores")
plt.grid(True)

# Save instead of show:
outpath = "/workspace/JAKS/results/test1_edge_scores_hist.png"
plt.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved plot to: {outpath}")


# To see where exactly are these crazy values coming from
high_score_edges = {k: v for k, v in final_activation_matrix.items() if v >= 1000}

#for edge, score in high_score_edges.items():
#    print(f"{edge}: {score}")

print(min(list_activation))
print(max(list_activation))