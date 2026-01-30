import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
path = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.csv"
df = pd.read_csv(path)

# Metrics (x-axis groups)
metrics = ["forb", "grass", "shrub", "tree", "crust"]

# Calculate mean and standard error by "ab"
grouped = df.groupby("ab")[metrics]

means = grouped.mean().reindex(["below", "above"])
ses = (grouped.std() / np.sqrt(grouped.count())).reindex(["below", "above"])

below_means = means.loc["below"].to_numpy()
above_means = means.loc["above"].to_numpy()

below_se = ses.loc["below"].to_numpy()
above_se = ses.loc["above"].to_numpy()

# Plot grouped bars
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.bar(
    x - width / 2,
    below_means,
    width,
    yerr=below_se,
    #capsize=4,
    facecolor="white",
    edgecolor="black",
    label="Below 3700 feet"
)

ax.bar(
    x + width / 2,
    above_means,
    width,
    yerr=above_se,
    #capsize=4,
    facecolor="darkgray",
    edgecolor="black",
    label="Above 3700 feet"
)
# Custom x-axis labels
x_labels = ["Forb", "Grass", "Shrub", "Tree", "Biologic crust"]

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Cover (%)")
ax.legend()

plt.tight_layout()

# ---- Save figure ----
output_path = "Fgroup_AbvBlw.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
