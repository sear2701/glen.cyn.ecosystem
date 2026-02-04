import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data (once)
path = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df = pd.read_csv(path)

# ---------- Figure setup ----------
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(9, 9),
    sharey=True
)

width = 0.35

# =========================================================
# TOP PANEL — Origin / life history
# =========================================================
metrics_top = ["native", "nonnative", "annual", "perennial", "nrich", "nnrich"]
labels_top = [
    "Native plants",
    "Non-native plants",
    "Annual plants",
    "Perennial plants",
    "Native richness",
    "Non-native richness"
]

grouped_top = df.groupby("ab")[metrics_top]
means_top = grouped_top.mean().reindex(["below", "above"])
ses_top = (grouped_top.std() / np.sqrt(grouped_top.count())).reindex(["below", "above"])

x_top = np.arange(len(metrics_top))

axes[0].bar(
    x_top - width / 2,
    means_top.loc["below"],
    width,
    yerr=ses_top.loc["below"],
    facecolor="white",
    edgecolor="black",
    label="Below 3700 feet"
)

axes[0].bar(
    x_top + width / 2,
    means_top.loc["above"],
    width,
    yerr=ses_top.loc["above"],
    facecolor="darkgray",
    edgecolor="black",
    label="Above 3700 feet"
)

axes[0].set_xticks(x_top)
axes[0].set_xticklabels(labels_top)
axes[0].set_ylabel("Cover (%)")
axes[0].legend(frameon=False)
#axes[0].set_title("A) Plant origin and life history", loc="left")
# Add right-hand axis for species richness (same scale)
ax0_right = axes[0].twinx()
ax0_right.set_ylabel("Species richness")
ax0_right.set_ylim(axes[0].get_ylim())


# =========================================================
# BOTTOM PANEL — Functional groups
# =========================================================
metrics_bot = ["forb", "grass", "shrub", "tree", "total", "crust"]
labels_bot = ["Forb", "Grass", "Shrub", "Tree", "All plants", "Biologic crust"]

grouped_bot = df.groupby("ab")[metrics_bot]
means_bot = grouped_bot.mean().reindex(["below", "above"])
ses_bot = (grouped_bot.std() / np.sqrt(grouped_bot.count())).reindex(["below", "above"])

x_bot = np.arange(len(metrics_bot))

axes[1].bar(
    x_bot - width / 2,
    means_bot.loc["below"],
    width,
    yerr=ses_bot.loc["below"],
    facecolor="white",
    edgecolor="black"
)

axes[1].bar(
    x_bot + width / 2,
    means_bot.loc["above"],
    width,
    yerr=ses_bot.loc["above"],
    facecolor="darkgray",
    edgecolor="black"
)

axes[1].set_xticks(x_bot)
axes[1].set_xticklabels(labels_bot)
axes[1].set_ylabel("Cover (%)")
#axes[1].set_title("B) Functional groups", loc="left")

# ---------- Final formatting & save ----------
plt.tight_layout()

output_path = "Cover_AbvBlw_Combined.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

# =========================================================
# Create table of mean values for all bars (FIXED)
# =========================================================

# Top panel table: make metrics the rows, ab groups the columns
table_top = means_top.T.copy()  # now shape is (5 rows x 2 cols)
table_top.index = labels_top
table_top.columns = ["Below 3700 feet", "Above 3700 feet"]

# Bottom panel table
table_bot = means_bot.T.copy()  # now shape is (5 rows x 2 cols)
table_bot.index = labels_bot
table_bot.columns = ["Below 3700 feet", "Above 3700 feet"]

# Combine into one table with a category label
mean_table = pd.concat(
    [table_top, table_bot],
    keys=["Plant origin & life history", "Functional groups"],
    names=["Category", "Metric"]
)

# Save to CSV
table_output_path = "Cover_AbvBlw_MeanValues.csv"
mean_table.to_csv(table_output_path)

print(f"Saved mean values table to: {table_output_path}")

plt.show()
