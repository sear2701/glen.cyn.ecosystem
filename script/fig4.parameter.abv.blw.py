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
    figsize=(7, 9),
    sharey=True
)

width = 0.35

# =========================================================
# TOP PANEL — Origin / life history
# =========================================================
metrics_top = ["native", "nonnative", "annual", "perennial", "nrich", "nnrich"]
labels_top = [
    "Native\nplants",
    "Non-native\nplants",
    "Annual\nplants",
    "Perennial\nplants",
    "Native\nrichness",
    "Non-native\nrichness"
]

grouped_top = df.groupby("ab")[metrics_top]
means_top = grouped_top.mean().reindex(["below", "above"])
ses_top = (grouped_top.std() / np.sqrt(grouped_top.count())).reindex(["below", "above"])

x_top = np.arange(len(metrics_top))

# --- TOP bars (store returned containers) ---
bars_top_below = axes[0].bar(
    x_top - width / 2,
    means_top.loc["below"],
    width,
    yerr=ses_top.loc["below"],
    facecolor="white",
    edgecolor="black",
    label="Below 3700 feet"
)

bars_top_above = axes[0].bar(
    x_top + width / 2,
    means_top.loc["above"],
    width,
    yerr=ses_top.loc["above"],
    facecolor="darkgray",
    edgecolor="black",
    label="Above 3700 feet"
)

axes[0].set_xticks(x_top)
axes[0].set_xticklabels(labels_top, fontsize=10, ha="center")
axes[0].margins(x=0.03)
axes[0].set_ylabel("Cover (%)")
axes[0].legend(frameon=False)

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

# --- BOTTOM bars (store returned containers) ---
bars_bot_below = axes[1].bar(
    x_bot - width / 2,
    means_bot.loc["below"],
    width,
    yerr=ses_bot.loc["below"],
    facecolor="white",
    edgecolor="black"
)

bars_bot_above = axes[1].bar(
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

# =========================================================
# Add significance symbols (*, **)
# =========================================================
def add_sig_symbols(ax, metrics, ses, bars_below, bars_above, sig_map, fontsize=16):
    """
    sig_map keys: (metric_name, "below"/"above") -> symbol "*" or "**"
    Places symbol centered above the target bar at mean + SE + offset.
    """
    y0, y1 = ax.get_ylim()
    y_offset = 0.03 * (y1 - y0)

    for (metric, group), symbol in sig_map.items():
        idx = metrics.index(metric)
        container = bars_above if group == "above" else bars_below
        patch = container.patches[idx]

        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height() + float(ses.loc[group, metric]) + y_offset

        ax.text(
            x, y, symbol,
            ha="center", va="bottom",
            fontsize=fontsize,
            fontweight="bold"
        )

# ---- significance symbol locations ----
# Top panel: Native plants (Above) = *, Perennial plants (Above) = **
sig_top = {
    ("native", "above"): "*",
    ("perennial", "above"): "**",
}

# Bottom panel: Grass (Above) = **, Tree (Above) = *, All plants (Above) = **
sig_bot = {
    ("grass", "above"): "**",
    ("tree", "above"): "*",
    ("total", "above"): "**",
}

# Add symbols (call AFTER bars are drawn)
add_sig_symbols(
    ax=axes[0],
    metrics=metrics_top,
    ses=ses_top,
    bars_below=bars_top_below,
    bars_above=bars_top_above,
    sig_map=sig_top
)

add_sig_symbols(
    ax=axes[1],
    metrics=metrics_bot,
    ses=ses_bot,
    bars_below=bars_bot_below,
    bars_above=bars_bot_above,
    sig_map=sig_bot
)

# ---------- Final formatting & save ----------
plt.tight_layout()

output_path = "F4_Cover_AbvBlw2.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

# =========================================================
# Create table of mean values for all bars
# =========================================================

# Top panel table: make metrics the rows, ab groups the columns
table_top = means_top.T.copy()
table_top.index = labels_top
table_top.columns = ["Below 3700 feet", "Above 3700 feet"]

# Bottom panel table
table_bot = means_bot.T.copy()
table_bot.index = labels_bot
table_bot.columns = ["Below 3700 feet", "Above 3700 feet"]

# Combine into one table with a category label
mean_table = pd.concat(
    [table_top, table_bot],
    keys=["Plant origin & life history", "Functional groups"],
    names=["Category", "Metric"]
)

# Save to CSV
table_output_path = "F4_Cover_AbvBlw.png"
mean_table.to_csv(table_output_path)

print(f"Saved mean values table to: {table_output_path}")

plt.show()
