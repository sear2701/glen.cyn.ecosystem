import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv("/workspaces/glen.cyn.ecosystem/data/richness.canyon.trim.csv")

# --- Columns / settings ---
x_col = "canyon"
cat_col = "abvblw"
vars_ = ["nnrich", "nrich", "trich"]
colors = ["whitesmoke", "lightgray", "dimgray"]

panels = [
    ("abv", "abvblw = abv"),
    ("blw", "abvblw = blw"),
    ("all", "abvblw = all"),
]

# Optional: keep canyon order consistent across panels
canyon_order = sorted(df[x_col].dropna().astype(str).unique())

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

def plot_panel(ax, cat_value, title):
    sub = df[df[cat_col] == cat_value].copy()

    # Mean per canyon in case there are multiple rows per canyon/category
    g = (sub.groupby(x_col, as_index=False)[vars_]
            .mean())

    # Reindex to fixed canyon order so bars align even if a category is missing a canyon
    g[x_col] = g[x_col].astype(str)
    g = g.set_index(x_col).reindex(canyon_order).reset_index()

    x = np.arange(len(canyon_order))
    width = 0.25

    for i, (v, c) in enumerate(zip(vars_, colors)):
        ax.bar(
            x + (i - 1) * width,
            g[v].values,
            width=width,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            label=v
        )

    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)

# Plot panels
for ax, (val, title) in zip(axes, panels):
    plot_panel(ax, val, title)

# X-axis (shared)
#axes[-1].set_xlabel("canyon")
#axes[-1].set_xticks(np.arange(len(canyon_order)))
#axes[-1].set_xticklabels(canyon_order, rotation=45, ha="right")

# Put x-axis labels and tick labels on ALL panels
for ax in axes:
    ax.set_xlabel("canyon")
    ax.set_xticks(np.arange(len(canyon_order)))
    ax.set_xticklabels(canyon_order, rotation=45, ha="right")
    ax.tick_params(axis="x", labelbottom=True)  # force labels to show


# One legend for whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=False)

plt.tight_layout()

output_path = "F_richness_canyon.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
