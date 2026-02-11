import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv("/workspaces/glen.cyn.ecosystem/data/richness.canyon.trim.csv")

# --- Columns / settings ---
x_col = "canyon"
cat_col = "abvblw"          # categorical values (e.g., "abv", "blw")
panel_vars = ["nnrich", "nrich"]

# Colors for abvblw values (paired bars)
color_map = {
    "abv": "whitesmoke",
    "blw": "dimgray",
}

# Order of canyons (consistent across panels)
canyon_order = sorted(df[x_col].dropna().astype(str).unique())

# Order of abvblw categories (keep only those present, prefer abv then blw)
cat_order_pref = ["abv", "blw"]
cat_order = [c for c in cat_order_pref if c in set(df[cat_col].dropna().astype(str))]

# --- Prep y-axis max so both panels share the same scale ---
ymax = 0
for v in panel_vars:
    tmp = (df[df[cat_col].isin(cat_order)]
           .groupby([x_col, cat_col], as_index=False)[v]
           .mean())
    if not tmp.empty:
        ymax = max(ymax, np.nanmax(tmp[v].values))

# add a little headroom
ymax = ymax * 1.05 if ymax > 0 else 1

# --- Make 2x1 figure (shared y for same scale) ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

def plot_panel(ax, value_col, title):
    sub = df[df[cat_col].isin(cat_order)].copy()
    sub[x_col] = sub[x_col].astype(str)
    sub[cat_col] = sub[cat_col].astype(str)

    # Mean per canyon + abvblw (in case multiple rows)
    g = (sub.groupby([x_col, cat_col], as_index=False)[value_col]
            .mean())

    x = np.arange(len(canyon_order))
    width = 0.38  # paired bars

    # Offsets for 2 categories: left/right
    offsets = [-width/2, width/2]

    for i, cval in enumerate(cat_order):
        gi = (g[g[cat_col] == cval]
              .set_index(x_col)
              .reindex(canyon_order))

        ax.bar(
            x + offsets[i],
            gi[value_col].values,
            width=width,
            color=color_map.get(cval, "gray"),
            edgecolor="black",
            linewidth=0.5,
            label=cval
        )

    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.3)

# --- Plot panels ---
plot_panel(axes[0], "nnrich", "nnrich")
plot_panel(axes[1], "nrich", "nrich")

# --- X axis labels on both panels (sharex hides top by default) ---
for ax in axes:
    ax.set_xlabel("canyon")
    ax.set_xticks(np.arange(len(canyon_order)))
    ax.set_xticklabels(canyon_order, rotation=45, ha="right")
    ax.tick_params(axis="x", labelbottom=True)

# --- One legend for whole figure ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=False)

plt.tight_layout()
plt.show()
