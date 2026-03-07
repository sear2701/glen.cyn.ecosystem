import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
path = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df = pd.read_csv(path)

# Columns to summarize
y_pairs = [
    ("nrich", "nnrich"),
    ("herbrich", "woodyrich"),
    ("conserve2", "wetland2")
]

# -----------------------------
# Compute means and 1 SE by ageclassn
# -----------------------------
agg = {}
for col1, col2 in y_pairs:
    agg[col1] = ["mean", "std", "count"]
    agg[col2] = ["mean", "std", "count"]

summ = df.groupby("ageclassn").agg(agg)

# Flatten column names (e.g., ('native','mean') -> 'native_mean')
summ.columns = [f"{c[0]}_{c[1]}" for c in summ.columns]
summ = summ.reset_index().sort_values("ageclassn")

x = summ["ageclassn"].to_numpy()

def se(prefix: str):
    """Compute standard error from *_std and *_count columns."""
    return summ[f"{prefix}_std"].to_numpy() / np.sqrt(summ[f"{prefix}_count"].to_numpy())

# Means + SE arrays
nrich_m, nrich_se = summ["nrich_mean"].to_numpy(), se("nrich")
nnrich_m, nnrich_se = summ["nnrich_mean"].to_numpy(), se("nnrich")

herbrich_m, herbrich_se = summ["herbrich_mean"].to_numpy(), se("herbrich")
woody_m, woody_se = summ["woodyrich_mean"].to_numpy(), se("woodyrich")

conserve_m, conserve_se = summ["conserve2_mean"].to_numpy(), se("conserve2")
wetland_m, wetland_se = summ["wetland2_mean"].to_numpy(), se("wetland2")

# -----------------------------
# Helpers: find index by x value + label just beyond error bar
# -----------------------------
def idx_at_x(x_arr, x_value):
    matches = np.where(x_arr == x_value)[0]
    if len(matches) == 0:
        raise ValueError(f"x={x_value} not found in ageclassn values: {x_arr}")
    return int(matches[0])

def add_label_at_errorbar(ax, x_arr, y_arr, yerr_arr, x_value, text, where="above",
                          ypad_frac=0.015, dx_pts=0, fontsize=12):
    """
    Place bold label just ABOVE or BELOW the error bar for the point at x_value.
    """
    i = idx_at_x(x_arr, x_value)

    y0, y1 = ax.get_ylim()
    ypad = ypad_frac * (y1 - y0)

    if where.lower() == "above":
        y_text = y_arr[i] + yerr_arr[i] + ypad
        va = "bottom"
    elif where.lower() == "below":
        y_text = y_arr[i] - yerr_arr[i] - ypad
        va = "top"
    else:
        raise ValueError("where must be 'above' or 'below'")

    ax.annotate(
        text,
        (x_arr[i], y_text),
        xytext=(dx_pts, 0),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=fontsize,
        fontweight="bold"
    )

# -----------------------------
# Figure setup
# -----------------------------
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 9), sharex=True)

line_kw = dict(color="black", linewidth=2)
marker1 = dict(marker="o", markersize=7, linestyle="-")
marker2 = dict(marker="s", markersize=7, linestyle="--")

# -----------------------------
# TOP — Native & Non-native richness
# -----------------------------
axes[0].errorbar(
    x, nrich_m, yerr=nrich_se,
    capsize=0, **line_kw, **marker1, label="Native"
)
axes[0].errorbar(
    x, nnrich_m, yerr=nnrich_se,
    capsize=0, **line_kw, **marker2, label="Non-native"
)
axes[0].set_ylabel("Richness", fontsize=14)
axes[0].legend(loc="center right", frameon=True)

# --- TOP PANEL LETTERS ---
add_label_at_errorbar(axes[0], x, nnrich_m, nnrich_se, x_value=1,  text="a", where="above")
add_label_at_errorbar(axes[0], x, nnrich_m, nnrich_se, x_value=25, text="b", where="above")
add_label_at_errorbar(axes[0], x, nnrich_m, nnrich_se, x_value=40, text="b", where="above")

# -----------------------------
# MIDDLE — Herbaceous and woody richness
# -----------------------------
axes[1].errorbar(
    x, herbrich_m, yerr=herbrich_se,
    capsize=0, **line_kw, **marker1, label="Herbaceous"
)
axes[1].errorbar(
    x, woody_m, yerr=woody_se,
    capsize=0, **line_kw, **marker2, label="Woody"
)
axes[1].set_ylabel("Richness", fontsize=14)
axes[1].legend(loc="upper right", frameon=True)

# -----------------------------
# BOTTOM — Conservation and wetland index
# -----------------------------
axes[2].errorbar(
    x, conserve_m, yerr=conserve_se,
    capsize=0, **line_kw, **marker1, label="Conservation index"
)
axes[2].errorbar(
    x, wetland_m, yerr=wetland_se,
    capsize=0, **line_kw, **marker2, label="Wetland index"
)
axes[2].set_ylabel("Indexed value", fontsize=14)
axes[2].set_xlabel("Landscape age", fontsize=14)
axes[2].legend(loc="upper right", frameon=True)

# --- TOP PANEL LETTERS ---
add_label_at_errorbar(axes[2], x, conserve_m, conserve_se, x_value=1,  text="a", where="above")
add_label_at_errorbar(axes[2], x, conserve_m, conserve_se, x_value=4,  text="b", where="above")
add_label_at_errorbar(axes[2], x, conserve_m, conserve_se, x_value=6,  text="b", where="above")
add_label_at_errorbar(axes[2], x, conserve_m, conserve_se, x_value=25,  text="b", where="above")
add_label_at_errorbar(axes[2], x, conserve_m, conserve_se, x_value=40,  text="b", where="above")
add_label_at_errorbar(axes[2], x, conserve_m, conserve_se, x_value=50,  text="b", where="above")
add_label_at_errorbar(axes[2], x, wetland_m, wetland_se, x_value=1,  text="x", where="below")
add_label_at_errorbar(axes[2], x, wetland_m, wetland_se, x_value=4,  text="y", where="below")
add_label_at_errorbar(axes[2], x, wetland_m, wetland_se, x_value=50,  text="y ", where="below")

# -----------------------------
# Save
# -----------------------------
plt.tight_layout()

output_path = "F7.5_wetland_conserve.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
