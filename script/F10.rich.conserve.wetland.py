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

# Flatten column names
summ.columns = [f"{c[0]}_{c[1]}" for c in summ.columns]
summ = summ.reset_index().sort_values("ageclassn")

x = summ["ageclassn"].to_numpy()

def se(prefix: str):
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

# ============================================================
# FIGURE 1: 2x2 PANELS (your current figure)
# ============================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7), sharex=True)

line_kw = dict(color="black", linewidth=2)
marker1 = dict(marker="o", markersize=7, linestyle="-")
marker2 = dict(marker="s", markersize=7, linestyle="--")

ax_tl = axes[0, 0]
ax_tr = axes[0, 1]
ax_bl = axes[1, 0]
ax_br = axes[1, 1]

# TOP LEFT — nrich & nnrich
ax_tl.errorbar(x, nrich_m, yerr=nrich_se, capsize=0, **line_kw, **marker1, label="Native")
ax_tl.errorbar(x, nnrich_m, yerr=nnrich_se, capsize=0, **line_kw, **marker2, label="Non-native")
ax_tl.set_ylabel("Richness", fontsize=14)
ax_tl.legend(loc="upper right", frameon=True)

add_label_at_errorbar(ax_tl, x, nnrich_m, nnrich_se, x_value=1,  text="a", where="above")
add_label_at_errorbar(ax_tl, x, nnrich_m, nnrich_se, x_value=25, text="b", where="above")
add_label_at_errorbar(ax_tl, x, nnrich_m, nnrich_se, x_value=40, text="b", where="above")

# BOTTOM LEFT — herbrich & woodyrich
ax_bl.errorbar(x, herbrich_m, yerr=herbrich_se, capsize=0, **line_kw, **marker1, label="Herbaceous")
ax_bl.errorbar(x, woody_m, yerr=woody_se, capsize=0, **line_kw, **marker2, label="Woody")
ax_bl.set_ylabel("Richness", fontsize=14)
ax_bl.set_xlabel("Landscape age", fontsize=14)
ax_bl.legend(loc="upper right", frameon=True)

# TOP RIGHT — conserve2 only
ax_tr.errorbar(x, conserve_m, yerr=conserve_se, capsize=0, **line_kw, **marker1)
ax_tr.set_ylabel("Indexed value", fontsize=14)

add_label_at_errorbar(ax_tr, x, conserve_m, conserve_se, x_value=1,  text="a", where="above")
add_label_at_errorbar(ax_tr, x, conserve_m, conserve_se, x_value=4,  text="b", where="above")
add_label_at_errorbar(ax_tr, x, conserve_m, conserve_se, x_value=6,  text="b", where="below")
add_label_at_errorbar(ax_tr, x, conserve_m, conserve_se, x_value=25, text="b", where="above")
add_label_at_errorbar(ax_tr, x, conserve_m, conserve_se, x_value=40, text="b", where="above")
add_label_at_errorbar(ax_tr, x, conserve_m, conserve_se, x_value=50, text="b", where="above")

# BOTTOM RIGHT — wetland2 only
ax_br.errorbar(x, wetland_m, yerr=wetland_se, capsize=0, **line_kw, **marker1)
ax_br.set_ylabel("Indexed value", fontsize=14)
ax_br.set_xlabel("Landscape age", fontsize=14)

add_label_at_errorbar(ax_br, x, wetland_m, wetland_se, x_value=1,  text="x", where="above")
add_label_at_errorbar(ax_br, x, wetland_m, wetland_se, x_value=4,  text="y", where="below")
add_label_at_errorbar(ax_br, x, wetland_m, wetland_se, x_value=50, text="y", where="below")

plt.tight_layout()

output_path = "F7.5_wetland_conserve_2x2.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()

# ============================================================
# FIGURE 2: 2 PANELS VERTICAL (conserve2 on top, wetland2 bottom)
# ============================================================
fig2, axes2 = plt.subplots(nrows=2, ncols=1, figsize=(5, 7), sharex=True)

ax_top = axes2[0]
ax_bot = axes2[1]

# TOP — conserve2 vs ageclassn
ax_top.errorbar(
    x, conserve_m, yerr=conserve_se,
    capsize=0, **line_kw, **marker1
)
ax_top.set_ylabel("Conservation index", fontsize=14)

add_label_at_errorbar(ax_top, x, conserve_m, conserve_se, x_value=1,  text="a", where="above")
add_label_at_errorbar(ax_top, x, conserve_m, conserve_se, x_value=4,  text="b", where="above")
add_label_at_errorbar(ax_top, x, conserve_m, conserve_se, x_value=6,  text="b", where="below")
add_label_at_errorbar(ax_top, x, conserve_m, conserve_se, x_value=25, text="b", where="above")
add_label_at_errorbar(ax_top, x, conserve_m, conserve_se, x_value=40, text="b", where="above")
add_label_at_errorbar(ax_top, x, conserve_m, conserve_se, x_value=50, text="b", where="above")

# BOTTOM — wetland2 vs ageclassn
ax_bot.errorbar(
    x, wetland_m, yerr=wetland_se,
    capsize=0, **line_kw, **marker1
)
ax_bot.set_ylabel("Wetland index", fontsize=14)
ax_bot.set_xlabel("Landscape age", fontsize=14)

add_label_at_errorbar(ax_bot, x, wetland_m, wetland_se, x_value=1,  text="x", where="above")
add_label_at_errorbar(ax_bot, x, wetland_m, wetland_se, x_value=4,  text="y", where="below")
add_label_at_errorbar(ax_bot, x, wetland_m, wetland_se, x_value=50, text="y", where="below")

plt.tight_layout()

output_path2 = "F10_conserve_wetland_vertical.png"
plt.savefig(output_path2, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path2}")

plt.show()
