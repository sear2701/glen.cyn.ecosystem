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
    ("native", "nonnative"),
    ("perennial", "annual"),
    ("herbaceous", "woody")
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
native_m, native_se = summ["native_mean"].to_numpy(), se("native")
nonnative_m, nonnative_se = summ["nonnative_mean"].to_numpy(), se("nonnative")

perennial_m, perennial_se = summ["perennial_mean"].to_numpy(), se("perennial")
annual_m, annual_se = summ["annual_mean"].to_numpy(), se("annual")

herbaceous_m, herbaceous_se = summ["herbaceous_mean"].to_numpy(), se("herbaceous")
woody_m, woody_se = summ["woody_mean"].to_numpy(), se("woody")

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
# TOP — Native & Non-native cover
# -----------------------------
axes[0].errorbar(
    x, native_m, yerr=native_se,
    capsize=0, **line_kw, **marker1, label="Native"
)
axes[0].errorbar(
    x, nonnative_m, yerr=nonnative_se,
    capsize=0, **line_kw, **marker2, label="Non-native"
)
axes[0].set_ylabel("Cover (%)")
axes[0].legend(loc="center right", frameon=True)

# --- TOP PANEL LETTERS (corrected b's: now on NATIVE, not NONNATIVE) ---
# a, x, y, y* (unchanged)
add_label_at_errorbar(axes[0], x, native_m,    native_se,    x_value=1,  text="a",  where="below")
add_label_at_errorbar(axes[0], x, nonnative_m, nonnative_se, x_value=1,  text="x",  where="above")
add_label_at_errorbar(axes[0], x, nonnative_m, nonnative_se, x_value=2,  text="y",  where="below")
add_label_at_errorbar(axes[0], x, nonnative_m, nonnative_se, x_value=6,  text="y*", where="below")
add_label_at_errorbar(axes[0], x, nonnative_m, nonnative_se, x_value=25, text="y",  where="below")
add_label_at_errorbar(axes[0], x, nonnative_m, nonnative_se, x_value=40, text="y",  where="above")
add_label_at_errorbar(axes[0], x, nonnative_m, nonnative_se, x_value=50, text="y",  where="below")

# b's (FIXED per your instructions)
add_label_at_errorbar(axes[0], x, native_m, native_se, x_value=25, text="b", where="above")
add_label_at_errorbar(axes[0], x, native_m, native_se, x_value=40, text="b", where="below")
add_label_at_errorbar(axes[0], x, native_m, native_se, x_value=50, text="b", where="above")

# -----------------------------
# MIDDLE — Perennial & Annual cover
# -----------------------------
axes[1].errorbar(
    x, perennial_m, yerr=perennial_se,
    capsize=0, **line_kw, **marker1, label="Perennial"
)
axes[1].errorbar(
    x, annual_m, yerr=annual_se,
    capsize=0, **line_kw, **marker2, label="Annual"
)
axes[1].set_ylabel("Cover (%)")
axes[1].legend(loc="center right", frameon=True)

# -----------------------------
# BOTTOM — Herbaceous and woody vegetation
# -----------------------------
axes[2].errorbar(
    x, herbaceous_m, yerr=herbaceous_se,
    capsize=0, **line_kw, **marker1, label="Herbaceous plants"
)
axes[2].errorbar(
    x, woody_m, yerr=woody_se,
    capsize=0, **line_kw, **marker2, label="Woody plants"
)
axes[2].set_ylabel("Cover (%)")
axes[2].set_xlabel("Landscape age")
axes[2].legend(loc="lower right", frameon=True)

# --- BOTTOM PANEL LETTERS (as previously requested) ---
add_label_at_errorbar(axes[2], x, woody_m, woody_se, x_value=1,  text="a",   where="below")
add_label_at_errorbar(axes[2], x, woody_m, woody_se, x_value=2,  text="ac",  where="below")
add_label_at_errorbar(axes[2], x, woody_m, woody_se, x_value=6,  text="b*c", where="above")
add_label_at_errorbar(axes[2], x, woody_m, woody_se, x_value=12, text="bc",  where="below")
add_label_at_errorbar(axes[2], x, woody_m, woody_se, x_value=25, text="b",   where="above")
add_label_at_errorbar(axes[2], x, woody_m, woody_se, x_value=40, text="bc",  where="above")
add_label_at_errorbar(axes[2], x, woody_m, woody_se, x_value=50, text="bc",  where="above")

# -----------------------------
# Save
# -----------------------------
plt.tight_layout()

output_path = "F5_ageclass_parameter.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
