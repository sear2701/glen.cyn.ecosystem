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

def se(mean_std_count_prefix: str):
    """Compute standard error from *_std and *_count columns."""
    return summ[f"{mean_std_count_prefix}_std"].to_numpy() / np.sqrt(summ[f"{mean_std_count_prefix}_count"].to_numpy())

# Means + SE arrays
native_m, native_se = summ["native_mean"].to_numpy(), se("native")
nonnative_m, nonnative_se = summ["nonnative_mean"].to_numpy(), se("nonnative")

perennial_m, perennial_se = summ["perennial_mean"].to_numpy(), se("perennial")
annual_m, annual_se = summ["annual_mean"].to_numpy(), se("annual")

herbaceous_m, herbaceous_se = summ["herbaceous_mean"].to_numpy(), se("herbaceous")
woody_m, woody_se = summ["woody_mean"].to_numpy(), se("woody")

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
#axes[0].set_title("Native & Non-native Cover")
legend_kw = dict(loc="center right", frameon=True)
axes[0].legend(**legend_kw)


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
#axes[1].set_title("Perennial & Annual Cover")
legend_kw = dict(loc="center right", frameon=True)
axes[1].legend(**legend_kw)
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
#axes[2].set_title("Native & Non-native Richness")
axes[2].set_xlabel("Landscape age")
legend_kw = dict(loc="lower right", frameon=True)
axes[2].legend(**legend_kw)

plt.tight_layout()

output_path = "F5_ageclass_parameter.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
