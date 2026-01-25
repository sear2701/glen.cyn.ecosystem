import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
csv_path = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"  # <-- your uploaded csv
df = pd.read_csv(csv_path)

# Use the same landscape-age order as the figure
age_order = [1, 2, 3, 6, 11, 21, 40, 50]
df = df[df["ageclassn"].isin(age_order)].copy()

# ----------------------------
# Helper: mean + 1 SE by ageclassn
# ----------------------------
def mean_se_by_age(data: pd.DataFrame, cols):
    g = data.groupby("ageclassn")[cols].agg(["mean", "std", "count"])
    means = g.xs("mean", level=1, axis=1).reindex(age_order)
    stds  = g.xs("std",  level=1, axis=1).reindex(age_order)
    ns    = g.xs("count",level=1, axis=1).reindex(age_order)
    ses   = stds / np.sqrt(ns)
    return means, ses

# Variables used in each panel (must match your csv column names)
cover_cols = ["perennial", "annual", "native", "nonnative"]
rich_cols  = ["nrich", "nnrich"]
coc_cols   = ["conserve"]

means_cover, se_cover = mean_se_by_age(df, cover_cols)
means_rich,  se_rich  = mean_se_by_age(df, rich_cols)
means_coc,   se_coc   = mean_se_by_age(df, coc_cols)

x = np.array(age_order)

# ----------------------------
# Plot: 2x2 panel
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150, sharex=True)
(ax1, ax2), (ax3, ax4) = axes

# --- Panel 1: Perennial & Annual Cover ---
ax1.errorbar(x, means_cover["perennial"], yerr=se_cover["perennial"],
             fmt="o-", color="black", linewidth=1.5, markersize=5, capsize=0, label="Perennial")
ax1.errorbar(x, means_cover["annual"], yerr=se_cover["annual"],
             fmt="s--", color="black", linewidth=1.5, markersize=5, capsize=0, label="Annual")
ax1.set_title("Perennial & Annual Cover")
ax1.set_ylabel("Plant cover (%)")
ax1.legend(loc="upper left", frameon=True)

# --- Panel 2: Native & Non-native Cover ---
ax2.errorbar(x, means_cover["native"], yerr=se_cover["native"],
             fmt="o-", color="black", linewidth=1.5, markersize=5, capsize=0, label="Native")
ax2.errorbar(x, means_cover["nonnative"], yerr=se_cover["nonnative"],
             fmt="s--", color="black", linewidth=1.5, markersize=5, capsize=0, label="Non-native")
ax2.set_title("Native & Non-native Cover")
ax2.set_ylabel("Plant cover (%)")
ax2.legend(loc="upper left", frameon=True)

# --- Panel 3: Native & Non-native Richness ---
ax3.errorbar(x, means_rich["nrich"], yerr=se_rich["nrich"],
             fmt="o-", color="black", linewidth=1.5, markersize=5, capsize=0, label="Native")
ax3.errorbar(x, means_rich["nnrich"], yerr=se_rich["nnrich"],
             fmt="s--", color="black", linewidth=1.5, markersize=5, capsize=0, label="Non-native")
ax3.set_title("Native & Non-native Richness")
ax3.set_ylabel("Richness")
ax3.set_xlabel("Landscape age (years)")
ax3.legend(loc="upper left", frameon=True)

# --- Panel 4: Coefficient of Conservation Values ---
ax4.errorbar(x, means_coc["conserve"], yerr=se_coc["conserve"],
             fmt="o-", color="black", linewidth=1.5, markersize=5, capsize=0)
ax4.set_title("Coefficient of Conservation Values")
ax4.set_ylabel("Conservation value")
ax4.set_xlabel("Landscape age (years)")

# Shared x formatting
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(age_order)

plt.tight_layout()

# Optional: save
# plt.savefig("panel_cover_richness_coc.png", dpi=300, bbox_inches="tight")
# ---- Save figure ----
output_path = "4panel.paramters.line.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")


plt.show()
