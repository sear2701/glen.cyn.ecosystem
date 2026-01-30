import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Load data ----
csv_path = "//workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df = pd.read_csv(csv_path)

# ---- X-axis groups (in requested order) ----
# ADDED "subshrub"
groups = ["forb", "grass", "shrub", "tree", "total", "crust"]

# ---- Use ageclassn for bar differentiation + legend ----
age_classes = sorted(df["ageclassn"].dropna().unique())

# ---- Mean + 1 SE by ageclassn ----
summary = df.groupby("ageclassn")[groups].agg(["mean", "std", "count"])

means = pd.DataFrame(index=age_classes, columns=groups, dtype=float)
ses   = pd.DataFrame(index=age_classes, columns=groups, dtype=float)

for a in age_classes:
    m = summary.loc[a].xs("mean", level=1)
    s = summary.loc[a].xs("std", level=1)
    n = summary.loc[a].xs("count", level=1)
    means.loc[a] = m
    ses.loc[a] = s / np.sqrt(n)

means = means.fillna(0.0)
ses   = ses.fillna(0.0)

# ---- Match the same monochrome + hatch scheme (first 8 classes) ----
# If you have more than 8 age classes, the code cycles through these styles.
styles = [
    dict(facecolor="white", edgecolor="black", hatch=None),   # 1
    dict(facecolor="white", edgecolor="black", hatch="//"),   # 2
    dict(facecolor="0.75",  edgecolor="black", hatch=None),   # 3
    dict(facecolor="white", edgecolor="black", hatch="xx"),   # 5
    dict(facecolor="0.55",  edgecolor="black", hatch=None),   # 12
    dict(facecolor="white", edgecolor="black", hatch="\\\\"), # 23
    dict(facecolor="black", edgecolor="black", hatch=None),   # 40
    dict(facecolor="white", edgecolor="black", hatch="oo"),   # 50
]

# ---- Grouped bar positions ----
x = np.arange(len(groups))
n_ages = len(age_classes)
bar_w = 0.10
offsets = (np.arange(n_ages) - (n_ages - 1) / 2.0) * bar_w

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

for i, a in enumerate(age_classes):
    style = styles[i % len(styles)]
    ax.bar(
        x + offsets[i],
        means.loc[a, groups].values,
        width=bar_w,
        yerr=ses.loc[a, groups].values,
        error_kw=dict(ecolor="black", elinewidth=1.5, capsize=0),
        linewidth=1.2,
        label=str(a),   # legend uses ageclassn
        **style,
    )

# ---- Axes formatting ----
ax.set_xticks(x)
ax.set_xticklabels(["Forb", "Grass", "Shrub", "Tree", "All plants", "Biologic crust"])
ax.set_ylabel("Cover (%)")
ax.set_xlabel("Functional Group")

ax.set_ylim(0, max(1, (means.values + ses.values).max() * 1.10))
ax.grid(axis="y", alpha=0.35)

ax.legend(title="Landscape age", loc="upper right", frameon=True)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()

# ---- Save figure ----
output_path = "fgroup_grouped_bar_ageclass_ShrubCombined.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
