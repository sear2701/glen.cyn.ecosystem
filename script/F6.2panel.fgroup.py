import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- TOP DATA ----------
csv_path_top = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df_top = pd.read_csv(csv_path_top)

groups = ["forb", "grass", "shrub", "tree", "total", "crust"]
age_classes = sorted(df_top["ageclassn"].dropna().unique())

summary = df_top.groupby("ageclassn")[groups].agg(["mean", "std", "count"])

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

# ---------- BOTTOM DATA ----------
csv_path_bottom = "/workspaces/glen.cyn.ecosystem/data/fgroup.age.proportion.csv"
df_bot = pd.read_csv(csv_path_bottom)

x_order = ["1", "2", "4", "6", "12", "25", "40", "50"]
fgroup_order = ["crust", "forb", "grass", "shrub", "vine", "tree"]

colors = [
    "darkslategray",
    "orange",
    "palegoldenrod",
    "green",
    "powderblue",
    "saddlebrown"
]

df_bot["fgroup"] = df_bot["fgroup"].astype(str).str.strip().str.lower()
df_bot["fgroup"] = df_bot["fgroup"].replace({"trees": "tree", "tree.": "tree"})

for c in x_order:
    df_bot[c] = pd.to_numeric(df_bot[c], errors="coerce")

wide = df_bot.groupby("fgroup")[x_order].sum()
wide = wide.reindex(fgroup_order).fillna(0)
pivot = wide.T

# ============================================================
# 2-panel figure
# ============================================================
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(8, 10),
    dpi=150,
    gridspec_kw={"height_ratios": [1.2, 1.0]}
)

# Create extra vertical space between panels so the bottom legend can sit above ax2
fig.subplots_adjust(hspace=0.35)

# -------------------- TOP plot --------------------
x = np.arange(len(groups))
n_ages = len(age_classes)
bar_w = 0.10
offsets = (np.arange(n_ages) - (n_ages - 1) / 2.0) * bar_w

for i, a in enumerate(age_classes):
    style = styles[i % len(styles)]
    ax1.bar(
        x + offsets[i],
        means.loc[a, groups].values,
        width=bar_w,
        yerr=ses.loc[a, groups].values,
        error_kw=dict(ecolor="black", elinewidth=1.5, capsize=0),
        linewidth=1.2,
        label=str(a),
        **style,
    )

ax1.set_xticks(x)
ax1.set_xticklabels(["Forb", "Grass", "Shrub", "Tree", "All plants", "Biologic crust"])
ax1.set_ylabel("Cover (%)")
ax1.set_xlabel("Functional Group")
ax1.set_ylim(0, max(1, (means.values + ses.values).max() * 1.10))
ax1.grid(axis="y", alpha=0.35)

# ✅ Top legend back inside the top panel (as in your original)
ax1.legend(title="Landscape age", loc="upper right", frameon=True)

# ------------------- BOTTOM plot -------------------
pivot.plot(kind="bar", stacked=True, color=colors, ax=ax2)

ax2.set_xlabel("Landscape age")
ax2.set_ylabel("Proportion of cover")
ax2.set_xticks(range(len(pivot.index)))
ax2.set_xticklabels(pivot.index.astype(str), rotation=0, ha="center")
ax2.set_ylim(0, 1)

# ✅ Bottom legend: wide, horizontal categories, above the bottom panel (outside axes)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend_.remove()  # remove the default legend created by pivot.plot

# Choose how many columns you want. 6 puts everything on 1 line here.
ncols = min(len(labels), 6)

ax2.legend(
    handles,
    labels,
    title=None,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),  # centered above ax2
    ncol=ncols,                  # horizontal layout
    frameon=True,
    columnspacing=1.2,
    handletextpad=0.6,
)

plt.tight_layout()

output_path = "F6_Fgroup_2panel.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
