import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- top DATA ----------
csv_path_top = "/workspaces/glen.cyn.ecosystem/data/fgroup.age.proportion.csv"
df_top = pd.read_csv(csv_path_top)

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

df_top["fgroup"] = df_top["fgroup"].astype(str).str.strip().str.lower()
df_top["fgroup"] = df_top["fgroup"].replace({"trees": "tree", "tree.": "tree"})

for c in x_order:
    df_top[c] = pd.to_numeric(df_top[c], errors="coerce")

wide = df_top.groupby("fgroup")[x_order].sum()
wide = wide.reindex(fgroup_order).fillna(0)
pivot = wide.T

# ---------- Bottom DATA ----------
csv_path_bot = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df_bot = pd.read_csv(csv_path_bot)

groups = ["forbunkn", "grassunkn", "shrub", "tree", "totalunkn", "crust"]
age_classes = sorted(df_bot["ageclassn"].dropna().unique())

summary = df_bot.groupby("ageclassn")[groups].agg(["mean", "std", "count"])

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
    dict(facecolor="0.75",  edgecolor="black", hatch=None),   # 4
    dict(facecolor="white", edgecolor="black", hatch="xx"),   # 6
    dict(facecolor="0.55",  edgecolor="black", hatch=None),   # 12
    dict(facecolor="white", edgecolor="black", hatch="\\\\"), # 25
    dict(facecolor="black", edgecolor="black", hatch=None),   # 40
    dict(facecolor="white", edgecolor="black", hatch="oo"),   # 50
]

# ============================================================
# 2-panel figure
# ============================================================
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(8, 10),
    dpi=150,
    gridspec_kw={"height_ratios": [1.0, 1.2]}
)

fig.subplots_adjust(hspace=0.35)

# -------------------- TOP plot: STACKED BAR CHART --------------------
pivot.plot(kind="bar", stacked=True, color=colors, ax=ax1)

ax1.set_xlabel("Landscape age", fontsize=14)
ax1.set_ylabel("Proportion of cover", fontsize=14)
ax1.set_xticks(range(len(pivot.index)))

labels = pivot.index.astype(str).tolist()
labels[-1] = ">50"

ax1.set_xticklabels(labels, rotation=0, ha="center")
ax1.set_ylim(0, 1)

# Top legend
handles, labels = ax1.get_legend_handles_labels()
ax1.legend_.remove()

ncols = min(len(labels), 6)

ax1.legend(
    handles,
    labels,
    title=None,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=ncols,
    frameon=True,
    columnspacing=1.2,
    handletextpad=0.6,
)

# ------------------- BOTTOM plot: GROUPED BAR CHART -------------------
x = np.arange(len(groups))
n_ages = len(age_classes)
bar_w = 0.10
offsets = (np.arange(n_ages) - (n_ages - 1) / 2.0) * bar_w

# ---- SIGNIFICANCE MAP (BOTTOM PANEL) ----
sig = {
    ("forbunkn", 1): "a",
    ("forbunkn", 2): "ab",
    ("forbunkn", 6): "c",
    ("forbunkn", 12): "bc",
    ("forbunkn", 25): "c",
    ("forbunkn", 50): "bc",

    ("shrub", 1): "a",
    ("shrub", 2): "b",
    ("shrub", 4): "cd",
    ("shrub", 6): "cd",
    ("shrub", 12): "cd",
    ("shrub", 25): "c",
    ("shrub", 40): "cd",
    ("shrub", 50): "d",

    ("tree", 1): "a",
    ("tree", 4): "ac",
    ("tree", 12): "ac",
    ("tree", 40): "bc",
    ("tree", 50): "b",

    ("crust", 1): "a",
    ("crust", 2): "ac",
    ("crust", 12): "b",
    ("crust", 25): "b",
    ("crust", 40): "bc",
    ("crust", 50): "b",
}

bar_patches = {}

for i, a in enumerate(age_classes):
    style = styles[i % len(styles)]
    cont = ax2.bar(
        x + offsets[i],
        means.loc[a, groups].values,
        width=bar_w,
        yerr=ses.loc[a, groups].values,
        error_kw=dict(ecolor="black", elinewidth=1.5, capsize=0),
        linewidth=1.2,
        label=str(a),
        **style,
    )
    for j, g in enumerate(groups):
        bar_patches[(g, a)] = cont.patches[j]

ax2.set_xticks(x)
ax2.set_xticklabels(["Forb", "Grass", "Shrub", "Tree", "All plants", "SBC"], fontsize=13)
ax2.set_ylabel("Cover (%)", fontsize=14)
ax2.set_ylim(0, max(1, (means.values + ses.values).max() * 1.10))
ax2.grid(axis="y", alpha=0.35)

# Bottom legend
ax2.legend(title="Landscape age", loc="upper right", frameon=True)

# ---- Add significance letters above specified bars (BOTTOM PANEL) ----
sig_fontsize = 9
y0, y1 = ax2.get_ylim()
ypad = 0.015 * (y1 - y0)

for (g, a), txt in sig.items():
    if (g, a) not in bar_patches:
        continue

    patch = bar_patches[(g, a)]
    x_center = patch.get_x() + patch.get_width() / 2

    mean_val = float(means.loc[a, g]) if (a in means.index and g in means.columns) else 0.0
    se_val = float(ses.loc[a, g]) if (a in ses.index and g in ses.columns) else 0.0

    y_text = mean_val + se_val + ypad
    ax2.text(
        x_center, y_text, txt,
        ha="center", va="bottom",
        fontsize=sig_fontsize,
        fontweight="bold"
    )

plt.tight_layout()

output_path = "F6_Fgroup_2panel.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()