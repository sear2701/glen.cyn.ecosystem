import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
path = "/workspaces/glen.cyn.ecosystem/data/fgroup.age.proportion.csv"
df = pd.read_csv(path)

# Desired x-axis order (landscape age classes)
x_order = ["1", "2", "4", "6", "12", "25", "40", "50"]

# Desired stacking order
fgroup_order = ["crust", "forb", "grass", "shrub", "vine", "tree"]

# Color mapping (must match fgroup_order)
colors = [
    "black",        # crust
    "orange",       # forb
    "lightgreen",   # grass
    "green",        # shrub (same default matplotlib green)
    "lavender",     # vine
    "brown"         # tree
]

# Clean fgroup labels
df["fgroup"] = df["fgroup"].astype(str).str.strip().str.lower()
df["fgroup"] = df["fgroup"].replace({"trees": "tree", "tree.": "tree"})

# Ensure numeric values
for c in x_order:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Aggregate and reshape
wide = df.groupby("fgroup")[x_order].sum()
wide = wide.reindex(fgroup_order).fillna(0)

pivot = wide.T  # rows = landscape age, columns = fgroup

# --- Plot ---
ax = pivot.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    color=colors
)

# Axis labels
ax.set_xlabel("Landscape age")
ax.set_ylabel("Proportion of cover")

# Rotate x-axis tick labels vertically
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# Force y-axis to 0–1
ax.set_ylim(0, 1)

# Legend without title
ax.legend(title=None, bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()

# Save figure
output_path = "Fgroup.age.stackedbar.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
