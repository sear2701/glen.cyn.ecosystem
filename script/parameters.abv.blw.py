import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "//workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df = pd.read_csv(csv_path)

# Categories become the groups
categories = ["native", "nonnative", "total", "annual", "perennial"]
ab_values = df["ab"]

x = np.arange(len(categories))  # category positions
width = 0.35                    # width of bars

fig, ax = plt.subplots()

# Plot one bar per ab value inside each category
for i, ab in enumerate(ab_values):
    ax.bar(
        x + i * width,
        df.loc[i, categories],
        width,
        label=f"ab = {ab}"
    )

# Formatting
ax.set_xlabel("Category")
ax.set_ylabel("Value")
ax.set_title("Grouped Bar Chart by Category with ab Split")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()

# ---- Save figure ----
output_path = "Parameters_AbvBlw.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()
