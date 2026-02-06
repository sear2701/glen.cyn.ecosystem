import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors = dict(mcolors.CSS4_COLORS)
names = sorted(colors)

ncols = 4
nrows = (len(names) + ncols - 1) // ncols

fig, ax = plt.subplots(figsize=(14, nrows * 0.25))
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
ax.axis("off")

for i, name in enumerate(names):
    row = nrows - 1 - (i // ncols)
    col = i % ncols

    # color name
    ax.text(col + 0.05, row + 0.5, name, fontsize=9, va="center")

    # color box
    ax.add_patch(
        plt.Rectangle(
            (col + 0.6, row + 0.2),
            0.35,
            0.6,
            color=colors[name]
        )
    )

plt.tight_layout()

# ✅ Save as PNG
plt.savefig("css4_color_grid.png", dpi=300, bbox_inches="tight")

plt.show()
