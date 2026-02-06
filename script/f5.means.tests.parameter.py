import numpy as np
import pandas as pd

# Stats
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Optional (recommended) for nonparametric post-hoc
try:
    import scikit_posthocs as sp
    HAS_SCPH = True
except ImportError:
    HAS_SCPH = False

# ----------------------------
# Load data
# ----------------------------
csv_path = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df = pd.read_csv(csv_path)

# Ensure ageclassn is treated as a categorical factor
df["ageclassn"] = df["ageclassn"].astype("category")

vars_to_test = ["perennial", "annual", "native", "nonnative", "trich", "nrich", "nnrich", "wetland", "conserve", "herbaceous", "woody"]

# Drop rows with missing values in any tested variable or ageclassn
df_sub = df[["ageclassn"] + vars_to_test].dropna().copy()

# ----------------------------
# Helpers
# ----------------------------
def shapiro_by_group(data: pd.DataFrame, y: str, group: str = "ageclassn"):
    """Run Shapiro-Wilk normality test within each group (if n>=3)."""
    out = []
    for lvl, g in data.groupby(group):
        x = g[y].dropna().values
        if len(x) < 3:
            out.append((lvl, len(x), np.nan, np.nan))
            continue
        # Shapiro is sensitive for large n; still OK as a quick diagnostic
        W, p = stats.shapiro(x)
        out.append((lvl, len(x), W, p))
    return pd.DataFrame(out, columns=[group, "n", "W", "p_shapiro"])

def levene_test(data: pd.DataFrame, y: str, group: str = "ageclassn"):
    """Levene’s test for equal variances across groups."""
    groups = [g[y].dropna().values for _, g in data.groupby(group)]
    if len(groups) < 2:
        return np.nan, np.nan
    stat, p = stats.levene(*groups, center="median")
    return stat, p

def run_anova(data: pd.DataFrame, y: str):
    """One-way ANOVA via OLS."""
    model = ols(f"{y} ~ C(ageclassn)", data=data).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    return model, aov

def run_kw(data: pd.DataFrame, y: str):
    """Kruskal-Wallis test."""
    groups = [g[y].dropna().values for _, g in data.groupby("ageclassn")]
    if len(groups) < 2:
        return np.nan, np.nan
    H, p = stats.kruskal(*groups)
    return H, p

def eta_squared_from_anova_table(aov_table: pd.DataFrame) -> float:
    """Compute eta^2 effect size for one-way ANOVA from statsmodels table."""
    ss_between = aov_table.loc["C(ageclassn)", "sum_sq"]
    ss_total = ss_between + aov_table.loc["Residual", "sum_sq"]
    return float(ss_between / ss_total) if ss_total > 0 else np.nan

# ----------------------------
# Analysis loop
# ----------------------------
results = {}

for y in vars_to_test:
    # Normality per group + overall
    shapiro_tbl = shapiro_by_group(df_sub, y)
    normal_groups_ok = (shapiro_tbl["p_shapiro"].dropna() > 0.05).all() if shapiro_tbl["p_shapiro"].notna().any() else False

    # Homogeneity of variance
    lev_stat, lev_p = levene_test(df_sub, y)
    homovar_ok = (lev_p > 0.05) if not np.isnan(lev_p) else False

    # Decide parametric vs nonparametric:
    # Use ANOVA only if group-wise normality and homogeneity are both acceptable.
    use_parametric = bool(normal_groups_ok and homovar_ok)

    if use_parametric:
        model, aov = run_anova(df_sub, y)
        p_main = float(aov.loc["C(ageclassn)", "PR(>F)"])
        effect = eta_squared_from_anova_table(aov)

        posthoc = None
        if p_main < 0.05:
            posthoc = pairwise_tukeyhsd(endog=df_sub[y].values, groups=df_sub["ageclassn"].values, alpha=0.05)

        results[y] = {
            "test_used": "ANOVA",
            "levene_p": lev_p,
            "all_groups_shapiro_ok": normal_groups_ok,
            "anova_table": aov,
            "anova_p": p_main,
            "eta_sq": effect,
            "tukey": posthoc,
            "shapiro_by_group": shapiro_tbl,
        }

    else:
        H, p_kw = run_kw(df_sub, y)

        # Post-hoc for Kruskal-Wallis: Dunn's test with Holm (or Bonferroni)
        dunn_tbl = None
        if (p_kw is not np.nan) and (p_kw < 0.05):
            if not HAS_SCPH:
                print(
                    f"[{y}] Kruskal-Wallis is significant, but scikit-posthocs is not installed.\n"
                    "Install it to run Dunn's post-hoc:\n"
                    "  pip install scikit-posthocs\n"
                )
            else:
                dunn_tbl = sp.posthoc_dunn(
                    df_sub, val_col=y, group_col="ageclassn", p_adjust="holm"
                )

        results[y] = {
            "test_used": "Kruskal-Wallis",
            "levene_p": lev_p,
            "all_groups_shapiro_ok": normal_groups_ok,
            "kw_H": H,
            "kw_p": p_kw,
            "dunn_holm_pvals": dunn_tbl,
            "shapiro_by_group": shapiro_tbl,
        }

# ----------------------------
# Print a compact summary
# ----------------------------
for y in vars_to_test:
    r = results[y]
    print("\n" + "=" * 70)
    print(f"Variable: {y}")
    print(f"Test used: {r['test_used']}")
    print(f"Levene p (homogeneity): {r['levene_p']:.4g}" if not np.isnan(r["levene_p"]) else "Levene p: NA")
    print(f"All groups Shapiro p>0.05: {r['all_groups_shapiro_ok']}")

    if r["test_used"] == "ANOVA":
        print("\nANOVA table:")
        print(r["anova_table"])
        print(f"\nANOVA p (ageclassn): {r['anova_p']:.4g}")
        print(f"Eta-squared: {r['eta_sq']:.4g}")

        print("\nShapiro by group:")
        print(r["shapiro_by_group"])

        if r["tukey"] is not None:
            print("\nTukey HSD (pairwise differences among ageclassn):")
            print(r["tukey"].summary())
        else:
            print("\nTukey HSD: not run (ANOVA not significant at alpha=0.05).")

    else:
        print(f"\nKruskal-Wallis H: {r['kw_H']:.4g}")
        print(f"Kruskal-Wallis p: {r['kw_p']:.4g}")
        print("\nShapiro by group:")
        print(r["shapiro_by_group"])

        if r["dunn_holm_pvals"] is not None:
            print("\nDunn's post-hoc (Holm-adjusted p-values):")
            print(r["dunn_holm_pvals"])
        else:
            print("\nDunn's post-hoc: not run (KW not significant or scikit-posthocs missing).")
# ----------------------------
# SAVE RESULTS TO FILES (add this at the very end)
# ----------------------------
import os

out_dir = "anova_results_parameters"
os.makedirs(out_dir, exist_ok=True)

summary_rows = []

for y in vars_to_test:
    r = results[y]

    # Save Shapiro table for each variable
    r["shapiro_by_group"].to_csv(f"{out_dir}/{y}_shapiro_by_group.csv", index=False)

    if r["test_used"] == "ANOVA":
        # Save ANOVA table
        r["anova_table"].to_csv(f"{out_dir}/{y}_anova_table.csv")

        # Save Tukey results (if run)
        if r["tukey"] is not None:
            tukey_df = pd.DataFrame(r["tukey"].summary().data[1:], columns=r["tukey"].summary().data[0])
            tukey_df.to_csv(f"{out_dir}/{y}_tukey_hsd.csv", index=False)

        summary_rows.append({
            "variable": y,
            "test_used": "ANOVA",
            "levene_p": r["levene_p"],
            "anova_p": r["anova_p"],
            "eta_sq": r["eta_sq"]
        })

    else:
        # Save Dunn results (if run)
        if r["dunn_holm_pvals"] is not None:
            r["dunn_holm_pvals"].to_csv(f"{out_dir}/{y}_dunn_holm.csv")

        summary_rows.append({
            "variable": y,
            "test_used": "Kruskal-Wallis",
            "levene_p": r["levene_p"],
            "kw_H": r["kw_H"],
            "kw_p": r["kw_p"]
        })

# Save one combined summary table
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{out_dir}/SUMMARY_tests.csv", index=False)

print(f"\nSaved all outputs to folder: {out_dir}/")
