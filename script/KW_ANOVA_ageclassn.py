import pandas as pd
import numpy as np

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# ----------------------------
# Load data
# ----------------------------
path = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df = pd.read_csv(path)

# Clean/ensure ageclassn is treated consistently
# (If ageclassn is numeric but categorical, keep it as category for ANOVA)
df["ageclassn"] = df["ageclassn"]
df["ageclassn_cat"] = df["ageclassn"].astype(str)

variables = ["native", "nonnative", "annual", "perennial", "nrich", "nnrich", "trich", "herbrich", "woodyrich"]
#variables = ["forb", "grass", "shrub", "tree", "crust", "total"]

# ----------------------------
# Helpers
# ----------------------------
def shapiro_p(x: np.ndarray):
    """Shapiro-Wilk normality test p-value (np.nan if not applicable)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or len(x) > 5000:
        return np.nan
    try:
        return stats.shapiro(x).pvalue
    except Exception:
        return np.nan

def benjamini_hochberg(pvals):
    """BH-FDR q-values for a list/array of p-values (keeps NaNs)."""
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    if mask.sum() == 0:
        return qvals
    rej, q, _, _ = multipletests(pvals[mask], method="fdr_bh")
    qvals[mask] = q
    return qvals

def dunn_posthoc_with_bh(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Pairwise Dunn-style posthoc using Mann-Whitney U between groups
    with BH-FDR correction across all pairwise comparisons for this variable.
    Returns a DataFrame of pairwise results.
    """
    # Gather groups
    groups = [g for g in sorted(data[group_col].unique())]
    vals = {g: data.loc[data[group_col] == g, value_col].dropna().to_numpy(dtype=float) for g in groups}

    pairs = []
    pvals = []
    stats_list = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            x1, x2 = vals[g1], vals[g2]
            if len(x1) < 2 or len(x2) < 2:
                u_stat = np.nan
                p = np.nan
            else:
                u = stats.mannwhitneyu(x1, x2, alternative="two-sided")
                u_stat, p = float(u.statistic), float(u.pvalue)
            pairs.append((g1, g2))
            stats_list.append(u_stat)
            pvals.append(p)

    qvals = benjamini_hochberg(pvals)

    out = pd.DataFrame({
        "group1": [p[0] for p in pairs],
        "group2": [p[1] for p in pairs],
        "mw_u_stat": stats_list,
        "p_value": pvals,
        "q_value_BH_FDR": qvals
    })
    return out

# ----------------------------
# Main analysis
# ----------------------------
overall_results = []
posthoc_results = {}  # variable -> posthoc df

age_levels = sorted(df["ageclassn_cat"].dropna().unique(), key=lambda z: float(z) if str(z).replace(".","",1).isdigit() else str(z))

for var in variables:
    if var not in df.columns:
        overall_results.append({
            "variable": var,
            "test_used": "MISSING COLUMN",
            "statistic": np.nan,
            "p_value": np.nan,
            "normality_note": np.nan
        })
        continue

    sub = df[["ageclassn_cat", var]].dropna()

    # Build group arrays
    group_arrays = []
    group_ns = []
    group_shapiro_ps = []
    for g in age_levels:
        arr = sub.loc[sub["ageclassn_cat"] == g, var].to_numpy(dtype=float)
        group_arrays.append(arr)
        group_ns.append(len(arr))
        group_shapiro_ps.append(shapiro_p(arr))

    # Normality decision rule:
    # - If any group has Shapiro p <= 0.05 (and is not NaN), treat as non-normal.
    # - If Shapiro can't run (NaNs) for many groups (too small n), we default to Kruskal.
    shapiro_valid = [p for p in group_shapiro_ps if not np.isnan(p)]
    any_non_normal = any(p <= 0.05 for p in shapiro_valid) if shapiro_valid else True  # default to nonparametric if no valid tests

    # Homogeneity of variance (Levene) if enough data
    levene_p = np.nan
    try:
        # Need >=2 groups with >=2 obs each to be meaningful
        arrays_for_levene = [a for a in group_arrays if len(a) >= 2]
        if len(arrays_for_levene) >= 2:
            levene_p = float(stats.levene(*arrays_for_levene, center="median").pvalue)
    except Exception:
        levene_p = np.nan

    # Choose ANOVA vs Kruskal-Wallis
    use_anova = (not any_non_normal) and (not np.isnan(levene_p) and levene_p > 0.05)

    stat = np.nan
    pval = np.nan
    test_used = ""

    # Need at least 2 groups with data
    nonempty_groups = sum(n > 0 for n in group_ns)
    if nonempty_groups < 2:
        test_used = "INSUFFICIENT GROUPS"
    else:
        if use_anova:
            # One-way ANOVA via OLS
            tmp = sub.rename(columns={var: "y"}).copy()
            model = smf.ols("y ~ C(ageclassn_cat)", data=tmp).fit()
            anova_tbl = sm.stats.anova_lm(model, typ=2)
            stat = float(anova_tbl.loc["C(ageclassn_cat)", "F"])
            pval = float(anova_tbl.loc["C(ageclassn_cat)", "PR(>F)"])
            test_used = "ANOVA (1-way)"

            # Posthoc: Tukey HSD (appropriate means test after ANOVA)
            if np.isfinite(pval) and pval <= 0.1:
                tuk = pairwise_tukeyhsd(endog=tmp["y"], groups=tmp["ageclassn_cat"], alpha=0.05)
                posthoc_results[var] = pd.DataFrame(
                    tuk.summary().data[1:],
                    columns=tuk.summary().data[0]
                )
        else:
            # Kruskal-Wallis across age classes
            arrays_for_kw = [a for a in group_arrays if len(a) > 0]
            kw = stats.kruskal(*arrays_for_kw, nan_policy="omit")
            stat = float(kw.statistic)
            pval = float(kw.pvalue)
            test_used = "Kruskal-Wallis"

            # Posthoc: pairwise Mann-Whitney U with BH-FDR (Dunn-like)
            # (Good, simple nonparametric "means test" analogue for pairwise differences)
            if np.isfinite(pval) and pval <= 0.1:
                posthoc_results[var] = dunn_posthoc_with_bh(sub, "ageclassn_cat", var)

    overall_results.append({
        "variable": var,
        "test_used": test_used,
        "statistic": stat,
        "p_value": pval,
        "levene_p": levene_p,
        "min_group_n": int(np.min([n for n in group_ns if n > 0])) if any(n > 0 for n in group_ns) else np.nan,
        "normality_note": (
            "All groups Shapiro p>0.05 (where test ran)" if (shapiro_valid and not any_non_normal)
            else "Non-normal group(s) or insufficient n for Shapiro; used nonparametric"
        )
    })

overall_df = pd.DataFrame(overall_results)
overall_df["q_value_BH_FDR_across_vars"] = benjamini_hochberg(overall_df["p_value"].to_numpy())

# ----------------------------
# Output / save (PRINT TO TERMINAL)
# ----------------------------
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

print("\n" + "=" * 80)
print("OVERALL TESTS: Differences among age classes (ageclassn)")
print("=" * 80)

print(
    overall_df[[
        "variable",
        "test_used",
        "statistic",
        "p_value",
        "q_value_BH_FDR_across_vars",
        "levene_p",
        "min_group_n",
        "normality_note"
    ]].to_string(index=False)
)

overall_out = "ageclassn_overall_tests.csv"
overall_df.to_csv(overall_out, index=False)
print(f"\nSaved overall results to: {overall_out}")

# ----------------------------
# Print posthoc results (if any)
# ----------------------------
for var, ph in posthoc_results.items():
    print("\n" + "-" * 80)
    print(f"POSTHOC RESULTS for variable: {var}")
    print("-" * 80)

    print(ph.to_string(index=False))

    out = f"ageclassn_posthoc_{var}.csv"
    ph.to_csv(out, index=False)
    print(f"Saved posthoc results for {var} to: {out}")
