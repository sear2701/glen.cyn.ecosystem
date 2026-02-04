import pandas as pd
import numpy as np

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ----------------------------
# Load data
# ----------------------------
path = "/workspaces/glen.cyn.ecosystem/data/cover.site.ecosystem.shrub.combined.csv"
df = pd.read_csv(path)

# Ensure expected group labels (adjust here if your file uses different text)
df["ab"] = df["ab"].astype(str).str.strip().str.lower()

categories = [
    "native", "nonnative", "total", "annual", "perennial",
    "forb", "grass", "shrub", "tree", "crust", "nrich", "nnrich"
]

# ----------------------------
# Helper functions
# ----------------------------
def shapiro_p(x: np.ndarray):
    """Return Shapiro-Wilk p-value (or np.nan if not applicable)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    # Shapiro requires at least 3 and at most 5000 observations
    if len(x) < 3 or len(x) > 5000:
        return np.nan
    try:
        return stats.shapiro(x).pvalue
    except Exception:
        return np.nan

def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values aligned with pvals."""
    pvals = np.array(pvals, dtype=float)
    n = np.sum(~np.isnan(pvals))
    qvals = np.full_like(pvals, np.nan)

    if n == 0:
        return qvals

    idx = np.where(~np.isnan(pvals))[0]
    pv = pvals[idx]
    order = np.argsort(pv)
    pv_sorted = pv[order]
    ranks = np.arange(1, len(pv_sorted) + 1)

    q_sorted = pv_sorted * len(pv_sorted) / ranks
    # enforce monotonicity from largest to smallest
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q = np.empty_like(pv_sorted)
    q[order] = q_sorted
    qvals[idx] = q
    return qvals

# ----------------------------
# Run tests per category
# ----------------------------
results = []

for var in categories:
    if var not in df.columns:
        results.append({
            "variable": var,
            "n_below": np.nan, "n_above": np.nan,
            "mean_below": np.nan, "mean_above": np.nan,
            "normality_p_below": np.nan, "normality_p_above": np.nan,
            "levene_p": np.nan,
            "test_used": "MISSING COLUMN",
            "statistic": np.nan,
            "p_value": np.nan
        })
        continue

    sub = df.loc[df["ab"].isin(["below", "above"]), ["ab", var]].dropna()
    below = sub.loc[sub["ab"] == "below", var].to_numpy(dtype=float)
    above = sub.loc[sub["ab"] == "above", var].to_numpy(dtype=float)

    n_below, n_above = len(below), len(above)
    mean_below = np.mean(below) if n_below else np.nan
    mean_above = np.mean(above) if n_above else np.nan

    # Normality tests (Shapiro) — run per group when possible
    p_norm_below = shapiro_p(below)
    p_norm_above = shapiro_p(above)

    # Equal variance test (Levene) — needs at least 2 per group
    levene_p = np.nan
    if n_below >= 2 and n_above >= 2:
        try:
            levene_p = stats.levene(below, above, center="median").pvalue
        except Exception:
            levene_p = np.nan

    # Decide ANOVA vs Kruskal-Wallis
    # Rule: if both groups look normal (p>0.05) AND variances look equal (p>0.05), use ANOVA.
    # Otherwise use Kruskal-Wallis (nonparametric).
    use_anova = (
        (not np.isnan(p_norm_below) and p_norm_below > 0.05) and
        (not np.isnan(p_norm_above) and p_norm_above > 0.05) and
        (not np.isnan(levene_p) and levene_p > 0.05)
    )

    stat = np.nan
    pval = np.nan
    test_used = ""

    if n_below < 2 or n_above < 2:
        test_used = "INSUFFICIENT DATA"
    else:
        if use_anova:
            # One-way ANOVA with 2 groups using statsmodels (equivalent to t-test under assumptions)
            # Avoid formula parsing issues by using a safe temporary column name
            tmp = sub.rename(columns={var: "y"}).copy()
            model = smf.ols("y ~ C(ab)", data=tmp).fit()
            anova_tbl = sm.stats.anova_lm(model, typ=2)
            stat = float(anova_tbl.loc["C(ab)", "F"])
            pval = float(anova_tbl.loc["C(ab)", "PR(>F)"])
            test_used = "ANOVA (1-way, 2 groups)"
        else:
            # Kruskal-Wallis (2 groups) as requested
            kw = stats.kruskal(below, above, nan_policy="omit")
            stat = float(kw.statistic)
            pval = float(kw.pvalue)
            test_used = "Kruskal-Wallis (2 groups)"

    results.append({
        "variable": var,
        "n_below": n_below,
        "n_above": n_above,
        "mean_below": mean_below,
        "mean_above": mean_above,
        "normality_p_below": p_norm_below,
        "normality_p_above": p_norm_above,
        "levene_p": levene_p,
        "test_used": test_used,
        "statistic": stat,
        "p_value": pval
    })

res = pd.DataFrame(results)

# Optional: add multiple-testing correction across all categories that produced a p-value
res["q_value_BH_FDR"] = benjamini_hochberg(res["p_value"].to_numpy())

# Pretty print + save
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 140)
print(res[[
    "variable", "n_below", "n_above", "mean_below", "mean_above",
    "normality_p_below", "normality_p_above", "levene_p",
    "test_used", "statistic", "p_value", "q_value_BH_FDR"
]].to_string(index=False))

out_csv = "ab_above_vs_below_mean_tests.csv"
res.to_csv(out_csv, index=False)
print(f"\nSaved results table to: {out_csv}")
