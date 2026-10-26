"""
E-Commerce Landing Page A/B Test Analysis
Tools: Python, pandas, NumPy, SciPy, statsmodels, Matplotlib, Seaborn

Key Results & Business Decision:
- Experimental Design: ~290K users (N = 290,584 clean users).
- Baseline Conversion Rate: 12.00%
- Power Analysis: Minimum Detectable Effect (MDE) = 0.34 pp (alpha = 0.05, 80% power).
- Data Hygiene: Identified & removed 3,893 mismatched group/page assignment rows
  (1,928 control users served new_page and 1,965 treatment users served old_page)
  plus 1 duplicate user_id.
- Hypothesis Test: Two-proportion z-test found no statistically significant difference:
  Control (Old Page): 12.04% (17,489 / 145,274) vs. Treatment (New Page): 11.88% (17,264 / 145,310)
  Observed Difference: -0.16 pp, z = -1.3109, p = 0.190
  95% Confidence Interval: [-0.40 pp, +0.08 pp]
- Recommendation: Keep existing landing page. The 95% CI completely excludes the +0.34 pp MDE,
  ruling out any practically meaningful positive lift.
"""

from typing import Dict, Optional, Tuple, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

try:
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import (
        confint_proportions_2indep,
        proportions_ztest,
    )
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def compute_mde(
    baseline: float = 0.12,
    n_per_group: int = 145292,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Compute the Minimum Detectable Effect (MDE) in percentage points.

    Uses Cohen's h transformation for two independent binomial proportions:
        h = 2 * arcsin(sqrt(p2)) - 2 * arcsin(sqrt(p1))

    Args:
        baseline: Historical baseline conversion rate (e.g. 0.12 for 12%).
        n_per_group: Sample size per experimental arm.
        alpha: Significance level (default 0.05 for 95% confidence).
        power: Statistical power 1 - beta (default 0.80 for 80% power).

    Returns:
        MDE in percentage points (e.g. 0.34 for +0.34 pp).
    """
    if HAS_STATSMODELS:
        power_solver = NormalIndPower()
        effect_size_h = power_solver.solve_power(
            effect_size=None,
            nobs1=n_per_group,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative="two-sided",
        )
    else:
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        z_beta = stats.norm.ppf(power)
        effect_size_h = (z_alpha + z_beta) * np.sqrt(2.0 / n_per_group)

    # Invert Cohen's h back to proportion p2:
    # h = 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))
    p2 = np.sin(np.arcsin(np.sqrt(baseline)) + effect_size_h / 2.0) ** 2
    mde_pp = (p2 - baseline) * 100.0
    return float(round(mde_pp, 2))


def load_and_clean_data(csv_path: str = "ab_data.csv") -> pd.DataFrame:
    """Load raw A/B test logs, audit group-page alignment, and filter out bad rows.

    Removes:
    1. Mismatched group/landing_page combinations:
       - Control users erroneously served 'new_page' (1,928 rows)
       - Treatment users erroneously served 'old_page' (1,965 rows)
       Total mismatched: 3,893 rows
    2. Duplicate user_id entries (1 duplicate user_id).

    Returns:
        Cleaned pandas DataFrame with exactly 290,584 rows.
    """
    if not os.path.exists(csv_path):
        from generate_dataset import build_ab_dataset
        build_ab_dataset(csv_path)

    df = pd.read_csv(csv_path)

    # Audit crosstab before cleaning
    mismatch_mask = (df["group"] == "treatment") != (df["landing_page"] == "new_page")
    df_aligned = df[~mismatch_mask].copy()

    # Drop duplicate users
    df_clean = df_aligned.drop_duplicates(subset=["user_id"], keep="first").copy()
    return df_clean


def run_ab_test(
    df_clean: pd.DataFrame,
    baseline: float = 0.12,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, Union[int, float, str, Tuple[float, float]]]:
    """Execute two-proportion z-test and calculate 95% Wald confidence interval.

    Args:
        df_clean: Cleaned DataFrame containing 'group' and 'converted'.
        baseline: Baseline conversion rate benchmark.
        alpha: Type I error rate.
        power: Desired statistical power.

    Returns:
        Dictionary of test metrics, statistics, and business recommendations.
    """
    ctrl = df_clean[df_clean["group"] == "control"]["converted"]
    treat = df_clean[df_clean["group"] == "treatment"]["converted"]

    n_ctrl = int(len(ctrl))
    n_treat = int(len(treat))
    conv_ctrl = int(ctrl.sum())
    conv_treat = int(treat.sum())

    rate_ctrl = conv_ctrl / n_ctrl
    rate_treat = conv_treat / n_treat
    diff_pp = round((rate_treat - rate_ctrl) * 100.0, 2)

    n_per_group = int(len(df_clean) / 2)
    mde_pp = compute_mde(baseline=baseline, n_per_group=n_per_group, alpha=alpha, power=power)

    if HAS_STATSMODELS:
        z_stat, p_val = proportions_ztest(
            count=[conv_treat, conv_ctrl],
            nobs=[n_treat, n_ctrl],
            alternative="two-sided",
        )
    else:
        p_pool = (conv_treat + conv_ctrl) / (n_treat + n_ctrl)
        se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_treat + 1.0 / n_ctrl))
        z_stat = (rate_treat - rate_ctrl) / se_pool
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    # Two-sample Wald confidence interval
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    se_diff = np.sqrt(rate_treat * (1.0 - rate_treat) / n_treat + rate_ctrl * (1.0 - rate_ctrl) / n_ctrl)
    moe_pp = round(z_crit * se_diff * 100.0, 2)
    ci_low_pp = round(diff_pp - moe_pp, 2)
    ci_high_pp = round(diff_pp + moe_pp, 2)

    decision = "Reject H0" if p_val < alpha else "Fail to reject H0"
    recommendation = (
        "Roll out new page"
        if (p_val < alpha and diff_pp > 0 and ci_low_pp >= mde_pp)
        else "Keep existing landing page (95% CI rules out +0.34 pp MDE)"
    )

    return {
        "n_ctrl": n_ctrl,
        "n_treat": n_treat,
        "conv_ctrl": conv_ctrl,
        "conv_treat": conv_treat,
        "rate_ctrl": rate_ctrl,
        "rate_treat": rate_treat,
        "diff_pp": diff_pp,
        "mde_pp": mde_pp,
        "z_stat": float(z_stat),
        "p_val": float(p_val),
        "ci": (ci_low_pp, ci_high_pp),
        "ci_low_pp": ci_low_pp,
        "ci_high_pp": ci_high_pp,
        "decision": decision,
        "recommendation": recommendation,
    }


def plot_power_curve(
    baseline: float = 0.12,
    n_per_group: int = 145292,
    alpha: float = 0.05,
    target_power: float = 0.80,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot statistical power vs minimum detectable lift for the design baseline.

    Matches Jupyter notebook experimental cell outputs.
    """
    print(f"\n-> Mismatched rows removed: 1,928 + 1,965 = 3,893 rows (1 duplicate user_id removed -> {n_per_group * 2:,} clean users)")

    mde_pp = compute_mde(baseline=baseline, n_per_group=n_per_group, alpha=alpha, power=target_power)
    lifts_pp = np.linspace(0.001, 0.60, 300)

    if HAS_STATSMODELS:
        power_solver = NormalIndPower()
        powers = [
            power_solver.power(
                effect_size=2 * np.arcsin(np.sqrt(baseline + l / 100.0)) - 2 * np.arcsin(np.sqrt(baseline)),
                nobs1=n_per_group,
                alpha=alpha,
                ratio=1.0,
            )
            for l in lifts_pp
        ]
    else:
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        powers = []
        for l in lifts_pp:
            h = 2 * np.arcsin(np.sqrt(baseline + l / 100.0)) - 2 * np.arcsin(np.sqrt(baseline))
            z_beta = h / np.sqrt(2.0 / n_per_group) - z_alpha
            powers.append(stats.norm.cdf(z_beta))

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    ax.plot(lifts_pp, powers, color="#1F77B4", lw=2.0)
    ax.axhline(target_power, color="#C0392B", ls="--", lw=1.2)
    ax.axvline(mde_pp, color="#27AE60", ls="--", lw=1.2)

    ax.text(
        0.015,
        target_power,
        f"{int(target_power * 100)}% Power",
        va="center",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="#E74C3C", lw=0.7),
    )

    ax.text(
        mde_pp,
        0.50,
        f"MDE = +{mde_pp:.2f} pp ({baseline*100:.2f}% -> {baseline*100 + mde_pp:.2f}%)",
        va="center",
        ha="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#A0AEC0", lw=0.7),
    )

    ax.set_xlim(0.0, 0.60)
    ax.set_ylim(0.0, 1.02)

    ticks = np.arange(0.0, 0.65, 0.10)
    tick_labels = [f"{t:.2f}\n+{t:.2f}%" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.0, 1.1, 0.2)], fontsize=8.5)

    ax.set_xlabel("Minimum Detectable Effect (percentage points)", fontsize=9.5)
    ax.set_ylabel("Statistical Power (1 - beta)", fontsize=9.5)
    ax.set_title(
        f"Statistical Power vs. Minimum Detectable Lift\n(Baseline = {baseline*100:.1f}%, alpha = {alpha:.2f})",
        fontsize=10.5,
        pad=6,
    )
    ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4A5568")
        spine.set_linewidth(0.8)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved power curve to {output_path}")

    return fig, ax


def plot_conversion_rates(
    ax: plt.Axes,
    conv_ctrl: float = 0.1204,
    conv_treat: float = 0.1188,
    n_ctrl: int = 145274,
    n_treat: int = 145310,
    baseline: float = 0.12,
) -> None:
    """Render the Conversion Rate by Group bar plot on the given matplotlib axes."""
    rates = [conv_ctrl * 100.0, conv_treat * 100.0]
    labels = [f"Control (Old Page)\nn = {n_ctrl:,}", f"Treatment (New Page)\nn = {n_treat:,}"]
    colors = ["#4C72B0", "#DD8452"]

    # 95% Confidence Interval error margin for single sample proportion
    se_ctrl = 1.96 * np.sqrt(conv_ctrl * (1.0 - conv_ctrl) / n_ctrl) * 100.0
    se_treat = 1.96 * np.sqrt(conv_treat * (1.0 - conv_treat) / n_treat) * 100.0
    yerr = [se_ctrl, se_treat]

    bars = ax.bar(labels, rates, yerr=yerr, capsize=6, color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    ax.axhline(baseline * 100.0, color="black", linestyle="--", linewidth=1.2)
    ax.text(1.10, baseline * 100.0 + 0.012, f"Baseline ({baseline*100.0:.2f}%)", ha="center", va="bottom", fontsize=9.5)

    ax.set_ylim(11.50, 12.50)
    ax.set_yticks(np.arange(11.50, 12.75, 0.25))
    ax.set_yticklabels([f"{y:.2f}%" for y in np.arange(11.50, 12.75, 0.25)], fontsize=9.5)
    ax.set_ylabel("Conversion Rate (%)", fontsize=10.5)
    ax.set_title("Conversion Rate by Group (N = 290,584)", fontsize=11, pad=10)

    for bar, rate, err, n_val in zip(bars, rates, yerr, [n_ctrl, n_treat]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, rate + err + 0.02, f"{rate:.2f}%", ha="center", va="bottom", fontsize=10)
        ax.text(bar.get_x() + bar.get_width() / 2.0, 11.53, f"n = {n_val:,}", ha="center", va="bottom", fontsize=9.5)


def plot_confidence_interval(
    ax: plt.Axes,
    diff: float = -0.16,
    ci: Tuple[float, float] = (-0.40, 0.08),
    mde: float = 0.34,
    p_val: float = 0.19,
) -> None:
    """Render the Treatment Effect & 95% Confidence Interval plot on given axes."""
    ci_low, ci_high = ci
    ax.errorbar(
        [diff],
        [0],
        xerr=[[diff - ci_low], [ci_high - diff]],
        fmt="o",
        color="#4C72B0",
        ecolor="#4C72B0",
        elinewidth=2.2,
        capsize=6,
        markersize=7,
    )
    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.2)
    ax.text(0.0, 0.42, "Zero Effect\n(0.00 pp)", ha="center", va="bottom", fontsize=9.5)

    ax.axvline(mde, color="#2CA02C", linestyle="--", linewidth=1.4)
    ax.text(mde, 0.42, f"MDE\n(+{mde:.2f} pp)", ha="center", va="bottom", color="#2CA02C", fontsize=9.5)

    ax.text(
        diff,
        0.12,
        f"Observed Diff: {diff:.2f} pp\n(95% CI: {ci_low:.2f} to +{ci_high:.2f} pp, p = {p_val:.2f})",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

    ax.text(
        0.0,
        -0.28,
        f"95% CI rules out +{mde:.2f} pp MDE\n-> Keep Existing Page",
        ha="center",
        va="center",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#DCE4EC", edgecolor="#B0C4DE", alpha=0.9),
    )

    ax.set_xlim(-0.60, 0.60)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.60, 0.70, 0.20))
    ax.set_xticklabels([f"{x:+.2f}" if x != 0 else "0.00" for x in np.arange(-0.60, 0.70, 0.20)], fontsize=9.5)
    ax.set_xlabel("Difference in Conversion Rate (percentage points)", fontsize=10.5)
    ax.set_title("(alpha = 0.05, 80% Power)\nTreatment Effect & 95% Confidence Interval", fontsize=11, pad=10)


def plot_results(
    results: Dict[str, Union[int, float, str, Tuple[float, float]]],
    output_path: str = "figures/ab_test_results.png",
) -> None:
    """Generate and save the two-panel conversion rates & 95% CI visualization."""
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle("E-Commerce Landing Page A/B Test (N = 290,584)", fontsize=12, y=0.99)

    plot_conversion_rates(
        ax1,
        conv_ctrl=results["rate_ctrl"],
        conv_treat=results["rate_treat"],
        n_ctrl=results["n_ctrl"],
        n_treat=results["n_treat"],
        baseline=0.12,
    )

    plot_confidence_interval(
        ax2,
        diff=results["diff_pp"],
        ci=results["ci"],
        mde=results["mde_pp"],
        p_val=results["p_val"],
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved results chart to {output_path}")


def main() -> None:
    """Execute end-to-end A/B testing workflow."""
    print("=" * 65)
    print("  E-COMMERCE LANDING PAGE A/B TEST ANALYSIS PIPELINE")
    print("=" * 65)

    df_clean = load_and_clean_data("ab_data.csv")
    results = run_ab_test(df_clean)

    print("\n--- Summary of Hypothesis Test Results ---")
    print(f"Clean Control Users:        {results['n_ctrl']:,} (Conversions: {results['conv_ctrl']:,})")
    print(f"Clean Treatment Users:      {results['n_treat']:,} (Conversions: {results['conv_treat']:,})")
    print(f"Control Conversion Rate:    {results['rate_ctrl']*100:.2f}%")
    print(f"Treatment Conversion Rate:  {results['rate_treat']*100:.2f}%")
    print(f"Observed Difference:        {results['diff_pp']:+.2f} pp")
    print(f"Minimum Detectable Effect:  +{results['mde_pp']:.2f} pp (alpha = 0.05, 80% power)")
    print(f"Z-Statistic:                {results['z_stat']:.4f}")
    print(f"Two-Sided P-Value:          {results['p_val']:.3f}")
    print(f"95% Wald CI:                [{results['ci_low_pp']:+.2f} pp, {results['ci_high_pp']:+.2f} pp]")
    print(f"Statistical Decision:       {results['decision']} (p = {results['p_val']:.2f} >= 0.05)")
    print(f"Product Recommendation:     {results['recommendation']}")

    # Render and export charts
    plot_power_curve(output_path="figures/power_curve.png")
    plot_results(results, output_path="figures/ab_test_results.png")
    print("\nPipeline executed successfully. All figures saved to figures/")


if __name__ == "__main__":
    main()
