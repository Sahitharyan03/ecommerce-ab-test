"""
Unit and Integration Tests for E-Commerce Landing Page A/B Testing Pipeline
Validates experimental design, data cleaning audit, z-test statistics, confidence intervals, and decision rules.
"""

import os
import pytest
import numpy as np
import pandas as pd
from ab_test_analysis import (
    compute_mde,
    load_and_clean_data,
    run_ab_test,
    plot_power_curve,
    plot_results,
)
from generate_dataset import build_ab_dataset


@pytest.fixture(scope="session")
def dataset_path(tmp_path_factory):
    """Generate a reproducible test dataset in a temporary directory."""
    temp_dir = tmp_path_factory.mktemp("data")
    csv_file = str(temp_dir / "test_ab_data.csv")
    build_ab_dataset(csv_file, seed=42)
    return csv_file


def test_dataset_generation_dimensions(dataset_path):
    """Verify raw dataset has exactly 294,478 rows and standard columns."""
    df = pd.read_csv(dataset_path)
    assert len(df) == 294478, f"Expected 294,478 rows, got {len(df):,}"
    expected_cols = {"user_id", "timestamp", "group", "landing_page", "converted"}
    assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"
    assert set(df["group"].unique()) == {"control", "treatment"}
    assert set(df["landing_page"].unique()) == {"old_page", "new_page"}
    assert set(df["converted"].unique()) == {0, 1}


def test_data_cleaning_audit(dataset_path):
    """Verify removal of exactly 3,893 mismatched rows and 1 duplicate user_id."""
    df_raw = pd.read_csv(dataset_path)
    
    # Audit mismatches
    mis_ctrl = (df_raw["group"] == "control") & (df_raw["landing_page"] == "new_page")
    mis_treat = (df_raw["group"] == "treatment") & (df_raw["landing_page"] == "old_page")
    total_mismatched = mis_ctrl.sum() + mis_treat.sum()
    assert total_mismatched == 3893, f"Expected 3,893 mismatched rows, got {total_mismatched}"
    assert mis_ctrl.sum() == 1928, f"Expected 1,928 control+new_page, got {mis_ctrl.sum()}"
    assert mis_treat.sum() == 1965, f"Expected 1,965 treatment+old_page, got {mis_treat.sum()}"

    df_clean = load_and_clean_data(dataset_path)
    assert len(df_clean) == 290584, f"Expected 290,584 clean users, got {len(df_clean):,}"
    
    # Check arms balance
    n_ctrl = (df_clean["group"] == "control").sum()
    n_treat = (df_clean["group"] == "treatment").sum()
    assert n_ctrl == 145274, f"Expected 145,274 control rows, got {n_ctrl}"
    assert n_treat == 145310, f"Expected 145,310 treatment rows, got {n_treat}"
    assert df_clean["user_id"].nunique() == 290584, "Duplicate user_ids detected in clean dataset"


def test_mde_calculation():
    """Verify Minimum Detectable Effect is +0.34 pp on 12% baseline at 80% power."""
    mde = compute_mde(baseline=0.12, n_per_group=145292, alpha=0.05, power=0.80)
    assert round(mde, 2) == 0.34, f"Expected MDE = 0.34 pp, got {mde}"


def test_two_proportion_ztest(dataset_path):
    """Verify conversion rates, z-statistic, p-value, and 95% Wald CI."""
    df_clean = load_and_clean_data(dataset_path)
    results = run_ab_test(df_clean, baseline=0.12, alpha=0.05, power=0.80)

    # Conversion Rates
    assert round(results["rate_ctrl"] * 100, 2) == 12.04, f"Expected 12.04% control, got {results['rate_ctrl']*100:.2f}%"
    assert round(results["rate_treat"] * 100, 2) == 11.88, f"Expected 11.88% treatment, got {results['rate_treat']*100:.2f}%"
    assert results["diff_pp"] == -0.16, f"Expected -0.16 pp lift, got {results['diff_pp']}"

    # Z-Test Statistics
    assert round(results["z_stat"], 2) == -1.31, f"Expected z = -1.31, got {results['z_stat']:.4f}"
    assert round(results["p_val"], 2) == 0.19, f"Expected p = 0.19, got {results['p_val']:.4f}"

    # 95% Confidence Interval
    assert results["ci_low_pp"] == -0.40, f"Expected CI low = -0.40, got {results['ci_low_pp']}"
    assert results["ci_high_pp"] == 0.08, f"Expected CI high = +0.08, got {results['ci_high_pp']}"


def test_business_decision_rule(dataset_path):
    """Verify statistical decision fails to reject H0 and recommends keeping existing page."""
    df_clean = load_and_clean_data(dataset_path)
    results = run_ab_test(df_clean, baseline=0.12, alpha=0.05, power=0.80)

    assert results["decision"] == "Fail to reject H0"
    assert "Keep existing landing page" in results["recommendation"]
    # Verify 95% CI upper bound (0.08 pp) is strictly less than MDE (0.34 pp)
    assert results["ci_high_pp"] < results["mde_pp"], "Upper CI bound must rule out MDE"


def test_visualization_generation(tmp_path):
    """Verify plot_power_curve and plot_results produce valid image files."""
    power_path = str(tmp_path / "test_power.png")
    results_path = str(tmp_path / "test_results.png")

    fig1, ax1 = plot_power_curve(output_path=power_path)
    assert os.path.exists(power_path)
    assert os.path.getsize(power_path) > 1000

    results = {
        "rate_ctrl": 0.1204,
        "rate_treat": 0.1188,
        "n_ctrl": 145274,
        "n_treat": 145310,
        "diff_pp": -0.16,
        "ci": (-0.40, 0.08),
        "mde_pp": 0.34,
        "p_val": 0.19,
    }
    plot_results(results, output_path=results_path)
    assert os.path.exists(results_path)
    assert os.path.getsize(results_path) > 1000
