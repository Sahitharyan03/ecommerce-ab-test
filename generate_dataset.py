"""
Reproducible Dataset Generator for E-Commerce Landing Page A/B Test (ab_data.csv)
Matches the exact row counts and conversion counts of the public Kaggle E-Commerce A/B Test dataset:
- Total raw rows: 294,478
- Clean Control (control + old_page): 145,274 rows (17,489 converted -> 12.0386%)
- Clean Treatment (treatment + new_page): 145,310 unique users + 1 duplicate user_id (773192) = 145,311 rows (17,264 converted -> 11.8808%)
- Mismatched rows:
  - control + new_page: 1,928 rows (234 converted)
  - treatment + old_page: 1,965 rows (250 converted)
  - Total mismatched: 3,893 rows
"""

import numpy as np
import pandas as pd


def build_ab_dataset(output_path: str = "ab_data.csv", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 1. Clean Control: 145,274 rows, 17,489 conversions
    n_ctrl = 145274
    conv_ctrl = np.zeros(n_ctrl, dtype=int)
    conv_ctrl[:17489] = 1
    rng.shuffle(conv_ctrl)

    # 2. Clean Treatment: 145,310 unique rows, 17,264 conversions
    n_treat = 145310
    conv_treat = np.zeros(n_treat, dtype=int)
    conv_treat[:17264] = 1
    rng.shuffle(conv_treat)

    # 3. Mismatched control + new_page: 1,928 rows, 234 conversions
    n_mis_ctrl = 1928
    conv_mis_ctrl = np.zeros(n_mis_ctrl, dtype=int)
    conv_mis_ctrl[:234] = 1
    rng.shuffle(conv_mis_ctrl)

    # 4. Mismatched treatment + old_page: 1,965 rows, 250 conversions
    n_mis_treat = 1965
    conv_mis_treat = np.zeros(n_mis_treat, dtype=int)
    conv_mis_treat[:250] = 1
    rng.shuffle(conv_mis_treat)

    total_unique = n_ctrl + n_treat + n_mis_ctrl + n_mis_treat  # 294,477 unique rows
    user_ids = np.arange(630000, 630000 + total_unique)
    rng.shuffle(user_ids)
    dup_user_id = int(user_ids[n_ctrl])

    groups = (
        ["control"] * n_ctrl
        + ["treatment"] * n_treat
        + ["control"] * n_mis_ctrl
        + ["treatment"] * n_mis_treat
    )
    pages = (
        ["old_page"] * n_ctrl
        + ["new_page"] * n_treat
        + ["new_page"] * n_mis_ctrl
        + ["old_page"] * n_mis_treat
    )
    converted = np.concatenate([conv_ctrl, conv_treat, conv_mis_ctrl, conv_mis_treat])

    # Generate realistic timestamps across a 22-day test window in January
    start_ts = pd.Timestamp("2017-01-02 13:42:05")
    random_seconds = rng.integers(0, 22 * 86400, size=total_unique)
    timestamps = (start_ts + pd.to_timedelta(random_seconds, unit="s")).strftime("%Y-%m-%d %H:%M:%S.%f")

    df = pd.DataFrame({
        "user_id": user_ids,
        "timestamp": timestamps,
        "group": groups,
        "landing_page": pages,
        "converted": converted,
    })

    # Add the 1 duplicate treatment user_id row to reach 294,478 raw rows
    dup_row = pd.DataFrame([{
        "user_id": dup_user_id,
        "timestamp": "2017-01-14 02:55:59.590927",
        "group": "treatment",
        "landing_page": "new_page",
        "converted": 0,
    }])
    df = pd.concat([df, dup_row], ignore_index=True)

    # Shuffle row order reproducibly
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
    return df


if __name__ == "__main__":
    build_ab_dataset("ab_data.csv")
