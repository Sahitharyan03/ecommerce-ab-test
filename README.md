# E-Commerce Landing Page A/B Test

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Statistical Stack](https://img.shields.io/badge/stack-SciPy%20|%20statsmodels%20|%20pandas-orange.svg)](https://www.statsmodels.org/)
[![Tests Passing](https://img.shields.io/badge/tests-6%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** Sahith Aryan  
**Tech Stack:** Python, `pandas`, `NumPy`, `SciPy`, `statsmodels`, `Matplotlib`, `Seaborn`, `pytest`  
**Dataset:** Public E-Commerce A/B Testing Dataset (~294K raw observations, ~290K clean users)

---

## Executive Summary & Key Highlights

- **Designed the analysis for a ~290K-user landing page test:** Computed a **0.34 percentage point (pp) minimum detectable effect (MDE)** on a **12.0% baseline** at $\alpha = 0.05$ (two-sided) and **80% statistical power**, and removed **~3.9K rows with mismatched group assignments** before testing.
- **Hypothesis Testing & Decision:** A two-proportion $z$-test found **no significant difference** (**control 12.04%** vs. **new page 11.88%**, $p = 0.190$, $z = -1.3109$, observed lift = **$-0.16\text{ pp}$**; $\alpha = 0.05$, 80% power; **95% CI: $-0.40\text{ pp}$ to $+0.08\text{ pp}$**); recommended **keeping the existing landing page**, since the confidence interval completely ruled out any lift near the $+0.34\text{ pp}$ MDE.

---

## Visual Proof & Experimental Evidence

### 1. Group Alignment Audit & Statistical Power Curve
Before running hypothesis testing, assignment integrity was verified using a cross-tabulation of user groups against served landing pages. A statistical power curve was computed across candidate lifts on the 12.00% baseline:

![Power Curve and Alignment Verification](docs/screenshots/jupyter_crosstab_power_curve.jpg)

### 2. Conversion Rate Comparison & 95% Confidence Interval vs. MDE
The observed conversion rate and 95% confidence interval for the treatment effect versus the $+0.34\text{ pp}$ Minimum Detectable Effect threshold:

![A/B Test Results and Confidence Interval](docs/screenshots/jupyter_ab_test_results.jpg)

---

## Key Experimental Results Summary

| Metric / Parameter | Value | Interpretation & Notes |
| :--- | :--- | :--- |
| **Raw Traffic Logs** | 294,478 rows | Initial raw event dataset |
| **Mismatched Records Removed** | 3,893 rows | 1,928 (`control` + `new_page`) + 1,965 (`treatment` + `old_page`) |
| **Duplicate Users Removed** | 1 row | Duplicate `user_id = 773192` in treatment group |
| **Clean Sample Size ($N$)** | **290,584 users** | Control ($n = 145,274$) \| Treatment ($n = 145,310$) |
| **Baseline Conversion Rate ($p_0$)** | 12.00% | Historical control benchmark |
| **Significance Level ($\alpha$)** | 0.05 | Two-sided critical value ($Z_{0.025} = 1.96$) |
| **Target Statistical Power ($1 - \beta$)** | 80.0% | Minimum probability of detecting true effect if present |
| **Minimum Detectable Effect (MDE)** | **$+0.34\text{ pp}$** | Lift from $12.00\% \to 12.34\%$ required for statistical detection |
| **Control Group Conversion Rate** | **12.04%** | $17,489 / 145,274$ conversions |
| **Treatment Group Conversion Rate** | **11.88%** | $17,264 / 145,310$ conversions |
| **Observed Treatment Effect ($\Delta$)** | **$-0.16\text{ pp}$** | $-0.001578$ difference between variants |
| **Two-Sample Z-Statistic** | **$-1.3109$** | Within acceptance region $[-1.96, +1.96]$ |
| **Two-Sided P-Value** | **0.190** | $p > 0.05 \implies$ Fail to reject null hypothesis $H_0$ |
| **95% Wald Confidence Interval** | **$[-0.40\text{ pp}, +0.08\text{ pp}]$** | Upper bound ($+0.08\text{ pp}$) is far below $+0.34\text{ pp}$ MDE |
| **Final Product Recommendation** | **Keep Existing Page** | Maintain `old_page`; avoid migration overhead |

---

## Methodology & Statistical Framework

### 1. Statistical Power & Sample Size Formulation
To detect a meaningful business improvement, we parameterized the test using Cohen's $h$ effect size for two independent binomial proportions:
$$h = 2 \arcsin(\sqrt{p_2}) - 2 \arcsin(\sqrt{p_1})$$

For sample size per group $n = 145,292$, $\alpha = 0.05$, and power $1 - \beta = 0.80$:
$$h = (Z_{1 - \alpha/2} + Z_{1 - \beta}) \sqrt{\frac{2}{n}} = (1.960 + 0.8416) \sqrt{\frac{2}{145,292}} \approx 0.01040$$

Inverting back to the required target proportion $p_2$:
$$p_2 = \sin\left(\arcsin(\sqrt{0.12}) + \frac{0.01040}{2}\right)^2 \approx 0.12339 \implies \text{MDE} = +0.339\text{ pp} \approx +0.34\text{ pp}$$

### 2. Data Hygiene & Assignment Audit
Real-world tracking pipelines occasionally suffer from edge-case delivery anomalies (e.g. browser cache collisions, redirect delays, race conditions). Prior to testing, a contingency table audit revealed:
- **1,928** control participants who erroneously received the `new_page`.
- **1,965** treatment participants who erroneously received the `old_page`.
- **1** user (`user_id = 773192`) logged twice in the treatment arm.

Including contaminated users would introduce severe bias into the intention-to-treat analysis. Removing these 3,894 non-conforming rows yielded an uncompromised, clean sample of **290,584 unique visitors** (balanced 50.0% / 50.0%).

### 3. Hypothesis Testing: Two-Proportion Z-Test
We formulated a two-sided test of proportions:
$$\begin{cases} H_0: p_{\text{treatment}} - p_{\text{control}} = 0 \\ H_1: p_{\text{treatment}} - p_{\text{control}} \neq 0 \end{cases}$$

Using the pooled conversion rate under $H_0$:
$$p_{\text{pool}} = \frac{17,489 + 17,264}{145,274 + 145,310} = \frac{34,753}{290,584} \approx 0.119597$$
$$SE_{\text{pool}} = \sqrt{p_{\text{pool}}(1 - p_{\text{pool}})\left(\frac{1}{n_{\text{ctrl}}} + \frac{1}{n_{\text{treat}}}\right)} \approx 0.0012038$$
$$Z = \frac{0.118808 - 0.120386}{0.0012038} = -1.3109 \implies p = 2 \times \Phi(-1.3109) = 0.1899 \approx 0.190$$

Because $p = 0.190 \ge 0.05$, we fail to reject the null hypothesis.

### 4. Confidence Interval & Equivalence Rule-Out
The 95% Wald confidence interval for the true difference in proportions is:
$$CI_{95\%} = (\hat{p}_{\text{treat}} - \hat{p}_{\text{ctrl}}) \pm 1.96 \sqrt{\frac{\hat{p}_{\text{treat}}(1 - \hat{p}_{\text{treat}})}{n_{\text{treat}}} + \frac{\hat{p}_{\text{ctrl}}(1 - \hat{p}_{\text{ctrl}})}{n_{\text{ctrl}}}}$$
$$CI_{95\%} = -0.16\text{ pp} \pm 0.24\text{ pp} = [-0.40\text{ pp}, +0.08\text{ pp}]$$

**Crucial Business Insight:** Failing to reject $H_0$ only indicates lack of evidence for a difference. However, our confidence interval proves that the true lift cannot reasonably exceed $+0.08\text{ pp}$—which is far below the $+0.34\text{ pp}$ MDE needed to justify site re-architecture. We thus conclude with high statistical confidence that the new landing page provides no viable commercial benefit.

---

## Repository Structure

```text
ecommerce-ab-test/
├── README.md                      # Comprehensive project documentation & experimental report
├── LICENSE                        # MIT License
├── requirements.txt               # Pinned Python package dependencies
├── ab_test_analysis.py            # End-to-end Python pipeline (cleaning, z-test, visualizations)
├── generate_dataset.py            # Reproducible data generator matching public Kaggle dataset
├── ab_testing.ipynb               # Fully executed Jupyter Notebook with rendered outputs & charts
├── figures/                       # Publication-ready exported vector & raster figures
│   ├── power_curve.png            # Statistical power vs minimum detectable effect
│   └── ab_test_results.png        # Conversion rate by group & 95% CI vs MDE
├── docs/
│   └── screenshots/               # Visual proof from interactive Jupyter sessions
│       ├── jupyter_crosstab_power_curve.jpg
│       └── jupyter_ab_test_results.jpg
└── tests/
    └── test_ab_testing.py         # Pytest test suite (6 passing tests covering all calculations)
```

---

## Quickstart & Reproduction

### 1. Clone the repository & install dependencies
```bash
git clone https://github.com/Sahitharyan03/ecommerce-ab-test.git
cd ecommerce-ab-test
pip install -r requirements.txt
```

### 2. Run the end-to-end analysis pipeline
```bash
python ab_test_analysis.py
```
*Note: If `ab_data.csv` is not present, the script automatically calls `generate_dataset.py` to create the exact 294,478-row dataset with identical distributions in under 4 seconds.*

### 3. Run the automated test suite
```bash
pytest -v tests/
```
Output:
```text
tests/test_ab_testing.py::test_dataset_generation_dimensions PASSED      [ 16%]
tests/test_ab_testing.py::test_data_cleaning_audit PASSED                [ 33%]
tests/test_ab_testing.py::test_mde_calculation PASSED                    [ 50%]
tests/test_ab_testing.py::test_two_proportion_ztest PASSED               [ 66%]
tests/test_ab_testing.py::test_business_decision_rule PASSED             [ 83%]
tests/test_ab_testing.py::test_visualization_generation PASSED           [100%]
============================== 6 passed in 2.68s ===============================
```

### 4. Interactive Exploration
Launch Jupyter Notebook to inspect the live analysis:
```bash
jupyter notebook ab_testing.ipynb
```
