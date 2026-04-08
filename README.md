# 📐 Multiple Testing & Statistical Inference — Controlling False Discoveries in Data Science

> **Skills Demonstrated:** Multiple Hypothesis Testing · FWER Control · FDR Control · Bonferroni · Holm Step-Down · Benjamini-Hochberg · p-value Analysis · Type I Error Control · Cherry-Picking Bias · Python · Statsmodels · SciPy

---

## 🎯 Project Overview

This project applies **Multiple Testing** methods to prevent false discoveries when simultaneously testing many hypotheses — a critical problem in genomics, finance, A/B testing, and clinical research.

> *"When you test 100 hypotheses at α=0.05, you expect 5 false positives just by chance. How do you know which rejections are real?"*

Two exercises are covered across two datasets:

1. **Carseats Dataset** — Testing which store variables truly predict Sales, applying Type I error control, FWER control (Holm), and FDR control (Benjamini-Hochberg)
2. **Simulated Fund Manager Data** — Testing 100 fund managers' returns, demonstrating why cherry-picking inflates false discoveries and corrupts statistical validity

---

## 📁 Datasets Used

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| **Carseats** | ISLP Simulated | 400 rows, 11 features | Which variables predict Sales? |
| **Simulated (Fund Managers)** | `numpy.random.default_rng(1)` | n=20 months, m=100 managers | Do any managers genuinely outperform? |

**Carseats Quantitative Variables Tested:**

| Variable | Description |
|----------|-------------|
| CompPrice | Competitor's price at each location |
| Income | Community income level |
| Advertising | Local advertising budget |
| Population | Population size in region |
| Price | Price charged for car seats |
| Age | Average age of local population |
| Education | Education level in community |

---

## 🔧 Techniques & Tools Applied

| Technique | Library | Purpose |
|-----------|---------|---------|
| Simple Linear Regression (OLS) | `statsmodels.api.OLS` | Individual p-value extraction per predictor |
| Type I Error Control (α=0.05) | Manual threshold | Naive uncorrected hypothesis testing |
| **FWER Control — Holm Step-Down** | `statsmodels.stats.multitest.multipletests` | Family-wise error rate correction |
| **FDR Control — Benjamini-Hochberg** | `multipletests(method='fdr_bh')` | False discovery rate correction |
| One-Sample t-test | `scipy.stats.ttest_1samp` | Testing fund manager mean returns = 0 |
| p-value Histogram | `matplotlib` | Visualizing null distribution of p-values |
| Cherry-Picking Analysis | Manual selection | Demonstrating selection bias danger |

**Libraries:** `numpy` · `pandas` · `statsmodels` · `scipy` · `matplotlib` · `ISLP`

---

## 📊 Key Results

### Exercise 7 — Carseats Sales Prediction: Multiple Testing Comparison

#### (a) Raw p-values from 7 Individual Linear Regressions

| Variable | p-value | Scientific Notation |
|----------|---------|-------------------|
| **Price** | **7.62 × 10⁻²¹** | Extremely significant |
| **Advertising** | **4.38 × 10⁻⁸** | Very significant |
| **Age** | **2.79 × 10⁻⁶** | Very significant |
| **Income** | **2.31 × 10⁻³** | Significant |
| CompPrice | 2.01 × 10⁻¹ | Not significant |
| Education | 3.00 × 10⁻¹ | Not significant |
| Population | 3.14 × 10⁻¹ | Not significant |

---

#### (b) Naive Type I Error Control (α = 0.05)

| Variable | p-value | Reject H₀? |
|----------|---------|-----------|
| Price | 7.62e-21 | ✅ Yes |
| Advertising | 4.38e-08 | ✅ Yes |
| Age | 2.79e-06 | ✅ Yes |
| Income | 2.31e-03 | ✅ Yes |
| CompPrice | 0.201 | ❌ No |
| Education | 0.300 | ❌ No |
| Population | 0.314 | ❌ No |

> **Rejected: 4 out of 7 hypotheses** (Price, Advertising, Age, Income)

---

#### (c) FWER Control — Holm Step-Down Method (α = 0.05)

Holm's method sorts p-values and applies a stricter threshold to each, controlling the probability of *any* false positive.

| Rank | Variable | p-value | Holm Threshold | Reject? |
|------|----------|---------|----------------|---------|
| 1 | **Price** | 7.62e-21 | α/7 = 0.0071 | ✅ Yes |
| 2 | **Advertising** | 4.38e-08 | α/6 = 0.0083 | ✅ Yes |
| 3 | **Age** | 2.79e-06 | α/5 = 0.010 | ✅ Yes |
| 4 | **Income** | 2.31e-03 | α/4 = 0.0125 | ✅ Yes |
| 5 | CompPrice | 0.201 | α/3 = 0.0167 | ❌ No |
| 6 | Education | 0.300 | α/2 = 0.025 | ❌ No |
| 7 | Population | 0.314 | α/1 = 0.05 | ❌ No |

> **FWER Result: 4 rejections** — same as naive approach here because the 4 significant predictors have extremely small p-values, easily surviving even strict correction. CompPrice, Education, Population all fail FWER correction.

---

#### (d) FDR Control — Benjamini-Hochberg Method (α = 0.2)

| Variable | p-value | Reject (FDR ≤ 0.2)? |
|----------|---------|---------------------|
| Price | 7.62e-21 | ✅ Yes |
| Advertising | 4.38e-08 | ✅ Yes |
| Age | 2.79e-06 | ✅ Yes |
| Income | 2.31e-03 | ✅ Yes |
| CompPrice | 0.201 | ❌ No |
| Population | 0.314 | ❌ No |
| Education | 0.300 | ❌ No |

> **FDR Result: 4 rejections** — at FDR = 0.2, same four variables are selected. BH allows up to 20% of rejected hypotheses to be false discoveries — a more lenient threshold useful in exploratory analysis.

**Summary Comparison Across All Methods:**

| Method | Rejections | Variables Rejected |
|--------|-----------|-------------------|
| Naive (α=0.05) | **4** | Price, Advertising, Age, Income |
| FWER — Holm | **4** | Price, Advertising, Age, Income |
| FDR — BH (α=0.2) | **4** | Price, Advertising, Age, Income |

> **Key Finding:** All three methods agree — **Price, Advertising, Age, and Income** are the genuine predictors of car seat Sales. The consistent result across all correction methods provides strong confidence in these findings.

---

### Exercise 8 — Fund Manager Returns: 100 Simultaneous Tests

**Setup:** m=100 fund managers, n=20 months, all true null hypotheses (population mean = 0)

#### (a) p-value Histogram

> **Shape:** Approximately uniform distribution — as expected when all null hypotheses are true. No clustering near zero indicates no genuine signal.

---

#### (b) Naive Type I Error Control (α = 0.05) — Without Correction

| Metric | Value |
|--------|-------|
| Total managers tested | 100 |
| **Null hypotheses rejected** | **4** |
| Expected false positives (5% × 100) | ~5 |
| All rejections are false positives? | ✅ Yes (all H₀ are true by design) |

> **Problem:** 4 managers appear significant purely by chance — even though none have genuine skill. This is the multiple testing problem in action.

---

#### (c) FWER Control — Holm Step-Down (α = 0.05)

```python
reject = array([False, False, False, ..., False])  # All 100 = False
```

| Metric | Value |
|--------|-------|
| **Null hypotheses rejected** | **0** |
| False positives controlled? | ✅ Yes — perfectly |

> **Result:** Holm correctly rejects zero hypotheses — no false discoveries. FWER control is conservative but reliable.

---

#### (d) FDR Control — Benjamini-Hochberg (α = 0.05)

```python
reject_fdr = array([False, False, False, ..., False])  # All 100 = False
```

| Metric | Value |
|--------|-------|
| **Null hypotheses rejected** | **0** |
| False positives controlled? | ✅ Yes — perfectly |

> **Result:** BH-FDR also correctly rejects zero hypotheses. Both FWER and FDR control successfully protect against all false discoveries when none of the 100 managers have genuine skill.

---

#### (e) Cherry-Picking: Top 10 Best-Performing Managers

**Top 10 managers selected by smallest p-values:**

| Rank | Manager Index | p-value |
|------|--------------|---------|
| 1 | **Manager 14** | **0.000807** |
| 2 | Manager 44 | 0.009551 |
| 3 | Manager 39 | 0.022933 |
| 4 | Manager 27 | 0.035539 |
| 5 | Manager 72 | 0.050266 |
| 6 | Manager 0 | 0.053880 |
| 7 | Manager 50 | 0.065646 |
| 8 | Manager 20 | 0.075501 |
| 9 | Manager 9 | 0.077960 |
| 10 | Manager 35 | 0.087938 |

**Cherry-Picked FWER (Holm) on top 10 only:**

```
reject = [True, False, False, False, False, False, False, False, False, False]
```
> **Rejected: Manager 14** — 1 false rejection

**Cherry-Picked FDR (BH) on top 10 only:**

```
reject = [True, True, False, False, False, False, False, False, False, False]
```
> **Rejected: Managers 14 & 44** — 2 false rejections

| Approach | Rejections | Reality |
|----------|-----------|---------|
| Full 100 managers — FWER | 0 | ✅ Correct |
| Full 100 managers — FDR | 0 | ✅ Correct |
| **Cherry-picked 10 — FWER** | **1** | ❌ False positive |
| **Cherry-picked 10 — FDR** | **2** | ❌ False positives |

---

#### (f) Why Cherry-Picking is Misleading ⚠️

> Cherry-picking the 10 best managers and then applying FWER/FDR correction is **statistically invalid** for three reasons:

1. **Correction methods assume all m hypotheses are tested together.** Holm and BH use the total number of tests (m) in their threshold calculations. When you reduce m from 100 to 10 by cherry-picking, you artificially inflate the per-test threshold — making it far easier to reject.

2. **Selection bias inflates apparent significance.** The top 10 p-values were chosen *because* they were small — they are not a random sample of all 100. By construction they will include the most extreme values from random noise.

3. **False discoveries become inevitable.** Manager 14 (p=0.0008) appears highly significant in isolation, but this is simply the most extreme of 100 random outcomes — expected even when no manager has real skill. Testing only this manager without accounting for the 99 others who were discarded creates a misleading narrative.

**The analogy:** Rolling 100 dice and declaring the highest roll "special" — then testing only that die ignores that you rolled 100 times to find it.

---

## 💡 Business Insights

1. **Price & Advertising Drive Sales — Robustly:** All three correction methods (Naive, FWER, FDR) agree that Price and Advertising predict car seat sales. This consistency across methods means these findings are not statistical artifacts — they are actionable for pricing and marketing strategy.

2. **Most Fund Managers Add No Value:** When 100 managers are properly tested simultaneously, FWER and FDR both reject zero hypotheses — confirming no manager genuinely outperforms. The 4 "significant" managers under naive testing are pure noise. This has major implications for active fund management fees.

3. **Cherry-Picking Destroys Statistical Validity:** Selecting top performers and then testing them in isolation is a common but invalid practice in finance, genomics, and A/B testing. The correction methods lose their protective properties when the selection process is ignored.

4. **FDR is More Powerful than FWER:** At α=0.2, BH-FDR allows up to 20% false discoveries — accepting some errors in exchange for more power to detect true signals. In exploratory data analysis and drug discovery screening, FDR control is preferred over the more conservative FWER.

5. **Real-World Application — A/B Testing:** Running 100 A/B tests simultaneously at α=0.05 expects 5 false positives. Product teams should always apply FDR/FWER correction before declaring test winners — otherwise marketing decisions are based on statistical noise.

---

## 🗂️ File Structure

```
Chapter_13_Applied_Exercise_Solutions/
│
├── Chapter_13.ipynb          ← Main analysis notebook (all exercises)
├── Chapter_13.html           ← Rendered HTML version (easy browser viewing)
├── Chapter_13.qmd            ← Quarto source file
└── README.md                 ← This file
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install ISLP statsmodels scipy pandas numpy matplotlib

# Launch notebook
jupyter notebook Chapter_13.ipynb
```

---

## 📚 Reference

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
*An Introduction to Statistical Learning with Applications in Python.* Springer.
Chapter 13: Multiple Testing — Applied Exercises 7–8.

---

## 🙏 Acknowledgements

Special thanks to **Karim Aboussel Ham** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM) provided useful guidance and reference during the completion of this project.

---

## 👤 About the Author

**Shad Ali Shah**
🎓 MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
💡 Passionate about the intersection of **Economics**, **Data Science**, and **Machine Learning**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shadalishah)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shadalishah)

---

*Part of the [ML Portfolio](../README.md) by Shad Ali Shah*
