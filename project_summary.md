# Project Summary — Superconductivity Critical Temperature Prediction

## Dataset
UCI Superconductivity dataset — 81 atomic/chemical features, ~21,000 materials, target: `critical_temp` (Kelvin). No missing values, clean after duplicate removal.

---

## Core Hypothesis

> **Atomic and chemical properties of a material are sufficient to predict its superconducting critical temperature. The physically meaningful threshold of 77K (liquid nitrogen boiling point) divides superconductors into two distinct regimes — low-temperature and high-temperature — where different atomic features drive superconductivity. A classification model can reliably identify which regime a material belongs to, and regime-specific regression models will outperform a single global model.**

---

## Why 77K?

Materials that superconduct above 77K (high-temperature superconductors, HTS) can be cooled using liquid nitrogen, which is ~50x cheaper than the liquid helium required for low-Tc materials. This makes the boundary practically and scientifically significant — it is not an arbitrary split.

**Class distribution: 81.4% below 77K, 18.6% above.**

The imbalance is not a data quality problem — it reflects physical reality. HTS materials are genuinely rare. This means a naive classifier predicting "below 77K" always would achieve 81% accuracy while being completely useless. Our models must be evaluated on AUC, F1, and minority-class recall, and trained with class weights to properly learn the rare but important HTS class. This is a scientifically motivated modelling decision.

---

## Project Framing (connects the two scenarios)

The 77K liquid nitrogen threshold structures the entire project as a two-stage pipeline:

1. **Classifier** (Scenario 1) — predicts which regime a material belongs to (below/above 77K)
2. **Regime-specific regression** (Scenario 1 + 3) — predicts exact Tc using features selected per regime (Scenario 3)

```
New material (81 features)
        ↓
[Classifier: is critical_temp >= 77K?]
        ↓
   YES (HTS)              NO (low-Tc)
      ↓                       ↓
[HTS regression         [Low-Tc regression
   model]                   model]
      ↓                       ↓
   predicted Tc           predicted Tc
```

This pipeline is **not a third scenario** — it is the narrative that motivates why the two scenarios are done together. The global vs regime-specific comparison is evaluated within Scenario 1 by comparing RMSE of a single global model against the two-stage pipeline on the same held-out test set.

If the classifier misclassifies a material, errors compound — so the two-stage approach only wins if classification is accurate enough. This trade-off is explicitly discussed and tested.

---

## Scenarios

### Scenario 3 — Feature Selection
- Algorithmic: forward + backward stepwise selection
- Embedded: Lasso, Ridge, Elastic Net
- Run on both the full dataset and separately within each regime
- Key question: *do the same features matter for low-Tc and high-Tc materials?*

### Scenario 1 — Model Comparison
- Compare 3+ regression methods (linear regression, random forest, gradient boosting) for Tc prediction
- Compare 2 partitioning approaches for classification (regression tree, random forest classifier)
- Evaluate global model vs two-stage regime-specific pipeline on the same held-out test set

---

## Target Variable
`log1p(critical_temp)` for regression — right-skewed distribution, log transformation reduces skewness significantly. Back-transform predictions for reporting RMSE in original Kelvin scale.

---

## Key Modelling Decisions to Justify

| Decision | Rationale |
|---|---|
| `log1p` transform on target | Right-skewed distribution, near-zero values present |
| 77K classification threshold | Liquid nitrogen boundary — physical and economic significance |
| Class weights over SMOTE | Simpler, interpretable, avoids synthetic data generation |
| Stratified train/test split | Preserves 81/19 class ratio in both sets |
| Regime-specific feature selection | Multicollinearity differs between groups; different physics |
| Two-stage pipeline | Tests whether physically motivated split improves prediction |

---

## Deliverables
- R Markdown with full reproducible workflow
- Shiny app: data explorer + model selector + parameter sliders
- One-page executive summary (written by you, no LLM text)
- **Deadline: May 1, 2026**
