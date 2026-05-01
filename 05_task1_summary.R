# =============================================================================
# TASK 1 SUMMARY — All models, both thresholds, all metrics
# =============================================================================
# Models:  LR-A (20 features)  LR-B (9 features)  SVM-A (20 features)  SVM-B (9 features)
#          RF (81 features)    DT (81 features)
# Metrics: Accuracy  Sensitivity  Specificity  Balanced Acc  False Neg  F1  AUC
# =============================================================================


# ── Threshold 0.5 ─────────────────────────────────────────────────────────────
#
# Model    Features   Accuracy  Sensitivity  Specificity  Bal.Acc  FN    F1    AUC
# LR-A     20 (EDA)    0.871     0.642        0.927        0.784   283   0.661  0.928
# LR-B     9  (dedup)  0.871     0.607        0.935        0.771   311   0.647  0.921
# SVM-A    20 (EDA)    0.867     0.564        0.940        0.752   345   0.623  0.916
# SVM-B    9  (dedup)  0.864     0.589        0.930        0.760   325   0.627  0.926
# RF       81 (all)    0.950<<<  0.879<<<     0.968<<<     0.923<<<  96<<< 0.874<<< 0.979<<<
# DT       81 (all)    0.924     0.800        0.954        0.877   158   0.803  0.955
#
# Best (0.5): RF wins every metric


# ── Youden threshold ──────────────────────────────────────────────────────────
#
# Model    Features   Accuracy  Sensitivity  Specificity  Bal.Acc  FN    F1    AUC
# LR-A     20 (EDA)    0.828     0.946        0.799        0.872    43   0.682  0.928
# LR-B     9  (dedup)  0.804     0.970<<<     0.764        0.867    24<<< 0.658 0.921
# SVM-A    20 (EDA)    0.828     0.938        0.801        0.870    49   0.680  0.916
# SVM-B    9  (dedup)  0.836     0.929        0.813        0.871    56   0.688  0.926
# RF       81 (all)    0.933<<<  0.954        0.928<<<     0.941<<<  36  0.848<<< 0.979<<<
# DT       81 (all)    0.902     0.906        0.901        0.904    74   0.784  0.955
#
# Best (Youden):
#   Sensitivity / False Neg : LR-B — highest recall (0.970), fewest missed (24) but lowest specificity so very high 
#                             false alarm rate which may not be worth it in practice.
#   Accuracy / Specificity  : RF   —  highest accuracy (0.933) and specificity (0.928), saving 539 false positives compared to LR-B (235 vs 774).
#                                     Which might be worth it for less Sensitivity - missing only 12 more HTC superconductors (36 vs 24). For lab
#                                     validation this is a strong improvement: 539 fewer wasted experiments to catch 12 extra candidates.
#   Balanced Accuracy / F1  : RF   — best joint optimum
#   AUC                     : RF   — strongest ranking ability (0.980)


# ───────────────────────────────────────────────────────────────
#
# Discovery goal: minimise missed superconductors (false negatives are permanently lost —
# a material never tested is never found), while keeping false positives manageable
# (each false positive is a real lab experiment that costs time and money).
# Sensitivity is the primary metric, but false positive count cannot be ignored in models.
#  
#   → LR-B Youden is the model with lowest false negatives: Sensitivity 0.970, only 24 missed.
#     Using reduced 9 features with collinearity removed — all coefficients significant, interpretable and stable.
#   → RF Youden is the best overall: AUC 0.979, Balanced Acc 0.941, F1 0.848.
#     Sensitivity 0.954 with only 36 missed — 12 more than LR-B, but saves 539
#     false positives (235 vs 774). In a lab setting that means 539 fewer wasted
#     experiments to recover those 12 extra candidates.
# 
#  -So best option is probably RF with Youden threshold even with lower Sensitivity than LR-B.-
#
# Interpretability ranking (high → low):
#   LR  >  DT  >  RF  >  SVM
#   (coefficients + p-values) (tree diagram) (Gini importance) (black box)
#
# Feature-space finding:
#   Collinearity helps  RF  (more features → more signal, subsampling handles redundancy).
#   Collinearity hurts  SVM (distorts margin: SVM-B 0.926 > SVM-A 0.916).
#   Collinearity hurts  LR  (inflated std. errors, non-significant coefficients in LR-A).
#   DT is collinearity-neutral (greedy split picks one feature at a time).


# ── RF feature importance vs EDA top-20 ──────────────────────────────────────
#
# Unique to RF (MeanDecreaseGini, not in EDA top-20):
#   wtd_std_Valence, wtd_range_Valence, wtd_mean_ThermalConductivity,
#   wtd_std_ElectronAffinity, std_atomic_mass, wtd_entropy_ThermalConductivity,
#   wtd_range_ThermalConductivity, wtd_gmean_ElectronAffinity,
#   wtd_gmean_ThermalConductivity, wtd_std_atomic_mass
#
# Unique to EDA (SMD ranking, not in RF top-20):
#   mean_Valence, range_fie, wtd_std_atomic_radius, gmean_Valence,
#   std_atomic_radius, entropy_Valence, wtd_std_fie, std_fie,
#   gmean_Density, range_atomic_mass
#
# Shared (10 features in both top-20s):
#   wtd_std_ThermalConductivity, range_ThermalConductivity, std_ThermalConductivity,
#   range_atomic_radius, wtd_mean_Valence, wtd_gmean_Valence,
#   wtd_entropy_atomic_mass, wtd_entropy_Valence, wtd_entropy_FusionHeat,
#   wtd_entropy_atomic_radius
#
# Takeaway: EDA (SMD) and RF (Gini) agree on 10 core features.
# RF additionally values other features that EDA did not show, likely because their predictive
# power comes from interactions with other features rather than marginal class separation (which SMD measures).
