library(tidyverse)
library(randomForest)
library(caret)        
library(pROC)         
library(ROCit)       


load("prepared_data.RData")
# Loaded: train_df, test_df, y_train, y_test, numeric_predictors

# =============================================================================
# MODEL 3: RANDOM FOREST all numeric features
# =============================================================================

set.seed(42)
model_rf <- randomForest(
  tc_class ~ .,
  data       = bind_cols(train_df %>% select(all_of(numeric_predictors)),
                         tc_class = y_train),
  ntree      = 500,
  mtry       = floor(sqrt(length(numeric_predictors))),
  importance = TRUE
)
# OOB sampling gives an unbiased estimate of test error without a separate validation set.
cat("RF OOB error:", round(model_rf$err.rate[500, "OOB"], 4), "\n")

class_rf <- predict(model_rf, newdata = test_df %>% select(all_of(numeric_predictors)))
prob_rf  <- predict(model_rf, newdata = test_df %>% select(all_of(numeric_predictors)),
                    type = "prob")[, "high_tc"]

print(confusionMatrix(class_rf, y_test, positive = "high_tc"))

roc_rf <- roc(y_test, prob_rf, levels = c("non_high_tc", "high_tc"), quiet = TRUE)


# ── threshold Youden ────────────────────
measure_rf    <- measureit(class = as.numeric(y_test == "high_tc"),
                           score = prob_rf, measure = c("SENS", "SPEC"))
youden_rf     <- measure_rf$SENS + measure_rf$SPEC - 1
best_idx_rf   <- which.max(youden_rf)
opt_cutoff_rf <- measure_rf$Cutoff[best_idx_rf]
cat("RF optimal cutoff (Youden):", round(opt_cutoff_rf, 4),
    "| Sensitivity:", round(measure_rf$SENS[best_idx_rf], 3),
    "| Specificity:", round(measure_rf$SPEC[best_idx_rf], 3), "\n")

class_rf2 <- factor(
  if_else(prob_rf >= opt_cutoff_rf, "high_tc", "non_high_tc"),
  levels = levels(y_train)
)
print(confusionMatrix(class_rf2, y_test, positive = "high_tc"))


# ── RF summary ────────────────────────────────────────────────────────────────
# Threshold  Accuracy  Sensitivity  Specificity  Balanced Acc  False Neg  AUC
#  0.500      0.951     0.877        0.968        0.923          97       0.980
#  Youden     0.939     0.947        0.937        0.942          42       0.980
#
# RF (Youden) is the better choice for our use case (superconductor discovery / screening):
# - Sensitivity 0.947 — catches 94.7% of true high_tc materials, missing only 42
# - Youden threshold recovers 55 additional true superconductors vs default 0.5
# - Specificity drop (0.968 → 0.937) is acceptable: 103 extra false alarms go to
#   experimental validation where they are filtered out, but the 55 recovered
#   candidates would otherwise be permanently missed
# - AUC unchanged at 0.980 — threshold shift moves the operating point on the ROC curve
# ---- Balanced Accuracy improves from 0.923 → 0.942 because Youden maximises Sens+Spec together

# ── ROC with Youden point ─────────────────────────────────────────────────────
plot(roc_rf, main = paste0("ROC — Random Forest (AUC = ", round(auc(roc_rf), 3), ")"))
points(
  x   = measure_rf$SPEC[best_idx_rf],
  y   = measure_rf$SENS[best_idx_rf],
  pch = 19, col = "red", cex = 1.5
)
legend("bottomright", legend = "Youden threshold", col = "red", pch = 19, bty = "n")



# ── Feature importance BETWEEN RF from all features and selected by EDA ─────
imp_df <- importance(model_rf) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  arrange(desc(MeanDecreaseGini))

top20_rf  <- imp_df %>% slice_head(n = 20) %>% pull(feature)

cat("\n── Top 20 RF (MeanDecreaseGini) vs Top 20 EDA (SMD) ──\n")
print(data.frame(rank = 1:20, RF_importance = top20_rf, EDA_smd = top20_eda))
cat("\nUnique to RF (not in EDA top 20):", paste(setdiff(top20_rf, top20_eda), collapse = ", "), "\n")
cat("Unique to EDA (not in RF top 20):", paste(setdiff(top20_eda, top20_rf), collapse = ", "), "\n")

# Unique to RF:  wtd_std_Valence, wtd_range_Valence, wtd_mean_ThermalConductivity,
#                wtd_std_ElectronAffinity, std_atomic_mass, wtd_entropy_ThermalConductivity,
#                wtd_range_ThermalConductivity, wtd_gmean_ElectronAffinity,
#                wtd_gmean_ThermalConductivity, wtd_std_atomic_mass
# Unique to EDA: mean_Valence, range_fie, wtd_std_atomic_radius, gmean_Valence,
#                std_atomic_radius, entropy_Valence, wtd_std_fie, std_fie,
#                gmean_Density, range_atomic_mass
# Shared (10):   wtd_std_ThermalConductivity, range_ThermalConductivity, std_ThermalConductivity,
#                range_atomic_radius, wtd_mean_Valence, wtd_gmean_Valence,
#                wtd_entropy_atomic_mass, wtd_entropy_Valence, wtd_entropy_FusionHeat,
#                wtd_entropy_atomic_radius

# ── Explainability & feature-space benefits ───────────────────────────────────
# Explainability: RF is partially interpretable — MeanDecreaseGini ranks feature
# importance globally but gives no directional effect. We cannot say directly
# "higher wtd_mean_Valence → more likely high_tc" from importance alone only if the feature is important.
#
# Feature-space approach (all 81 numeric features):
#   Benefit: RF handles collinearity natively via random feature subsampling per split.
#   Retaining all features allows the forest to exploit weak signals not captured by EDA.
#   Cost: no coefficient interpretation — model is a black box at the individual level.
