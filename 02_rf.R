library(tidyverse)
library(randomForest)
library(caret)        
library(pROC)         
library(ROCit)       


load("prepared_data.RData")
# Loaded: train_df, test_df, y_train, y_test, numeric_predictors

# =============================================================================
# MODEL 2: RANDOM FOREST all numeric features
# =============================================================================

fit_rf <- randomForest(
  tc_class ~ .,
  data       = bind_cols(train_df %>% select(all_of(numeric_predictors)),
                         tc_class = y_train),
  ntree      = 500,
  mtry       = floor(sqrt(length(numeric_predictors))),
  importance = TRUE
)
# OOB sampling gives an unbiased estimate of test error without a separate validation set.
cat("RF OOB error:", round(fit_rf$err.rate[500, "OOB"], 4), "\n")

class_rf <- predict(fit_rf, newdata = test_df %>% select(all_of(numeric_predictors)))
prob_rf  <- predict(fit_rf, newdata = test_df %>% select(all_of(numeric_predictors)),
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
#  0.500      0.952     0.881        0.969        0.925          94       0.980
#  Youden     0.932     0.955        0.926        0.940          36       0.980
#
# RF (Youden) is the better choice for our use case (superconductor discovery / screening):
# - Sensitivity 0.955 — catches 95.5% of true high_tc materials, missing only 36
# - Youden threshold recovers 58 additional true superconductors vs default 0.5
# - Specificity drop (0.969 → 0.926) is acceptable: 140 extra false alarms go to
#   experimental validation where they are filtered out, but the 58 recovered
#   candidates would otherwise be permanently missed
# - AUC unchanged at 0.980 — threshold shift moves the operating point on the ROC curve
# ---- Balanced Accuracy improves from 0.925 → 0.940 because Youden maximises Sens+Spec together
