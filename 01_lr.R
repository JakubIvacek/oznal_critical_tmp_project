library(tidyverse)
library(caret)   
library(pROC)    
library(broom)  
library(ROCit)   


load("prepared_data.RData")
# Loaded: train_df, test_df, y_train, y_test, top20_eda, lr2_features, numeric_predictors

# =============================================================================
# MODEL 1a: LOGISTIC REGRESSION — 20 top features (all shortlisted)
# =============================================================================

fit_lr <- glm(
  tc_class ~ .,
  data   = bind_cols(train_df %>% select(all_of(top20_eda)), tc_class = y_train),
  family = binomial(link = "logit")
)
cat("\nLR-A converged:", fit_lr$converged, "\n")
print(tidy(fit_lr), n = 21)
# Non-significant (p > 0.05): mean_Valence (p=0.176), gmean_Valence (p=0.756)

# ── threshold 0.5 ──
prob_lr  <- predict(fit_lr, newdata = test_df %>% select(all_of(top20_eda)), type = "response")
class_lr <- factor(if_else(prob_lr >= 0.5, "high_tc", "non_high_tc"), levels = levels(y_train))
print(confusionMatrix(class_lr, y_test, positive = "high_tc"))
roc_lr <- roc(y_test, prob_lr, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_lr, main = paste0("ROC — LR-A (20 feat)  (AUC = ", round(auc(roc_lr), 3), ")"))

# ── threshold Youden ──
measure_lra    <- measureit(class = as.numeric(y_test == "high_tc"),
                            score = prob_lr, measure = c("SENS", "SPEC"))
youden_lra     <- measure_lra$SENS + measure_lra$SPEC - 1
opt_cutoff_lra <- measure_lra$Cutoff[which.max(youden_lra)]
cat("LR-A Youden cutoff:", round(opt_cutoff_lra, 4),
    "| Sensitivity:", round(measure_lra$SENS[which.max(youden_lra)], 3),
    "| Specificity:", round(measure_lra$SPEC[which.max(youden_lra)], 3), "\n")
class_lra_y <- factor(if_else(prob_lr >= opt_cutoff_lra, "high_tc", "non_high_tc"),
                      levels = levels(y_train))
print(confusionMatrix(class_lra_y, y_test, positive = "high_tc"))

# ── LR-A summary ──────────────────────────────────────────────────────────────
# Threshold  Sensitivity  Specificity  Balanced Acc  False Neg  AUC
#  0.500      0.636        0.924        0.780         311       0.928
#  Youden     0.946        0.799        0.873          43       0.928
#
# LR-A Youden is preferred for discovery: sensitivity jumps from 0.636 → 0.946,
# recovering 268 additional true superconductors at the cost of more false alarms.
# 2 features non-significant (mean_Valence, gmean_Valence) — collinearity inflates std.errors.
# ---- Balanced Accuracy improves from 0.780 → 0.873 with Youden threshold.



# =============================================================================
# MODEL 1b: LOGISTIC REGRESSION — 9 deduplicated features (collinearity removed)
# =============================================================================


fit_lr2 <- glm(
  tc_class ~ .,
  data   = bind_cols(train_df %>% select(all_of(lr2_features)), tc_class = y_train),
  family = binomial(link = "logit")
)
cat("\nLR-B converged:", fit_lr2$converged, "\n")
print(tidy(fit_lr2), n = length(lr2_features) + 1)

# ── LR-B threshold 0.5 ──
prob_lr2  <- predict(fit_lr2, newdata = test_df %>% select(all_of(lr2_features)), type = "response")
class_lr2 <- factor(if_else(prob_lr2 >= 0.5, "high_tc", "non_high_tc"), levels = levels(y_train))
print(confusionMatrix(class_lr2, y_test, positive = "high_tc"))
roc_lr2 <- roc(y_test, prob_lr2, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_lr2, main = paste0("ROC — LR-B (9 feat)  (AUC = ", round(auc(roc_lr2), 3), ")"))

# ── LR-B threshold Youden ──
measure_lrb    <- measureit(class = as.numeric(y_test == "high_tc"),
                            score = prob_lr2, measure = c("SENS", "SPEC"))
youden_lrb     <- measure_lrb$SENS + measure_lrb$SPEC - 1
opt_cutoff_lrb <- measure_lrb$Cutoff[which.max(youden_lrb)]
cat("LR-B Youden cutoff:", round(opt_cutoff_lrb, 4),
    "| Sensitivity:", round(measure_lrb$SENS[which.max(youden_lrb)], 3),
    "| Specificity:", round(measure_lrb$SPEC[which.max(youden_lrb)], 3), "\n")
class_lrb_y <- factor(if_else(prob_lr2 >= opt_cutoff_lrb, "high_tc", "non_high_tc"),
                      levels = levels(y_train))
print(confusionMatrix(class_lrb_y, y_test, positive = "high_tc"))

# ── LR-B summary ──────────────────────────────────────────────────────────────
# Threshold  Sensitivity  Specificity  Balanced Acc  False Neg  AUC
#  0.500      0.607        0.935        0.771         311       0.921
#  Youden     0.970        0.764        0.867          24       0.921
#
# LR-B Youden is the best linear model for discovery: sensitivity 0.970, missing only 24.
# All 9 coefficients stable and significant — collinearity resolved and reduced features.
# ---- Balanced Accuracy improves from 0.771 → 0.867 with Youden threshold.
