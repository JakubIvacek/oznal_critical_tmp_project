library(tidyverse)
library(e1071)
library(caret)
library(pROC)
library(ROCit)


load("prepared_data.RData")
# Loaded: train_df, test_df, y_train, y_test, lr2_features

# =============================================================================
# MODEL 3: SVM — radial kernel, 9 deduplicated features
# =============================================================================
# SVM is sensitive to feature scale → center and scale based on training data only.
# RBF kernel handles non-linear class boundaries without manual feature engineering.
# lr2_features reused deduplicated: collinearity hurts SVM

preproc          <- preProcess(train_df %>% select(all_of(lr2_features)),
                               method = c("center", "scale"))
train_svm_scaled <- predict(preproc, train_df %>% select(all_of(lr2_features)))
test_svm_scaled  <- predict(preproc, test_df  %>% select(all_of(lr2_features)))

fit_svm <- svm(
  x           = train_svm_scaled,
  y           = y_train,
  kernel      = "radial",
  probability = TRUE
)

# ── threshold 0.5 ──
prob_svm  <- attr(predict(fit_svm, test_svm_scaled, probability = TRUE),
                  "probabilities")[, "high_tc"]
class_svm <- factor(if_else(prob_svm >= 0.5, "high_tc", "non_high_tc"),
                    levels = levels(y_train))
print(confusionMatrix(class_svm, y_test, positive = "high_tc"))

roc_svm <- roc(y_test, prob_svm, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_svm, main = paste0("ROC — SVM (AUC = ", round(auc(roc_svm), 3), ")"))

# ── threshold Youden ──
measure_svm    <- measureit(class = as.numeric(y_test == "high_tc"),
                            score = prob_svm, measure = c("SENS", "SPEC"))
youden_svm     <- measure_svm$SENS + measure_svm$SPEC - 1
opt_cutoff_svm <- measure_svm$Cutoff[which.max(youden_svm)]
cat("SVM Youden cutoff:", round(opt_cutoff_svm, 4),
    "| Sensitivity:", round(measure_svm$SENS[which.max(youden_svm)], 3),
    "| Specificity:", round(measure_svm$SPEC[which.max(youden_svm)], 3), "\n")

class_svm_y <- factor(if_else(prob_svm >= opt_cutoff_svm, "high_tc", "non_high_tc"),
                      levels = levels(y_train))
print(confusionMatrix(class_svm_y, y_test, positive = "high_tc"))

# ── SVM summary ───────────────────────────────────────────────────────────────
# Threshold  Accuracy  Sensitivity  Specificity  Balanced Acc  False Neg  AUC
#  0.500      0.864     0.589        0.930        0.760         325       0.926
#  Youden     0.836     0.929        0.813        0.871          56       0.926
#
# SVM (Youden) recovers 269 additional HT superconductors vs default 0.5 threshold.
# AUC 0.926 — comparable to LR-B (0.921), weaker than RF (0.980).
# Default threshold heavily biased toward non_high_tc (Sensitivity only 0.589).
# Youden restores balance: Sensitivity 0.929, missing only 56 true superconductors.

plot(roc_svm, main = paste0("ROC — SVM (AUC = ", round(auc(roc_svm), 3), ")"))
points(
  x   = measure_svm$SPEC[which.max(youden_svm)],
  y   = measure_svm$SENS[which.max(youden_svm)],
  pch = 19, col = "red", cex = 1.5
)
legend("bottomright", legend = "Youden threshold", col = "red", pch = 19, bty = "n")

# ── Explainability & feature-space benefits ───────────────────────────────────
# Explainability: SVM is the least interpretable model — RBF kernel maps data
# into an implicit high-dimensional space. No feature coefficients exist.
#
# Feature-space (lr2_features, 9 features):
#   Benefit: collinearity removed — SVM margin estimation is more stable.
#   SVM maximises the margin between classes; correlated features can distort it.