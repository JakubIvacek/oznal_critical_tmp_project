library(tidyverse)
library(e1071)
library(caret)
library(pROC)
library(ROCit)
library(MLmetrics)


load("prepared_data.RData")
# Loaded: train_df, test_df, y_train, y_test, top20_eda, lr2_features

# =============================================================================
# MODEL 2a: SVM — radial kernel, 20 top features
# =============================================================================
# SVM is sensitive to feature scale → center and scale based on training data only.
# RBF kernel handles non-linear class boundaries without manual feature engineering.

preproc_a          <- preProcess(train_df %>% select(all_of(top20_eda)),
                                 method = c("center", "scale"))
train_svm_a_scaled <- predict(preproc_a, train_df %>% select(all_of(top20_eda)))
test_svm_a_scaled  <- predict(preproc_a, test_df  %>% select(all_of(top20_eda)))

model_svm_a <- svm(
  x           = train_svm_a_scaled,
  y           = y_train,
  kernel      = "radial",
  probability = TRUE
)

# ── threshold 0.5 ──
prob_svm_a  <- attr(predict(model_svm_a, test_svm_a_scaled, probability = TRUE),
                    "probabilities")[, "high_tc"]
class_svm_a <- factor(if_else(prob_svm_a >= 0.5, "high_tc", "non_high_tc"),
                      levels = levels(y_train))
print(confusionMatrix(class_svm_a, y_test, positive = "high_tc"))

cat("F1 (0.5):", round(MLmetrics::F1_Score(y_true = y_test, y_pred = class_svm_a, positive = "high_tc"), 3), "\n")

roc_svm_a <- roc(y_test, prob_svm_a, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_svm_a, main = paste0("ROC — SVM-A (20 feat)  (AUC = ", round(auc(roc_svm_a), 3), ")"))

# ── threshold Youden ──
measure_svm_a    <- measureit(class = as.numeric(y_test == "high_tc"),
                              score = prob_svm_a, measure = c("SENS", "SPEC"))
youden_svm_a     <- measure_svm_a$SENS + measure_svm_a$SPEC - 1
opt_cutoff_svm_a <- measure_svm_a$Cutoff[which.max(youden_svm_a)]
cat("SVM-A Youden cutoff:", round(opt_cutoff_svm_a, 4),
    "| Sensitivity:", round(measure_svm_a$SENS[which.max(youden_svm_a)], 3),
    "| Specificity:", round(measure_svm_a$SPEC[which.max(youden_svm_a)], 3), "\n")
class_svm_a_y <- factor(if_else(prob_svm_a >= opt_cutoff_svm_a, "high_tc", "non_high_tc"),
                        levels = levels(y_train))
print(confusionMatrix(class_svm_a_y, y_test, positive = "high_tc"))

cat("F1 (Youden):", round(MLmetrics::F1_Score(y_true = y_test, y_pred = class_svm_a_y, positive = "high_tc"), 3), "\n")

# ── SVM-A ROC with Youden point ───────────────────────────────────────────────
plot(roc_svm_a, main = paste0("ROC — SVM-A (20 feat)  (AUC = ", round(auc(roc_svm_a), 3), ")"))
points(
  x   = measure_svm_a$SPEC[which.max(youden_svm_a)],
  y   = measure_svm_a$SENS[which.max(youden_svm_a)],
  pch = 19, col = "red", cex = 1.5
)
legend("bottomright", legend = "Youden threshold", col = "red", pch = 19, bty = "n")

# ── SVM-A summary ─────────────────────────────────────────────────────────────
# Threshold  Accuracy  Sensitivity  Specificity  Balanced Acc  False Neg   F1    AUC
#  0.500      0.867     0.564        0.940        0.752         345        0.623  0.916
#  Youden     0.828     0.938        0.801        0.870          49        0.680  0.916


# =============================================================================
# MODEL 2b: SVM — radial kernel, 9 deduplicated features
# =============================================================================
# lr2_features reused: collinearity hurts SVM margin estimation similarly to LR.

preproc_b          <- preProcess(train_df %>% select(all_of(lr2_features)),
                                 method = c("center", "scale"))
train_svm_b_scaled <- predict(preproc_b, train_df %>% select(all_of(lr2_features)))
test_svm_b_scaled  <- predict(preproc_b, test_df  %>% select(all_of(lr2_features)))

model_svm_b <- svm(
  x           = train_svm_b_scaled,
  y           = y_train,
  kernel      = "radial",
  probability = TRUE
)

# ── threshold 0.5 ──
prob_svm_b  <- attr(predict(model_svm_b, test_svm_b_scaled, probability = TRUE),
                    "probabilities")[, "high_tc"]
class_svm_b <- factor(if_else(prob_svm_b >= 0.5, "high_tc", "non_high_tc"),
                      levels = levels(y_train))
print(confusionMatrix(class_svm_b, y_test, positive = "high_tc"))

cat("F1 (0.5):", round(MLmetrics::F1_Score(y_true = y_test, y_pred = class_svm_b, positive = "high_tc"), 3), "\n")

roc_svm_b <- roc(y_test, prob_svm_b, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_svm_b, main = paste0("ROC — SVM-B (9 feat)  (AUC = ", round(auc(roc_svm_b), 3), ")"))

# ── threshold Youden ──
measure_svm_b    <- measureit(class = as.numeric(y_test == "high_tc"),
                              score = prob_svm_b, measure = c("SENS", "SPEC"))
youden_svm_b     <- measure_svm_b$SENS + measure_svm_b$SPEC - 1
opt_cutoff_svm_b <- measure_svm_b$Cutoff[which.max(youden_svm_b)]
cat("SVM-B Youden cutoff:", round(opt_cutoff_svm_b, 4),
    "| Sensitivity:", round(measure_svm_b$SENS[which.max(youden_svm_b)], 3),
    "| Specificity:", round(measure_svm_b$SPEC[which.max(youden_svm_b)], 3), "\n")
class_svm_b_y <- factor(if_else(prob_svm_b >= opt_cutoff_svm_b, "high_tc", "non_high_tc"),
                        levels = levels(y_train))
print(confusionMatrix(class_svm_b_y, y_test, positive = "high_tc"))

cat("F1 (Youden):", round(MLmetrics::F1_Score(y_true = y_test, y_pred = class_svm_b_y, positive = "high_tc"), 3), "\n")

# ── SVM-B ROC with Youden point ───────────────────────────────────────────────
plot(roc_svm_b, main = paste0("ROC — SVM-B (9 feat)  (AUC = ", round(auc(roc_svm_b), 3), ")"))
points(
  x   = measure_svm_b$SPEC[which.max(youden_svm_b)],
  y   = measure_svm_b$SENS[which.max(youden_svm_b)],
  pch = 19, col = "red", cex = 1.5
)
legend("bottomright", legend = "Youden threshold", col = "red", pch = 19, bty = "n")

# ── SVM-B summary ─────────────────────────────────────────────────────────────
# Threshold  Accuracy  Sensitivity  Specificity  Balanced Acc  False Neg   F1    AUC
#  0.500      0.864     0.589        0.930        0.760         325        0.627  0.926
#  Youden     0.836     0.929        0.813        0.871          56        0.688  0.926
#
# SVM-B Youden recovers 269 additional superconductors vs default 0.5 threshold.
# AUC 0.926 — comparable to LR-B (0.921), weaker than RF (0.980).

# ── Explainability & feature-space benefits ───────────────────────────────────
# Explainability: SVM is the least interpretable model — RBF kernel maps data
# into an implicit high-dimensional space. No feature coefficients exist.
#
# Feature-space A (top20_eda, 20 features):
#   Benefit: more signal available for the margin optimisation.
#   Cost: correlated features can distort the margin — unlike RF, SVM does not
#   subsample features, so redundancy directly affects the decision boundary.
#
# Feature-space B (lr2_features, 9 features):
#   Benefit: collinearity removed — SVM margin estimation is more stable.
#   Cost: marginal signal loss vs 20-feature variant.
#
# Confirmed by results: SVM-B (9 feat) AUC 0.926 > SVM-A (20 feat) AUC 0.916 —
# unlike RF where more features help, collinearity actively hurts SVM by
# duplicating dimensions in the kernel space and distorting the margin.
