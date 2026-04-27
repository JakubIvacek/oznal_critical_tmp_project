library(tidyverse)
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(ROCit)


load("prepared_data.RData")
# Loaded: train_df, test_df, y_train, y_test, numeric_predictors

# =============================================================================
# MODEL 4: DECISION TREE — all numeric features, pruned via cp
# =============================================================================

set.seed(42)
model_dt <- rpart(
  tc_class ~ .,
  data    = bind_cols(train_df %>% select(all_of(numeric_predictors)),
                      tc_class = y_train),
  method  = "class",
  control = rpart.control(cp = 0.001)
)

# prune to the cp with lowest cross-validated error
best_cp <- model_dt$cptable[which.min(model_dt$cptable[, "xerror"]), "CP"]
model_dt_pruned <- prune(model_dt, cp = best_cp)
cat("Best cp:", round(best_cp, 6),
    "| Terminal nodes:", model_dt_pruned$numresp, "\n")

# full pruned tree
rpart.plot(model_dt_pruned, type = 4, extra = 104,
           main = "Decision Tree")
print(model_dt_pruned)

# ── threshold 0.5 ──
prob_dt  <- predict(model_dt_pruned,
                    newdata = test_df %>% select(all_of(numeric_predictors)),
                    type = "prob")[, "high_tc"]
class_dt <- factor(if_else(prob_dt >= 0.5, "high_tc", "non_high_tc"),
                   levels = levels(y_train))
print(confusionMatrix(class_dt, y_test, positive = "high_tc"))

roc_dt <- roc(y_test, prob_dt, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_dt, main = paste0("ROC — Decision Tree (AUC = ", round(auc(roc_dt), 3), ")"))

# ── threshold Youden ──
measure_dt    <- measureit(class = as.numeric(y_test == "high_tc"),
                           score = prob_dt, measure = c("SENS", "SPEC"))
youden_dt     <- measure_dt$SENS + measure_dt$SPEC - 1
best_idx_dt   <- which.max(youden_dt)
opt_cutoff_dt <- measure_dt$Cutoff[best_idx_dt]
cat("DT Youden cutoff:", round(opt_cutoff_dt, 4),
    "| Sensitivity:", round(measure_dt$SENS[best_idx_dt], 3),
    "| Specificity:", round(measure_dt$SPEC[best_idx_dt], 3), "\n")

class_dt_y <- factor(if_else(prob_dt >= opt_cutoff_dt, "high_tc", "non_high_tc"),
                     levels = levels(y_train))
print(confusionMatrix(class_dt_y, y_test, positive = "high_tc"))

# ── ROC with Youden point ─────────────────────────────────────────────────────
plot(roc_dt, main = paste0("ROC — Decision Tree (AUC = ", round(auc(roc_dt), 3), ")"))
points(
  x   = measure_dt$SPEC[best_idx_dt],
  y   = measure_dt$SENS[best_idx_dt],
  pch = 19, col = "red", cex = 1.5
)
legend("bottomright", legend = "Youden threshold", col = "red", pch = 19, bty = "n")

# ── DT summary ────────────────────────────────────────────────────────────────
# Threshold  Accuracy  Sensitivity  Specificity  Balanced Acc  False Neg  AUC
#  0.500      0.924     0.800        0.954        0.877         158       0.955
#  Youden     0.902     0.906        0.901        0.904          74       0.955

# ── Explainability & feature-space benefits ───────────────────────────────────
# Explainability: DT is highly interpretable — tree diagram directly
# shows the decision path for any prediction. Each split has an explicit threshold
# and readable feature. More interpretable than RF (no aggregation over 500 trees)
# and SVM (no kernel), b,ut less statistically formal than LR (no p-values or confidence intervals).
#
# Feature-space (all 81 numeric features):
#   Benefit: DT performs built-in feature selection — only splits on features
#   that reduce Gini impurity. Collinearity is not a problem since the tree
#   picks one feature per split greedily, ignoring redundant ones.
#   Cost: single tree is prone to overfitting — pruning via cp is required.
#   AUC 0.955 is lower than RF (0.980) because RF averages 500 trees,
#   reducing variance that a single tree cannot avoid.