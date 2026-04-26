library(tidyverse)
library(ggplot2)
library(patchwork)
library(magrittr)
library(pheatmap)
library(MASS)         # lda() / qda()
library(randomForest) # randomForest()
library(pROC)         # roc(), auc()
library(broom)        # tidy()
library(ggrepel)

select <- dplyr::select   # prevent MASS::select from masking dplyr::select

# Load the dataset
data <- read_csv("train.csv")


# EXPLORATORY DATA ANALYSIS (EDA)

# Quick look at the structure of the dataset
glimpse(data)

# Summary statistics
summary(data)

# Check missing values
cat("Sum of missing values:", sum(is.na(data)), "\n")
# Check duplicates
cat("Total duplicate rows:", sum(duplicated(data)), "\n")
print(data[duplicated(data), ])

# Drop duplicates
data <- distinct(data)
print(dim(data))

# Create target variable: high_tc vs non_high_tc based on critical_temp >= 77 K.
data <- data %>%
  mutate(
    tc_class = factor(
      case_when(
        critical_temp >= 77 ~ "high_tc",
        TRUE ~ "non_high_tc"
      ),
      levels = c("non_high_tc", "high_tc")
    ),
    tc_binary = case_when(
      tc_class == "high_tc" ~ 1,
      TRUE ~ 0
    )
  )

data %>%
  count(tc_class) %>%
  mutate(prop = n / sum(n))

data %>%
  group_by(tc_class) %>%
  summarise(
    n = n(),
    mean_tc = mean(critical_temp),
    median_tc = median(critical_temp),
    sd_tc = sd(critical_temp),
    min_tc = min(critical_temp),
    max_tc = max(critical_temp),
    .groups = "drop"
  )

class_balance <- data %>%
  count(tc_class) %>%
  mutate(
    proportion = n / sum(n),
    proportion_pct = 100 * proportion
  )

class_balance

p_target_hist <- ggplot(data, aes(x = critical_temp)) +
  geom_histogram(bins = 50, fill = "grey75", color = "black") +
  geom_vline(xintercept = 77, color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Distribution of critical_temp with 77 K threshold",
    x = "critical_temp",
    y = "Count"
  ) +
  theme_minimal()

p_class_bar <- ggplot(class_balance,aes(x = tc_class, y = n, fill = tc_class)) +
  geom_col() +
  geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)), vjust = -0.3) +
  labs(
    title = "Class balance",
    x = "Class",
    y = "Count"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
  

p_target_hist + p_class_bar

# prepare numeric predictors
numeric_predictors <- data %>%
  select(where(is.numeric), -critical_temp, -tc_binary) %>%
  names()

length(numeric_predictors)

# Feature screening by class separation
# Goal: identify predictors that separate high_tc and non_high_tc.
feature_screening <- data %>%
  select(tc_class, all_of(numeric_predictors)) %>%
  pivot_longer(
    cols = -tc_class,
    names_to = "feature",
    values_to = "value"
  ) %>%
  group_by(feature, tc_class) %>%
  summarise(
    mean_value = mean(value),
    median_value = median(value),
    sd_value = sd(value),
    q25 = quantile(value, 0.25),
    q75 = quantile(value, 0.75),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = tc_class,
    values_from = c(mean_value, median_value, sd_value, q25, q75),
    names_sep = "_"
  ) %>%
  mutate(
    mean_diff = mean_value_high_tc - mean_value_non_high_tc,
    pooled_sd = sqrt((sd_value_high_tc^2 + sd_value_non_high_tc^2) / 2),
    abs_smd = if_else(pooled_sd > 0, abs(mean_diff / pooled_sd), NA_real_)
  ) %>%
  arrange(desc(abs_smd))

feature_screening %>%
  slice_head(n = 15)

# Correlation of each predictor with class label
# Point-biserial style correlation: numeric predictor vs binary class.

class_correlations <- data %>%
  select(tc_binary, all_of(numeric_predictors)) %>%
  pivot_longer(
    cols = -tc_binary,
    names_to = "feature",
    values_to = "value"
  ) %>%
  group_by(feature) %>%
  summarise(
    corr_to_class = cor(value, tc_binary),
    .groups = "drop"
  ) %>%
  arrange(desc(abs(corr_to_class)))

class_correlations %>%
  slice_head(n = 15)

feature_ranking <- feature_screening %>%
  left_join(class_correlations, by = "feature") %>%
  arrange(desc(abs_smd))

feature_ranking %>%
  select(feature, abs_smd, corr_to_class, mean_diff) %>%
  slice_head(n = 15)

# Plot top features 

top_features <- feature_ranking %>%
  slice_head(n = 8) %>%
  pull(feature)

plot_data <- data %>%
  select(tc_class, all_of(top_features)) %>%
  pivot_longer(
    cols = -tc_class,
    names_to = "feature",
    values_to = "value"
  )

p_box <- ggplot(plot_data, aes(x = tc_class, y = value, fill = tc_class)) +
  geom_boxplot(alpha = 0.75, outlier.alpha = 0.10) +
  facet_wrap(~ feature, scales = "free_y", ncol = 2) +
  labs(
    title = "Top predictors by class separation: boxplots",
    x = "Class",
    y = "Value"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

p_density <- ggplot(plot_data, aes(x = value, fill = tc_class)) +
  geom_density(alpha = 0.3) +
  facet_wrap(~ feature, scales = "free", ncol = 2) +
  labs(
    title = "Top predictors by class separation: densities",
    x = "Value",
    y = "Density"
  ) +
  theme_minimal()

p_box 

p_density


# Correlation matrix among shortlisted predictors. 
# Goal: identify redundancy before feature selection.

cor_top <- data %>%
  select(all_of(top_features)) %>%
  cor()

pheatmap(
  cor_top,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  main = "Correlation matrix of top shortlisted predictors"
)

strong_pairs <- cor_top %>%
  as.data.frame() %>%
  rownames_to_column("feature_1") %>%
  pivot_longer(
    cols = -feature_1,
    names_to = "feature_2",
    values_to = "correlation"
  ) %>%
  filter(feature_1 != feature_2) %>%
  mutate(
    pair_id = purrr::map2_chr(feature_1, feature_2, ~ paste(sort(c(.x, .y)), collapse = " | ")),
    abs_correlation = abs(correlation)
  ) %>%
  arrange(desc(abs_correlation)) %>%
  distinct(pair_id, .keep_all = TRUE)



strong_pairs %>%
  filter(abs_correlation >= 0.70) %>%
  select(feature_1, feature_2, correlation, abs_correlation) %>%
  slice_head(n = 20)


# Save shortlist
feature_shortlist <- feature_ranking %>%
  slice_head(n = 15) %>%
  pull(feature)

feature_shortlist

# Outlier counts per feature (IQR method)
outlier_counts <- data %>%
  select(all_of(numeric_predictors)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "feature",
    values_to = "value"
  ) %>%
  group_by(feature) %>%
  summarise(
    q1 = quantile(value, 0.25, names = FALSE),
    q3 = quantile(value, 0.75, names = FALSE),
    iqr_value = q3 - q1,
    outlier_count = sum(value < q1 - 1.5 * iqr_value | value > q3 + 1.5 * iqr_value),
    .groups = "drop"
  ) %>%
  arrange(desc(outlier_count))

outlier_counts


# =============================================================================
# TASK 1 — MODELS: Three Methods × Two Feature-Space Partitioning Families
#
# Family A — Linear hyperplane:  (1) Logistic Regression  (2) QDA
# Family B — Recursive binary:   (3) Random Forest
#
# Family A methods use top-20 features (from feature_ranking above).
# Family B uses all 81 numeric predictors.
# =============================================================================

set.seed(42)

# top-20 features for linear methods (reuse feature_ranking from EDA)
top20 <- feature_ranking %>% slice_head(n = 20) %>% pull(feature)

# ── STRATIFIED TRAIN / TEST SPLIT (80 / 20) ───────────────────────────────────

data_split <- data %>% mutate(row_id = row_number())

train_df <- data_split %>%
  group_by(tc_class) %>%
  slice_sample(prop = 0.8) %>%
  ungroup()

test_df <- data_split %>% anti_join(train_df, by = "row_id")

y_train <- train_df$tc_class
y_test  <- test_df$tc_class

cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")
cat("Train class balance:\n"); print(count(train_df, tc_class))
cat("Test class balance:\n");  print(count(test_df,  tc_class))

# ── MODEL 1: LOGISTIC REGRESSION — linear hyperplane ──────────────────────────

fit_lr <- glm(
  tc_class ~ .,
  data   = bind_cols(train_df %>% select(all_of(top20)), tc_class = y_train),
  family = binomial(link = "logit")
)
cat("\nLogistic Regression converged:", fit_lr$converged, "\n")

prob_lr  <- predict(fit_lr, newdata = test_df %>% select(all_of(top20)), type = "response")
class_lr <- factor(if_else(prob_lr >= 0.5, "high_tc", "non_high_tc"),
                   levels = levels(y_train))

# ── MODEL 2: QDA — quadratic discriminant analysis (linear hyperplane family) ─
# QDA relaxes LDA's equal-covariance assumption: each class gets its own
# covariance matrix, yielding a quadratic decision boundary.

fit_qda  <- qda(x = as.matrix(train_df %>% select(all_of(top20))), grouping = y_train)
pred_qda  <- predict(fit_qda, newdata = as.matrix(test_df %>% select(all_of(top20))))
class_qda <- pred_qda$class
prob_qda  <- pred_qda$posterior[, "high_tc"]

cat("QDA prior probabilities:", round(fit_qda$prior, 3), "\n")

# ── MODEL 3: RANDOM FOREST — recursive binary partitioning ────────────────────

fit_rf <- randomForest(
  tc_class ~ .,
  data       = bind_cols(train_df %>% select(all_of(numeric_predictors)),
                         tc_class = y_train),
  ntree      = 500,
  mtry       = floor(sqrt(length(numeric_predictors))),
  importance = TRUE
)

class_rf <- predict(fit_rf, newdata = test_df %>% select(all_of(numeric_predictors)))
prob_rf  <- predict(fit_rf, newdata = test_df %>% select(all_of(numeric_predictors)),
                    type = "prob")[, "high_tc"]

# OOB sampling is used so OOB error is an unbiased estimate of
# test error without needing a separate validation set.
cat("RF OOB error:", round(fit_rf$err.rate[500, "OOB"], 4), "\n")

# ── PERFORMANCE METRICS ───────────────────────────────────────────────────────

compute_metrics <- function(pred_class, pred_prob, truth, label, partition) {
  cm <- tibble(pred = pred_class, truth = truth) %>%
    summarise(
      tp = sum(pred == "high_tc"     & truth == "high_tc"),
      tn = sum(pred == "non_high_tc" & truth == "non_high_tc"),
      fp = sum(pred == "high_tc"     & truth == "non_high_tc"),
      fn = sum(pred == "non_high_tc" & truth == "high_tc")
    )
  precision   <- cm$tp / (cm$tp + cm$fp)
  sensitivity <- cm$tp / (cm$tp + cm$fn)
  tibble(
    Method       = label,
    Partitioning = partition,
    Accuracy     = round((cm$tp + cm$tn) / (cm$tp + cm$tn + cm$fp + cm$fn), 4),
    Sensitivity  = round(sensitivity, 4),
    Specificity  = round(cm$tn / (cm$tn + cm$fp), 4),
    F1           = round(2 * precision * sensitivity / (precision + sensitivity), 4),
    AUC          = round(as.numeric(auc(
      roc(truth, pred_prob, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
    )), 4)
  )
}

perf_table <- bind_rows(
  compute_metrics(class_lr,  prob_lr,  y_test, "Logistic Regression", "Linear hyperplane"),
  compute_metrics(class_qda, prob_qda, y_test, "QDA",                 "Linear hyperplane"),
  compute_metrics(class_rf,  prob_rf,  y_test, "Random Forest",       "Recursive binary")
)

cat("\n=== PERFORMANCE TABLE ===\n")
print(perf_table, width = 120)

# ── CONFUSION MATRICES ────────────────────────────────────────────────────────

show_cm <- function(pred_class, truth, label) {
  cat(sprintf("\n── %s ──\n", label))
  tibble(Predicted = pred_class, Actual = truth) %>%
    count(Predicted, Actual) %>%
    pivot_wider(names_from = Actual, values_from = n, values_fill = 0L) %>%
    print()
}

show_cm(class_lr,  y_test, "Logistic Regression")
show_cm(class_qda, y_test, "QDA")
show_cm(class_rf,  y_test, "Random Forest")

# ── ROC CURVES ────────────────────────────────────────────────────────────────

roc_to_df <- function(pred_prob, truth, label) {
  r <- roc(truth, pred_prob, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
  tibble(FPR    = 1 - r$specificities,
         TPR    = r$sensitivities,
         Method = sprintf("%s  (AUC = %.3f)", label, as.numeric(auc(r))))
}

p_roc <- bind_rows(
  roc_to_df(prob_lr,  y_test, "Logistic Regression"),
  roc_to_df(prob_qda, y_test, "QDA"),
  roc_to_df(prob_rf,  y_test, "Random Forest")
) %>%
  ggplot(aes(x = FPR, y = TPR, color = Method)) +
  geom_line(linewidth = 1.1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey55") +
  scale_color_manual(values = c("#E41A1C", "#984EA3", "#4DAF4A")) +
  labs(title    = "ROC curves — three classifiers",
       subtitle = "Linear hyperplane (LR, QDA) vs Recursive binary (RF)",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", legend.title = element_blank())

p_roc

# =============================================================================
# RESULTS SUMMARY (threshold = 0.5, test set)
#
# Method               Partitioning       Accuracy  Sensitivity  Specificity  F1     AUC
# Logistic Regression  Linear hyperplane  0.870     0.636        0.924        0.646  0.928
# QDA                  Linear hyperplane  0.776     0.957        0.734        0.614  0.909
# Random Forest        Recursive binary   0.948     0.847        0.971        0.858  0.980
#
# Key observations:
# - RF is the best overall (highest accuracy, F1, AUC, specificity),
#   with strong sensitivity (0.847) — misses ~15% of true high_tc cases, but
#   produces very few false alarms on non_high_tc (specificity 0.971).
#
# - QDA has the highest sensitivity (0.957) — misses only 4.3% of true high_tc cases, but
#   produces many false alarms on non_high_tc as we can see on the low specificity 0.734.
#
# - LR has the highest specificity among Family A (0.924) but low sensitivity
#   (0.636) — misses over a third of true high_tc materials at the 0.5 threshold.
#
# - LR and QDA use the same 20 features yet have opposite sensitivity/specificity
#   profiles because the 0.5 threshold is too high for LR given the class imbalance
#   (~76% non_high_tc), pushing predictions toward the majority class.
#





# SUGGESTED NEXT STEPS:
# 1. Youden index threshold — replace the fixed 0.5 cutoff with the threshold that
#    maximises (sensitivity + specificity - 1) on the ROC curve. This is the lecture-
#    recommended approach and would likely close the sensitivity gap for LR.
#    coords(roc_obj, "best", best.method = "youden") from pROC does this in one line.
#
# 2. Cross-validation — the current single 80/20 split gives one estimate of
#    performance. k-fold CV (e.g. k=5 or k=10) would give a more stable estimate
#    with confidence intervals, especially important for QDA which showed volatile
#    sensitivity/specificity.
#
# 3. RF variable importance plot — importance=TRUE was set during training, so
#    varImpPlot(fit_rf) can show which of the 81 features drive RF's predictions.
#    Useful for comparing against the top-20 selected for LR/QDA.
# =============================================================================


# =============================================================================
# TASK 3 — FEATURE SELECTION: Algorithmic + Embedded Methods
#
# Goal: compare one algorithmic and two embedded selection methods on the same
# classification task, report features retained and significance changes.
#
# ALGORITHMIC — Stepwise selection (forward / backward / mixed):
#   Use stepAIC() from MASS (already loaded) on a logistic regression fit.
#   direction = "forward"  starts with intercept only, adds features one by one
#   direction = "backward" starts with all features, removes the least useful
#   direction = "both"     mixed — recommended, combines both directions
#   Example:
#     fit_step <- stepAIC(
#       glm(tc_class ~ ., data = train_top20, family = binomial),
#       direction = "both", trace = FALSE
#     )
#     summary(fit_step)  # see which features remain and their p-values
#
# EMBEDDED METHOD 1 — Lasso (alpha = 1):
#   Lasso adds an L1 penalty that shrinks some coefficients exactly to zero,
#   effectively performing feature selection. Requires glmnet package.
#     library(glmnet)
#     x_train <- as.matrix(train_df %>% select(all_of(numeric_predictors)))
#     y_train_bin <- as.numeric(y_train == "high_tc")
#     cv_lasso <- cv.glmnet(x_train, y_train_bin, family = "binomial", alpha = 1)
#     coef(cv_lasso, s = "lambda.min")  # non-zero coefficients = retained features
#
# EMBEDDED METHOD 2 — Elastic Net (0 < alpha < 1, e.g. alpha = 0.5):
#   Combines L1 (lasso) and L2 (ridge) penalties. Ridge alone (alpha = 0) shrinks
#   but never zeros out coefficients so it does not select features — elastic net
#   is the better second embedded method to pair with lasso.
#     cv_enet <- cv.glmnet(x_train, y_train_bin, family = "binomial", alpha = 0.5)
#     coef(cv_enet, s = "lambda.min")
#
# REPORTING:
#   - Count non-zero coefficients in lasso / elastic net at lambda.min and lambda.1se
#   - Compare retained feature sets across stepwise, lasso, and elastic net
#   - Note which features appear in all three (most stable) vs only one (fragile)
#   - Check if features that were significant in LR (Task 1) lose significance
#     when other predictors are added / removed during stepwise search
# =============================================================================
