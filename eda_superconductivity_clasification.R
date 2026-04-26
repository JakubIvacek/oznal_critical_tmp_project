library(tidyverse)
library(ggplot2)
library(patchwork)
library(magrittr)
library(pheatmap)
library(e1071)        # svm()
library(randomForest) # randomForest()
library(caret)        # confusionMatrix()
library(pROC)         # roc(), auc()
library(broom)        # tidy()
library(ggrepel)
library(ROCit)

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
  slice_head(n = 20)

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
  slice_head(n = 20)

feature_ranking <- feature_screening %>%
  left_join(class_correlations, by = "feature") %>%
  arrange(desc(abs_smd))

feature_ranking %>%
  select(feature, abs_smd, corr_to_class, mean_diff) %>%
  slice_head(n = 20)

# Plot top features 

top_features <- feature_ranking %>%
  slice_head(n = 20) %>%
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
  slice_head(n = 20) %>%
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

# Not that many outliers save to remove models which are sensitive to them.

# ── OUTLIER REMOVAL for LR and SVM (IQR method for top 20 features) ───
# Random Forests (RF) are robust to outliers we will use full data.
# LR and SVM are sensitive — remove rows with extreme values in any top-20
# which could be possibly used in these models.


keep_rows <- rep(TRUE, nrow(data))
for (feat in feature_shortlist) {
  vals  <- data[[feat]]
  q1    <- quantile(vals, 0.25)
  q3    <- quantile(vals, 0.75)
  iqr_v <- q3 - q1
  keep_rows <- keep_rows &
    vals >= q1 - 1.5 * iqr_v &
    vals <= q3 + 1.5 * iqr_v
}

data_clean <- data[keep_rows, ]
cat("Rows original:", nrow(data),
    "| After outlier removal:", nrow(data_clean),
    "| Removed:", sum(!keep_rows), "\n")





# =============================================================================
# TASK 1 — MODELS: Three Methods × Two Feature-Space Partitioning Families
#
# A — Linear hyperplane:  (1) Logistic Regression  (2) SVM (linear kernel)??
# B — Recursive binary:   (3) Random Forest
#
# =============================================================================

set.seed(42)

# top20_eda defined in EDA outlier-removal block
top20_eda <- feature_shortlist

# ── TRAIN / TEST SPLIT (80 / 20) ───────────────────────────────────

# LR and SVM use data_clean (outliers removed)
data_split <- data_clean %>% mutate(row_id = row_number())

train_df <- data_split %>%
  group_by(tc_class) %>%
  slice_sample(prop = 0.8) %>%
  ungroup()

test_df <- data_split %>% anti_join(train_df, by = "row_id")

y_train <- train_df$tc_class
y_test  <- test_df$tc_class

cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")
cat("Train class balance:\n")
print(count(train_df, tc_class) %>% mutate(pct = scales::percent(n / sum(n), accuracy = 0.01)))
cat("Test class balance:\n")
print(count(test_df,  tc_class) %>% mutate(pct = scales::percent(n / sum(n), accuracy = 0.01)))

# =============================================================================
# MODEL 1a: LOGISTIC REGRESSION — linear hyperplane — all top features
# =============================================================================

fit_lr <- glm(
  tc_class ~ .,
  data   = bind_cols(train_df %>% select(all_of(top20_eda)), tc_class = y_train),
  family = binomial(link = "logit")
)
cat("\nLogistic Regression converged:", fit_lr$converged, "\n")
print(tidy(fit_lr), n = 21)

# term                         estimate std.error statistic   p.value
#  1 (Intercept)                   5.50     1.47         3.74  1.82e-  4
#  2 wtd_std_ThermalConductivity   0.0406   0.00353     11.5   1.13e- 30  *
#  3 range_ThermalConductivity    -0.0361   0.00455     -7.94  2.05e- 15  *
#  4 std_ThermalConductivity       0.0822   0.00949      8.67  4.43e- 18  *
#  5 range_atomic_radius           0.0476   0.00502      9.47  2.75e- 21  *
#  6 wtd_mean_Valence              4.60     1.59         2.89  3.89e-  3  *
#  7 wtd_gmean_Valence           -11.9      2.08        -5.70  1.23e-  8  *
#  8 mean_Valence                  1.01     0.750        1.35  1.76e-  1  *  
#  9 wtd_entropy_atomic_mass       4.85     0.653        7.42  1.18e- 13  *
# 10 range_fie                     0.00908  0.00168      5.40  6.54e-  8  *
# 11 wtd_std_atomic_radius         0.0348   0.0121       2.87  4.14e-  3  *
# 12 wtd_entropy_atomic_radius    -5.47     1.01        -5.43  5.54e-  8  *
# 13 gmean_Valence                -0.302    0.972       -0.311 7.56e-  1    
# 14 wtd_entropy_Valence          -6.37     0.595      -10.7   9.61e- 27  *
# 15 std_atomic_radius            -0.0799   0.0140      -5.73  1.01e-  8  *
# 16 entropy_Valence               2.03     0.664        3.06  2.22e-  3  *
# 17 wtd_std_fie                  -0.0237   0.00273     -8.68  3.81e- 18  *
# 18 wtd_entropy_FusionHeat        7.39     0.565       13.1   3.49e- 39  *
# 19 std_fie                      -0.0231   0.00446     -5.18  2.20e-  7  *
# 20 gmean_Density                -0.00376  0.000236   -15.9   3.10e- 57  *
# 21 range_atomic_mass             0.0361   0.00163     22.2   6.68e-109  *
# Non-significant (p > 0.05): mean_Valence (p=0.176), gmean_Valence (p=0.756)

prob_lr  <- predict(fit_lr, newdata = test_df %>% select(all_of(top20_eda)), type = "response")
class_lr <- factor(if_else(prob_lr >= 0.5, "high_tc", "non_high_tc"), levels = levels(y_train))

print(confusionMatrix(class_lr, y_test, positive = "high_tc"))

roc_lr <- roc(y_test, prob_lr, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_lr, main = paste0("ROC — Logistic Regression  (AUC = ", round(auc(roc_lr), 3), ")"))


# =============================================================================
# MODEL 1b: LOGISTIC REGRESSION — features with too much correlation removed
# =============================================================================
# Collinearity check revealed clusters in the top-20 features:
#   wtd_mean_Valence, wtd_gmean_Valence, mean_Valence, gmean_Valence (~0.99)
#   std_ThermalConductivity, wtd_std_ThermalConductivity, range_ThermalConductivity (~0.96-0.99)
#   wtd_entropy_atomic_mass, wtd_entropy_Valence, wtd_entropy_atomic_radius, entropy_Valence (~0.90-0.96)
#   std_atomic_radius, range_atomic_radius, wtd_std_atomic_radius, range_fie (~0.87-0.97)
#   range_fie, wtd_std_fie, std_fie (~0.87+)
# Strategy: keep one representative per tight cluster
#           keep two from the entropy group
#           keep all not in any cluster

lr2_features <- c(
  "wtd_mean_Valence",            # Valence location — 1 of 4 (r≈0.99)
  "wtd_std_ThermalConductivity", # TC spread        — 1 of 3 (r≈0.96-0.99)
  "range_atomic_radius",         # atomic_radius spread — 1 of 4
  "wtd_entropy_Valence",         # Entropy group    — 2 of 4 
  "wtd_entropy_atomic_mass",
  "wtd_std_fie",                 # fie spread       — 1 of 3 (r≈0.87+)
  "wtd_entropy_FusionHeat",      # not in any cluster
  "gmean_Density",               # not in any cluster
  "range_atomic_mass"            # not in any cluster
)

fit_lr2 <- glm(
  tc_class ~ .,
  data   = bind_cols(train_df %>% select(all_of(lr2_features)), tc_class = y_train),
  family = binomial(link = "logit")
)
cat("\nLR2 (deduplicated) converged:", fit_lr2$converged, "\n")
print(tidy(fit_lr2), n = length(lr2_features) + 1)

prob_lr2  <- predict(fit_lr2, newdata = test_df %>% select(all_of(lr2_features)), type = "response")
class_lr2 <- factor(if_else(prob_lr2 >= 0.5, "high_tc", "non_high_tc"), levels = levels(y_train))

print(confusionMatrix(class_lr2, y_test, positive = "high_tc"))

roc_lr2 <- roc(y_test, prob_lr2, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_lr2, main = paste0("ROC — LR2 deduplicated  (AUC = ", round(auc(roc_lr2), 3), ")"))


# =============================================================================
# MODEL 1c: LR3 — LR2 features + optimal threshold (Youden Index ROCit)
# =============================================================================
# Youden Index finds the cutoff that maximises Sensitivity + Specificity.

# reuse fit_lr2 probabilities on test set
roc_rocit <- rocit(
  class = as.numeric(y_test == "high_tc"),
  score = prob_lr2
)

# find Youden Index optimal cutoff
measure_lr3 <- measureit(
  class = as.numeric(y_test == "high_tc"),
  score = prob_lr2,
  measure = c("ACC", "SENS", "SPEC", "FSCR")
)

youden      <- measure_lr3$SENS + measure_lr3$SPEC - 1
best_idx    <- which.max(youden)
opt_cutoff  <- measure_lr3$Cutoff[best_idx]
cat("Optimal cutoff (Youden):", round(opt_cutoff, 4),
    "| Sensitivity:", round(measure_lr3$SENS[best_idx], 3),
    "| Specificity:", round(measure_lr3$SPEC[best_idx], 3), "\n")

plot(roc_rocit, values = TRUE)

# classify with optimal Youden threshold
class_lr3 <- factor(
  if_else(prob_lr2 >= opt_cutoff, "high_tc", "non_high_tc"),
  levels = levels(y_train)
)
print(confusionMatrix(class_lr3, y_test, positive = "high_tc"))


# ── LR1 vs LR2 vs LR3 ─────────────────
#
# Variant          Threshold  Sensitivity  Specificity  Balanced Acc  False Neg  AUC
# LR1 (20 feat)     0.500      0.636        0.924        0.780         311       0.928
# LR2 (9 feat)      0.500      0.607        0.935        0.771         311       0.921
# LR3 (9 feat)      Youden     0.970        0.764        0.867          24       0.921
#
# LR3 is the best choice for our use case (superconductor discovery / screening):
# - Sensitivity 0.970 — catches 97% of true high_tc materials, missing only 24
# - The 773 false positives (non_high_tc flagged as high_tc) are an acceptable cost:
#   candidates go to experimental validation where false alarms are filtered out if they
#   dont perform, but missing a true superconductor would be a missed opportunity so
#   (false negative) is far more costly than a false alarm in our context.
#
# ---- Balanced Accuracy = (Sensitivity + Specificity) / 2 (very useful for imbalanced datasets):
#      improves from 0.771 → 0.867 because Youden maximises Sens+Spec together






















## Robiim ten hore ja 

# =============================================================================
# MODEL 2: SVM — linear kernel (linear hyperplane family)
# =============================================================================
# SVM finds the maximum-margin hyperplane between classes.
# Linear kernel keeps the decision boundary linear, comparable to LR.
# probability = TRUE enables Platt scaling to produce class probabilities for ROC.

fit_svm <- svm(
  tc_class ~ .,
  data        = bind_cols(train_df %>% select(all_of(top20_eda)), tc_class = y_train),
  kernel      = "linear",
  probability = TRUE
)
cat("SVM support vectors:", nrow(fit_svm$SV), "\n")

pred_svm  <- predict(fit_svm,
                     newdata     = test_df %>% select(all_of(top20_eda)),
                     probability = TRUE)
class_svm <- pred_svm
prob_svm  <- attr(pred_svm, "probabilities")[, "high_tc"]

print(confusionMatrix(class_svm, y_test, positive = "high_tc"))

roc_svm <- roc(y_test, prob_svm, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_svm, main = paste0("ROC — SVM linear  (AUC = ", round(auc(roc_svm), 3), ")"))

# =============================================================================
# MODEL 3: RANDOM FOREST — recursive binary partitioning
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
plot(roc_rf, main = paste0("ROC — Random Forest  (AUC = ", round(auc(roc_rf), 3), ")"))




roc_to_df <- function(pred_prob, truth, label) {
  r <- roc(truth, pred_prob, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
  tibble(FPR    = 1 - r$specificities,
         TPR    = r$sensitivities,
         Method = sprintf("%s  (AUC = %.3f)", label, as.numeric(auc(r))))
}

p_roc <- bind_rows(
  roc_to_df(prob_lr,  y_test, "Logistic Regression"),
  roc_to_df(prob_svm, y_test, "SVM (linear)"),
  roc_to_df(prob_rf,  y_test, "Random Forest")
) %>%
  ggplot(aes(x = FPR, y = TPR, color = Method)) +
  geom_line(linewidth = 1.1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey55") +
  scale_color_manual(values = c("#E41A1C", "#377EB8", "#4DAF4A")) +
  labs(title    = "ROC curves — three classifiers",
       subtitle = "Linear hyperplane (LR, SVM) vs Recursive binary (RF)",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", legend.title = element_blank())

p_roc

# =============================================================================
# RESULTS SUMMARY (threshold = 0.5, test set)
#
# Method               Partitioning       Accuracy  Sensitivity  Specificity  F1     AUC
# Logistic Regression  Linear hyperplane  0.870     0.636        0.924        0.646  0.928
# SVM (linear)         Linear hyperplane  —         —            —            —      —
# Random Forest        Recursive binary   0.948     0.847        0.971        0.858  0.980
# (update SVM row after running)
#
# Key observations:
# - RF is the best overall (highest accuracy, F1, AUC, specificity),
#   with strong sensitivity (0.847) — misses ~15% of true high_tc cases, but
#   produces very few false alarms on non_high_tc (specificity 0.971).
#
# - SVM (linear kernel) finds the maximum-margin hyperplane between classes,
#   complementing LR which finds the maximum-likelihood hyperplane on the same features.
#
# - LR has the highest specificity among Family A (0.924) but low sensitivity
#   (0.636) — misses over a third of true high_tc materials at the 0.5 threshold.
#
# - LR and SVM both use the same 20 features with a linear boundary; differences
#   in their metrics reflect the different training objectives (likelihood vs margin).
#



