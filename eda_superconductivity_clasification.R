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
# A — Linear hyperplane:  (1) Logistic Regression
# B — Recursive binary:   (2) Random Forest
#                      (3) ??? este nejaky treba pridat
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

# ── LR-A threshold 0.5 ──
prob_lr  <- predict(fit_lr, newdata = test_df %>% select(all_of(top20_eda)), type = "response")
class_lr <- factor(if_else(prob_lr >= 0.5, "high_tc", "non_high_tc"), levels = levels(y_train))
print(confusionMatrix(class_lr, y_test, positive = "high_tc"))
roc_lr <- roc(y_test, prob_lr, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_lr, main = paste0("ROC — LR-A (20 feat)  (AUC = ", round(auc(roc_lr), 3), ")"))

# ── LR-A threshold Youden ──
measure_lra <- measureit(class = as.numeric(y_test == "high_tc"),
                         score = prob_lr, measure = c("SENS", "SPEC"))
youden_lra    <- measure_lra$SENS + measure_lra$SPEC - 1
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
# recovering 268 additional HT superconductors at the cost of more false alarms.
# 2 features non-significant (mean_Valence, gmean_Valence) — collinearity inflates std.errors.
# ---- Balanced Accuracy improves from 0.780 → 0.873 with Youden threshold.


# =============================================================================
# MODEL 1b: LOGISTIC REGRESSION — 9 deduplicated features (collinearity removed)
# =============================================================================
# Collinearity clusters in top-20 (r > 0.87) — keep one per group:
#   Valence location:     wtd_mean_Valence, wtd_gmean_Valence, mean_Valence, gmean_Valence
#   TC spread:            std/wtd_std/range_ThermalConductivity
#   atomic_radius spread: std/range/wtd_std_atomic_radius
#   Entropy (keep 2):     wtd_entropy_atomic_mass, wtd_entropy_Valence, wtd_entropy_atomic_radius, entropy_Valence
#   fie spread:           range_fie, wtd_std_fie, std_fie

lr2_features <- c(
  "wtd_mean_Valence",            # Valence location — 1 of 4 (r≈0.99)
  "wtd_std_ThermalConductivity", # TC spread        — 1 of 3 (r≈0.96-0.99)
  "range_atomic_radius",         # atomic_radius spread — 1 of 4 (r≈0.87-0.97)
  "wtd_entropy_Valence",         # Entropy group    — 2 of 4 (r≈0.90-0.96)
  "wtd_entropy_atomic_mass",
  "wtd_std_fie",                 # fie spread       — 1 of 3 (r≈0.87+)
  "wtd_entropy_FusionHeat",      # singleton
  "gmean_Density",               # singleton
  "range_atomic_mass"            # singleton
)

fit_lr2 <- glm(
  tc_class ~ .,
  data   = bind_cols(train_df %>% select(all_of(lr2_features)), tc_class = y_train),
  family = binomial(link = "logit")
)
cat("\nLR-B converged:", fit_lr2$converged, "\n")
print(tidy(fit_lr2), n = length(lr2_features) + 1)
# All 9 features significant (p < 0.05) — collinearity resolved

# ── LR-B threshold 0.5 ──
prob_lr2  <- predict(fit_lr2, newdata = test_df %>% select(all_of(lr2_features)), type = "response")
class_lr2 <- factor(if_else(prob_lr2 >= 0.5, "high_tc", "non_high_tc"), levels = levels(y_train))
print(confusionMatrix(class_lr2, y_test, positive = "high_tc"))
roc_lr2 <- roc(y_test, prob_lr2, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_lr2, main = paste0("ROC — LR-B (9 feat)  (AUC = ", round(auc(roc_lr2), 3), ")"))

# ── LR-B threshold Youden ──
measure_lrb <- measureit(class = as.numeric(y_test == "high_tc"),
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
# All 9 coefficients stable and significant — collinearity fully resolved vs LR-A.
# fully interpretable, trustworthy coefficients.
# ---- Balanced Accuracy improves from 0.771 → 0.867 with Youden threshold.





# =============================================================================
# MODEL 2: RANDOM FOREST — recursive binary partitioning
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



# ── RF threshold Youden ──
measure_rf <- measureit(
  class   = as.numeric(y_test == "high_tc"),
  score   = prob_rf,
  measure = c("SENS", "SPEC")
)

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

roc_rf2 <- roc(y_test, prob_rf, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
plot(roc_rf2, main = paste0("ROC — RF Youden threshold  (AUC = ", round(auc(roc_rf2), 3), ")"))

# ── RF vs RF (Youden threshold) ──────────────────────────────────────────────
#
# Variant       Threshold  Sensitivity  Specificity  Balanced Acc  False Neg  AUC
# RF  (0.5)      0.500      0.881        0.969        0.925          94       0.980
# RF (Youden)   Youden     0.955        0.926        0.940          36       0.980
#
# RF (Youden) is the best choice for our use case (superconductor discovery / screening):
# - Sensitivity 0.955 — catches 95.5% of true high_tc materials, missing only 36
# - Youden threshold recovers 58 additional true superconductors
# - Specificity drop (0.969 → 0.926) is acceptable: 140 extra false alarms go to
#   validation where they are filtered out, but the 58 recovered
#   candidates would otherwise be permanently missed
#
# ---- Balanced Accuracy = (Sensitivity + Specificity) / 2:
#      improves from 0.925 → 0.940 because Youden maximises Sens+Spec together



# =============================================================================
# MODEL 3: ??????
# =============================================================================































