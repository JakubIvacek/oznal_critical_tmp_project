library(tidyverse)
library(ggplot2)
library(patchwork)
library(magrittr)
library(pheatmap)

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
