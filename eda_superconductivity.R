library(tidyverse)
library(ggplot2)
library(patchwork)
library(corrplot)
library(gridExtra)

# LOAD DATA
# Set working directory to the data folder
setwd("C:/Users/ivace/Downloads/superconductivty+data")
# Load the dataset
data <- read_csv("train.csv")


# EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------

# Quick look at the structure of the dataset
glimpse(data)

# Summary statistics
summary(data)
# critical temp ma male min tak aj tu sa dokazuje ze ten log1p je dobry 
# a max je 185 K outlier ... niekolko features maju min 0 ale to asi nebude problem ??

# Check missing values
cat("Sum of missing values:", sum(is.na(data)), "\n")
# Check duplicates
cat("Total duplicate rows:", sum(duplicated(data)), "\n")
print(data[duplicated(data), ])

# Drop duplicates
data <- distinct(data)
print(dim(data))

# Outlier counts per feature (IQR method)
outlier_counts <- sapply(data, function(col) {
  Q1 <- quantile(col, 0.25)
  Q3 <- quantile(col, 0.75)
  IQR_val <- Q3 - Q1
  sum(col < Q1 - 1.5 * IQR_val | col > Q3 + 1.5 * IQR_val)
})

print(sort(outlier_counts, decreasing = TRUE))

# Check target variable distribution

mean_ct <- mean(data$critical_temp)
median_ct <- median(data$critical_temp)

# ------------- HISTROGRAMS of target variable
# Histogram
p1 <- ggplot(data, aes(x = critical_temp)) +
  geom_histogram(bins = 60, fill = "blue", color = "black") +
  geom_vline(aes(xintercept = mean_ct), color = "red", linetype = "dashed") +
  geom_vline(aes(xintercept = median_ct), color = "orange", linetype = "dashed") +
  annotate("text", x = mean_ct, y = Inf, label = paste("Mean:", round(mean_ct, 1)),
           color = "red", vjust = 2, hjust = -0.15) +
  annotate("text", x = median_ct, y = Inf, label = paste("Median:", round(median_ct, 1)),
           color = "orange", vjust = 5, hjust = -0.05) +
  labs(title = "Critical_temp distribution", x = "Critical Temperature (K)", y = "Count") +
  theme_minimal()
p1

# Log-transformed histogram
p2 <- ggplot(data, aes(x = log(critical_temp))) +
  geom_histogram(bins = 60, fill = "darkgreen", color = "black", alpha = 0.85) +
  geom_vline(aes(xintercept = log(mean_ct)), color = "red", linetype = "dashed") +
  geom_vline(aes(xintercept = log(median_ct)), color = "orange", linetype = "dashed") +
  labs(title = "Distribution of log(critical_temp)", x = "log(Critical Temperature)", y = "Count") +
  theme_minimal()
p2
# Log-transformed + 1 histogram
p3 <- ggplot(data, aes(x = log1p(critical_temp))) +
  geom_histogram(bins = 60, fill = "darkred", color = "black", alpha = 0.85) +
  geom_vline(aes(xintercept = log1p(mean_ct)), color = "red", linetype = "dashed") +
  geom_vline(aes(xintercept = log1p(median_ct)), color = "orange", linetype = "dashed") +
  labs(title = "Distribution of log(critical_temp + 1)", x = "log(Critical Temperature + 1)", y = "Count") +
  theme_minimal()
p3

# Combined distributions of target variable
p1 | p2 | p3
# Asi obsahuje velmi male hodnoty pri ctritical_temp tak log + 1 dava lepsu distribuciu
sum(data$critical_temp < 1)

# ----------- BOXPLOTS
# Boxplot
bp1 <- ggplot(data, aes(y = critical_temp)) +
  geom_boxplot(fill = "blue", alpha = 0.6) +
  labs(title = "Boxplot of critical_temp", y = "Critical Temperature (K)") +
  theme_minimal()
# Boxplot log
bp2 <- ggplot(data, aes(y = log1p(critical_temp))) +
  geom_boxplot(fill = "blue", alpha = 0.6) +
  labs(title = "Boxplot of log(critical_temp + 1)", y = "log(Critical Temperature + 1)") +
  theme_minimal()

bp1  | bp2


# ----------- OTHER FEATURES
# Correlation matrix
cor_matrix <- cor(data)
# ----------- Correlation with target 
cor_target <- cor_matrix[, "critical_temp"]
sort(cor_target, decreasing = TRUE)

# Top 10 positive
top10_pos <- names(sort(cor_target, decreasing = TRUE)[2:11])
cat("Top 10 positive:\n")
print(sort(cor_target, decreasing = TRUE)[2:11])

# Top 10 negative
top10_neg <- names(sort(cor_target)[1:10])
cat("\nTop 10 negative:\n")
print(sort(cor_target)[1:10])

#  ----------- Correlation with each other top features
# Combine top 10 pos + neg + target
top20 <- c(top10_pos, top10_neg, "critical_temp")
# Correlation matrix medzi nimi
cor_top20 <- cor(data[, top20])
# Plot
corrplot(cor_top20, method = "color", tl.cex = 0.6, tl.col = "black",
         addCoef.col = "black", number.cex = 0.5,
         title = "Top 10 Pos + Neg Features Correlation Matrix", mar = c(0,0,1,0))

# Features dost na seba coreluju v oboch blokoch pre pozitivnu aj negativnu korelaciu
# Bude treba vybrat len s kazdej skupiny nejakych zastupcov co nemaju velku korelaciu


# -------------- Distribution of corelated features

# Histograms top 20 features
plots <- lapply(top20[top20 != "critical_temp"], function(col) {
  ggplot(data, aes(x = .data[[col]])) +
    geom_histogram(bins = 40, fill = "blue", color = "black", alpha = 0.8) +
    labs(title = col, x = "", y = "Count") +
    theme_minimal() +
    theme(plot.title = element_text(size = 8))
})

grid.arrange(grobs = plots, ncol = 4)

# Boxplots top 20 features
box_plots <- lapply(top20[top20 != "critical_temp"], function(col) {
  ggplot(data, aes(y = .data[[col]])) +
    geom_boxplot(fill = "blue", alpha = 0.6) +
    labs(title = col, y = "") +
    theme_minimal() +
    theme(plot.title = element_text(size = 8))
})

grid.arrange(grobs = box_plots, ncol = 4)

# Scatter plots top 20 features vs critical_temp
scatter_plots <- lapply(top20[top20 != "critical_temp"], function(col) {
  ggplot(data, aes(x = .data[[col]], y = critical_temp)) +
    geom_point(alpha = 0.2, size = 0.5, color = "blue") +
    geom_smooth(method = "lm", color = "red", se = FALSE, linewidth = 0.7) +
    labs(title = col, x = "", y = "critical_temp") +
    theme_minimal() +
    theme(plot.title = element_text(size = 8))
})

grid.arrange(grobs = scatter_plots, ncol = 4)

# Outlier counts pre top 20 features
outlier_top20 <- outlier_counts[names(outlier_counts) %in% top20]
print(sort(outlier_top20, decreasing = TRUE))

# ESTE CO TREBA DOPLNIT
#1. Skewness/Kurtosis ?? 
# ESte nieco ????
