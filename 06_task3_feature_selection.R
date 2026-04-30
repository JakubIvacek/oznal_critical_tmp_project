library(tidyverse)
library(glmnet)
library(MASS)
library(caret)
library(pROC)
library(magrittr)

load("prepared_data.RData")

candidate_features <- numeric_predictors

train_task3 <- train_df %>%
  dplyr::select(tc_class, all_of(candidate_features))

test_task3 <- test_df %>%
  dplyr::select(tc_class, all_of(candidate_features))

y_train <- if_else(train_task3$tc_class == "high_tc", 1, 0)
y_test  <- if_else(test_task3$tc_class == "high_tc", 1, 0)

x_train <- model.matrix(tc_class ~ ., data = train_task3)[, -1]
x_test  <- model.matrix(tc_class ~ ., data = test_task3)[, -1]


# Helper function: evaluate classification performance
evaluate_classifier <- function(truth, prob, threshold = 0.5) {
  # Convert predicted probabilities to class labels using a selected threshold.
  pred <- if_else(prob >= threshold, "high_tc", "non_high_tc") %>%
    factor(levels = c("non_high_tc", "high_tc"))
  
  # Ensure that the true labels have the correct factor levels.
  truth <- factor(truth, levels = c("non_high_tc", "high_tc"))
  
  # Compute confusion matrix metrics.
  cm <- confusionMatrix(pred, truth, positive = "high_tc")
  
  # Compute ROC curve and AUC.
  roc_obj <- roc(response = truth, predictor = prob, levels = c("non_high_tc", "high_tc"),   quiet = TRUE)
  
  # Return all relevant metrics in one tibble.
  tibble(
    threshold = threshold,
    accuracy = unname(cm$overall["Accuracy"]),
    sensitivity = unname(cm$byClass["Sensitivity"]),
    specificity = unname(cm$byClass["Specificity"]),
    precision = unname(cm$byClass["Precision"]),
    recall = unname(cm$byClass["Recall"]),
    f1 = unname(cm$byClass["F1"]),
    auc = as.numeric(auc(roc_obj))
  )
}


# Helper function: compute Youden threshold

get_youden_threshold <- function(truth, prob) {
  # Ensure that the true labels have the correct factor levels.
  truth <- factor(truth, levels = c("non_high_tc", "high_tc"))
  
  # Build ROC curve.
  roc_obj <- pROC::roc(
    response = truth,
    predictor = prob,
    levels = c("non_high_tc", "high_tc"),
    quiet = TRUE
  )
  
  # Select the threshold that maximizes Youden's index:
  # sensitivity + specificity - 1.
  pROC::coords(
    roc_obj,
    x = "best",
    best.method = "youden",
    ret = "threshold",
    transpose = FALSE
  ) %>%
    as.numeric() %>%
    first()
}



# 1. Algorithmic feature selection: Backward stepwise logistic regression

# Build the full logistic regression formula using all candidate features.
full_formula <- as.formula(
  paste("tc_class ~", paste(candidate_features, collapse = " + "))
)

# Fit full logistic regression model.
lr_full <- glm(
  full_formula,
  data = train_task3,
  family = binomial()
)


# Apply backward stepwise selection using AIC as the selection criterion.
lr_backward <- MASS::stepAIC(
  lr_full,
  direction = "backward",
  trace = FALSE
)

# Show summary of the final backward-selected model.
summary(lr_backward)



# Extract features retained by backward selection.
backward_features <- names(coef(lr_backward)) %>%
  setdiff("(Intercept)")

backward_features
length(backward_features)

# Predict probabilities on the test set.
prob_backward <- predict(
  lr_backward,
  newdata = test_task3,
  type = "response"
)


# Evaluation with default threshold 0.5
backward_metrics_05 <- evaluate_classifier(
  truth = test_task3$tc_class,
  prob = prob_backward,
  threshold = 0.5
) %>%
  mutate(threshold_type = "0.5")

# Find Youden threshold from ROC curve
backward_youden_threshold <- get_youden_threshold(
  truth = test_task3$tc_class,
  prob = prob_backward
)

backward_youden_threshold

# Evaluation with Youden threshold
backward_metrics_youden <- evaluate_classifier(
  truth = test_task3$tc_class,
  prob = prob_backward,
  threshold = backward_youden_threshold
) %>%
  mutate(threshold_type = "Youden")

# Combined result table
backward_metrics <- bind_rows(
  backward_metrics_05,
  backward_metrics_youden
) %>%
  dplyr::select(
    threshold_type,
    threshold,
    accuracy,
    sensitivity,
    specificity,
    precision,
    recall,
    f1,
    auc
  )

backward_metrics


# 2. Embedded feature selection: Lasso logistic regression

# Fit cross-validated lasso logistic regression.
# alpha = 1 means pure lasso.
# type.measure = "auc" selects lambda based on cross-validated AUC.
cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1,
  type.measure = "auc"
)

# Store the fitted glmnet model.
lasso_model <- cv_lasso$glmnet.fit

#lambda.min 
# lambda.min gives the best cross-validated AUC.
lasso_coef_min <- coef(cv_lasso, s = "lambda.min")


# Extract selected features with non-zero coefficients.
lasso_features_min <- rownames(lasso_coef_min)[as.vector(lasso_coef_min != 0)] %>%
  setdiff("(Intercept)")

lasso_features_min
length(lasso_features_min)


# Predict probabilities using lambda.min.
prob_lasso_min <- predict(
  cv_lasso,
  newx = x_test,
  s = "lambda.min",
  type = "response"
) %>%
  as.vector()


# Evaluate lasso lambda.min with default threshold 0.5.
lasso_metrics_min_05 <- evaluate_classifier(
  test_task3$tc_class,
  prob_lasso_min,
  threshold = 0.5
)


# Compute Youden threshold for lasso lambda.min.
lasso_youden_threshold_min <- get_youden_threshold(
  truth = test_task3$tc_class,
  prob = prob_lasso_min
)


# Evaluate lasso lambda.min with Youden threshold.
lasso_metrics_min_youden <- evaluate_classifier(
  test_task3$tc_class,
  prob_lasso_min,
  threshold = lasso_youden_threshold_min
)

# Combine lasso lambda.min results.
lasso_metrics_min <- bind_rows(
  lasso_metrics_min_05 %>% mutate(lambda_type = "lambda.min", threshold_type = "0.5"),
  lasso_metrics_min_youden %>% mutate(lambda_type = "lambda.min", threshold_type = "Youden")
) %>%
  dplyr::select(lambda_type, threshold_type, everything())

lasso_metrics_min


# Lasso using lambda.1se
# lambda.1se gives a simpler model whose CV performance is within one standard
# error of the best model.
lasso_lambda_1se <- cv_lasso$lambda.1se

lasso_coef_1se <- coef(cv_lasso, s = "lambda.1se")


# Extract selected features with non-zero coefficients.
lasso_features_1se <- rownames(lasso_coef_1se)[as.vector(lasso_coef_1se != 0)] %>%
  setdiff("(Intercept)")

lasso_features_1se
length(lasso_features_1se)

# Predict probabilities using lambda.1se.
prob_lasso_1se <- predict(
  cv_lasso,
  newx = x_test,
  s = "lambda.1se",
  type = "response"
) %>%
  as.vector()


# Evaluate lasso lambda.1se with default threshold 0.5.
lasso_metrics_1se_05 <- evaluate_classifier(
  test_task3$tc_class,
  prob_lasso_1se,
  threshold = 0.5
)


# Compute Youden threshold for lasso lambda.1se.
lasso_youden_threshold_1se <- get_youden_threshold(
  truth = test_task3$tc_class,
  prob = prob_lasso_1se
)


# Evaluate lasso lambda.1se with Youden threshold.
lasso_metrics_1se_youden <- evaluate_classifier(
  test_task3$tc_class,
  prob_lasso_1se,
  threshold = lasso_youden_threshold_1se
)


# Combine lasso lambda.1se results.
lasso_metrics_1se <- bind_rows(
  lasso_metrics_1se_05 %>% mutate(lambda_type = "lambda.1se", threshold_type = "0.5"),
  lasso_metrics_1se_youden %>% mutate(lambda_type = "lambda.1se", threshold_type = "Youden")
) %>%
  dplyr::select(lambda_type, threshold_type, everything())

lasso_metrics_1se

# Combined lasso summary
lasso_summary <- bind_rows(
  lasso_metrics_min %>%
    mutate(
      lambda_type = "lambda.min",
      retained_features = length(lasso_features_min)
    ),
  
  lasso_metrics_1se %>%
    mutate(
      lambda_type = "lambda.1se",
      retained_features = length(lasso_features_1se)
    )
  
) %>%
  dplyr::select(
    lambda_type,
    threshold_type,
    retained_features,
    threshold,
    accuracy,
    sensitivity,
    specificity,
    precision,
    recall,
    f1,
    auc
  )

lasso_summary



# 3. Embedded feature selection: Elastic net logistic regression
# Test several alpha values.
# alpha = 0 is ridge, alpha = 1 is lasso, values between 0 and 1 are elastic net.
alpha_grid <- c(0.2, 0.5, 0.8)


# Fit cross-validated elastic net models for each alpha.
enet_cv_results <- tibble(alpha = alpha_grid) %>%
  mutate(
    cv_fit = purrr::map(
      alpha,
      ~ cv.glmnet(
        x = x_train,
        y = y_train,
        family = "binomial",
        alpha = .x,
        type.measure = "auc"
      )
    ),
    best_auc = purrr::map_dbl(cv_fit, ~ max(.x$cvm))
  ) %>%
  arrange(desc(best_auc))

enet_cv_results


# Select alpha with the highest cross-validated AUC.
best_enet <- enet_cv_results$cv_fit[[1]]
best_alpha <- enet_cv_results$alpha[[1]]

best_alpha
best_enet$lambda.min


# Elastic net using lambda.min
enet_coef_min <- coef(best_enet, s = "lambda.min")


# Extract selected features with non-zero coefficients.
enet_features_min <- rownames(enet_coef_min)[as.vector(enet_coef_min != 0)] %>%
  setdiff("(Intercept)")

enet_features_min
length(enet_features_min)


# Predict probabilities using lambda.min.
prob_enet_min <- predict(
  best_enet,
  newx = x_test,
  s = "lambda.min",
  type = "response"
) %>%
  as.vector()


# Evaluate elastic net lambda.min with threshold 0.5.
enet_metrics_min_05 <- evaluate_classifier(
  truth = test_task3$tc_class,
  prob = prob_enet_min,
  threshold = 0.5
) %>%
  mutate(
    lambda_type = "lambda.min",
    threshold_type = "0.5",
    retained_features = length(enet_features_min)
  )


# Compute Youden threshold for elastic net lambda.min.
enet_youden_threshold_min <- get_youden_threshold(
  truth = test_task3$tc_class,
  prob = prob_enet_min
)


# Evaluate elastic net lambda.min with Youden threshold.
enet_metrics_min_youden <- evaluate_classifier(
  truth = test_task3$tc_class,
  prob = prob_enet_min,
  threshold = enet_youden_threshold_min
) %>%
  mutate(
    lambda_type = "lambda.min",
    threshold_type = "Youden",
    retained_features = length(enet_features_min)
  )


# Elastic net using lambda.1se
best_enet$lambda.1se

enet_coef_1se <- coef(best_enet, s = "lambda.1se")


# Extract selected features with non-zero coefficients.
enet_features_1se <- rownames(enet_coef_1se)[as.vector(enet_coef_1se != 0)] %>%
  setdiff("(Intercept)")

enet_features_1se
length(enet_features_1se)

# Predict probabilities using lambda.1se.
prob_enet_1se <- predict(
  best_enet,
  newx = x_test,
  s = "lambda.1se",
  type = "response"
) %>%
  as.vector()


# Evaluate elastic net lambda.1se with threshold 0.5.
enet_metrics_1se_05 <- evaluate_classifier(
  truth = test_task3$tc_class,
  prob = prob_enet_1se,
  threshold = 0.5
) %>%
  mutate(
    lambda_type = "lambda.1se",
    threshold_type = "0.5",
    retained_features = length(enet_features_1se)
  )

# Compute Youden threshold for elastic net lambda.1se.
enet_youden_threshold_1se <- get_youden_threshold(
  truth = test_task3$tc_class,
  prob = prob_enet_1se
)

# Evaluate elastic net lambda.1se with Youden threshold.
enet_metrics_1se_youden <- evaluate_classifier(
  truth = test_task3$tc_class,
  prob = prob_enet_1se,
  threshold = enet_youden_threshold_1se
) %>%
  mutate(
    lambda_type = "lambda.1se",
    threshold_type = "Youden",
    retained_features = length(enet_features_1se)
  )


# Combined elastic net summary
enet_summary <- bind_rows(
  enet_metrics_min_05,
  enet_metrics_min_youden,
  enet_metrics_1se_05,
  enet_metrics_1se_youden
) %>%
  mutate(
    method = paste0("Elastic Net LR (alpha=", best_alpha, ")")
  ) %>%
  dplyr::select(
    method,
    lambda_type,
    threshold_type,
    retained_features,
    threshold,
    accuracy,
    sensitivity,
    specificity,
    precision,
    recall,
    f1,
    auc
  )

enet_summary



# Final Task 3 summary
task3_summary <- bind_rows(
  backward_metrics %>%
    mutate(
      method = "Backward LR",
      lambda_type = "AIC",
      retained_features = length(backward_features)
    ) %>%
    dplyr::select(
      method,
      lambda_type,
      threshold_type,
      retained_features,
      threshold,
      accuracy,
      sensitivity,
      specificity,
      precision,
      recall,
      f1,
      auc
    ),
  
  lasso_summary %>%
    mutate(
      method = "Lasso LR"
    ) %>%
    dplyr::select(
      method,
      lambda_type,
      threshold_type,
      retained_features,
      threshold,
      accuracy,
      sensitivity,
      specificity,
      precision,
      recall,
      f1,
      auc
    ),
  
  enet_summary
)

task3_summary


# =============================================================================
# Feature retention comparison
# =============================================================================

feature_retention <- tibble(feature = candidate_features) %>%
  mutate(
    backward = feature %in% backward_features,
    lasso_min = feature %in% lasso_features_min,
    lasso_1se = feature %in% lasso_features_1se,
    elastic_net_min = feature %in% enet_features_min,
    elastic_net_1se = feature %in% enet_features_1se,
    selected_count = backward + lasso_min + lasso_1se + elastic_net_min + elastic_net_1se
  ) %>%
  arrange(desc(selected_count), feature)

feature_retention  
  



# Features selected by at least 4 out of 5 feature-selection variants
stable_features <- feature_retention %>%
  filter(selected_count == 5) %>%
  arrange(desc(selected_count), feature)

stable_features
  


# Number of retained features per method
feature_selection_counts <- feature_retention %>%
  summarise(
    backward = sum(backward),
    lasso_min = sum(lasso_min),
    lasso_1se = sum(lasso_1se),
    elastic_net_min = sum(elastic_net_min),
    elastic_net_1se = sum(elastic_net_1se)
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "method",
    values_to = "retained_features"
  ) %>%
  arrange(retained_features)

feature_selection_counts

