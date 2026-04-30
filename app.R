library(shiny)
library(tidyverse)
library(caret)
library(pROC)
library(e1071)
library(randomForest)
library(rpart)
library(rpart.plot)
library(broom)
library(MLmetrics)
library(DT)
library(ROCit)
library(glmnet)
library(MASS)

select <- dplyr::select


# ── Load prepared data and fit all models once at startup ─────────────────────
load("prepared_data.RData")
# Provides: train_df, test_df, y_train, y_test,
#           top20_eda, lr2_features, numeric_predictors, feature_ranking

raw_data <- read_csv("train.csv", show_col_types = FALSE) %>% distinct()

# ── Fit models or load from cache ─────────────────────────────────────────────
if (file.exists("fitted_models.RData")) {
  message("Loading cached models ...")
  load("fitted_models.RData")
} else {
  message("First run: fitting all models (takes ~1h, cached afterwards) ...")
  set.seed(42)

  model_lr_a <- glm(tc_class ~ ., family = binomial,
    data = bind_cols(train_df %>% select(all_of(top20_eda)), tc_class = y_train))

  model_lr_b <- glm(tc_class ~ ., family = binomial,
    data = bind_cols(train_df %>% select(all_of(lr2_features)), tc_class = y_train))

  preproc_a   <- preProcess(train_df %>% select(all_of(top20_eda)), method = c("center","scale"))
  model_svm_a <- svm(x = predict(preproc_a, train_df %>% select(all_of(top20_eda))),
                     y = y_train, kernel = "radial", probability = TRUE)

  preproc_b   <- preProcess(train_df %>% select(all_of(lr2_features)), method = c("center","scale"))
  model_svm_b <- svm(x = predict(preproc_b, train_df %>% select(all_of(lr2_features))),
                     y = y_train, kernel = "radial", probability = TRUE)

  model_rf <- randomForest(tc_class ~ ., importance = TRUE, ntree = 500,
    mtry = floor(sqrt(length(numeric_predictors))),
    data = bind_cols(train_df %>% select(all_of(numeric_predictors)), tc_class = y_train))

  dt_raw   <- rpart(tc_class ~ ., method = "class", control = rpart.control(cp = 0.001),
    data = bind_cols(train_df %>% select(all_of(numeric_predictors)), tc_class = y_train))
  model_dt <- prune(dt_raw, cp = dt_raw$cptable[which.min(dt_raw$cptable[, "xerror"]), "CP"])

  # Task 3 — feature selection models
  # Prepare matrix form needed by glmnet
  x_train <- model.matrix(tc_class ~ ., data = bind_cols(
    train_df %>% select(all_of(numeric_predictors)), tc_class = y_train))[, -1]
  x_test  <- model.matrix(tc_class ~ ., data = bind_cols(
    test_df  %>% select(all_of(numeric_predictors)), tc_class = y_test))[, -1]
  y_train_bin <- as.integer(y_train == "high_tc")

  # Backward stepwise LR
  lr_full     <- glm(tc_class ~ ., family = binomial(),
    data = bind_cols(train_df %>% select(all_of(numeric_predictors)), tc_class = y_train))
  model_backward <- MASS::stepAIC(lr_full, direction = "backward", trace = FALSE)

  # Lasso (lambda.min)
  cv_lasso    <- cv.glmnet(x_train, y_train_bin, family = "binomial", alpha = 1,
                           type.measure = "auc")
  model_lasso <- cv_lasso

  # Elastic Net (best alpha from 0.2, 0.5, 0.8)
  enet_results <- tibble(alpha = c(0.2, 0.5, 0.8)) %>%
    mutate(cv_fit  = map(alpha, ~cv.glmnet(x_train, y_train_bin, family = "binomial",
                                           alpha = .x, type.measure = "auc")),
           best_auc = map_dbl(cv_fit, ~max(.x$cvm))) %>%
    arrange(desc(best_auc))
  model_enet       <- enet_results$cv_fit[[1]]
  best_enet_alpha  <- enet_results$alpha[[1]]

  save(model_lr_a, model_lr_b,
       model_svm_a, preproc_a,
       model_svm_b, preproc_b,
       model_rf, model_dt,
       model_backward,
       model_lasso, x_test,
       model_enet, best_enet_alpha,
       file = "fitted_models.RData")
  message("Models cached to fitted_models.RData — next startup will be instant.")
}

# Test-set predictions (always recomputed from loaded/fitted models)
prob_lr_a  <- predict(model_lr_a, test_df %>% select(all_of(top20_eda)), type = "response")
prob_lr_b  <- predict(model_lr_b, test_df %>% select(all_of(lr2_features)), type = "response")
prob_svm_a <- attr(predict(model_svm_a, predict(preproc_a, test_df %>% select(all_of(top20_eda))),
                           probability = TRUE), "probabilities")[, "high_tc"]
prob_svm_b <- attr(predict(model_svm_b, predict(preproc_b, test_df %>% select(all_of(lr2_features))),
                           probability = TRUE), "probabilities")[, "high_tc"]
prob_rf       <- predict(model_rf, test_df %>% select(all_of(numeric_predictors)), type = "prob")[, "high_tc"]
prob_dt       <- predict(model_dt, test_df %>% select(all_of(numeric_predictors)), type = "prob")[, "high_tc"]
prob_backward <- predict(model_backward,
                         newdata = bind_cols(test_df %>% select(all_of(numeric_predictors)), tc_class = y_test),
                         type = "response")
# Task 3 — lambda variants for feature selection models
prob_lasso_min <- as.vector(predict(model_lasso, newx = x_test, s = "lambda.min", type = "response"))
prob_lasso_1se <- as.vector(predict(model_lasso, newx = x_test, s = "lambda.1se", type = "response"))

prob_enet_min <- as.vector(predict(model_enet, newx = x_test, s = "lambda.min", type = "response"))
prob_enet_1se <- as.vector(predict(model_enet, newx = x_test, s = "lambda.1se", type = "response"))

# Keep lambda.min as the default version used in the main model comparison
prob_lasso <- prob_lasso_min
prob_enet  <- prob_enet_min

# ── Shared constants ──────────────────────────────────────────────────────────
MODEL_NAMES <- c("LR-A (20 feat)", "LR-B (9 feat)", "SVM-A (20 feat)",
                 "SVM-B (9 feat)", "Random Forest", "Decision Tree",
                 "Backward LR", "Lasso LR", "Elastic Net LR")
MODEL_COLORS <- setNames(
  c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628",
    "#1abc9c", "#8e44ad", "#e67e22"),
  MODEL_NAMES
)

all_probs <- list(
  "LR-A (20 feat)"  = prob_lr_a,
  "LR-B (9 feat)"   = prob_lr_b,
  "SVM-A (20 feat)" = prob_svm_a,
  "SVM-B (9 feat)"  = prob_svm_b,
  "Random Forest"   = prob_rf,
  "Decision Tree"   = prob_dt,
  "Backward LR"     = prob_backward,
  "Lasso LR"        = prob_lasso,
  "Elastic Net LR"  = prob_enet
)

all_rocs <- lapply(all_probs, function(p)
  roc(y_test, p, levels = c("non_high_tc", "high_tc"), quiet = TRUE))

# RF feature importance table
imp_df <- importance(model_rf) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  arrange(desc(MeanDecreaseGini))

# ── Helper functions ──────────────────────────────────────────────────────────
get_metrics_row <- function(probs, truth, thr, model_name) {
  pred    <- factor(if_else(probs >= thr, "high_tc", "non_high_tc"), levels = c("non_high_tc", "high_tc"))
  truth_f <- factor(as.character(truth), levels = c("non_high_tc", "high_tc"))
  cm      <- suppressWarnings(confusionMatrix(pred, truth_f, positive = "high_tc"))
  tibble(
    Model       = model_name,
    AUC         = round(as.numeric(auc(all_rocs[[model_name]])), 3),
    Accuracy    = round(unname(cm$overall["Accuracy"]), 3),
    Sensitivity = round(unname(cm$byClass["Sensitivity"]), 3),
    Specificity = round(unname(cm$byClass["Specificity"]), 3),
    `Bal.Acc`   = round(unname(cm$byClass["Balanced Accuracy"]), 3),
    F1          = round(unname(cm$byClass["F1"]), 3),
    TP          = cm$table["high_tc", "high_tc"],
    FN          = cm$table["non_high_tc", "high_tc"],
    FP          = cm$table["high_tc", "non_high_tc"]
  )
}




# ── Task 3: Feature selection helpers ─────────────────────────────────────────

get_basic_metrics <- function(probs, truth, threshold) {
  pred <- factor(
    if_else(probs >= threshold, "high_tc", "non_high_tc"),
    levels = c("non_high_tc", "high_tc")
  )
  
  truth_f <- factor(as.character(truth), levels = c("non_high_tc", "high_tc"))
  
  cm <- suppressWarnings(confusionMatrix(pred, truth_f, positive = "high_tc"))
  roc_obj <- roc(truth_f, probs, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
  
  tibble(
    threshold = round(threshold, 3),
    accuracy = round(unname(cm$overall["Accuracy"]), 3),
    sensitivity = round(unname(cm$byClass["Sensitivity"]), 3),
    specificity = round(unname(cm$byClass["Specificity"]), 3),
    precision = round(unname(cm$byClass["Precision"]), 3),
    recall = round(unname(cm$byClass["Recall"]), 3),
    f1 = round(unname(cm$byClass["F1"]), 3),
    auc = round(as.numeric(auc(roc_obj)), 3)
  )
}

get_youden_threshold_basic <- function(probs, truth) {
  truth_f <- factor(as.character(truth), levels = c("non_high_tc", "high_tc"))
  
  roc_obj <- roc(
    response = truth_f,
    predictor = probs,
    levels = c("non_high_tc", "high_tc"),
    quiet = TRUE
  )
  
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

get_glmnet_coef_table <- function(model, lambda_type) {
  coef_mat <- coef(model, s = lambda_type) %>%
    as.matrix()
  
  tibble(
    feature = rownames(coef_mat),
    coefficient = as.numeric(coef_mat[, 1])
  ) %>%
    filter(feature != "(Intercept)", coefficient != 0) %>%
    mutate(abs_coefficient = abs(coefficient)) %>%
    arrange(feature)
}

get_backward_coef_table <- function(model) {
  broom::tidy(model) %>%
    filter(term != "(Intercept)") %>%
    transmute(
      feature = term,
      coefficient = estimate,
      abs_coefficient = abs(estimate)
    ) %>%
    arrange(feature)
}

get_feature_selection_summary <- function() {
  fs_summary_base <- tibble(
    method = c(
      "Backward LR",
      "Lasso LR",
      "Lasso LR",
      paste0("Elastic Net LR alpha=", best_enet_alpha),
      paste0("Elastic Net LR alpha=", best_enet_alpha)
    ),
    selection_setting = c(
      "AIC",
      "lambda.min",
      "lambda.1se",
      "lambda.min",
      "lambda.1se"
    ),
    probs = list(
      prob_backward,
      prob_lasso_min,
      prob_lasso_1se,
      prob_enet_min,
      prob_enet_1se
    ),
    retained_features = c(
      nrow(get_backward_coef_table(model_backward)),
      nrow(get_glmnet_coef_table(model_lasso, "lambda.min")),
      nrow(get_glmnet_coef_table(model_lasso, "lambda.1se")),
      nrow(get_glmnet_coef_table(model_enet, "lambda.min")),
      nrow(get_glmnet_coef_table(model_enet, "lambda.1se"))
    )
  )
  
  fs_summary_base %>%
    mutate(
      youden_threshold = map_dbl(
        probs,
        ~ get_youden_threshold_basic(.x, y_test)
      ),
      metrics_05 = map(
        probs,
        ~ get_basic_metrics(.x, y_test, threshold = 0.5) %>%
          mutate(threshold_type = "0.5", .before = threshold)
      ),
      metrics_youden = map2(
        probs,
        youden_threshold,
        ~ get_basic_metrics(.x, y_test, threshold = .y) %>%
          mutate(threshold_type = "Youden", .before = threshold)
      ),
      metrics = map2(metrics_05, metrics_youden, bind_rows)
    ) %>%
    select(
      method,
      selection_setting,
      retained_features,
      metrics
    ) %>%
    unnest(metrics) %>%
    select(
      method,
      selection_setting,
      retained_features,
      threshold_type,
      threshold,
      accuracy,
      sensitivity,
      specificity,
      precision,
      recall,
      f1,
      auc
    ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 3)))
}

get_feature_selection_matrix <- function() {
  feature_lists <- list(
    "Backward LR" = get_backward_coef_table(model_backward)$feature,
    "Lasso - lambda.min" = get_glmnet_coef_table(model_lasso, "lambda.min")$feature,
    "Lasso - lambda.1se" = get_glmnet_coef_table(model_lasso, "lambda.1se")$feature,
    "Elastic Net - lambda.min" = get_glmnet_coef_table(model_enet, "lambda.min")$feature,
    "Elastic Net - lambda.1se" = get_glmnet_coef_table(model_enet, "lambda.1se")$feature
  )
  
  tibble(feature = sort(numeric_predictors)) %>%
    mutate(
      `Backward LR` = feature %in% feature_lists[["Backward LR"]],
      `Lasso - lambda.min` = feature %in% feature_lists[["Lasso - lambda.min"]],
      `Lasso - lambda.1se` = feature %in% feature_lists[["Lasso - lambda.1se"]],
      `Elastic Net - lambda.min` = feature %in% feature_lists[["Elastic Net - lambda.min"]],
      `Elastic Net - lambda.1se` = feature %in% feature_lists[["Elastic Net - lambda.1se"]]
    ) %>%
    mutate(
      retained_count = rowSums(
        across(
          c(
            `Backward LR`,
            `Lasso - lambda.min`,
            `Lasso - lambda.1se`,
            `Elastic Net - lambda.min`,
            `Elastic Net - lambda.1se`
          ),
          ~ as.integer(.x)
        )
      ),
      .after = feature
    ) %>%
    arrange(desc(retained_count), feature)
}



# ── UI ────────────────────────────────────────────────────────────────────────
ui <- fluidPage(
  titlePanel("Superconductor Classification — Interactive Model Explorer"),
  tags$head(tags$style(HTML(".navbar { margin-bottom: 10px; }"))),

  navbarPage("",
    # ── Tab 1: Data Explorer ──────────────────────────────────────────────────
    tabPanel("Data Explorer",
      sidebarLayout(
        sidebarPanel(width = 3,
          h5("Feature explorer"),
          selectInput("eda_feat", "Feature:",
                      choices = sort(numeric_predictors), selected = "wtd_mean_Valence"),
          radioButtons("eda_type", "Plot type:",
                       c("Density" = "density", "Boxplot" = "box", "Histogram" = "hist")),
          hr(),
          h5("Data table"),
          numericInput("eda_rows", "Rows:", 200, 50, 21263, 50)
        ),
        mainPanel(width = 9,
          fluidRow(
            column(6, plotOutput("eda_class_bar", height = "220px")),
            column(6, plotOutput("eda_tc_hist",   height = "220px"))
          ),
          plotOutput("eda_feat_plot", height = "270px"),
          hr(),
          DTOutput("eda_table")
        )
      )
    ),

    # ── Tab 2: Model Comparison ───────────────────────────────────────────────
    tabPanel("Model Comparison",
      sidebarLayout(
        sidebarPanel(width = 3,
          h5("Select model"),
          selectInput("cmp_model", NULL, choices = MODEL_NAMES, selected = "Random Forest"),
          hr(),
          h5("Classification threshold"),
          sliderInput("cmp_thr", NULL, 0.01, 0.99, 0.5, 0.01),
          actionButton("cmp_reset",  "Reset to 0.5",        class = "btn-sm btn-default"),
          br(), br(),
          actionButton("cmp_youden", "Set Youden threshold", class = "btn-sm btn-info")
        ),
        mainPanel(width = 9,
          h4(textOutput("cmp_model_title")),
          DTOutput("cmp_table"),
          hr(),
          plotOutput("cmp_roc", height = "420px")
        )
      )
    ),

    # ── Tab 3: Decision Tree Explorer ────────────────────────────────────────
    tabPanel("Decision Tree",
      sidebarLayout(
        sidebarPanel(width = 3,
          h5("Tree parameters"),
          sliderInput("dt_cp",       "Complexity (cp):",   0.0001, 0.05, 0.001, 0.0001),
          sliderInput("dt_maxdepth", "Max depth:",         1, 15, 10, 1),
          sliderInput("dt_minsplit", "Min split (nodes):", 2, 100, 20, 1),
          hr(),
          h5("Classification threshold"),
          sliderInput("dt_thr", NULL, 0.01, 0.99, 0.5, 0.01),
          hr()
        ),
        mainPanel(width = 9,
          h4("Decision Tree structure"),
          plotOutput("dt_plot", height = "500px"),
          hr(),
          fluidRow(
            column(6,
              h5("Custom tree metrics"),
              DTOutput("dt_metrics")
            ),
            column(6,
              h5("vs. original pruned DT"),
              DTOutput("dt_metrics_orig")
            )
          )
        )
      )
    ),

    # ── Tab 4: Random Forest Explorer ────────────────────────────────────────
    tabPanel("Random Forest",
      sidebarLayout(
        sidebarPanel(width = 3,
          h5("Classification threshold"),
          sliderInput("rf_thr", NULL, 0.01, 0.99, 0.5, 0.01),
          actionButton("rf_reset",  "Reset to 0.5",        class = "btn-sm btn-default"),
          br(), br(),
          actionButton("rf_youden", "Set Youden threshold", class = "btn-sm btn-info")
        ),
        mainPanel(width = 9,
          fluidRow(
            column(7,
              h4("OOB error vs number of trees"),
              plotOutput("rf_oob", height = "380px")
            ),
            column(5,
              h4("Metrics at threshold"),
              DTOutput("rf_metrics")
            )
          )
        )
      )
    ),

    # ── Tab 5: Summary ───────────────────────────────────────────────────────
    tabPanel("Summary",
      sidebarLayout(
        sidebarPanel(width = 2,
          h5("ROC visibility"),
          checkboxGroupInput("sum_models", NULL,
            choices  = MODEL_NAMES,
            selected = MODEL_NAMES
          ),
          hr(),
          actionButton("sum_all",  "Select all",   width = "100%"),
          br(), br(),
          actionButton("sum_none", "Deselect all", width = "100%")
        ),
        mainPanel(width = 10,
          h4("All models at Youden threshold"),
          p(em("Each model evaluated at its own optimal Youden threshold (maximises sensitivity + specificity).")),
          DTOutput("sum_table"),
          hr(),
          h4("ROC curves with Youden operating points"),
          plotOutput("sum_roc", height = "480px")
        )
      )
    ),
    
    
    # ── Tab 5: Feature Selection ───────────────────────────────────────────────
    tabPanel("Feature Selection",
             fluidRow(
               column(
                 width = 12,
                 h4("Feature selection model performance"),
                 p("The table compares feature selection methods and settings. Each method is evaluated with both the default 0.5 threshold and the Youden threshold."),
                 DTOutput("fs_metrics_table")
               )
             ),
             hr(),
             fluidRow(
               column(
                 width = 12,
                 h4("Retained features across feature selection methods"),
                 p("The table shows whether each feature was retained by each feature selection method. Features retained by more methods are shown first."),
                 DTOutput("fs_feature_matrix")
               )
             )
    ),

    # ── Tab 5: Feature Importance ─────────────────────────────────────────────
    tabPanel("Feature Importance",
      sidebarLayout(
        sidebarPanel(width = 3,
          sliderInput("fi_n", "Top N features (RF):", 5, 40, 20, 1),
          hr(),
          p("RF ranks features by mean decrease in node impurity (Gini)."),
          p("LR-B shows log-odds coefficients — interpretable direction and magnitude.")
        ),
        mainPanel(width = 9,
          h4("Random Forest — Mean Decrease Gini"),
          plotOutput("fi_rf", height = "430px"),
          hr(),
          h4("LR-B — Coefficients (log-odds ± 95% CI)"),
          plotOutput("fi_lr", height = "320px"),
          hr(),
          h4("EDA vs RF Top-20 Feature Ranking"),
          DTOutput("fi_rank_table")
        )
      )
    )
  )
)

# ── Server ────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  # ── Tab 1: Data Explorer ──────────────────────────────────────────────────
  output$eda_class_bar <- renderPlot({
    bind_rows(
      train_df %>% select(tc_class),
      test_df  %>% select(tc_class)
    ) %>%
      count(tc_class) %>%
      mutate(p = n / sum(n)) %>%
      ggplot(aes(tc_class, n, fill = tc_class)) +
      geom_col(show.legend = FALSE) +
      geom_text(aes(label = scales::percent(p, 0.1)), vjust = -0.4, size = 3.5) +
      scale_fill_manual(values = c("high_tc" = "#27ae60", "non_high_tc" = "#e74c3c")) +
      labs(title = "Class balance (train + test)", x = NULL, y = "Count") +
      theme_minimal(base_size = 12)
  })

  output$eda_tc_hist <- renderPlot({
    ggplot(raw_data, aes(critical_temp)) +
      geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.8) +
      geom_vline(xintercept = 77, color = "red", linetype = "dashed", linewidth = 1) +
      annotate("text", x = 82, y = Inf, label = "77 K threshold",
               color = "red", vjust = 2, hjust = 0, size = 3.5) +
      labs(title = "critical_temp distribution", x = "critical_temp (K)", y = "Count") +
      theme_minimal(base_size = 12)
  })

  output$eda_feat_plot <- renderPlot({
    f  <- input$eda_feat
    df <- bind_rows(
      train_df %>% transmute(tc_class, v = .data[[f]], split = "Train"),
      test_df  %>% transmute(tc_class, v = .data[[f]], split = "Test")
    )
    cls_colors <- c("high_tc" = "#27ae60", "non_high_tc" = "#e74c3c")
    if (input$eda_type == "density") {
      ggplot(df, aes(v, fill = tc_class)) +
        geom_density(alpha = 0.45) +
        scale_fill_manual(values = cls_colors) +
        labs(title = f, x = f, fill = "Class") + theme_minimal(base_size = 12)
    } else if (input$eda_type == "box") {
      ggplot(df, aes(tc_class, v, fill = tc_class)) +
        geom_boxplot(outlier.alpha = 0.15, show.legend = FALSE) +
        scale_fill_manual(values = cls_colors) +
        labs(title = f, x = NULL, y = f) + theme_minimal(base_size = 12)
    } else {
      ggplot(df, aes(v, fill = tc_class)) +
        geom_histogram(bins = 40, alpha = 0.55, position = "identity") +
        scale_fill_manual(values = cls_colors) +
        labs(title = f, x = f, fill = "Class") + theme_minimal(base_size = 12)
    }
  })

  output$eda_table <- renderDT({
    bind_rows(
      train_df %>% select(tc_class, all_of(top20_eda)) %>% mutate(split = "Train"),
      test_df  %>% select(tc_class, all_of(top20_eda)) %>% mutate(split = "Test")
    ) %>%
      head(input$eda_rows) %>%
      datatable(options = list(scrollX = TRUE, pageLength = 10, dom = "ltp"), rownames = FALSE)
  })

  # ── Tab 2: Model Comparison ────────────────────────────────────────────────
  observeEvent(input$cmp_reset, updateSliderInput(session, "cmp_thr", value = 0.5))

  observeEvent(input$cmp_youden, {
    coords     <- pROC::coords(all_rocs[[input$cmp_model]], x = "best",
                               best.method = "youden", ret = "threshold", transpose = FALSE)
    updateSliderInput(session, "cmp_thr", value = round(as.numeric(coords[1]), 2))
  })

  output$cmp_model_title <- renderText({
    paste("Metrics —", input$cmp_model, "at threshold", input$cmp_thr)
  })

  output$cmp_table <- renderDT({
    row <- get_metrics_row(all_probs[[input$cmp_model]], y_test, input$cmp_thr, input$cmp_model)
    datatable(row, rownames = FALSE, options = list(dom = "t")) %>%
      formatStyle("AUC", fontWeight = "bold") %>%
      formatStyle("Sensitivity",
        color = styleInterval(c(0.8, 0.9), c("black", "darkorange", "darkgreen")))
  })

  output$cmp_roc <- renderPlot({
    nm  <- input$cmp_model
    thr <- input$cmp_thr
    r   <- all_rocs[[nm]]
    p   <- all_probs[[nm]]
    sens_pt <- mean(p[as.character(y_test) == "high_tc"]    >= thr)
    spec_pt <- mean(p[as.character(y_test) == "non_high_tc"] <  thr)
    plot(r, main = sprintf("ROC — %s  (AUC = %.3f)", nm, auc(r)),
         col = MODEL_COLORS[nm], lwd = 2.5)
    points(spec_pt, sens_pt, pch = 19, col = "black", cex = 1.8)
    legend("bottomright",
           legend = c(sprintf("AUC = %.3f", round(as.numeric(auc(r)), 3)),
                      sprintf("Threshold = %.2f", thr)),
           col = c(MODEL_COLORS[nm], "black"), lwd = c(2, NA), pch = c(NA, 19),
           bty = "n", cex = 0.9)
  })

  # ── Tab 4: Summary ─────────────────────────────────────────────────────────
  # Precompute Youden threshold and metrics for every model (static)
  youden_thrs <- setNames(map_dbl(MODEL_NAMES, function(nm) {
    coords <- pROC::coords(all_rocs[[nm]], x = "best", best.method = "youden",
                           ret = "threshold", transpose = FALSE)
    round(as.numeric(coords[1]), 3)
  }), MODEL_NAMES)

  sum_rows <- map_dfr(MODEL_NAMES, function(nm) {
    get_metrics_row(all_probs[[nm]], y_test, youden_thrs[[nm]], nm) %>%
      mutate(Threshold = youden_thrs[[nm]], .after = Model)
  })

  output$sum_table <- renderDT({
    datatable(sum_rows, rownames = FALSE, options = list(dom = "t", pageLength = 10)) %>%
      formatStyle("AUC", fontWeight = "bold") %>%
      formatStyle("Sensitivity",
        color = styleInterval(c(0.8, 0.9), c("black", "darkorange", "darkgreen"))) %>%
      formatStyle("Model",
        target = "row",
        backgroundColor = styleEqual("Random Forest", "#eaf4fb"))
  })

  observeEvent(input$sum_all,  updateCheckboxGroupInput(session, "sum_models", selected = MODEL_NAMES))
  observeEvent(input$sum_none, updateCheckboxGroupInput(session, "sum_models", selected = character(0)))

  output$sum_roc <- renderPlot({
    visible <- input$sum_models
    plot(NA, xlim = c(1, 0), ylim = c(0, 1),
         xlab = "Specificity", ylab = "Sensitivity",
         main = "ROC Curves — All Models at Youden Threshold", cex.main = 1.2)
    abline(a = 1, b = -1, lty = 2, col = "grey70")
    if (length(visible) == 0) return()
    for (nm in visible) {
      r   <- all_rocs[[nm]]
      thr <- youden_thrs[[nm]]
      p   <- all_probs[[nm]]
      lines(r$specificities, r$sensitivities, col = MODEL_COLORS[nm], lwd = 2.2)
      sens_pt <- mean(p[as.character(y_test) == "high_tc"]    >= thr)
      spec_pt <- mean(p[as.character(y_test) == "non_high_tc"] <  thr)
      points(spec_pt, sens_pt, pch = 19, col = MODEL_COLORS[nm], cex = 1.6)
    }
    aucs <- sapply(visible, function(n) round(as.numeric(auc(all_rocs[[n]])), 3))
    legend("bottomright",
           legend = sprintf("%-20s  AUC=%.3f  thr=%.3f", visible, aucs, youden_thrs[visible]),
           col = MODEL_COLORS[visible], lwd = 2, pch = 19, bty = "n", cex = 0.82)
  })

  # ── Tab 3: Decision Tree Explorer ─────────────────────────────────────────
  dt_custom <- reactive({
    raw <- rpart(tc_class ~ ., method = "class",
      control = rpart.control(
        cp       = input$dt_cp,
        maxdepth = input$dt_maxdepth,
        minsplit = input$dt_minsplit
      ),
      data = bind_cols(train_df %>% select(all_of(numeric_predictors)), tc_class = y_train))
    prune(raw, cp = input$dt_cp)
  })

  output$dt_plot <- renderPlot({
    rpart.plot(dt_custom(), type = 4, extra = 104,
               main = sprintf("DT  cp=%.4f  maxdepth=%d  minsplit=%d",
                              input$dt_cp, input$dt_maxdepth, input$dt_minsplit),
               cex = 0.7)
  })

  dt_metrics_tbl <- function(tree, thr) {
    prob <- predict(tree, test_df %>% select(all_of(numeric_predictors)), type = "prob")[, "high_tc"]
    pred <- factor(if_else(prob >= thr, "high_tc", "non_high_tc"), levels = c("non_high_tc", "high_tc"))
    truth_f <- factor(as.character(y_test), levels = c("non_high_tc", "high_tc"))
    cm  <- suppressWarnings(confusionMatrix(pred, truth_f, positive = "high_tc"))
    roc_obj <- roc(y_test, prob, levels = c("non_high_tc", "high_tc"), quiet = TRUE)
    tibble(
      Metric = c("AUC", "Accuracy", "Sensitivity", "Specificity", "Bal. Acc", "F1", "FN", "FP"),
      Value  = c(
        round(as.numeric(auc(roc_obj)), 3),
        round(unname(cm$overall["Accuracy"]), 3),
        round(unname(cm$byClass["Sensitivity"]), 3),
        round(unname(cm$byClass["Specificity"]), 3),
        round(unname(cm$byClass["Balanced Accuracy"]), 3),
        round(unname(cm$byClass["F1"]), 3),
        cm$table["non_high_tc", "high_tc"],
        cm$table["high_tc", "non_high_tc"]
      )
    )
  }

  output$dt_metrics <- renderDT({
    datatable(dt_metrics_tbl(dt_custom(), input$dt_thr),
              rownames = FALSE, options = list(dom = "t", pageLength = 10))
  })

  output$dt_metrics_orig <- renderDT({
    datatable(dt_metrics_tbl(model_dt, input$dt_thr),
              rownames = FALSE, options = list(dom = "t", pageLength = 10))
  })

  # ── Tab 4: Random Forest Explorer ─────────────────────────────────────────
  observeEvent(input$rf_reset, updateSliderInput(session, "rf_thr", value = 0.5))
  observeEvent(input$rf_youden, {
    coords <- pROC::coords(all_rocs[["Random Forest"]], x = "best",
                           best.method = "youden", ret = "threshold", transpose = FALSE)
    updateSliderInput(session, "rf_thr", value = round(as.numeric(coords[1]), 2))
  })

  output$rf_oob <- renderPlot({
    data.frame(
      trees       = seq_len(nrow(model_rf$err.rate)),
      OOB         = model_rf$err.rate[, "OOB"],
      high_tc     = model_rf$err.rate[, "high_tc"],
      non_high_tc = model_rf$err.rate[, "non_high_tc"]
    ) %>%
      pivot_longer(-trees, names_to = "type", values_to = "error") %>%
      ggplot(aes(trees, error, color = type)) +
      geom_line(linewidth = 0.8) +
      scale_color_manual(
        values = c("OOB" = "black", "high_tc" = "#27ae60", "non_high_tc" = "#e74c3c"),
        name = NULL
      ) +
      labs(title = "OOB error stabilises as more trees are added",
           x = "Number of trees", y = "Error rate") +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom")
  })

  output$rf_metrics <- renderDT({
    row <- get_metrics_row(prob_rf, y_test, input$rf_thr, "Random Forest")
    row %>%
      select(-Model) %>%
      pivot_longer(everything(), names_to = "Metric", values_to = "Value") %>%
      datatable(rownames = FALSE, options = list(dom = "t", pageLength = 15))
  })
  
  # ── Tab 5: Feature Selection ───────────────────────────────────────────────
  
  output$fs_metrics_table <- renderDT({
    fs_summary <- get_feature_selection_summary()
    
    datatable(
      fs_summary,
      rownames = FALSE,
      options = list(
        pageLength = 10,
        scrollX = TRUE,
        dom = "tip"
      )
    ) %>%
      formatStyle("auc", fontWeight = "bold") %>%
      formatStyle(
        "retained_features",
        backgroundColor = styleInterval(
          c(10, 30),
          c("#eafaf1", "#fff9e6", "#fdecea")
        )
      )
  })
  
  output$fs_feature_matrix <- renderDT({
    fs_matrix <- get_feature_selection_matrix()
    
    method_cols <- c(
      "Backward LR",
      "Lasso - lambda.min",
      "Lasso - lambda.1se",
      "Elastic Net - lambda.min",
      "Elastic Net - lambda.1se"
    )
    
    datatable(
      fs_matrix,
      rownames = FALSE,
      filter = "top",
      options = list(
        pageLength = 25,
        scrollX = TRUE,
        dom = "tip"
      )
    ) %>%
      formatStyle(
        columns = method_cols,
        backgroundColor = styleEqual(
          c(TRUE, FALSE),
          c("#eafaf1", "#fdecea")
        )
      ) %>%
      formatStyle(
        "retained_count",
        fontWeight = "bold",
        backgroundColor = styleInterval(
          c(0, 2, 4),
          c("#fdecea", "#fff9e6", "#eafaf1", "#d5f5e3")
        )
      )
  })

  # ── Tab 6: Feature Importance ──────────────────────────────────────────────
  output$fi_rf <- renderPlot({
    imp_df %>%
      slice_head(n = input$fi_n) %>%
      mutate(feature = fct_reorder(feature, MeanDecreaseGini),
             in_eda  = feature %in% top20_eda) %>%
      ggplot(aes(MeanDecreaseGini, feature, fill = in_eda)) +
      geom_col(alpha = 0.85) +
      scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "grey60"),
                        labels = c("TRUE" = "Also in EDA top-20", "FALSE" = "RF only"),
                        name = NULL) +
      labs(title = paste("Top", input$fi_n, "RF features by Mean Decrease Gini"),
           x = "Mean Decrease Gini", y = NULL) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom")
  })

  output$fi_lr <- renderPlot({
    tidy(model_lr_b, conf.int = TRUE) %>%
      filter(term != "(Intercept)") %>%
      mutate(term      = fct_reorder(term, estimate),
             direction = estimate > 0) %>%
      ggplot(aes(estimate, term, xmin = conf.low, xmax = conf.high, color = direction)) +
      geom_pointrange(linewidth = 0.8) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
      scale_color_manual(values = c("TRUE" = "#1e8449", "FALSE" = "#922b21"),
                         labels = c("TRUE" = "Positive effect", "FALSE" = "Negative effect"),
                         name = NULL) +
      labs(x = "Coefficient (log-odds)", y = NULL,
           title = "LR-B — all 9 coefficients significant (p < 0.05)") +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom")
  })

  output$fi_rank_table <- renderDT({
    eda_top <- feature_ranking %>%
      select(feature, abs_smd, corr_to_class) %>%
      mutate(eda_rank = row_number()) %>%
      filter(eda_rank <= 20)

    rf_top <- imp_df %>%
      select(feature, MeanDecreaseGini) %>%
      mutate(rf_rank = row_number()) %>%
      filter(rf_rank <= 20)

    full_join(eda_top, rf_top, by = "feature") %>%
      arrange(coalesce(eda_rank, 99L)) %>%
      mutate(across(where(is.numeric), ~round(., 3)),
             shared = !is.na(eda_rank) & !is.na(rf_rank)) %>%
      datatable(rownames = FALSE, options = list(pageLength = 25, dom = "t")) %>%
      formatStyle("shared",
        target = "row",
        backgroundColor = styleEqual(TRUE, "#eafaf1"))
  })
}

shinyApp(ui, server)
