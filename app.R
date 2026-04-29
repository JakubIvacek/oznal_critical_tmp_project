library(shiny)
library(tidyverse)
library(caret)
library(pROC)
library(e1071)
library(randomForest)
library(rpart)
library(broom)
library(MLmetrics)
library(DT)
library(ROCit)

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
  message("First run: fitting all models (takes ~4-5 min, cached afterwards) ...")
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

  save(model_lr_a, model_lr_b,
       model_svm_a, preproc_a,
       model_svm_b, preproc_b,
       model_rf, model_dt,
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
prob_rf    <- predict(model_rf, test_df %>% select(all_of(numeric_predictors)), type = "prob")[, "high_tc"]
prob_dt    <- predict(model_dt, test_df %>% select(all_of(numeric_predictors)), type = "prob")[, "high_tc"]

# ── Shared constants ──────────────────────────────────────────────────────────
MODEL_NAMES <- c("LR-A (20 feat)", "LR-B (9 feat)", "SVM-A (20 feat)",
                 "SVM-B (9 feat)", "Random Forest", "Decision Tree")
MODEL_COLORS <- setNames(
  c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"),
  MODEL_NAMES
)

all_probs <- list(
  "LR-A (20 feat)"  = prob_lr_a,
  "LR-B (9 feat)"   = prob_lr_b,
  "SVM-A (20 feat)" = prob_svm_a,
  "SVM-B (9 feat)"  = prob_svm_b,
  "Random Forest"   = prob_rf,
  "Decision Tree"   = prob_dt
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
    FN          = cm$table["non_high_tc", "high_tc"],
    FP          = cm$table["high_tc", "non_high_tc"]
  )
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
          h5("Classification threshold"),
          sliderInput("cmp_thr", NULL, 0.01, 0.99, 0.5, 0.01),
          actionButton("cmp_reset", "Reset to 0.5", class = "btn-sm btn-default"),
          actionButton("cmp_youden", "Set Youden threshold", class = "btn-sm btn-info"),
          selectInput("cmp_youden_model", "Youden for model:",
                      choices = MODEL_NAMES, selected = "Random Forest"),
          hr(),
          h5("ROC curves to show"),
          checkboxGroupInput("cmp_show", NULL,
                             choices = MODEL_NAMES, selected = MODEL_NAMES)
        ),
        mainPanel(width = 9,
          h4("Metrics at selected threshold"),
          DTOutput("cmp_table"),
          hr(),
          plotOutput("cmp_roc", height = "420px")
        )
      )
    ),

    # ── Tab 3: Feature Importance ─────────────────────────────────────────────
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
    roc_obj <- all_rocs[[input$cmp_youden_model]]
    coords  <- pROC::coords(roc_obj, x = "best", best.method = "youden",
                            ret = "threshold", transpose = FALSE)
    youden_thr <- round(as.numeric(coords[1]), 2)
    updateSliderInput(session, "cmp_thr", value = youden_thr)
  })

  output$cmp_table <- renderDT({
    thr  <- input$cmp_thr
    rows <- map_dfr(MODEL_NAMES, ~get_metrics_row(all_probs[[.x]], y_test, thr, .x))
    datatable(rows, rownames = FALSE, options = list(dom = "t", pageLength = 10)) %>%
      formatStyle("AUC", fontWeight = "bold") %>%
      formatStyle("Sensitivity",
        color = styleInterval(c(0.8, 0.9), c("black", "darkorange", "darkgreen"))) %>%
      formatStyle("Model",
        target = "row",
        backgroundColor = styleEqual("Random Forest", "#eaf4fb"))
  })

  output$cmp_roc <- renderPlot({
    sel <- input$cmp_show
    if (!length(sel)) return(NULL)
    plot(NA, xlim = c(1, 0), ylim = c(0, 1),
         xlab = "Specificity", ylab = "Sensitivity",
         main = "ROC Curves — All Models", cex.main = 1.2)
    abline(a = 1, b = -1, lty = 2, col = "grey70")
    for (nm in sel) {
      r <- all_rocs[[nm]]
      lines(r$specificities, r$sensitivities, col = MODEL_COLORS[nm], lwd = 2.2)
    }
    # Mark operating point at current threshold
    thr <- input$cmp_thr
    for (nm in sel) {
      p    <- all_probs[[nm]]
      sens <- mean(p[as.character(y_test) == "high_tc"]    >= thr)
      spec <- mean(p[as.character(y_test) == "non_high_tc"] <  thr)
      points(spec, sens, pch = 19, col = MODEL_COLORS[nm], cex = 1.4)
    }
    aucs <- sapply(sel, function(n) round(as.numeric(auc(all_rocs[[n]])), 3))
    legend("bottomright",
           legend = sprintf("%-20s  AUC = %.3f", sel, aucs),
           col = MODEL_COLORS[sel], lwd = 2, bty = "n", cex = 0.82,
           text.font = 1)
  })

  # ── Tab 3: Feature Importance ──────────────────────────────────────────────
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
