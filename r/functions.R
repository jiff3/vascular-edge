# Shared helpers for cohort models. Every attempted model produces a status row.
suppressPackageStartupMessages({
  library(ggplot2)
  library(broom)
})

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

write_status <- function(model, family, analysis_class, status, reason = NA_character_, n = NA_integer_) {
  data.frame(model = model, family = family, analysis_class = analysis_class,
             status = status, reason = reason, n = n, stringsAsFactors = FALSE)
}

tidy_lm_safe <- function(formula, data, model, family, analysis_class, min_n = 12L) {
  variables <- all.vars(formula)
  complete <- data[complete.cases(data[, variables, drop = FALSE]), , drop = FALSE]
  if (nrow(complete) < max(min_n, length(variables) + 3L)) {
    return(list(table = data.frame(), status = write_status(model, family, analysis_class,
      "not_estimable", "insufficient complete cases", nrow(complete))))
  }
  fit <- try(lm(formula, data = complete), silent = TRUE)
  if (inherits(fit, "try-error") || qr(fit)$rank < length(coef(fit))) {
    return(list(table = data.frame(), status = write_status(model, family, analysis_class,
      "not_estimable", "fit failed or design matrix was rank deficient", nrow(complete))))
  }
  table <- broom::tidy(fit, conf.int = TRUE)
  table$model <- model; table$family <- family; table$analysis_class <- analysis_class
  list(table = table, status = write_status(model, family, analysis_class, "estimated", n = nrow(complete)))
}

tidy_mixed_safe <- function(formula, data, model, family, analysis_class, min_subjects = 10L) {
  variables <- all.vars(formula)
  complete <- data[complete.cases(data[, variables, drop = FALSE]), , drop = FALSE]
  n_subjects <- length(unique(complete$subject))
  if (!requireNamespace("lme4", quietly = TRUE) || !requireNamespace("lmerTest", quietly = TRUE) || !requireNamespace("broom.mixed", quietly = TRUE)) {
    return(list(table = data.frame(), status = write_status(model, family, analysis_class,
      "not_estimable", "lme4, lmerTest, and broom.mixed are required", nrow(complete))))
  }
  if (n_subjects < min_subjects || nrow(complete) < 2L * n_subjects || length(unique(complete$distance_region)) < 2L) {
    return(list(table = data.frame(), status = write_status(model, family, analysis_class,
      "not_estimable", "repeated-measures structure or sample size was inadequate", nrow(complete))))
  }
  fit <- try(lmerTest::lmer(formula, data = complete, REML = FALSE), silent = TRUE)
  if (inherits(fit, "try-error") || lme4::isSingular(fit, tol = 1e-4)) {
    return(list(table = data.frame(), status = write_status(model, family, analysis_class,
      "not_estimable", "model failed or random-effect fit was singular", nrow(complete))))
  }
  table <- broom.mixed::tidy(fit, effects = "fixed", conf.int = TRUE)
  table$model <- model; table$family <- family; table$analysis_class <- analysis_class
  list(table = table, status = write_status(model, family, analysis_class, "estimated", n = nrow(complete)))
}

adjust_families <- function(table) {
  if (nrow(table) == 0L) return(table)
  table$q_bh <- NA_real_
  for (name in unique(table$family)) {
    use <- which(table$family == name & !is.na(table$p.value) & table$term != "(Intercept)")
    if (length(use)) table$q_bh[use] <- p.adjust(table$p.value[use], method = "BH")
  }
  table
}
