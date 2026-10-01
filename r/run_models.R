#!/usr/bin/env Rscript
# Prespecified and exploratory cohort models for Vascular Edge.
# Usage: Rscript r/run_models.R analysis_table.csv perilesional_profiles.csv output_dir [seed]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) stop("Usage: Rscript r/run_models.R analysis_table.csv perilesional_profiles.csv output_dir [seed]")
table_file <- args[[1]]; profiles_file <- args[[2]]; output_dir <- args[[3]]
seed <- if (length(args) >= 4L) as.integer(args[[4]]) else 2026L
set.seed(seed); dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1]]))) else "r"
source(file.path(script_dir, "functions.R"))

analysis <- read.csv(table_file, check.names = FALSE, na.strings = c("", "NA", "n/a", "xx"))
profiles <- read.csv(profiles_file, check.names = FALSE, na.strings = c("", "NA", "n/a", "xx"))
tables <- list(); statuses <- list(); k <- 0L
add_result <- function(result) {
  k <<- k + 1L; tables[[k]] <<- result$table; statuses[[k]] <<- result$status
}

# 1. Prespecified perilesional repeated-measure models, separately by physiology map.
if (nrow(profiles) && all(c("subject", "session", "region", "map", "mean") %in% names(profiles))) {
  covars <- intersect(c("subject", "session", "age", "sex", "baseline_wmh_volume_ml", "wmh_volume_ml"), names(analysis))
  repeated <- merge(profiles, analysis[, covars, drop = FALSE], by = c("subject", "session"), all.x = TRUE)
  repeated <- repeated[repeated$map != "geometry", , drop = FALSE]
  repeated$distance_region <- factor(repeated$region, levels = unique(repeated$region[order(repeated$region_order)]))
  burden <- if ("baseline_wmh_volume_ml" %in% names(repeated)) "baseline_wmh_volume_ml" else if ("wmh_volume_ml" %in% names(repeated)) "wmh_volume_ml" else NULL
  for (map_name in unique(repeated$map)) {
    d <- repeated[repeated$map == map_name, , drop = FALSE]
    fixed <- c("distance_region", intersect(c("age", "sex", burden), names(d)))
    formula <- as.formula(paste("mean ~", paste(fixed, collapse = " + "), "+ (1 | subject)"))
    add_result(tidy_mixed_safe(formula, d, paste0("perilesional_", map_name),
      "primary_perilesional", "prespecified_primary"))
  }
} else statuses[[length(statuses) + 1L]] <- write_status("perilesional", "primary_perilesional",
  "prespecified_primary", "not_estimable", "perilesional profile columns unavailable", 0L)

# 2. Prespecified longitudinal progression model. Prefer core perfusion, then any baseline physiology summary.
outcome <- intersect(c("lesion_progression_ml", "absolute_wmh_change_ml"), names(analysis))[1]
exposure_candidates <- grep("(perfusion|cvr).*(wmh_core|stable_nawm|converting).*(mean|median)$", names(analysis), value = TRUE)
if (!is.na(outcome) && length(exposure_candidates)) {
  exposure <- exposure_candidates[[1]]
  covars <- intersect(c("age", "sex", "education_years", "baseline_wmh_volume_ml", "baseline_wmh_ml"), names(analysis))
  formula <- as.formula(paste(outcome, "~", paste(c(exposure, covars), collapse = " + ")))
  add_result(tidy_lm_safe(formula, analysis, "longitudinal_progression", "primary_progression", "prespecified_primary"))
} else statuses[[length(statuses) + 1L]] <- write_status("longitudinal_progression", "primary_progression",
  "prespecified_primary", "not_estimable", "progression outcome or baseline physiology unavailable", 0L)

# 3. Cognitive associations are exploratory unless a score/exposure pair was prespecified externally.
cognitive <- grep("^cog_", names(analysis), value = TRUE)
vascular <- grep("(perfusion|cvr).*(mean|median)$", names(analysis), value = TRUE)
if (length(cognitive) && length(vascular)) {
  for (score in cognitive) for (exposure in vascular) {
    covars <- intersect(c("age", "sex", "education_years"), names(analysis))
    formula <- as.formula(paste(score, "~", paste(c(exposure, covars), collapse = " + ")))
    add_result(tidy_lm_safe(formula, analysis, paste0("cognition_", score, "__", exposure),
      "exploratory_cognition", "exploratory"))
  }
} else statuses[[length(statuses) + 1L]] <- write_status("cognition", "exploratory_cognition", "exploratory",
  "not_estimable", "usable cognition or vascular exposure unavailable", 0L)

nonempty <- Filter(function(x) is.data.frame(x) && nrow(x) > 0L, tables)
model_table <- if (length(nonempty)) adjust_families(do.call(rbind, nonempty)) else data.frame()
status_table <- do.call(rbind, statuses)
write.csv(model_table, file.path(output_dir, "model_estimates.csv"), row.names = FALSE, na = "")
write.csv(status_table, file.path(output_dir, "model_status.csv"), row.names = FALSE, na = "")
writeLines(capture.output(sessionInfo()), file.path(output_dir, "r_session_info.txt"))
writeLines(paste0("random_seed: ", seed), file.path(output_dir, "r_analysis_config.txt"))

if (nrow(model_table)) {
  plot_data <- model_table[model_table$term != "(Intercept)" & !is.na(model_table$estimate), , drop = FALSE]
  if (nrow(plot_data)) {
    plot_data$label <- paste(plot_data$model, plot_data$term, sep = ": ")
    p <- ggplot(plot_data, aes(x = estimate, y = reorder(label, estimate))) +
      geom_vline(xintercept = 0, colour = "grey60") + geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = .15) +
      geom_point(size = 2) + facet_wrap(~ family, scales = "free_y") +
      labs(x = "Estimate (95% CI)", y = NULL, title = "Cohort model estimates") + theme_minimal(base_size = 11)
    ggsave(file.path(output_dir, "model_estimates.png"), p, width = 10, height = max(5, .25*nrow(plot_data)), dpi = 300)
  }
}
