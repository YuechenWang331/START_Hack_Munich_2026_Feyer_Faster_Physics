# Install missing packages
packages <- c("jsonlite", "ggplot2", "tidyr")
missing <- setdiff(packages, rownames(installed.packages()))

if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

library(jsonlite)
library(ggplot2)
library(tidyr)

# Select benchmark-20261121-152209.json
input_file <- file.choose()
report <- fromJSON(input_file)

output_dir <- dirname(input_file)

# First-run phase timings
first <- report$phases
first$label <- factor(
  first$label,
  levels = rev(first$label)
)

p_first <- ggplot(
  first,
  aes(x = label, y = seconds, fill = category)
) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = sprintf("%.3f", seconds)),
    hjust = -0.1,
    size = 3
  ) +
  coord_flip() +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.18))
  ) +
  labs(
    title = "First complete run: phase timings",
    subtitle = sprintf(
      "Total: %.3f s | One observation | Validation excluded",
      report$timing_seconds$end_to_end
    ),
    x = NULL,
    y = "Elapsed time (s)",
    fill = "Category"
  ) +
  theme_minimal(base_size = 12)

print(p_first)

# Repeated calculation timings
phase_names <- c(
  "forward", "sensitivity", "gradient", "host_transfer"
)

runs <- report$repeat_samples
runs$run <- seq_len(nrow(runs))
runs$total <- rowSums(runs[, phase_names])

long <- pivot_longer(
  runs,
  cols = all_of(phase_names),
  names_to = "phase",
  values_to = "seconds"
)

long$phase <- factor(
  long$phase,
  levels = phase_names
)

mean_total <- mean(runs$total)

p_repeats <- ggplot(
  long,
  aes(x = factor(run), y = seconds, fill = phase)
) +
  geom_col(
    width = 0.7,
    position = position_stack(reverse = TRUE)
  ) +
  geom_hline(
    yintercept = mean_total,
    linetype = "dashed",
    color = "firebrick"
  ) +
  geom_text(
    data = runs,
    aes(
      x = factor(run),
      y = total,
      label = sprintf("%.3f", total)
    ),
    inherit.aes = FALSE,
    vjust = -0.5,
    size = 3
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.12))
  ) +
  labs(
    title = "Ten repeated calculations after compilation",
    subtitle = sprintf(
      "Mean: %.3f s | Median: %.3f s | Dashed line: mean",
      mean_total,
      median(runs$total)
    ),
    x = "Repeat",
    y = "Elapsed time (s)",
    fill = "Phase"
  ) +
  theme_minimal(base_size = 12)

print(p_repeats)


#------------------------------------------------------------------
library(ggplot2)

# Read phase data from the selected report
first <- report$phases
runs <- report$repeat_samples

phase_keys <- c(
  "forward", "sensitivity", "gradient", "host_transfer"
)

phase_labels <- c(
  "Forward simulation + AD residuals",
  "Sensitivity, powers and scalar loss",
  "Reverse-mode gradient",
  "Transfer results to host"
)

category_colors <- c(
  compile = "#F8766D",
  execute = "#7CAE00",
  output = "#00BFC4",
  setup = "#C77CFF"
)

n_runs <- nrow(runs)

plot_dir <- file.path(output_dir, "r-phase-details")
dir.create(plot_dir, showWarnings = FALSE)

# Use the same time scale for all individual repeat plots
repeat_limit <- max(
  as.matrix(runs[, phase_keys]),
  first$seconds[first$key %in% phase_keys]
) * 1.22

# Create one horizontal phase chart for each repeat
for (i in seq_len(n_runs)) {
  values <- as.numeric(
    unlist(runs[i, phase_keys], use.names = FALSE)
  )
  
  phase_data <- data.frame(
    label = factor(
      phase_labels,
      levels = rev(phase_labels)
    ),
    seconds = values,
    category = c("execute", "execute", "execute", "output")
  )
  
  p <- ggplot(
    phase_data,
    aes(x = label, y = seconds, fill = category)
  ) +
    geom_col(width = 0.7) +
    geom_text(
      aes(label = sprintf("%.4f", seconds)),
      hjust = -0.12,
      size = 3.5
    ) +
    coord_flip() +
    scale_y_continuous(
      limits = c(0, repeat_limit),
      expand = expansion(mult = c(0, 0))
    ) +
    scale_fill_manual(values = category_colors) +
    labs(
      title = sprintf(
        "Repeat %02d of %02d: phase timings",
        i, n_runs
      ),
      subtitle = sprintf(
        "Total: %.3f s | Setup and compilation not repeated",
        sum(values)
      ),
      x = NULL,
      y = "Elapsed time (s)",
      fill = "Category"
    ) +
    theme_minimal(base_size = 12)
  
  print(p)
  
  ggsave(
    file.path(
      plot_dir,
      sprintf("repeat-%02d-phases.png", i)
    ),
    p,
    width = 11,
    height = 4.8,
    dpi = 300
  )
}

# Compare every first-run phase with the repeat mean
repeat_means <- colMeans(runs[, phase_keys])

comparison <- first[, c("key", "label", "seconds")]
names(comparison)[3] <- "first_seconds"

comparison$repeat_seconds <- NA_real_

matched <- match(comparison$key, names(repeat_means))
is_repeated <- !is.na(matched)

comparison$repeat_seconds[is_repeated] <-
  unname(repeat_means[matched[is_repeated]])

first_data <- data.frame(
  label = comparison$label,
  seconds = comparison$first_seconds,
  status = "Measured",
  series = "First complete run"
)

repeat_data <- data.frame(
  label = comparison$label,
  seconds = comparison$repeat_seconds,
  status = ifelse(is_repeated, "Measured", "Not repeated"),
  series = sprintf("Repeat mean (n = %d)", n_runs)
)

chart_data <- rbind(first_data, repeat_data)

chart_data$label <- factor(
  chart_data$label,
  levels = rev(first$label)
)

chart_data$series <- factor(
  chart_data$series,
  levels = c(
    "First complete run",
    sprintf("Repeat mean (n = %d)", n_runs)
  )
)

# Zero is only a label position for phases that were not repeated
chart_data$plot_seconds <- ifelse(
  is.na(chart_data$seconds),
  0,
  chart_data$seconds
)

chart_data$value_label <- ifelse(
  chart_data$status == "Not repeated",
  "Not repeated",
  sprintf("%.4f s", chart_data$seconds)
)

comparison_limit <- max(chart_data$seconds, na.rm = TRUE) * 1.35

p_compare <- ggplot(
  chart_data,
  aes(x = label, y = plot_seconds)
) +
  geom_col(
    aes(fill = series),
    width = 0.7,
    show.legend = FALSE
  ) +
  geom_text(
    aes(label = value_label),
    hjust = -0.08,
    size = 3
  ) +
  coord_flip() +
  facet_wrap(~series, nrow = 1) +
  scale_y_continuous(
    limits = c(0, comparison_limit),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_fill_manual(values = c("#5470C6", "#00A087")) +
  labs(
    title = "First complete run vs repeated calculation",
    subtitle = "Same time scale | Not repeated is not a measured zero",
    x = NULL,
    y = "Elapsed time (s)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    strip.text = element_text(face = "bold")
  )

print(p_compare)

ggsave(
  file.path(plot_dir, "first-vs-repeat-mean.png"),
  p_compare,
  width = 15,
  height = 8,
  dpi = 300
)

# Decompose the time difference
first_total <- sum(comparison$first_seconds)

not_repeated_time <- sum(
  comparison$first_seconds[!is_repeated]
)

first_calculation_time <- sum(
  comparison$first_seconds[is_repeated]
)

repeat_mean_total <- sum(repeat_means)

breakdown <- data.frame(
  metric = c(
    "First complete run",
    "Setup and compilation not repeated",
    "First-run calculation phases",
    "Repeat mean calculation phases",
    "Calculation-phase difference",
    "Total difference"
  ),
  seconds = c(
    first_total,
    not_repeated_time,
    first_calculation_time,
    repeat_mean_total,
    first_calculation_time - repeat_mean_total,
    first_total - repeat_mean_total
  )
)

print(breakdown, row.names = FALSE, digits = 6)

write.csv(
  comparison,
  file.path(plot_dir, "phase-comparison.csv"),
  row.names = FALSE,
  na = "Not repeated"
)

write.csv(
  breakdown,
  file.path(plot_dir, "time-difference-breakdown.csv"),
  row.names = FALSE
)

cat("\nSaved to:", normalizePath(plot_dir), "\n")









# Summary statistics for repeated calculations
metrics <- c(phase_names, "total")

summary_table <- data.frame(
  phase = metrics,
  mean_seconds = sapply(runs[metrics], mean),
  median_seconds = sapply(runs[metrics], median),
  sd_seconds = sapply(runs[metrics], sd),
  min_seconds = sapply(runs[metrics], min),
  max_seconds = sapply(runs[metrics], max),
  row.names = NULL
)

print(summary_table, digits = 6)

# Save plots and tables beside the selected JSON
ggsave(
  file.path(output_dir, "r-first-run-phases.png"),
  p_first,
  width = 11, height = 7, dpi = 300
)

ggsave(
  file.path(output_dir, "r-repeat-timings.png"),
  p_repeats,
  width = 10, height = 6, dpi = 300
)

write.csv(
  runs,
  file.path(output_dir, "r-repeat-samples.csv"),
  row.names = FALSE
)

write.csv(
  summary_table,
  file.path(output_dir, "r-repeat-summary.csv"),
  row.names = FALSE
)