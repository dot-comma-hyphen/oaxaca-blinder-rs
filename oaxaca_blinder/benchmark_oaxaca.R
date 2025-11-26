#!/usr/bin/env Rscript

# Load the oaxaca package
library(oaxaca)

# Load the data
cat("Loading benchmark_100k.csv...\n")
data <- read.csv("benchmark_100k.csv")

cat("Data dimensions:", nrow(data), "rows,", ncol(data), "columns\n")
cat("Columns:", paste(names(data), collapse = ", "), "\n\n")

# Convert gender to numeric (0/1) as required by oaxaca package
# Let's define female = 1, male = 0
data$female <- ifelse(data$gender == "F", 1, 0)
cat("Created 'female' indicator variable: 1 = Female, 0 = Male\n")

# Prepare the formula with 10 predictors and the group variable
# Using: education, experience, age, tenure, ability, training, performance, hours, metro, married
# Syntax: outcome ~ predictors | group_variable
formula <- wage ~ education + experience + age + tenure + ability + training + performance + hours + metro + married | female

cat("Running Oaxaca-Blinder decomposition with:\n")
cat("- Outcome: wage\n")
cat("- Group: female (0 = Male [Group A], 1 = Female [Group B])\n")
cat("- Predictors: education, experience, age, tenure, ability, training, performance, hours, metro, married (10 predictors)\n")
cat("- Bootstrap replications: 500\n")
cat("- Reference group: Female (Group B)\n\n")

# Time the decomposition
cat("Starting decomposition...\n")
start_time <- Sys.time()

# Run the Oaxaca decomposition with bootstrap
# group.weights = 0 means using Group B (Female) coefficients as reference
result <- oaxaca(
  formula = formula,
  data = data,
  group.weights = 0,  # Use Group B (Female) as reference
  R = 500, # Number of bootstrap replications
  type = "twofold"
)

end_time <- Sys.time()
elapsed_time <- end_time - start_time

cat("\n")
cat("========================================\n")
cat("TIMING RESULTS\n")
cat("========================================\n")
cat("Total execution time:", elapsed_time, attr(elapsed_time, "units"), "\n")
cat("========================================\n\n")

# Print the results
cat("========================================\n")
cat("DECOMPOSITION RESULTS\n")
cat("========================================\n")
print(summary(result))

# Save results to file
cat("\n\nSaving detailed results to benchmark_oaxaca_results.txt...\n")
sink("benchmark_oaxaca_results.txt")
cat("Oaxaca-Blinder Decomposition Results\n")
cat("=====================================\n\n")
cat("Execution time:", elapsed_time, attr(elapsed_time, "units"), "\n\n")
print(summary(result))
sink()

cat("Done! Results saved to benchmark_oaxaca_results.txt\n")
