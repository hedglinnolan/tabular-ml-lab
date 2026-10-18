# Sullivan 2021 reference (R only): the fits, their weights and the paper's printed numbers.
# Sourced by prep.R and fit.R. The paper's numbers are kept as printed (strings), so the comparison
# rounds ours to the paper's own precision. Source: estimate_candidates.md §1 (Table 4 and the
# supplement's Appendix Table 2), checked at PMC8523030 and NIHMS1722501-supplement-1.pdf.

# Glucose and insulin use the fasting subsample weight; HbA1c and BMI the dietary two-day weight.
outcome_weight <- c(
  glucose = "w_fasting_glu",
  insulin = "w_fasting_ins",
  hba1c = "w_diet2d",
  bmi = "w_diet2d"
)
weight_source <- c(
  w_fasting_glu = "WTSAF2YR / 4",
  w_fasting_ins = "WTSAF2YR / 4",
  w_diet2d = "WTDR2D / 4"
)

all_covariates <- c("age", "sex", "race", "educ", "pir", "INDFMPIR", "smoke", "rec_pa", "bmi",
                    "energy", "glucose_med")

m2 <- c("age", "sex", "race", "educ", "pir")
ladder <- list(
  M0 = character(),
  M1 = c("age", "sex"),
  M2 = m2,
  M3 = c(m2, "smoke", "rec_pa", "energy"),
  M4 = c(m2, "glucose_med")
)
table4_covariates <- c("age", "sex", "race", "educ", "pir", "smoke", "rec_pa", "bmi")

# The paper's numbers: n, beta per rating step, SE and p as printed; NA where not printed.
paper <- list(
  table4_glucose = list(n = 6832, beta = "-2.0", se = "0.4", p = "<0.001"),
  table4_insulin = list(n = 6655, beta = "-0.6", se = "0.2", p = "0.003"),
  table4_hba1c = list(n = 14795, beta = "-0.05", se = "0.01", p = "<0.001"),
  table4_bmi = list(n = 15278, beta = "-1.6", se = "0.1", p = "<0.001"),
  M0_glucose = list(beta = "-2.7", se = "0.37"),
  M1_glucose = list(beta = "-4.1", se = "0.36"),
  M2_glucose = list(beta = "-3.7", se = "0.44"),
  M3_glucose = list(beta = "-3.6", se = "0.45"),
  M4_glucose = list(beta = "-1.1", se = "0.36", p = "0.004"),
  M0_insulin = list(beta = "-2.3", se = "0.22"),
  M1_insulin = list(beta = "-2.5", se = "0.21"),
  M2_insulin = list(beta = "-2.2", se = "0.23"),
  M3_insulin = list(beta = "-2.1", se = "0.23"),
  M4_insulin = list(beta = "-0.4", se = "0.19", p = "0.03"),
  M0_hba1c = list(beta = "-0.08", se = "0.01"),
  M1_hba1c = list(beta = "-0.13"),
  M2_hba1c = list(beta = "-0.10"),
  M3_hba1c = list(beta = "-0.09"),
  M4_hba1c = list(beta = "-0.03", se = "0.01")
)

# Every fit for one outcome: Table 4 first, then the ladder (not for BMI).
fits_for <- function(outcome) {
  t4 <- if (outcome == "bmi") setdiff(table4_covariates, "bmi") else table4_covariates
  out <- list(list(id = paste0("table4_", outcome), table = "Table 4", model = "Table 4",
                   outcome = outcome, covariates = t4))
  if (outcome != "bmi") {
    for (m in names(ladder)) {
      out[[length(out) + 1]] <- list(id = paste0(m, "_", outcome), table = "Appendix Table 2",
                                     model = m, outcome = outcome, covariates = ladder[[m]])
    }
  }
  out
}

fit_formula <- function(f) {
  rhs <- paste(c("rating", f$covariates), collapse = " + ")
  as.formula(paste(f$outcome, "~", rhs))
}
