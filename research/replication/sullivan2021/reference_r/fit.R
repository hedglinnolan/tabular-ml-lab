# Sullivan 2021 reference (R only): fit Table 4 and the Appendix Table 2 ladder, write results.json.
#
# Run from the repository root, after prep.R:
#   Rscript research/replication/sullivan2021/reference_r/fit.R
#
# Each fit is survey-weighted linear regression (survey::svyglm) on the design of all 39,156
# records of the four cycles: svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = <outcome's
# weight>, nest = TRUE). The analysis rows are a domain, taken with subset() of that design, so
# the SE is the domain (linearization) SE and not that of a subset data frame.

suppressPackageStartupMessages({
  library(survey)
  library(digest)
  library(jsonlite)
})

here <- file.path("research", "replication", "sullivan2021", "reference_r")
derived <- file.path("research", "data", "derived", "reference_r")
source(file.path(here, "models.R"))

d <- readRDS(file.path(derived, "stacked.rds"))
prep <- fromJSON(file.path(derived, "sample_steps.json"), simplifyVector = FALSE)

decimals <- function(s) if (grepl("\\.", s)) nchar(sub("^[^.]*\\.", "", s)) else 0
matches <- function(ours, printed) {
  if (is.null(printed)) return(NULL)
  isTRUE(round(ours, decimals(printed)) == as.numeric(printed))
}

designs <- list()
design_for <- function(w) {
  if (is.null(designs[[w]])) {
    designs[[w]] <<- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = as.formula(paste0("~", w)),
                               nest = TRUE, data = d)
  }
  designs[[w]]
}

# One survey-weighted fit of `f` over the domain `keep` (the analytic sample unless a check narrows
# it), as a subset() of the full design.
fit_domain <- function(f, keep = d$in_analytic) {
  oc <- f$outcome
  w <- outcome_weight[[oc]]
  full <- design_for(w)
  vars <- c(oc, "rating", f$covariates)
  domain <- keep & !is.na(d[[oc]]) & d[[w]] > 0 & complete.cases(d[, vars])
  form <- fit_formula(f)
  m <- svyglm(form, design = subset(full, domain), family = gaussian())
  co <- summary(m)$coefficients
  stopifnot(sum(domain) == length(m$residuals))
  beta <- unname(co["rating", "Estimate"])
  se <- unname(co["rating", "Std. Error"])
  list(form = form, w = w, full = full, domain = domain, model = m, n = sum(domain), beta = beta,
       se = se, t = beta / se, df = degf(full), p = 2 * pt(-abs(beta / se), degf(full)),
       svyglm_df_residual = m$df.residual, svyglm_p = unname(co["rating", "Pr(>|t|)"]))
}

fits <- list()
for (oc in names(outcome_weight)) {
  table_rows <- read.csv(file.path(derived, paste0("analytic_", oc, ".csv")))
  for (f in fits_for(oc)) {
    r <- fit_domain(f)
    stopifnot(r$n == sum(table_rows[[paste0("in_", f$id)]]))
    pp <- paper[[f$id]]
    fits[[length(fits) + 1]] <- list(
      id = f$id,
      table = f$table,
      model = f$model,
      outcome = oc,
      formula = paste(deparse(r$form, width.cutoff = 500), collapse = ""),
      weight = weight_source[[r$w]],
      weight_column = r$w,
      n = r$n,
      beta = r$beta,
      se = r$se,
      t = r$t,
      df = r$df,
      p = r$p,
      svyglm_df_residual = r$svyglm_df_residual,
      svyglm_p = r$svyglm_p,
      beta_on_raw_dbq700 = -r$beta,
      paper = pp,
      beta_matches_paper = matches(r$beta, pp$beta),
      se_matches_paper = matches(r$se, pp$se),
      n_minus_paper_n = if (is.null(pp$n)) NULL else r$n - pp$n
    )
    cat(sprintf("%-15s n %6d  beta %9.5f  se %8.5f  p %.2g   paper %s (%s) n %s\n", f$id, r$n, r$beta,
                r$se, r$p, pp$beta, if (is.null(pp$se)) "-" else pp$se, if (is.null(pp$n)) "-" else pp$n))
  }
}

# ---- The domain variance, derived by hand ---------------------------------------------------------
# Linearization for a weighted least-squares fit: V = A^-1 B A^-1, with A = X'WX and B the
# with-replacement variance of the PSU totals of u_i = w_i x_i e_i, which are zero outside the
# domain: B = sum_h n_h / (n_h - 1) sum_j (z_hj - zbar_h)(z_hj - zbar_h)', n_h counting the full
# design's PSUs in stratum h. The same SE from a design built on the domain's rows alone (a subset
# data frame) is reported beside it; it differs only where a stratum's PSU has no domain rows.
hand_domain_se <- function(r) {
  rows <- which(r$domain)
  X <- model.matrix(r$form, d[rows, ])
  wt <- d[[r$w]][rows]
  y <- d[[all.vars(r$form)[1]]][rows]
  A <- crossprod(X, X * wt)
  b <- solve(A, crossprod(X, wt * y))
  u <- X * as.vector(wt * (y - X %*% b))
  psu_all <- unique(d[, c("SDMVSTRA", "SDMVPSU")])
  key_all <- paste(psu_all$SDMVSTRA, psu_all$SDMVPSU)
  z <- matrix(0, nrow(psu_all), ncol(X), dimnames = list(key_all, colnames(X)))
  zs <- rowsum(u, paste(d$SDMVSTRA[rows], d$SDMVPSU[rows]))
  z[rownames(zs), ] <- zs
  B <- matrix(0, ncol(X), ncol(X))
  for (h in unique(psu_all$SDMVSTRA)) {
    zh <- z[psu_all$SDMVSTRA == h, , drop = FALSE]
    nh <- nrow(zh)
    dev <- sweep(zh, 2, colMeans(zh))
    B <- B + nh / (nh - 1) * crossprod(dev)
  }
  Ai <- solve(A)
  V <- Ai %*% B %*% Ai
  list(beta = unname(b["rating", 1]), se = sqrt(V["rating", "rating"]))
}
frame_se <- function(r) {
  des <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = as.formula(paste0("~", r$w)),
                   nest = TRUE, data = d[r$domain, ])
  tryCatch(unname(summary(svyglm(r$form, design = des))$coefficients["rating", "Std. Error"]),
           error = function(e) conditionMessage(e))
}
variance_checks <- list()
for (oc in names(outcome_weight)) {
  f <- fits_for(oc)[[1]]
  r <- fit_domain(f)
  h <- hand_domain_se(r)
  fs <- frame_se(r)
  psu_keys <- unique(paste(d$SDMVSTRA, d$SDMVPSU))
  empty <- setdiff(psu_keys, unique(paste(d$SDMVSTRA[r$domain], d$SDMVPSU[r$domain])))
  variance_checks[[length(variance_checks) + 1]] <- list(
    id = f$id, svyglm_se = r$se, hand_se = h$se, hand_beta = h$beta,
    rel_diff_se = abs(h$se - r$se) / r$se, rel_diff_beta = abs(h$beta - r$beta) / abs(r$beta),
    subset_frame_se = fs, psus_without_domain_rows = length(empty)
  )
  stopifnot(abs(h$se - r$se) / r$se < 1e-8, abs(h$beta - r$beta) / abs(r$beta) < 1e-8)
}

# ---- Premise checks (not results) ------------------------------------------------------------------
# Run after the primary fits were compared with the paper, to locate two differences; each is
# reported, none replaces a primary fit. Every variant tried is listed, in the order tried.
premise_checks <- list()

# 1. The water step. Ours removes 18 (a reliable day with zero energy and at least one item); the
#    paper removes 20. Also removing the 2 people with a reliable day on which nothing was reported
#    (zero energy, zero items) gives the paper's 20; the Table 4 fits under that reading:
nothing_day <- (d$DR1DRSTZ %in% 1 & d$DR1TNUMF %in% 0) | (d$DR2DRSTZ %in% 1 & d$DR2TNUMF %in% 0)
for (oc in names(outcome_weight)) {
  f <- fits_for(oc)[[1]]
  r <- fit_domain(f, keep = d$in_analytic & !nothing_day)
  premise_checks[[length(premise_checks) + 1]] <- list(
    check = "water step read as zero energy on a reliable recall day (removes 20)",
    id = f$id, n = r$n, beta = r$beta, se = r$se, p = r$p, paper = paper[[f$id]],
    beta_matches_paper = matches(r$beta, paper[[f$id]]$beta), se_matches_paper = matches(r$se, paper[[f$id]]$se),
    n_minus_paper_n = r$n - paper[[f$id]]$n
  )
}

# 2. M4. The stated M4 (M2 + glucose-lowering medication) misses the paper's M4 for all three
#    outcomes while M0-M3 match. Variants tried, in order:
m3 <- ladder$M3
m4_variants <- list(
  `M2 + medication + BMI` = c(m2, "glucose_med", "bmi"),
  `M2 + BMI` = c(m2, "bmi"),
  `M3 + medication` = c(m3, "glucose_med"),
  `M3 + medication + BMI` = c(m3, "glucose_med", "bmi"),
  `Table 4 model + medication` = c(table4_covariates, "glucose_med")
)
for (v in names(m4_variants)) {
  for (oc in c("glucose", "insulin", "hba1c")) {
    f <- list(id = paste0("M4_", oc), outcome = oc, covariates = m4_variants[[v]])
    r <- fit_domain(f)
    pp <- paper[[f$id]]
    premise_checks[[length(premise_checks) + 1]] <- list(
      check = paste("M4 read as", v), id = f$id,
      formula = paste(deparse(r$form, width.cutoff = 500), collapse = ""), n = r$n, beta = r$beta,
      se = r$se, p = r$p, paper = pp, beta_matches_paper = matches(r$beta, pp$beta),
      se_matches_paper = matches(r$se, pp$se)
    )
  }
}
for (pc in premise_checks) cat(sprintf("%-70s %-15s n %6d beta %9.5f se %8.5f  %s/%s\n", pc$check, pc$id, pc$n,
                                       pc$beta, pc$se, pc$beta_matches_paper, pc$se_matches_paper))

tables <- lapply(names(outcome_weight), function(oc) {
  path <- file.path(derived, paste0("analytic_", oc, ".csv"))
  x <- read.csv(path)
  t4 <- fits_for(oc)[[1]]
  # Complete cases: how many of the table's rows each Table 4 variable is missing on, and how many
  # rows the Table 4 fit drops in all.
  missing <- lapply(setNames(t4$covariates, t4$covariates), function(v) sum(is.na(x[[v]]) | x[[v]] == ""))
  list(outcome = oc, path = path, rows = nrow(x), sha256 = digest(file = path, algo = "sha256"),
       weight_column = outcome_weight[[oc]],
       rows_are = paste("analytic sample rows with the outcome and a positive weight; in_<fit> marks",
                        "the rows each fit uses (complete cases on its own variables)"),
       table4_rows_used = sum(x[[paste0("in_", t4$id)]]),
       table4_rows_dropped_for_missing_covariates = nrow(x) - sum(x[[paste0("in_", t4$id)]]),
       table4_missing_by_covariate = missing,
       nothing_reported_day_rows = sum(x$nothing_reported_day))
})

full_ref <- design_for("w_diet2d")
results <- list(
  paper = "Sullivan VK et al. Self-Rated Diet Quality and Cardiometabolic Health Among U.S. Adults, 2011-2018. Am J Prev Med 2021;61(4):563-575.",
  reference = "Blind R-only reference (package SU-R); inputs: estimate_candidates.md section 1 and the CDC codebook pages. Decisions in decisions.md.",
  generated = format(Sys.time(), "%Y-%m-%d", tz = "UTC"),
  software = list(r = R.version.string, survey = as.character(packageVersion("survey")),
                  haven = as.character(packageVersion("haven"))),
  exposure = "rating = 6 - DBQ700 (poor = 1 ... excellent = 5), codes 7 and 9 dropped; beta is per rating step. beta_on_raw_dbq700 = -beta.",
  design = list(
    call = "svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~<weight_column>, nest = TRUE, data = <all 39,156 records>)",
    domain = "subset(design, in_analytic & outcome present & weight > 0 & complete cases on the fit's variables)",
    fit = "svyglm(formula, design = domain, family = gaussian())",
    records = nrow(d),
    strata = length(unique(d$SDMVSTRA)),
    psus = nrow(unique(d[, c("SDMVSTRA", "SDMVPSU")])),
    df = degf(full_ref),
    p = "two-sided t with df = PSUs - strata of the full design (SAS survey procedures' default); svyglm_p uses svyglm's residual df"
  ),
  sample_steps = prep$sample_steps,
  nothing_reported_days_in_sample = prep$nothing_reported_days_in_sample,
  analytic_tables = tables,
  fits = fits,
  domain_variance_checks = variance_checks,
  premise_checks_note = paste(
    "Not results. Run after the primary fits were compared with the paper, to locate two",
    "differences (the water step's count; M4). Every variant tried is listed in the order tried;",
    "none replaces a primary fit."
  ),
  premise_checks = premise_checks
)
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null", na = "null"),
           file.path(here, "results.json"))
cat("wrote", file.path(here, "results.json"), "\n")
