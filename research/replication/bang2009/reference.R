# Bang 2009 reference (R only): the same models on the same participants, built from the NHANES
# transport files with no TurboTab code.
#
# Run from the repository root:
#   Rscript research/replication/bang2009/reference.R
#
# It reads the 1999-2004 files the intake downloaded (research/data/nhanes/<cycle>/*.xpt, with
# haven), joins each cycle's files to its DEMO by SEQN and stacks the cycles; derives the six
# predictors from the paper's rules (bang2009.md section 2.3) in its own code; builds the NCHS
# six-year fasting weight from the tutorial's formula (2/3 x WTSAF4YR on the 1999-2002 rows,
# 1/3 x WTSAF2YR on the 2003-2004 rows; section 2.6); applies the eligibility rules as prereg.json
# states them, each rule read as data; and fits every run run.py makes (prereg.json's sweep):
#   - weighted (the survey's population answer): survey::svyglm, quasibinomial, on the design
#     svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~w6, nest = TRUE) over every row with
#     a positive fasting weight, the analysis rows a domain taken with subset(); intervals on t with
#     degf(design) degrees of freedom; beside it the standard errors of a design over the analysis
#     rows alone (a subset, as section 6 choice 7 says SAS 9.1 may have done);
#   - unweighted (the sample answer, and every prediction run): glm, binomial, to 1e-12.
# Each fit's apparent AUC is pROC's on its own rows, unweighted; the default's weighted AUC is the
# weighted Mann-Whitney sum beside it. Writes research/replication/bang2009/reference.json (the
# numbers, each fit's row count and the SHA-256 of its sorted SEQN, one per line) and the SEQN
# lists under research/data/derived/bang2009/reference/ (ignored by git).

suppressPackageStartupMessages({
  library(haven)
  library(jsonlite)
  library(survey)
  library(pROC)
  library(digest)
})

here <- file.path("research", "replication", "bang2009")
prereg <- fromJSON(file.path(here, "prereg.json"), simplifyVector = FALSE)
out_dir <- file.path("research", "data", "derived", "bang2009", "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
root <- file.path("research", "data", "nhanes")
cycles <- list(list(dir = "1999-2000", sfx = "", lab = "LAB10AM"),
               list(dir = "2001-2002", sfx = "_B", lab = "L10AM_B"),
               list(dir = "2003-2004", sfx = "_C", lab = "L10AM_C"))

read_one <- function(cyc, stem, vars) {
  d <- as.data.frame(read_xpt(file.path(root, cyc$dir, paste0(stem, ".xpt"))))
  keep <- intersect(c("SEQN", vars), names(d))
  d <- d[, keep, drop = FALSE]
  d[] <- lapply(d, function(v) { attributes(v) <- NULL; as.numeric(v) })
  stopifnot(!anyDuplicated(d$SEQN))
  for (v in setdiff(vars, names(d))) d[[v]] <- NA_real_
  d
}

one_cycle <- function(cyc) {
  d <- read_one(cyc, paste0("DEMO", cyc$sfx),
                c("SDDSRVYR", "RIDAGEYR", "RIAGENDR", "RIDEXPRG", "SDMVSTRA", "SDMVPSU"))
  parts <- list(
    list(cyc$lab, c("LBXGLU", "WTSAF2YR", "WTSAF4YR")),
    list(paste0("DIQ", cyc$sfx), c("DIQ010", "DIQ050", "DIQ070")),
    list(paste0("BPQ", cyc$sfx), c("BPQ020", "BPQ040A", "BPQ050A")),
    list(paste0("BPX", cyc$sfx), c(paste0("BPXSY", 1:4), paste0("BPXDI", 1:4))),
    list(paste0("BMX", cyc$sfx), c("BMXBMI", "BMXWAIST")),
    list(paste0("PAQ", cyc$sfx), "PAQ520"),
    list(paste0("MCQ", cyc$sfx), c("MCQ250A", "MCQ260AA", "MCQ260AB", "MCQ260AG", "MCQ260AH")))
  for (p in parts) d <- merge(d, read_one(cyc, p[[1]], p[[2]]), by = "SEQN", all.x = TRUE, sort = FALSE)
  d
}

d <- do.call(rbind, lapply(cycles, one_cycle))
d <- d[order(d$SEQN), ]
stopifnot(!anyDuplicated(d$SEQN))

# ---- the six predictors, from the paper's words ---------------------------------------------
yes <- function(x, code = 1) !is.na(x) & x == code
d$age_band <- ifelse(d$RIDAGEYR >= 60, 3, ifelse(d$RIDAGEYR >= 50, 2, ifelse(d$RIDAGEYR >= 40, 1, 0)))
d$male <- as.numeric(d$RIAGENDR == 1)
d$famhx <- as.numeric(yes(d$MCQ250A))
d$famhx_close <- as.numeric(yes(d$MCQ260AA, 1) | yes(d$MCQ260AB, 2) | yes(d$MCQ260AG, 7) |
                              yes(d$MCQ260AH, 8))
sy <- as.matrix(d[, paste0("BPXSY", 1:4)])
di <- as.matrix(d[, paste0("BPXDI", 1:4)])
row_mean <- function(m) { s <- rowSums(!is.na(m)); ifelse(s > 0, rowSums(m, na.rm = TRUE) / s, NA) }
first_of <- function(m) apply(m, 1, function(r) { r <- r[!is.na(r)]; if (length(r)) r[1] else NA })
high <- function(s, dbp) (!is.na(s) & s >= 140) | (!is.na(dbp) & dbp >= 90)
told <- yes(d$BPQ020)
d$htn <- as.numeric(told | yes(d$BPQ050A) | high(row_mean(sy), row_mean(di)))
d$htn_first <- as.numeric(told | yes(d$BPQ050A) | high(first_of(sy), first_of(di)))
d$htn_told_rx <- as.numeric(told | yes(d$BPQ040A) | high(row_mean(sy), row_mean(di)))
waist_in <- d$BMXWAIST / 2.54
lvl <- function(x, a, b, c) ifelse(is.na(x), NA, ifelse(x >= c, 3, ifelse(x >= b, 2, ifelse(x >= a, 1, 0))))
by_bmi <- lvl(d$BMXBMI, 25, 30, 40)
by_waist <- ifelse(d$RIAGENDR == 1, lvl(waist_in, 37, 40, 50), lvl(waist_in, 31.5, 35, 49))
d$obesity <- suppressWarnings(pmax(by_bmi, by_waist, na.rm = TRUE))
d$obesity_absent <- ifelse(is.na(d$obesity), 0, d$obesity)
d$active <- as.numeric(yes(d$PAQ520))
d$y <- as.numeric(d$LBXGLU >= 126)
d$w6 <- ifelse(d$SDDSRVYR %in% c(1, 2), 2 / 3 * d$WTSAF4YR, 1 / 3 * d$WTSAF2YR)
# Every predictor column by SEQN, for results.py to set beside the prep's derived table.
predictor_columns <- c("age_band", "male", "famhx", "famhx_close", "htn", "htn_first", "htn_told_rx",
                       "obesity", "obesity_absent", "active")
write.csv(d[, c("SEQN", predictor_columns)], file.path(out_dir, "predictors.csv"), row.names = FALSE,
          na = "")

# ---- the rules, read as data -----------------------------------------------------------------
keeps <- function(rule) {
  x <- d[[rule$column]]
  missing_keep <- identical(rule$missing, "keep")
  if (identical(rule$kind, "range")) {
    ok <- !is.na(x)
    if (!is.null(rule$low)) ok <- ok & x >= rule$low
    if (!is.null(rule$high)) ok <- ok & x <= rule$high
    if (missing_keep) ok <- ok | is.na(x)
    return(ok)
  }
  unconfirmed <- is.na(x) | x %in% as.numeric(unlist(rule$refused)) |
    x %in% as.numeric(unlist(rule$dont_know))
  at <- !is.na(x) & x %in% as.numeric(unlist(rule$exclude))
  !at & (missing_keep | !unconfirmed)
}

decisions <- prereg$decisions
settings <- function(lane, variant) {
  changed <- prereg$sweep$variants[[variant]]
  predictors <- lapply(prereg$model$predictors, function(p) p$column)
  rules <- prereg$eligibility$rules
  survey_answer <- "population"
  for (dd in decisions) {
    option <- if (!is.null(changed[[dd$id]])) changed[[dd$id]] else dd$default
    apply <- dd$options[[option]]$apply
    for (k in names(apply$predictors)) predictors[[k]] <- apply$predictors[[k]]
    for (k in unlist(apply$drop_rules)) rules[[k]] <- NULL
    for (k in names(apply$rules)) for (f in names(apply$rules[[k]])) rules[[k]][[f]] <- apply$rules[[k]][[f]]
    if (!is.null(apply$survey)) survey_answer <- apply$survey
  }
  if (lane == "predict") survey_answer <- "sample"  # the engine fits no weight under prediction
  list(predictors = predictors, rules = rules, survey = survey_answer)
}

runs <- character(0)
for (lane in c("estimate", "predict")) for (v in names(prereg$sweep$variants)) {
  ch <- names(prereg$sweep$variants[[v]])
  only <- unlist(lapply(decisions, function(dd) if (dd$id %in% ch) dd$lanes else NULL))
  if (is.null(only) || lane %in% only) runs <- c(runs, paste0(lane, ":", v))
}

design_all <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~w6, nest = TRUE,
                        data = d[!is.na(d$w6) & d$w6 > 0, ])

seqn_hash <- function(s) digest(paste(sprintf("%d", as.integer(sort(s))), collapse = "\n"),
                                algo = "sha256", serialize = FALSE)
weighted_auc <- function(y, p, w) {
  cases <- which(y == 1); controls <- which(y == 0)
  num <- 0
  for (i in cases) num <- num + w[i] * (sum(w[controls] * (p[i] > p[controls])) +
                                        0.5 * sum(w[controls] * (p[i] == p[controls])))
  num / (sum(w[cases]) * sum(w[controls]))
}
term_of <- function(name, predictors) {
  for (k in names(predictors)) {
    col <- predictors[[k]]
    if (startsWith(name, paste0("factor(", col, ")"))) return(c(col, sub(".*\\)", "", name)))
    if (name == col) return(c(col, "1"))
  }
  c(name, NA)
}

fits <- list()
counts <- NULL
for (name in runs) {
  parts <- strsplit(name, ":")[[1]]
  s <- settings(parts[1], parts[2])
  eligible <- !is.na(d$y)
  for (r in s$rules) eligible <- eligible & keeps(r)
  cols <- unlist(s$predictors)
  complete <- eligible & complete.cases(d[, cols])
  rhs <- sapply(cols, function(c) if (c %in% c(s$predictors$age, s$predictors$obesity))
    paste0("factor(", c, ")") else c)
  form <- as.formula(paste("y ~", paste(rhs, collapse = " + ")))
  out <- list(name = name, n_analyzed = sum(complete), seqn_sha256 = seqn_hash(d$SEQN[complete]))
  writeLines(sprintf("%d", as.integer(sort(d$SEQN[complete]))),
             file.path(out_dir, paste0(sub(":", "-", name), ".seqn")))
  if (s$survey == "population") {
    domain <- complete & !is.na(d$w6) & d$w6 > 0
    keep_idx <- d$SEQN[domain]
    sub <- subset(design_all, design_all$variables$SEQN %in% keep_idx)
    m <- svyglm(form, design = sub, family = quasibinomial(),
                control = glm.control(epsilon = 1e-12, maxit = 100))
    df <- degf(design_all)
    ci <- confint(m, ddf = df)
    alone <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~w6, nest = TRUE,
                       data = d[domain, ])
    m_alone <- svyglm(form, design = alone, family = quasibinomial(),
                      control = glm.control(epsilon = 1e-12, maxit = 100))
    se_subset <- sqrt(diag(vcov(m_alone)))
    rows <- d[domain, ]
    out$n_estimate <- sum(domain)
    out$df <- df
    out$estimator <- "survey::svyglm, quasibinomial, Taylor linearization (domain of the design)"
  } else {
    rows <- d[complete, ]
    m <- glm(form, data = rows, family = binomial(),
             control = glm.control(epsilon = 1e-12, maxit = 100))
    ci <- confint.default(m)
    se_subset <- NULL
    out$n_estimate <- sum(complete)
    out$estimator <- "glm, binomial, model-based standard errors"
  }
  co <- summary(m)$coefficients
  p_hat <- as.numeric(fitted(m))
  stopifnot(length(p_hat) == nrow(rows), all(as.numeric(m$y) == rows$y))
  out$auc_apparent <- as.numeric(auc(roc(rows$y, p_hat, direction = "<", levels = c(0, 1), quiet = TRUE)))
  if (name == "estimate:default") out$auc_apparent_weighted <- weighted_auc(rows$y, p_hat, rows$w6)
  terms <- list()
  for (i in seq_len(nrow(co))) {
    t <- rownames(co)[i]
    key <- if (t == "(Intercept)") c("(intercept)", NA) else term_of(t, s$predictors)
    terms[[length(terms) + 1]] <- list(
      term = t, column = key[1], level = key[2], log_or = unname(co[i, 1]), se = unname(co[i, 2]),
      ci_low = unname(ci[i, 1]), ci_high = unname(ci[i, 2]),
      se_subset = if (is.null(se_subset)) NULL else unname(se_subset[t]))
  }
  out$terms <- terms
  out$converged <- isTRUE(m$converged)
  fits[[name]] <- out
  if (name == "estimate:default") {
    base <- !is.na(d$y)
    for (r in s$rules[c("age", "pregnant", "fasting")]) base <- base & keeps(r)
    known <- yes(d$DIQ010) | yes(d$DIQ050) | yes(d$DIQ070)
    eligible_rows <- base & !known
    g <- d$LBXGLU
    counts <- list(
      about = paste("aged 20 or older, not pregnant, a positive fasting weight and a glucose result;",
                    "known diabetes is a yes to DIQ010, DIQ050 or DIQ070; the rest by glucose"),
      known_diabetes = sum(base & known),
      normal = sum(eligible_rows & g < 100), impaired = sum(eligible_rows & g >= 100 & g < 126),
      undiagnosed = sum(eligible_rows & g >= 126), eligible = sum(eligible_rows),
      complete_cases = sum(complete))
  }
}

result <- list(
  about = paste("Bang 2009 reference fits in R, independent of TurboTab: reference.R reads the",
                "NHANES transport files, derives the predictors and applies prereg.json's rules"),
  r = R.version.string, survey = as.character(packageVersion("survey")),
  pROC = as.character(packageVersion("pROC")), haven = as.character(packageVersion("haven")),
  rows_read = nrow(d), counts = counts, fits = fits)
write(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE),
      file.path(here, "reference.json"))
cat("wrote", length(fits), "fits\n")
