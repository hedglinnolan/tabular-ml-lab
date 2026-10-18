# Sullivan 2021 reference (R only): build the analytic tables from the public NHANES files.
#
# Run from the repository root, after download.R:
#   Rscript research/replication/sullivan2021/reference_r/prep.R
#
# Reads research/data/nhanes/<cycle>/*.xpt and writes, under research/data/derived/reference_r/
# (ignored by git):
#   stacked.rds          every DEMO record of the four cycles (39,156) with the derived variables,
#                        the sample-step flags and the analysis weights; fit.R builds the survey
#                        design over all of it, so each analysis is a domain of the full design.
#   analytic_<outcome>.csv
#                        one per outcome: the analytic sample's rows that have the outcome and a
#                        positive weight, with every model's variables and a flag per fit saying
#                        which rows that fit uses (complete cases on its own variables).
#   sample_steps.json    the paper's sample steps with our count after each one.
# Every choice the paper left unstated is in decisions.md, under the label cited beside it here.

suppressPackageStartupMessages({
  library(haven)
  library(jsonlite)
})

data_root <- file.path("research", "data", "nhanes")
out_dir <- file.path("research", "data", "derived", "reference_r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cycles <- c(G = "2011-2012", H = "2013-2014", I = "2015-2016", J = "2017-2018")
n_cycles <- length(cycles)

read_component <- function(name, suffix, vars) {
  path <- file.path(data_root, cycles[[suffix]], paste0(name, "_", suffix, ".xpt"))
  d <- as.data.frame(read_xpt(path, col_select = all_of(c("SEQN", vars))))
  d[] <- lapply(d, function(v) { attributes(v) <- NULL; v })
  d
}

# ---- Merge each cycle's components onto its DEMO file by SEQN, then stack the cycles ----------

one_cycle <- function(s) {
  d <- read_component("DEMO", s, c("RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
                                   "RIDEXPRG", "SDMVPSU", "SDMVSTRA"))
  add <- function(d, name, vars, rename = NULL) {
    x <- read_component(name, s, vars)
    if (!is.null(rename)) names(x)[match(names(rename), names(x))] <- unname(rename)
    stopifnot(!anyDuplicated(x$SEQN))
    merge(d, x, by = "SEQN", all.x = TRUE, sort = FALSE)
  }
  d <- add(d, "DBQ", "DBQ700")
  d <- add(d, "DR1TOT", c("DR1DRSTZ", "DR1TKCAL", "DR1TNUMF", "WTDR2D"))
  d <- add(d, "DR2TOT", c("DR2DRSTZ", "DR2TKCAL", "DR2TNUMF"))
  # Glucose and its fasting weight; in 2011-2012 insulin is in the same file (INS_G does not exist).
  if (s == "G") {
    d <- add(d, "GLU", c("LBXGLU", "LBXIN", "WTSAF2YR"))
    d$WTSAF2YR_INS <- d$WTSAF2YR
  } else {
    d <- add(d, "GLU", c("LBXGLU", "WTSAF2YR"))
    d <- add(d, "INS", c("LBXIN", "WTSAF2YR"), rename = c(WTSAF2YR = "WTSAF2YR_INS"))
  }
  d <- add(d, "GHB", "LBXGH")
  d <- add(d, "BMX", "BMXBMI")
  d <- add(d, "SMQ", c("SMQ020", "SMQ040"))
  d <- add(d, "PAQ", c("PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675"))
  d <- add(d, "DIQ", c("DIQ050", "DIQ070"))
  d$cycle <- cycles[[s]]
  d[order(d$SEQN), ]
}

d <- do.call(rbind, lapply(names(cycles), one_cycle))
rownames(d) <- NULL
stopifnot(!anyDuplicated(d$SEQN))

# ---- Derived variables ------------------------------------------------------------------------

# Exposure: DBQ700 is coded excellent = 1 ... poor = 5, with 7 (refused) and 9 (don't know). The
# paper reverses it to poor = 1 ... excellent = 5 and models it as one continuous score.
d$rating <- ifelse(d$DBQ700 %in% 1:5, 6 - d$DBQ700, NA_real_)

# Covariates.
d$age <- d$RIDAGEYR                                                    # D-age: linear, years
d$sex <- factor(ifelse(d$RIAGENDR == 1, "male", "female"), levels = c("male", "female"))
race_map <- c("1" = "hispanic", "2" = "hispanic", "3" = "nh_white", "4" = "nh_black",
              "6" = "nh_asian", "7" = "other")                         # D-race
d$race <- factor(unname(race_map[as.character(d$RIDRETH3)]),
                 levels = c("nh_white", "nh_black", "hispanic", "nh_asian", "other"))
educ_map <- c("1" = "lt_high_school", "2" = "lt_high_school", "3" = "high_school",
              "4" = "some_college", "5" = "college_grad")              # D-educ: 7, 9 missing
d$educ <- factor(unname(educ_map[as.character(d$DMDEDUC2)]),
                 levels = c("college_grad", "some_college", "high_school", "lt_high_school"))
# D-pir: the three Table 1 bands (<130%, 130-185%, >185% of the poverty threshold).
d$pir <- factor(ifelse(is.na(d$INDFMPIR), NA,
                       ifelse(d$INDFMPIR < 1.30, "lt_130",
                              ifelse(d$INDFMPIR <= 1.85, "130_185", "gt_185"))),
                levels = c("gt_185", "130_185", "lt_130"))

# D-smoke: current smoking = smoked at least 100 cigarettes in life and now smokes every day or
# some days; never smokers and former smokers are not current. Refused or don't know is missing.
d$smoke <- ifelse(d$SMQ020 %in% 2, 0,
                  ifelse(d$SMQ020 %in% 1 & d$SMQ040 %in% c(1, 2), 1,
                         ifelse(d$SMQ020 %in% 1 & d$SMQ040 %in% 3, 0, NA_real_)))

# D-pa: recreational activity, minutes per week = moderate + 2 x vigorous, each days x minutes per
# day. "No" to the activity gives 0 minutes; refused or don't know (7/9, 77/99, 7777/9999) is
# missing.
weekly_minutes <- function(any_q, days, minutes) {
  ok_days <- days %in% 1:7
  ok_min <- !is.na(minutes) & minutes > 0 & minutes < 7777
  ifelse(any_q %in% 2, 0, ifelse(any_q %in% 1 & ok_days & ok_min, days * minutes, NA_real_))
}
d$vig_min <- weekly_minutes(d$PAQ650, d$PAQ655, d$PAD660)
d$mod_min <- weekly_minutes(d$PAQ665, d$PAQ670, d$PAD675)
d$rec_pa <- d$mod_min + 2 * d$vig_min

d$bmi <- d$BMXBMI
# D-energy (M3): the mean of the two recall days' total energy, kcal per day.
d$energy <- (d$DR1TKCAL + d$DR2TKCAL) / 2
# D-med (M4): glucose-lowering medication = now taking insulin (DIQ050) or diabetic pills
# (DIQ070). DIQ070 is skipped for people never told of diabetes or prediabetes; a skip is "no".
# Refused or don't know with no "yes" is missing.
d$glucose_med <- ifelse(d$DIQ050 %in% 1 | d$DIQ070 %in% 1, 1,
                        ifelse(d$DIQ050 %in% c(7, 9) | d$DIQ070 %in% c(7, 9) | is.na(d$DIQ050),
                               NA_real_, 0))

# Outcomes.
d$glucose <- d$LBXGLU
d$insulin <- d$LBXIN
d$hba1c <- d$LBXGH

# D-weights: two-year weights divided by the number of cycles (NCHS's rule for pooled cycles).
# Records outside a subsample (no lab record, or not fasting 8 to <24 hours) have weight 0.
d$w_fasting_glu <- ifelse(is.na(d$WTSAF2YR), 0, d$WTSAF2YR) / n_cycles
d$w_fasting_ins <- ifelse(is.na(d$WTSAF2YR_INS), 0, d$WTSAF2YR_INS) / n_cycles
d$w_diet2d <- ifelse(is.na(d$WTDR2D), 0, d$WTDR2D) / n_cycles
stopifnot(all.equal(d$w_fasting_glu, d$w_fasting_ins))

# ---- The paper's sample steps, in its order ----------------------------------------------------

reliable1 <- d$DR1DRSTZ %in% 1
reliable2 <- d$DR2DRSTZ %in% 1
# D-water: a reliable recall day with zero energy and at least one item reported; the codebook
# says a water-only day is coded reliable, has its water items counted in DRxTNUMF and has zero
# energy. A reliable day with no items at all is not "only water" and is counted apart below.
water_only <- (reliable1 & d$DR1TKCAL %in% 0 & d$DR1TNUMF > 0) |
              (reliable2 & d$DR2TKCAL %in% 0 & d$DR2TNUMF > 0)
nothing_day <- (reliable1 & d$DR1TNUMF %in% 0) | (reliable2 & d$DR2TNUMF %in% 0)
d$nothing_reported_day <- as.integer(nothing_day)

steps <- list(
  list(step = "Start (four cycles combined)", keep = rep(TRUE, nrow(d)), paper_removed = NA, paper_remaining = 39156),
  list(step = "Age < 20", keep = d$RIDAGEYR >= 20, paper_removed = 16539, paper_remaining = 22617),
  list(step = "Pregnant", keep = !(d$RIDEXPRG %in% 1), paper_removed = 247, paper_remaining = 22370),
  list(step = "No self-rating (DBQ700 missing, 7 or 9)", keep = !is.na(d$rating), paper_removed = 10, paper_remaining = 22360),
  list(step = "No dietary recall (day 1 not complete and reliable)", keep = reliable1, paper_removed = 2982, paper_remaining = 19378),
  list(step = "Only one recall (day 2 not complete and reliable)", keep = reliable2, paper_removed = 2445, paper_remaining = 16933),
  list(step = "Only water reported on a recall day", keep = !water_only, paper_removed = 20, paper_remaining = 16913)
)
in_sample <- rep(TRUE, nrow(d))
step_rows <- list()
for (st in steps) {
  before <- sum(in_sample)
  in_sample <- in_sample & st$keep
  step_rows[[length(step_rows) + 1]] <- list(
    step = st$step, paper_removed = st$paper_removed, ours_removed = if (is.na(st$paper_removed)) NA else before - sum(in_sample),
    paper_remaining = st$paper_remaining, ours_remaining = sum(in_sample)
  )
}
d$in_analytic <- in_sample
for (r in step_rows) cat(sprintf("%-55s paper -%-6s %6s   ours -%-6s %6d\n", r$step,
                                 ifelse(is.na(r$paper_removed), "", r$paper_removed), r$paper_remaining,
                                 ifelse(is.na(r$ours_removed), "", r$ours_removed), r$ours_remaining))

# ---- The fits' variables and their rows ----------------------------------------------------------

source(file.path("research", "replication", "sullivan2021", "reference_r", "models.R"))

for (oc in names(outcome_weight)) {
  w <- outcome_weight[[oc]]
  base <- d$in_analytic & !is.na(d[[oc]]) & d[[w]] > 0
  cols <- c("SEQN", "cycle", "SDMVSTRA", "SDMVPSU", w, "DBQ700", "rating", oc,
            setdiff(all_covariates, oc), "nothing_reported_day")
  tab <- d[base, cols]
  for (f in fits_for(oc)) {
    vars <- c(oc, "rating", f$covariates)
    tab[[paste0("in_", f$id)]] <- as.integer(complete.cases(tab[, vars]))
  }
  path <- file.path(out_dir, paste0("analytic_", oc, ".csv"))
  write.csv(tab, path, row.names = FALSE, na = "")
  cat(sprintf("%-8s %-14s base rows %6d\n", oc, w, nrow(tab)))
}

# Rows outside the sample are kept: each analysis is a domain of the full design.
saveRDS(d, file.path(out_dir, "stacked.rds"))
writeLines(toJSON(list(sample_steps = step_rows,
                       nothing_reported_days_in_sample = sum(nothing_day & d$in_analytic),
                       water_only_any_age = sum(water_only)),
                  auto_unbox = TRUE, pretty = TRUE, na = "null"),
           file.path(out_dir, "sample_steps.json"))
