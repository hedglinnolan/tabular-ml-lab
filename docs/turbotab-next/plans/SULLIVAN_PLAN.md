# The Sullivan 2021 replication (swarm path, step 1)

Written 2026-10-18 by the orchestrator, under Nolan's go-ahead of the same day: "Do everything
right up until we are ready to start the swarm and I will have a discussion with you about it."

## TL;DR

- **What:** run Sullivan et al. 2021 (self-rated diet quality and fasting glucose, insulin, HbA1c
  and BMI, NHANES 2011–2018) through TurboTab's Estimate lane, from the public files, and compare
  our numbers with the paper's.
- **Why:** it is the first end-to-end check of the Estimate lane on a real published analysis, and
  the swarm's replication runs need a hand-checked reference to be scored against.
- **What it reports:**
  - the decisions the paper stated;
  - the ones it left unstated;
  - how far each unstated option moves the published number;
  - every place the engine could not do what the paper did (pull requirements for later waves).
- **How we know our numbers are right:** a second agent, blind to the first one's code, builds
  the same table in R alone from the paper and the codebooks. The two must agree with each other
  before either is compared with the paper.
- **Cost:** about 5–6M fresh tokens: over the 3M step line, said here in advance.

## 1. What is known

The paper, its sample, weights, covariates, exposure coding, known answers, data files and the
engine gaps are in `notes/replication/estimate_candidates.md` §1. They were checked at their
sources on 2026-10-15, and that section is this plan's premise. As every premise, it is checked
in the code and the files first.

The known answers this step reproduces:
- **Table 4** (BMI-adjusted): β per rating step, SE and n for glucose (−2.0, 0.4, 6,832), insulin
  (−0.6, 0.2, 6,655), HbA1c (−0.05, 0.01, 14,795) and BMI with no BMI adjustment (−1.6, 0.1,
  15,278).
- **Appendix Table 2:** the nested ladder M0–M4 for glucose, insulin and HbA1c.

## 2. Rulings (the orchestrator's, as the methods expert)

1. **Where things live.** Code in `research/replication/` and data under `research/data/`, both
   outside `turbotab/` (SWARM_PLAN §4: the product never imports them). The NHANES files are not
   committed: `research/data/` is ignored by git. A committed manifest lists, per file:
   - the URL;
   - the date CDC states for the file, where the page states one;
   - the download date;
   - the size and SHA-256.

   This is the first instance of SWARM_PLAN §5's provenance.
2. **The prep does only what the engine cannot do today.** Each such step is named in the report
   as an engine gap:
   - merging the files by SEQN;
   - stacking the four cycles;
   - deriving recreational activity minutes (moderate + 2 × vigorous) and current smoking;
   - counting each person's reliable recalls (the DRxDRSTZ status);
   - flagging glucose-lowering medication for M4.

   Everything the engine can do goes through it, through the API as a researcher would answer:
   - the exclusions (age under 20, pregnancy, no rating, DBQ700 codes 7 and 9, fewer than two
     reliable recalls, a recall with only water);
   - the outcome, the exposure and its linear form, the covariates;
   - the survey design (weights, strata, PSUs) and the estimand.

   So the exclusions stay visible in the methods the engine writes.
3. **The exposure's direction.** NHANES codes DBQ700 with excellent = 1, and the paper reverses it.
   The engine has no recode, and the prep must not hide one. So we fit on the raw codes, compare
   −β with the paper, and file the missing recode as an engine gap.
4. **Weights:**
   - WTSAF2YR for glucose and insulin;
   - WTDR2D for HbA1c and BMI;
   - each divided by the number of cycles, as the engine does for a stacked table and as NCHS
     advises.

   The division is unstated in the paper. Its shift is measured like any other unstated option
   (§3), not assumed. By derivation, a common factor on every weight moves neither β nor a
   linearized SE, so the expected shift is zero; the run confirms it.
5. **One project per outcome and model:** 4 for Table 4 and 15 for the ladder. Each runs as a
   journey: `turbotab/core/reference/journeys.py`'s loop and capture, with the paper's stated
   decisions as the journey's answers. That spec is SWARM_PLAN's design 1 "plan file" in its first
   form. The driver step later grows it into the analyst's command set, so it is written to be
   reused, not thrown away.
6. **Engine gaps are filed, not fixed here.** Each is an INBOX line naming the replication. One
   exception: a gap that stops the replication from reaching a number (a refusal with no exit, a
   crash). It comes back to the orchestrator with its evidence and is ruled on then.
7. **The independent reference is blind.** The second agent reads the paper's facts
   (`estimate_candidates.md` §1) and the CDC codebooks, never the first agent's code or table.
   It builds its own analytic table and fits with R's `survey` package (`svyglm`, the design from
   `svydesign` with `nest = TRUE`). R 4.6.1 with `survey` and `haven` is installed. Nothing is
   installed outside a scratch library.

## 3. Unstated decisions, each measured

Each one is refit through the engine with its options, and the shift in β and SE is reported
against the paper's rounding:
- the form of age (linear, three bands, a spline);
- the form of PIR (continuous, the paper's three Table 1 bands, either with a missing category);
- the division of the weights across cycles;
- how "only water on a recall day" is identified;
- missing covariates: complete cases, as SAS survey procedures drop them; the count dropped is
  reported per outcome.

**The analytic choices worth discussing** (`estimate_candidates.md` §1) are alternatives, not
unstated decisions. Each one's shift is measured and reported apart, labeled as an alternative:
- BMI in or out;
- excluding diagnosed diabetes;
- energy on its own;
- insulin on the log scale.

## 4. Acceptance

1. **The two pipelines agree:** the engine's β and SE equal R's to 1e-6 relative on each one's
   own table. Where the tables differ, the rows that differ are listed and explained.
2. **n per outcome** is within 1% of the paper's, or each difference is traced to a named decision.
3. **Table 4 and the ladder:** each β and SE matches the paper at its printed precision, or the
   gap is explained by a named, measured unstated option, or it is reported as unexplained. No
   number is tuned toward the paper.
4. **The methods** the engine writes state every stated decision the paper made, and nothing it
   did not.
5. **The report** (`notes/replication/sullivan2021.md`, plain TL;DR first) gives:
   - the stated and unstated decisions table;
   - the materiality of each;
   - our numbers against the paper's;
   - the engine gaps with their INBOX lines;
   - the run's spend and fit times.

## 5. Packages

| Package | Agent | Work | Estimate |
|---|---|---|---|
| SU-A | Opus builder | intake and manifest; the prep; the journey specs; the 19 runs and the §3 refits; the report | 2.5–3.5M |
| SU-R | Opus builder, blind | the R-only reference for Table 4 and the ladder | 0.7–1M |
| Verify | Opus verifier | re-derives the comparison; probes for tuning toward the paper, a decision the methods misstate, a prep step the engine could have done | 0.5–1M |
| Repair | at most two rounds | | 1M |

SU-A and SU-R run in parallel. The verifier reads both.

**Machine:**
- the downloads are about 44 files, a few hundred MB;
- the fits are light (survey linear regressions on 7,000–15,000 rows);
- two workers, under the quiet-hours throttle at night;
- no heavy run.

**Web:** wwwn.cdc.gov only, for the data files and codebooks, with no identifier in any request.
