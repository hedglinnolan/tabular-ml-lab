# Exploring without overwhelming: the First look stage

*Design brief · TurboTab Next · 2026-10-05 · built on the calm foundation (`FOUNDATION.md` §3 page zones and §5 canvas grammar). The repo was only read. The only computations were small summaries of two sample files, run to supply the worked examples' numbers.*

---

## 0 · The answer on one page

Nolan asked whether exploration should be one dynamic canvas with sensible grouping. Yes. Four rules make it work:

1. **One stage and one canvas, with the same six groups in every lens.**
   - Each group is named for the question it asks of the data, not for a chart type.
   - The groups follow the STRATOS initial-data-analysis (IDA) domains: participants, missing values, each variable, variables together, plus "over time" when the data have that structure.
   - A lens fills the groups. It never adds a tab.
2. **The card becomes a guide.** In this stage the card holds a short reading list:
   - at most three items under "Worth a look";
   - then the six groups, each with a count.

   The card asks nothing until the user opens a decision.
3. **"Worth a look" means: this differs from what we expected, and it would change the plan.**
   - It never means "striking".
   - It is never ranked by the outcome.
   - Every highlight shows its measure beside its reference, plus one line on why it is listed.
4. **Looking and deciding stay separate.**
   - A look that needs an answer offers "Decide now". The card turns into the familiar one-question card, each option previews on the same canvas, and Esc returns to the list.
   - The outcome has its own door. Its views are recorded. Under inference they are never pointed at.

So the Q&A does not go away. It moves inside the browsing: people browse to find things and use the Q&A to decide on them. Voyager's study participants asked for this hybrid. 15 of 16 preferred the browsing tool for exploration, 15 of 16 preferred the manual tool for answering questions, and all but one wanted both.

---

## 1 · Layout in the page zones

### 1.1 Where the stage sits in the chain

`Data ✓ · First look · Participants · Exposure · Confounders · Energy · Model · Results`

- **After Data.** Looks read settled readings (BLUEPRINT §14.1): units, roles, the energy column, sample roles. A look built on an unsettled reading would put a guess on screen as fact.
- **Before Participants.** What the stage shows feeds the decisions that follow: exclusions, missing values, energy, and the QC steps.
- **The same place under both purposes.**
  - Under prediction the stage comes before the seal, and its outcome group opens once the split is recorded.
  - Under inference the outcome group is a recorded door from the start.

### 1.2 The card changes role, from a question to a guide

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Data ✓ · First look · Participants · Exposure · Confounders · Energy · Model · Results    │
├──┬────────────────────────────────┬───────────────────────────────────────────────────────┤
│M │ First look · before you plan   │ Your data now · 21,849 rows · 29 columns               │
│a │ 21,849 adults · 9 cycles pooled│                                                        │
│n │                                │ name             kind       shape          blank  look │
│u │ Worth a look                   │ SEQN             identifier                            │
│s │ 1 Sugar rises with energy      │ cycle_begin_year 9 levels   ▂▂▂▃▃▂▂▂▂                  │
│c │   Affects the energy model     │ age              number     ▃▅▅▅▄▃▂                    │
│r │ 2 1,614 adults report energy   │ kcal             number     ▂▇▅▂▁                 1 2  │
│i │   outside Willett's range      │ sugar            number     ▆▇▃▁                  1    │
│p │   Affects who is analyzed      │ triglycerides    the outcome · opens in its group      │
│t │ 3 Medication answers are blank │ meds_hbp         yes/no     ▇▁             71%    3    │
│  │   for 71–79%, falling with age │ meds_chol        yes/no     ▇▁             79%    3    │
│  │   Affects the missing values   │ imputed_bp_sys   fill flag of bp_sys   8.6% filled     │
│  │                                │ … every column, in column order, nothing hidden        │
│  │ Who is in the data          2  │                                                        │
│  │ What is missing             2  │                                                        │
│  │ Each variable               1  │                                                        │
│  │ Variables together          3  │                                                        │
│  │ Over time and by batch      1  │                                                        │
│  │ The outcome · each view is recorded                                                     │
│  │                                │                                                        │
│  │ Why does this matter?          │                                                        │
│  │             [ Look at the first ]                                                       │
└──┴────────────────────────────────┴───────────────────────────────────────────────────────┘
```

**The card, from top to bottom:**
- **Header.** "First look · before you plan", then one line on the shape of the data.
- **Worth a look.** At most three numbered items. Each is one claim (at most 20 words) plus the decision it affects, written as "Affects …". The number is in amber, the coach color, and always comes with words.
- **The six groups.** Each shows the count of its items. A group with nothing to show stays listed in gray with one line saying why, for example "No batches or repeated measures in this table". Show Me's rule: an inapplicable item stays visible and says why.
- **The outcome row**, with its rule in words.
- **"Why does this matter?"** One disclosure for the stage.
- **One primary action,** always at the bottom.

**The primary action:**
- On the first visit it walks the highlights: "Look at the first", then "Next · 2 of 3", then "Continue to Participants". This is BLUEPRINT §11.4's "next slot" walk.
- A T1 blocker (§3.2) replaces the label with "Decide this first".
- On a revisit, which is mastery, the button reads "Continue" from the start.
- The chain never blocks navigation.

### 1.3 How much room the canvas gets

- In this stage the card sits at its minimum width, 380 px.
- The manuscript rail stays collapsed. It can still be pinned at 1,680 px or wider, but it is off by default here.
- On a 1,440 px screen the canvas gets about 1,000 px, roughly 70%.

The canvas has two modes:

```
OVERVIEW (on arrival)                    ITEM (a look is open)
┌──────────────────────────────────┐     ┌────────┬──────────────────────────────┐
│ index: every column, one row     │     │ index  │ FOCAL VIEW (≈ 2/3)           │
│ name · kind · shape · blank ·    │     │ names  │ layout chosen by footprint   │
│ look number                      │     │ only,  ├──────────────────────────────┤
│                                  │     │ 160 px │ CONTEXT VIEW (≈ 1/3)         │
│                                  │     │        │ the same, by one structural  │
│                                  │     │        │ variable    More angles (1)  │
└──────────────────────────────────┘     └────────┴──────────────────────────────┘
```

- **The index is the Strip layout's row design** (FOUNDATION §5): every column at its "now" measure, in column order, with no before→after. It is navigation, not a fourth view. It follows Voyager's rule to show "univariate summaries of all variables prior to user interaction" to discourage premature fixation.
- **At most three views:** the focal view, the context view, and one behind "More angles" (§5 rule 2). The largest gets the most room (rule 1).
- **On narrow screens** the index becomes a list above the views, and the primary action moves to the fixed footer.

### 1.4 What the canvas draws, and in which colors

- **"Your data now" only.** A look changes nothing, so there is no flip. The flip appears only once the user points at a decision's option (rule 7: never draw a change that isn't there).
- **Colors by role:**
  - data in gray;
  - amber for the noticing, always with its words;
  - red only for a real T1 blocker;
  - comparison colors for structural groups (sex, batch, pooled QC versus participant);
  - indigo only when an option is pointed at.
- **Linked views:** pointing at a name in the index highlights it in every view (rule 4).
- **Keyboard:** ↑ and ↓ move through items and the canvas morphs. Axes hold still when the next item shares a column, which is Dziban's visual anchoring. Esc steps back one level.
- **Seen marks:** a plain gray check appears beside an item and its index name once it has been opened. Green stays reserved for recorded sentences (FOUNDATION §4), so it appears only in the manuscript, beside decisions and outcome-view sentences.

---

## 2 · Grouping

### 2.1 Six groups, fixed order, every lens

| # | Group | The question | STRATOS IDA domain | Usual layouts |
|---|---|---|---|---|
| 1 | Who is in the data | Who and what are the rows? | participants and samples | Flow, table focus |
| 2 | What is missing | Where are the blanks, and why? | missing values (M1–M4) | Focus, Strip |
| 3 | Each variable | Is each column what it says? | univariate (U1–U2) | Focus, Strip |
| 4 | Variables together | What moves together or duplicates? | multivariate (V1–V3) | Strip, matrix (new), embedding (new) |
| 5 | Over time and by batch | Did how or when it was measured change it? | structural variables; longitudinal | Focus; a relationship over order |
| 6 | The outcome | What does the outcome look like? (a door) | univariate, outcome only | Focus; recorded |

Why the order is fixed:
- Show Me's "stable grid of choices" and Voyager's "predictable and consistent ordering" both teach that a map learned once can be reused.
- An NHANES user who moves to a metabolomics panel finds the same six questions. Only the answers differ.

### 2.2 What each lens puts in the groups

| Group | Dietary | Clinical | Metabolomics | Genomics | Survey |
|---|---|---|---|---|---|
| Who | recalls per person; survey design, lonely PSU; pooled cycles; Kish n_eff (new) | visits per patient; sites; ages judged by age-right bands | sample roles (pooled QC, blanks, calibrants); repeated subjects; duplicate ids | bulk or single cell; replicates | respondents; design weights |
| Missing | whole recall vs nutrient absent; day-2 completers vs not (new) | censored results ("<5"); fill flags | blanks against abundance (detection limit); zeros or missing | zeros and low counts per gene | sentinel codes; skip patterns |
| Each variable | energy unit (Atwater); implausible intake under each convention; share of zero days (new) | mixed units; default-value spikes; impossible vs extreme; number format | already logged or scaled; empty and constant features; ion modes; QC RSD and D-ratio as looks (new) | data type (counts, CPM, FPKM, VST); gene-id corruption, versions, duplicates; library size | the response run; floor and ceiling; ordinal declared |
| Together | energy dependence; macronutrient closure; collinear nutrients; food-source clusters (new) | collinear labs | redundancy (effective number of quantities); PCA with the QCs (new view) | sample × sample correlation (new view); PCA by batch (new view); p ≫ n | item correlations; reverse-keyed items; reliability |
| Over time / batch | cycles; day of week; consecutive-day recalls (new) | dates out of order | run-order drift against chance; batch; total signal by run order | batch; batch × outcome verdict | waves; attrition by wave (new) |
| Outcome | the same door in every lens (§6) | | | | |

- "(new)" marks a reading that the domain review found no engine code for.
- The other entries map to existing finding ids (`pack::<lens>::…`, `voice::…`) or to stage readings. Each lens's review packet confirms them (§2.3).
- Each lens also declares its **context variables**, the structural variables a look may split by:
  - dietary: sex, age band, cycle, recall day;
  - clinical: site, visit;
  - metabolomics: batch, run order, sample type;
  - genomics: batch, lane;
  - survey: wave, mode, stratum.

  The outcome is never a context variable.

### 2.3 How a look is declared: the Look contract

A look enters the app the way a domain method does (BLUEPRINT §13): through a declared contract that the engine enforces. `Pack` gains one field, `looks`, beside `detectors`:

```python
@dataclass(frozen=True)
class Look:
    key: str                 # "dietary.energy_screens"
    group: Group             # who | missing | each | together | over_time | outcome
    question: str            # ≤ 14 words: the question this look answers
    purpose: Purpose         # one of BLUEPRINT §11.2's five (most: "Is my data okay?")
    outcome_class: OClass    # O0 | O1 | O2 | O3; never O4 (§6.1)
    rows: RowScope           # all (descriptive, pre-seal) | training | reference
    needs: tuple[str, ...]   # settled readings it requires (energy unit, sex, run order…)
    measure: Callable        # deterministic and replayable → value, reach, columns or features
    reference: Reference     # chance share | a named convention | the field's range; never pass/fail
    tier: Callable           # T1 | T2 | T3, from the thread's leash rung and the measure
    views: tuple[ViewSpec, ...]  # closed vocabulary; the layout follows the footprint
    context: tuple[str, ...] # structural variables it may split by (never the outcome)
    thread: str              # the QUESTION_KEY it returns at (the finding's routes_to)
    claim: Callable          # ≤ 20 words, about this table
    why: str                 # ≤ 22 words
    width: WidthRule         # per column | per-feature distribution + strip | clusters
    prefer: tuple[Soft, ...] # weighted view preferences
    sources: tuple[str, ...]
```

**Hard rules.** The engine enforces these for every lens, and no pack can loosen them:
- An O3 look is never ranked or highlighted.
- Under inference, an O3 look stays behind the outcome door.
- The outcome is never a context variable.
- A look whose `needs` are unsettled is not shown. Its reading is asked at Data instead (§14.2).
- No pass/fail stamps, which the hard stop in DOMAIN_SCIENCE §01.2 already forbids.
- No adult reference bands applied to children (`detectors/plausibility.py` already enforces this).

**Soft preferences.** Each pack declares these with weights, in Draco's style. Examples:
- a log axis for right-skewed intakes;
- the zero spike drawn as its own bar for foods many people never eat;
- metabolomics pairs ordered by run order.

These are declarative, so a domain expert can review them line by line. That gives the per-domain expert review packets a concrete object to vet.

**Reuse from the findings stage.** Findings already carry the fields a look needs:

| Finding field | Look field |
|---|---|
| `summary` (≤ 20 words) | the claim |
| `routes_to` | the thread |
| `lever_label` | the "Affects …" line |
| `group` (the pager key) | paging among same-kind items inside one look |

So most existing pack findings become looks by adding a group, a measure with a reference, and views. The engine also supplies **generic looks** for every lens: each column's shape and blank share (the index), the missing pattern, low variance, collinear pairs, p ≫ n, quality by group, fill flags, the outcome's own distribution, and the batch × outcome verdict.

**Purpose registry.** Every look declares which of the five pedagogical questions it answers (BLUEPRINT §11.2). The pedagogy reviewer audits looks the way it audits view kinds.

---

## 3 · Ranking: what "worth a look" means

### 3.1 The definition

**A look is worth a look when the data differ from what was expected and the difference would change the plan.**

This combines two research results:
- SeeDB: a view is interesting when it "displays large deviations from some reference".
- Profiler: "determining what constitutes an error is context-dependent and so requires human judgment."

The app measures the deviation. The plan supplies the consequence. The researcher judges.

### 3.2 How it is measured

Each look returns four things. The engine's drift reading (`detectors/assay.py`), which "states the share expected by chance beside the observed one", is the template.

- **Measure.** Deterministic and replayable.
- **Reference.** One of: the share expected by chance; a named convention; the field's usual range; or the user's own codebook range. It is always shown beside the measure.
- **Reach.** Which declared roles the look touches, and what share of rows or features it affects.
- **Thread.** The decision it returns at.

```
rank key = (tier, reach, share, excess, column order)

tier    T1  blocker: nothing downstream is valid until it is answered
        T2  changes a number a reviewer would challenge (it routes to a lever)
        T3  shapes the interpretation (it becomes a caveat in the manuscript)
        - taken from the thread's leash rung for this purpose (BLUEPRINT §11.3)
        - a look whose thread is already answered drops one tier and reads "Settled"
reach   3  touches the exposure (or the outcome, for O1/O2 looks only)
        2  touches the adjustment set or the declared predictors
        1  touches other analyzed columns
        0  touches unused columns
share   the share of rows, or of features, affected
excess  the measure beyond its reference, mapped to 0–1 by the look's contract
```

Each comparison can be explained in words. That explanation is the item's **"Why is this here?"** line (at most 22 words), for example: "Changes a number (the energy model) · touches your exposure · every row."

### 3.3 How many highlights, and how they are shown

- **At most three highlights** across all groups. This is BLUEPRINT §11 rule 7: "≤ 3 pushed findings, the rest counted and typed".
- **At most two highlights from any one group.**
- **Looks that tell one story merge.** If two looks share columns and a thread, the second becomes the first's context view. CompassQL warns that redundant suggestions add charts "without providing additional value".
- **When fewer than three things stand out**, the card says so: "Two things are worth a look", or "Nothing stands out against its reference. Each group is listed below." The primary action then reads "Continue".
- **Inside a group**, items follow the same key. The top three are shown, then "and N more", which follows Voyager 2's cap per category.
- **Inside a Strip**, the top 12 are shown, then "and N more".
- **Highlights are drawn as numbered lines in the card**, each a claim plus "Affects …". The same number appears beside the column in the index, so the list and the data stay linked.

### 3.4 How not to anchor people on the recommendations

Recommendations shape what people keep: in Voyager, 69% of bookmarked charts included a variable that the recommender had added. Free browsing also produces false findings: in Zgraggen et al. 2018, over 60% of reported insights were false. The design guards against both:

1. **The overview comes first.** The whole table is shown in column order, and the highlights are marked inside it, not instead of it.
2. **The order is stable.** The group order never changes. The index never re-sorts. Ranks are recomputed only when the data change, never while the user browses.
3. **Every highlight shows its measure, its reference and its reason.** There is never a pass/fail stamp.
4. **The outcome is never a ranking input.** O3 is never highlighted, under either purpose. This avoids the Lux and Sweetviz habit of ranking by correlation with the target.
5. **Silence is stated.** Each group says what it checked, for example "27 columns: no reading beyond its reference", so a missing highlight is not mistaken for something unexamined.
6. **One step at a time.** A context view adds one structural variable, never a combinatorial gallery. Voyager "looks ahead by only one variable at a time".
7. **The influence is auditable.** The IDA supplement lists what the app highlighted and what the user opened (§4.5).
8. **"Why does this matter?" teaches the reference, not the conclusion.** It says what a detection limit is. It does not say "your data have a problem".

---

## 4 · The drill-down path: overview → group → look → lever → decision

```
Overview ──click a group──▶ Group strip ──click / ↑↓──▶ Look ──"Decide now"──▶ Question card ──Record──▶ Look (Decided)
    ▲                             ▲                       ▲                          │
    └──────────── Esc ────────────┴────────── Esc ────────┴───────── Esc ────────────┘
```

### 4.1 Group

- Clicking a group in the card fills the canvas with that group's Strip: every item ranked by the group's measure, with the first one focused large.
- Examples:
  - **What is missing:** blank share per column, ranked, with the focused column's missing pattern.
  - **Variables together:** pairs ranked by |ρ|. At 30 columns or fewer, a clustered matrix with the values printed.

### 4.2 Look

A look's layout comes from its footprint, the same grammar as FOUNDATION §5, applied to what the look touches rather than to what an option changes:

| The look touches | Layout | Example |
|---|---|---|
| one column | Focus | energy by sex, with each screen's cuts marked |
| several columns or features | Strip | nutrients ranked by their correlation with energy |
| which rows | Flow | who a screen would remove, and how they differ from who stays |
| roles or structure | Routing | the 8 pooled QCs routed out of the cohort, kept as reference rows |
| competing conventions declared by a contract | Angles | QC RSD at 20, 25 and 30%, with the D-ratio beside it |

- The **context view** shows the same measure split by the declared context variable where it differs most. This is SeeDB's reference comparison and Profiler's coordinated view.
- If no context variable changes the picture, the canvas says so in one line, for example "Similar for women and men", and draws nothing.

### 4.3 Lever

- "Decide now" appears when the look's thread can be asked now: its `NEEDS` are met, and no unanswered earlier question that it depends on is open, per the contract relations. Primary action: "Next" stays primary; "Decide now" is a quiet link inside the item, unless the item is T1.
- Otherwise the item reads, in plain text, "Comes up at Energy". When the user reaches that question, its card offers a quiet "You looked at this" link that brings the look back as a secondary view.

### 4.4 Decision

The decision uses the existing mechanic:
- The card becomes that question's card. Its header names it: "Participants · asked from First look".
- Pointing at an option, or moving through options with the arrow keys, calls `POST /preview`. The flip ("Your data now ⇄ With this choice") and the storyboard appear.
- Indigo marks exactly what the pointed option touches.
- Record writes the sentence into the manuscript, marked green.
- Esc returns to the guide. The item now reads "Decided: Willett's range" in plain text.

### 4.5 After a decision, and saving

- **Changed since you looked.**
  - Any recorded decision that changes data (an exclusion, a repair, a transform) re-measures the looks on the affected columns.
  - A look that changed returns at the top of its group with its before → after, drawn by the existing generic diff (`diff_views`). This is Lux's history-based recommendation.
  - It becomes a highlight only if it now crosses its reference.
- **Save.**
  - Every view already has Save (`PreviewGrid.tsx`).
  - Save gains "Add to the supplement".
  - The supplement is part of the export: the IDA paragraph (§6.5) plus the saved figures, each captioned with its basis.
  - It is not a growing wall on the canvas. A Data Formulator 2 participant put it as: "I don't like to pollute my workspace".

---

## 5 · Wide data: aggregate first, then features

**The one rule:**
- A view about columns becomes one distribution of a per-feature statistic, plus a ranked Strip of the extreme features (top 12, then "and N more").
- A view about samples stays usable at any width.
- A feature × feature matrix becomes clusters, or a sample × sample view.

| Width | Per-column looks | Feature × feature | Missing | Engine limits already in place |
|---|---|---|---|---|
| up to ~30 | every column in the index | clustered matrix with values printed | full pattern | — |
| ~30–500 | Strip: top 12 by the look's measure | clustered matrix, no values, clusters named | top 10–15 patterns | `MAX_SHOWN = 12`, `MAX_PAIRS_P = 500` (explore.py) |
| ~500–4,000 | distribution of the statistic + top-12 Strip | redundancy clusters and an effective number of quantities; never the matrix | blank share per sample + per-feature histogram | `MAX_CLUSTERED = 4000` (assay.py) |
| 4,000 and above (p ≫ n) | the same | sample × sample, PCA scores, modules | the same | Arrow column-wise summaries |

- **The index at width:**
  - one row per metadata column;
  - one row per confirmed family, shown as a block, for example "392 features mz_0001 … mz_0392", carrying three tiny per-feature distributions (blank share, the lens's spread measure, the lens's drift or quality measure);
  - one samples row: total signal per sample by run order.
- **Families** are the ones confirmed as a block at Data (BLUEPRINT §14.2).
- **Rows scale separately.** Above ~5,000 rows, scatters become hexbins and relationships become quantile-binned. The `relationship` view caps at 800 points.
- **Search replaces lists at width.** The outcome group (§6) offers "Look at one feature…" instead of listing 20,000 curves. The engine's Explore today would list "the first 12 genes".
- **Cost.** Per-feature statistics are computed in one column-wise pass when the stage runs, and cached for that data state. Pointing and morphing read cached numbers. Nothing recomputes on hover.
- **Never drawn:**
  - a feature × feature matrix above ~500 features;
  - a heatmap of the top-k features chosen by the test being illustrated. METABOLOMICS_PACK §06.2 calls this "circular": it is an after-the-lock display, never a look.

---

## 6 · Inference versus prediction

### 6.1 How much a look involves the outcome

| Class | What it shows | Prediction | Inference, before the lock | Can it be highlighted? |
|---|---|---|---|---|
| **O0** outcome-free | data, structure, quality, predictor–predictor | free | free | yes |
| **O1** the outcome alone | its distribution, missingness, event count | after the split, on training rows; recorded | allowed; recorded | no; listed in its group |
| **O2** outcome × design | batch × outcome, total signal vs outcome | the verdict is computed as a gate and shown as a sentence; opening the table is recorded | the same | the verdict can be, as a T1 |
| **O3** outcome × exposure or predictor | binned curves, Table 1 by outcome, PCA colored by outcome group | training rows; recorded; levers offered in-fold first | **behind a counted door, never pointed at or ranked; recorded** | never |
| **O4** outcome-model estimate | Table 2, volcano, PLS-DA, sensitivity forests | model evaluation | after the lock only (FOUNDATION §5 rule 6) | not a look |

**Roles decide the class.** A view is O3 when it pairs the declared outcome with the declared exposure or predictors. In metabolomics this flips with the plan:
- When metabolites are the outcome (diet → metabolome), a PCA colored by diet group is O3.
- When metabolites predict a clinical outcome, a PCA colored by that outcome is O3.

### 6.2 The STRATOS boundary

Heinze et al. 2024 write that IDA "should – without good reason – not anticipate analysis directly related to the research question, implying that associations between outcome and predictors are not explored, neither numerically nor graphically."

The outcome door reconciles that line with MODELING_SEQUENCE §4 ("record and disclose; never block"):
- Under inference the door is never on the default path and is never pointed at.
- It is open to the user who chooses it.
- Every view through it is recorded and stated in the methods.

Two of the packs' signature exhibits are O4 and belong after the lock, in "Which of my decisions mattered?":
- nutrition's exclusion-sensitivity table;
- metabolomics' "significant list under method A vs B".

### 6.3 Which rows each look reads

| Look | Rows | Why |
|---|---|---|
| O0 (pre-seal) | every row | descriptive scope (BLUEPRINT §13). Its levers are fitted in-fold, as the D-ratio, ComBat and variance filters already are. A fixed, outcome-free rule chosen by hand carries no outcome information. |
| O1 and O3 under prediction | training rows, after the split | MODELING_SEQUENCE ruling 3; held-out rows are never read |
| O1 and O3 under inference | every analyzed row | the seal is purpose-scoped (BLUEPRINT §12 ruling 3) |
| O2 verdicts | every row | a structural guard, as the findings stage computes it today |

**One fix follows.** The `/columns` and histogram routes ignore the seal. The index must never draw the outcome column from them. It shows "The outcome · opens in its group" instead.

### 6.4 What is recorded

- **Opening an O0 look** goes to a looked-at ledger. It is not a decision and does not appear in the methods text. It drives the seen marks and the supplement.
- **Opening an O1 or O3 look** records the existing `view_outcome` decision. It folds to one entry per column, and keeps the lever answers the user had at the first look.
- **A lever changed after an outcome view** becomes a hand lever (`hand_levers`, explore.py:190-217):
  - under inference it is stated as a forking path;
  - under prediction it is stated as outside the corrected score.
- **Outcome views do not lock the plan.** Estimates do (`plan_lock.py`). This keeps the engine's two tiers.

### 6.5 How forking paths are disclosed in the manuscript

**The IDA paragraph (new; every journey):**

> "Before the analysis plan was set, the data were screened following the STRATOS initial-data-analysis framework: participants, missing values, each variable, variables together, and variation by survey cycle. Associations between the outcome and other variables were not examined (Heinze et al. 2024). Screens are shown in the supplement."

**If the door was opened under inference,** the existing sentences carry the disclosure:
- `view_sentence`: "The outcome's relationship with sugar was viewed in Explore on the 21,849 analyzed rows and recorded as looked at (forking paths: a choice made after it is disclosed)."
- `hand_levers`: "…was changed from … to … after its relationship with the outcome was viewed (forking paths)."
- The IDA paragraph's second sentence then reads: "The outcome's distribution and its relationship with sugar were viewed before the plan was locked; see Methods."

**Under prediction,** the existing `explore_sentence` already says: "Explore read the N training rows only (the M held-out rows were never read), and every lever was offered first as an in-fold rule the resampling repeats; …". The evaluation stage adds that the corrected score does not include the optimism of hand-set levers.

Huebner et al. 2020 found IDA usually goes unreported. The supplement closes that gap.

---

## 7 · Worked examples

### 7.1 Dietary: NHANES, 29 columns, inference

**The data.** This is the repo's 29-column extract (`_tt_tmp_nhanes.csv`): 21,849 adults aged 18–85, nine cycles from 2001 to 2017 pooled, with no weights, strata or PSUs.

**The assumed plan.** Exposure `sugar`, outcome `triglycerides`, roles settled at Data. The plan is an assumption; every number below was computed on the file.

**Screen 1 · Arrival** (the wireframe in §1.2)

- The card shows three highlights:
  1. **Sugar rises with energy.** "Affects the energy model." Group: Variables together. T2. Touches the exposure.
  2. **1,614 adults (7.4%) report energy outside Willett's range** (500–3,500 kcal for women, 800–4,000 for men). "Affects who is analyzed." Group: Each variable. T2. Every row.
  3. **Medication answers are blank for 71% (`meds_hbp`) and 79% (`meds_chol`), falling with age.** "Affects the missing values." Group: Missing. T2. Touches likely confounders.
- **Who is in the data** lists "No survey weights, strata or PSUs: estimates describe these 21,849 adults". It reads "Settled", because the survey question was answered at Data.
- **Each variable** says "kcal is in kilocalories: Atwater ratio 1.01 (5th–95th 0.84–1.04)". This is listed and not highlighted, because it sits within its reference.
- The canvas index shows all 29 columns. The `triglycerides` row is not drawn: "The outcome · opens in its group".

**Screen 2 · Highlight 1, energy dependence** (Strip)

- **Focal view:** the nutrient family ranked by rank correlation with kcal: fat_total 0.87, carb 0.86, fat_mon 0.84, fat_sat 0.80, protein 0.78, fat_poly 0.73, sugar 0.64.
  - `sugar` is focused because it is the exposure.
  - The large view is sugar against kcal, hexbinned for 21,849 rows.
- **Card:** "Measured 0.64 · Reference: nutrients that travel with energy exceed 0.3".
- **Why is this here?** "Your exposure carries energy; how you adjust changes its estimate."
- **Thread:** the item reads "Comes up at Energy". The energy question needs the proposals and is asked in sequence.

**Screen 3 · Highlight 2, implausible energy** (Flow)

- **Focal view:** the flow, 21,849 → 20,235, with the screen's step lit. Then who leaves: energy by gender, with the Willett cuts and the NHS/HPFS cuts (men to 4,200) both marked. Then how they differ from who stays: age, gender, BMI.
  - The outcome is left out of this comparison, because the plan is not yet locked.
- **"Decide now"** is available, since Participants is the next stage. The card becomes the exclusions question, with options as that question already serves them. Pointing at each option morphs the flow. Indigo marks the step the option adds.
- **Record** writes the exclusion sentence (green in the manuscript). Esc returns, and the item reads "Decided".

**Screen 4 · Highlight 3, medication blanks** (Focus)

- **Focal view:** the missing pattern of `meds_hbp` and `meds_chol`.
- **Context view:** blank share by age band, chosen because it differs most there: 94% under 40, 69% at 40–60, 43% over 60.
- **Card:** "Blank share falls with age: consistent with a question asked only of people told they had the condition." The app does not settle the reading. It asks, one tap with the evidence beside it (BLUEPRINT §14.2): "Was this asked only of people told they had the condition?"
- **Thread:** "Comes up at Missing values". Under inference, complete cases would arm the row-loss concern there.

**Screen 5 · The user opens Variables together** (group Strip, then a matrix)

- **Pairs ranked:** fat_total–fat_mon 0.97, fat_total–fat_sat 0.92, bmi–waist 0.90.
- **Focal view:** a clustered matrix of the seven nutrients with values printed. This is the new `matrix` view.
- **Thread:** under inference, "answer it in the adjustment set". There is no lever here; this is Explore's existing `collinear` routing.
- **Not highlighted:** closure. Protein, carbohydrate and fat add up to energy, so an all-parts model is singular. That would be T1 only if the plan models all parts with energy. Here it is T3.

**Screen 6 · The outcome door**

- **Card:** "The outcome · 0 viewed. Looking at how triglycerides move with other columns before the plan is set is a forking path. Each view is recorded and stated in the methods."
- **Inside the door:** "Its own distribution" (O1), and "With one column…" (O3). Columns are listed in plan order (exposure first), then column order. Nothing is ranked or amber.
- If the user opens sugar × triglycerides, the manuscript rail count rises and the `view_sentence` appears in green.

**Then:** "Continue to Participants". Over time and by batch still holds one unhighlighted item, T3 because blood pressure is not in this plan: "Blood-pressure fills range from 4.5% (2015) to 17% (2003) by cycle."

### 7.2 Metabolomics: untargeted, thousands of features, prediction

**The data.** This is the repo fixture `metabolomics_untargeted.csv`: 80 injections (72 participants, plus 8 pooled QCs at every tenth injection), two batches of 40, 392 features, and the outcome `responder`. At 4,000 features the screens are identical; only the counts change.

**Screen 1 · Arrival at width**

```
│ First look · before you plan      │ Your data now · 80 injections · 399 columns          │
│ 72 participants · 8 pooled QCs    │ sample_id     identifier                              │
│                                   │ sample_type   72 participant · 8 pooled_qc            │
│ Worth a look                      │ run_order     1–80, each once                         │
│ 1 Intensity tracks run order in   │ batch         B1 40 · B2 40                           │
│   64% of features; ~1% by chance  │ age, sex, bmi blank on QC rows, as expected           │
│   Affects drift correction        │ responder     the outcome · not available yet         │
│ 2 Blanks sit in the faintest      │ ───────────────────────────────────────────────────   │
│   features (rank corr. −0.99)     │ 392 features  mz_0001 … mz_0392 · one block            │
│   Affects how blanks are filled   │   blank share per feature   ▇▃▂▁▁  304 have any    2  │
│ 3 QC variation tops 20% in 60% of │   run-order |ρ| per feature ▁▂▅▇▃                  1  │
│   features, before correction     │   QC RSD per feature        ▂▅▇▄▂                  3  │
│   Affects the QC filters          │ ───────────────────────────────────────────────────   │
│ …six groups…                      │ injections: total signal by run order ▅▅▆▅▅ (QCs ◆)   │
│              [ Look at the first ]│                                                       │
```

- **Who is in the data:** "72 participants and 8 pooled QCs; the QCs are kept for quality checks, out of the model". This reads "Settled", from the sample-roles reading at Data.
- **Over time and by batch:** the O2 verdict, as a sentence: "Batch and responder are balanced (B1 20 / 16, B2 19 / 17)." The table is not drawn until it is opened.
- **Variables together:** "395 independent quantities among 396 features: none move together at 0.9." This fixture has no redundancy; real panels usually do (adducts, isotopes).

**Screen 2 · Highlight 1, run-order drift** (Strip)

- **Focal view:** the distribution of per-feature Spearman ρ with run order. The share expected by chance at 80 injections, about 0.7% beyond |ρ| 0.3, is drawn beside the observed share: 253 of 394 numeric columns (64%) pass and survive Benjamini–Hochberg.
  - Beneath it, the Strip of the 12 most drifting features, each a small trace over run order with the QCs marked and the batch boundary at injection 40.
- **Context view:** PCA scores. This is the new `embedding` view, on the 88 features with no blanks, log scale, standardized, display only.
  - PC1 (26%) and PC2 (3.6%), with the aspect proportional to variance.
  - QCs and batches drawn in comparison colors.
- **"More angles":** PC1 against run order, a relationship view. Rank correlation −0.99, with the QCs spread along it (0.87 of the participants' spread) instead of sitting at the center. METABOLOMICS_PACK §06.1 calls the PCA "the field's trust anchor"; here the anchor says the run was not stable.
- **"Decide now"** opens drift correction, which is reference-rows scope and runs before the seal.
  - QC-RLSC is labeled **"Not available yet"**, with its one-line reason. Each batch has 4 pooled QCs and the engine needs 5. Injections 32–40 and 72–80 fall after their batch's last QC, and the engine never extrapolates (`methods/qc_drift.py`).
  - The other options on the menu stay available, each previewed. One example is batch as a model term.
  - This is the leash at work: refused, with an exit.

**Screen 3 · Highlight 2, left-censoring** (Focus)

- **Focal view:** one point per feature, plotting blank share against log mean intensity. It is binned above 800 features.
  - Reference line: "If blanks were random: flat."
  - Measured: −0.99.
- **Context view:** the same by batch, or the line "Similar in both batches".
- **Thread:** "Comes up at Missing values". Detection-limit methods rank first there.
- **Card:** "The plain 80% rule would drop 118 features." The modified 80% rule reads the responder labels, so its cost appears only as that lever's preview on training rows after the split. It is O3 by construction.

**Screen 4 · Highlight 3, the QC filters** (Angles)

- **Panel "How many features does each RSD cut keep?"** Computed before drift correction:

  | RSD cut | Features dropped (of 391) | Kept |
  |---|---|---|
  | 20% (the engine's LC-MS cut) | 235 | 156 |
  | 25% | 198 | 193 |
  | 30% | 161 | 230 |

- **Panel "Is the noise small next to the biology?"** The D-ratio is over 50% in 104 features (27%). It is shown descriptively here. The filter itself runs in each training fold.
- The option × question table follows the pointer.
- **Card:** "RSD before correction penalizes drift that could be corrected; once drift is answered, both are reported" (METABOLOMICS_PACK §02).
- **Thread:** "Comes up after drift correction". The contract relation orders it.

**Screen 5 · The outcome group** (prediction)

- **Before the split:** "Not available yet · opens once the split is recorded."
- **After the split** (resampling at n = 72), reading training rows:
  - "responder: 33 of 72 (46%)". The class labels are shown, which fixes today's unlabeled class index. There is no imbalance lever at 46%.
  - "Look at one feature…" (a search).
  - "PCA colored by responder" (O3).
- Each view is recorded. Any lever pulled after a view is disclosed as outside the corrected score.

**Screen 6 · Continue to Participants.** The flow shows the 8 QCs on a line of their own, as reference rows.

---

## 8 · What the engine already serves, and what is new

| Piece | Served today | New work |
|---|---|---|
| Outcome-free readings, pre-seal, all rows | `findings` stage with the lens packs (findings.py) | a group key per finding; the rank key (today the sort is severity, confidence, then id, findings.py:225-232) |
| Finding fields | `summary`, `routes_to`, `lever_label`, `group` | reused as claim, thread, "Affects …", paging |
| Chance-referenced measures | drift with its chance share; redundancy with leave-one-out (assay.py) | a measure and reference on every look |
| Explore stage | findings, levers, `viewed`, `hand_levers`, sentence (explore.py); not routed (interview.py `QUESTION_KEYS`; INBOX:226) and not rendered (Record.tsx) | render it. Move its outcome-free kinds (low variance, wide, collinear, quality by group) into a pre-seal `look` stage. Keep its outcome views for the outcome group. Stop leading with 12 curves in column order (explore.py:404-418). Label classes. |
| Outcome-view recording and disclosure | `view_outcome`, `view_sentence`, `hand_levers`, the Explore and evaluation sentences | reused as they are |
| Lever previews | `explore_previews.py` (levers, selection, intended use, updating) | reused by "Decide now" |
| View vocabulary, grid, save | 5 kinds; `PreviewGrid` (1 primary + 2); Save on every card | `embedding` (reserved in consequences.py:28) and `matrix`, each needing the design decision the docstring asks for. The index, which reuses the Strip row. |
| Layout by footprint | none (no footprint anywhere in the code) | a look's footprint → Focus / Strip / Flow / Routing / Angles |
| Generic diff | `diff_views` ranks changes | reused for "Changed since you looked" |
| Coach notes | exclusions, missing values, energy, aggregation (coach.py:468-471) | one note per look, anchored to the noticing |
| Wide aggregates | Arrow column summaries; `MAX_CLUSTERED`, `MAX_PAIRS_P` | per-feature distribution + Strip views; the samples row |
| Pack contract | `Pack` (detectors, priors, reframings, hedges, recipes) | `Pack.looks` with the Look contract; per-lens review packets |
| Chain and Router | 15 stages fetched; no Explore stop | the "First look" stop; the walk; mastery |
| Ledger and export | export bundle with methods | the looked-at ledger; the IDA paragraph; the supplement |
| Seal | the `/columns` routes ignore the seal | the index never draws the outcome |
| Lens readings missing (domain review) | — | dietary λ strip, Kish n_eff, day-2 completer SMD, consecutive days, food-source clusters; metabolomics QC RSD and D-ratio as looks; PCA |

---

## 9 · Open questions for Nolan

1. **The outcome door under inference.**
   - My recommendation: a counted, recorded door inside First look, never pointed at.
   - The strict alternative: hide outcome views entirely until the plan locks, as STRATOS reads literally.
   - The door keeps MODELING_SEQUENCE's "never block". Hiding is cleaner for a reviewer.
2. **Walk or browse on the first visit.**
   - My recommendation: on the first visit the primary button walks the highlights ("Look at the first → Next → Continue"), and on a revisit it says "Continue".
   - The alternative: "Continue" always, with highlights as optional reading.
   - The walk guarantees the three most consequential looks are seen. Browsing respects experts from the first visit.
3. **Two new view kinds.** Do you approve adding `embedding` (PCA scores, the sample map) and `matrix` (clustered correlation, the PC × covariate grid, sample × sample) to the closed vocabulary? Metabolomics and genomics have no honest first look without them, and consequences.py says nothing joins the vocabulary without a design decision.


## Sources
- /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/calm/FOUNDATION.md (§1 test, §2 calm budget, §3 page zones, §4 color by role, §5 canvas grammar)
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/BLUEPRINT.md §11 (rules 1-9), §11.1, §11.2, §11.3, §11.4, §12 ruling 3, §13 method contract, §14.1-14.3
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/MODELING_SEQUENCE.md §0 ruling 3, §3, §4
- /Users/nhedglin/tabular-ml-lab/docs/turbotab-next/INBOX.md:226
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/explore.py (explore_stage 368-430; hand_levers 190-217; explore_sentence 226-260; view_sentence 794-814)
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/findings.py (fields 1-25; ranking 225-249)
- /Users/nhedglin/tabular-ml-lab/turbotab/core/consequences.py (docstring 1-48; embedding reserved at 28; MAX_VIEWS)
- /Users/nhedglin/tabular-ml-lab/turbotab/core/explore_previews.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/assay.py (drift chance share; redundancy; MAX_CLUSTERED)
- /Users/nhedglin/tabular-ml-lab/turbotab/core/methods/qc_drift.py (QC-RLSC MIN_QC=5, no extrapolation; QC RSD; D-ratio training-fold scope)
- /Users/nhedglin/tabular-ml-lab/turbotab/core/contracts.py (MethodContract 112-150)
- /Users/nhedglin/tabular-ml-lab/turbotab/core/interview.py (QUESTION_KEYS 99-106; NEEDS)
- /Users/nhedglin/tabular-ml-lab/turbotab/packs.py (Pack 5110-5136; metabolomics looks_for 5241-5290; lens looks_for ids)
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/components/stage/PreviewGrid.tsx
- /Users/nhedglin/tabular-ml-lab/turbotab/frontend/src/state/focus.tsx
- /Users/nhedglin/tabular-ml-lab/docs/turbotab/research/METABOLOMICS_PACK.md §02 (QC diagnostics, order of operations), §06 (two-tier EDA, PCA trust anchor, circular heatmap)
- /Users/nhedglin/tabular-ml-lab/turbotab/sample_data/metabolomics_untargeted.csv and .md (worked example 7.2; summaries computed read-only)
- /Users/nhedglin/tabular-ml-lab/_tt_tmp_nhanes.csv (29-column NHANES extract; worked example 7.1; summaries computed read-only)
- Engine inventory, research and domain reports supplied by the workflow (ENGINE, RESEARCH, DOMAINS)
- Heinze G, Baillie M, Lusa L, Sauerbrei W, Schmidt CO, Harrell FE, Huebner M; STRATOS TG2/TG3. Regression without regrets: initial data analysis is a prerequisite for multivariable regression. BMC Med Res Methodol 2024;24:178
- Huebner M, le Cessie S, Schmidt CO, Vach W. A contemporary conceptual framework for initial data analysis. Observational Studies 2018;4:171-192
- Huebner M, Vach W, le Cessie S, Schmidt CO, Lusa L. Hidden analyzes: a review of reporting practice and recommendations for more transparent reporting of initial data analyzes. BMC Med Res Methodol 2020;20:61
- Zgraggen E, Zhao Z, Zeleznik R, Kraska T. Investigating the effect of the multiple comparisons problem in visual analysis. CHI 2018
- Wongsuphasawat K, Moritz D, Anand A, Mackinlay J, Howe B, Heer J. Voyager: exploratory analysis via faceted browsing of visualization recommendations. IEEE TVCG 2016 (InfoVis 2015)
- Wongsuphasawat K, Qu Z, Moritz D, et al. Voyager 2: augmenting visual analysis with partial view specifications. CHI 2017
- Wongsuphasawat K, Moritz D, Anand A, Mackinlay J, Howe B, Heer J. Towards a general-purpose query language for visualization recommendation (CompassQL). HILDA 2016
- Mackinlay J, Hanrahan P, Stolte C. Show Me: automatic presentation for visual analysis. IEEE TVCG 2007
- Moritz D, Wang C, Nelson GL, et al. Formalizing visualization design knowledge as constraints (Draco). IEEE TVCG 2019
- Lin H, Moritz D, Heer J. Dziban: balancing agency and automation in visualization design via anchored recommendations. CHI 2020
- Lee DJL, Tang D, Agarwal K, et al. Lux: always-on visualization recommendations for exploratory dataframe workflows. PVLDB 15(3), 2021
- Wang C, et al. Data Formulator 2: iterative creation of data visualizations with AI. CHI 2025
- Wilkinson L, Anand A, Grossman R. Graph-theoretic scagnostics. InfoVis 2005
- Vartak M, Rahman S, Madden S, Parameswaran A, Polyzotis N. SeeDB: efficient data-driven visualization recommendations. PVLDB 2015
- Demiralp C, Haas PJ, Parthasarathy S, Pedapati T. Foresight: recommending visual insights. PVLDB 2017
- Kandel S, Parikh R, Paepcke A, Hellerstein JM, Heer J. Profiler: integrated statistical analysis and visualization for data quality assessment. AVI 2012
- Willett W. Nutritional Epidemiology (energy-intake plausibility ranges 500-3,500 / 800-4,000 kcal)
- Broadhurst D, et al. Guidelines and considerations for the use of system suitability and quality control samples in mass spectrometry assays. Metabolomics 2018;14:72
- Dunn WB, et al. Procedures for large-scale metabolic profiling of serum and plasma using GC and LC-MS. Nat Protoc 2011;6:1060