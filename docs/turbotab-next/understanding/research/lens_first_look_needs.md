## How to guide exploration without overwhelming people: the research and what it means for TurboTab's EDA phase

### Bottom line

Nolan's idea holds up: one dynamic canvas with views grouped sensibly. The literature adds three things to it.

1. **Group by the question being asked of the data, not by chart type.** STRATOS TG3 already defines the grouping: participants, missing values, each variable, and variables together. A fifth group, over time, applies when there are repeated measures. TG3 also defines the organizing axis, "structural variables", which keeps the number of multivariable views small. Each lens adds its checks inside these groups rather than adding new tabs, so the structure stays the same across all five lenses and only the contents differ.
2. **"Interesting" should mean "differs from what we expected, and would change the plan".** It should not mean "striking". Rank each group by that measure, show the top few, and keep everything else reachable in order. This is the canvas rule TurboTab already follows ("largest measured change gets the most room"), applied to findings instead of options.
3. **Separate looking from deciding, and keep looking blind to the outcome under inference.** STRATOS says initial data analysis (IDA) "does not touch the research question". Recommenders that rank by correlation with a target (Lux's Correlation action, Sweetviz's target analysis) do exactly what IDA forbids. Record every view automatically, because people are bad at telling a real pattern from noise: in Zgraggen et al., over 60% of reported insights were false.

A sketch inside the approved zones (FOUNDATION §3):

```
CHAIN   Data · Participants · [Look at your data] · Exposure · Confounders · ...
CARD (reading list)                      CANVAS (at most 3 views)
Look at your data · before you plan      1 Overview strip: every column, one row each
                                           (tiny distribution, % missing, a mark if a
Who is in the data          1 to look at   finding attaches, a tick once looked at)
What is missing             2            2 Focal view for the selected finding
Each variable               4              (Focus / Strip / Flow, chosen by footprint)
Variables together          1            3 Context: the same rows by a structural
(Over time, if repeated measures)          variable (sex, age band, wave, batch, stratum)
Looks at the outcome (recorded) >
Why does this matter?
                         Continue
```

- An item reads as one sentence tied to a lever, for example: "Energy intake: 31 recalls under 500 kcal. Affects the energy model." The arrow keys rifle through items and the canvas morphs.
- A finding that needs an answer opens the familiar one-question card ("Decide"); Esc returns to the list. You browse to find things and use Q&A to decide on them.

---

### (a) Visualization recommenders

**Voyager (Wongsuphasawat, Moritz, Anand, Mackinlay, Howe, Heer; InfoVis 2015, TVCG 2016)**

What it says:
- On overload: "The interface should not overwhelm users, yet must enable them to rapidly browse collections of visualizations with minimal cognitive load."
- C1: "Show data variation, not design variation… To discourage premature fixation and avoid the problem of empty results, Voyager shows univariate summaries of all variables prior to user interaction… To help users stay oriented, avoid combinatorial explosion, and reduce the risk of irrelevant displays, Voyager currently looks ahead by only one variable at a time."
- C4, on grouping: "Voyager organizes suggested charts by clustering encoding variations of the same data and showing a single top-ranked exemplar of each cluster… [and] partitions the main gallery into a section that involves only user-selected variables and a section that includes additional (non-selected) variables recommended by the system."
- On ranking: "Compass ranks the generated encodings using perceptual effectiveness metrics." Variable sets are deliberately ordered by type and name: "This approach provides a predictable and consistent ordering."

What the study found:
- "participants were exposed to over 3 times more variable sets and interacted with 1.5 times more when using Voyager."
- "Of the 179 total visualizations bookmarked in Voyager, 124 (69%) include a data variable automatically added by the recommendation engine." Recommendations shape what people end up keeping.
- "Subjects roundly preferred Voyager for exploration (15/16, 94%) and PoleStar for question answering (15/16, 94%)", and "All but one subject wished to use a hybrid of both tools."

What it means for TurboTab:
- Never open on a blank canvas. Start with every column's summary, which guards against premature fixation.
- Suggest one step at a time (this column, then this column by one structural variable).
- Keep "what you selected" separate from "what we suggest".
- Show one exemplar per item, with its alternatives behind "More angles".
- Keep the group order predictable.
- Combine browsing (to find things) with Q&A (to decide on them).

**Voyager 2 and CompassQL (CHI 2017; HILDA 2016)**

What they say:
- Voyager 2 abstract: "Visualization recommenders can encourage broad coverage, but irrelevant suggestions may distract users once they commit to specific questions."
- C8: "Presenting many similar recommendations may overwhelm or distract users… Voyager 2 groups suggestions into selectable categories, limits the default number of suggestions per category, and prunes the space of visual encodings to suggest distinct designs."
- "To avoid overwhelming users (C8), the system does not show related views when wildcards are in play."
- "Rather than starting with a blank screen, univariate summaries help users familiarize themselves with the different data fields."
- Results: 102 unique field sets seen versus 23 with the manual tool, and 41 interacted with versus 17.
- CompassQL formalizes ranking as group / choose / order: "Voyager groups visualizations backed by the same data query, applies an encoding-based metric to choose group representatives, and orders them using a data-based metric." It also warns: "Many candidate visualizations may be similar and thus redundant, increasing the number of charts the user has to consider without providing additional value."

What it means for TurboTab:
- Set a cap per group: the top 3 marked "worth a look", then "and N more".
- When a finding is open as a decision, hide the suggestions, since related views distract once the user is focused.
- Express each group's ranking as group by IDA domain, choose by view effectiveness (the closed vocabulary), and order by notability.

**Show Me (Mackinlay, Hanrahan, Stolte; InfoVis 2007)**

What it says:
- "Users typically start visual analysis with vague tasks in mind, which are refined and transformed as they see graphical views of data."
- "Selecting items from a long list can interrupt the flow of visual analysis."
- The Alternatives menu "contains only 14 items… items are only active when they can build appropriate charts."
- "a stable grid of choices with tooltips that describe the conditions for a choice to be active."
- "defaults to the highest ranked command whose condition is met… crafted to default Show Me to designs that embody best practices."

What it means for TurboTab:
- The view vocabulary is a stable, small grid, and the group order never moves between datasets or lenses.
- An item that does not apply still shows, inactive, with one line saying why. This matches the FOUNDATION label "Not available yet".
- The default view for each column type is chosen automatically, the way Show Me's Automatic Marks picks a mark type.

**Draco (Moritz et al.; InfoVis 2018, TVCG 2019)**

What it says:
- "We propose modeling visualization design knowledge as a collection of constraints, in conjunction with a method to learn weights for soft constraints from experimental data."
- "Draco can systematically enumerate the visualizations that do not violate the hard constraints and find the most preferred visualizations according to the soft constraints."
- "each violation of a soft constraint imposes a penalty (or cost) equal to its weight."

Related: Dziban (Lin, Moritz, Heer; CHI 2020) adds "anchored recommendations… perceptually similar to the anchor", because "Recommender systems are often forced to make decisions in the face of ambiguous user intent. Sometimes, these decisions will hamper exploration." This is visual continuity, not the cognitive anchoring discussed below.

What it means for TurboTab:
- Write each lens's "which view for which column" knowledge as declarative rules that domain experts can review.
- **Hard rules**, for example:
  - no outcome–predictor view in the default list under inference before the lock;
  - never judge children by adult bands, which `detectors/plausibility.py` already enforces.
- **Weighted soft preferences**, for example:
  - log axis for right-skewed intakes;
  - the zero spike as its own bar for foods many people never eat;
  - order pairs by run order for metabolomics.
- This gives the per-domain expert review packets a concrete object to vet.
- From Dziban: when moving between items, keep axes and encodings stable so the morph reads clearly.

**Lux (Lee et al.; PVLDB 15(3), 2021)**

What it says:
- "When users print a dataframe… Lux recommends visualizations to provide a quick overview of the patterns and trends and suggests promising analysis directions."
- "Visualizations are organized into sets called actions, displayed as tabs." The tabs are Correlation, Distribution, Occurrence, Temporal and Geographic, plus intent-driven Enhance and Filter.
- "the Correlation action plots pairwise relationships ranked by Pearson's correlation."
- "Lux displays history-based recommendations based on whether the dataframe has been filtered or aggregated in its recent history."
- An open problem the authors name: "surfacing the inferred implicit intent in a way that is interpretable and explains resulting recommendations choices."
- Field study: one participant who examines 100+ columns said "If not, it will take me forever to go through these many variables". Others wanted actions "similar to the default Lux actions, but with a different ranking". Custom actions are UDFs that fire "whenever the dataframe satisfies the user-specified condition."

What it means for TurboTab:
- Lens packs register items with an applicability condition, the way Lux's custom actions do.
- After a repair or transform, that column's before/after comes back automatically (history-based recommendation).
- Every item gets a one-line "Why is this here?" naming the expectation it deviates from and the lever it informs.
- Avoid Lux's default of ranking by correlation (see (c)).

**Data Formulator 1 and 2 (Wang et al.; VIS 2023; CHI 2025)**

What they say:
- Data Formulator 1 separates high-level intent ("concepts") from the data transformations an AI agent performs.
- Data Formulator 2: "DF2 lets users navigate their iteration history and reuse previous designs, eliminating the need to start from scratch each time."
- Data threads are a "tree-structured iteration history".
- One participant: "I don't like to pollute my workspace."

What it means for TurboTab:
- Pinning a plot (Nolan asked that any plot be saveable) goes to a small tray that feeds the IDA supplement. It does not go to a growing wall.
- History lives in the chain and the looked-at ledger, not on the canvas.

**Interestingness measures: scagnostics, SeeDB, Foresight, Profiler**

What they say:
- **Scagnostics** (Wilkinson, Anand, Grossman 2005) quotes Friedman and Stuetzle on Tukey: "scatterplot matrices lose their effectiveness when the number of variables is large… the computer could find the most interesting scatterplots to be presented to the user." It defines nine measures: Outlying, Skewed, Clumpy, Sparse, Striated, Convex, Skinny, Stringy, Monotonic. "unusual scatterplots could be identified from outliers in the scagnostic SPLOM."
- **SeeDB** (Vartak et al., PVLDB 2015): "a visualization is likely to be interesting if it displays large deviations from some reference (e.g. another dataset, historical data, or the rest of the data)."
- **Foresight** (Demiralp et al., PVLDB 2017):
  - "We define an insight as a strong manifestation of a distributional property of the data."
  - "Each carousel in the Foresight UI corresponds to a distinct class of insight. Visualizations within a carousel are ranked by the insight's ranking metric with the strongest insights displayed first."
  - On anchoring: "At any point during the EDA process, the user can step back and look at the overview visualization of an insight. This helps ensure that… the EDA process does not get inadvertently 'trapped' in some local 'neighborhood'."
- **Profiler** (Kandel et al., AVI 2012): "Automated methods can help identify anomalies, but determining what constitutes an error is context-dependent and so requires human judgment." Profiler "applies data mining methods to automatically flag problematic data and suggests coordinated summary visualizations for assessing the data in context."

What it means for TurboTab: notability = deviation from an expectation × consequence for the plan.
- **Expectations** come from metadata: codebook ranges, the NHANES reference bands in `plausibility.py`, lens packs, and settled readings. This is IDA Rule 4: metadata includes "expectations about distributional properties and associations".
- **Deviation** uses Foresight/scagnostics-style metrics: missing share, spike at zero, skew, extreme tails, redundancy (VIF), and gaps across structural strata.
- **Consequence** is the leash rung of the lever the finding informs.
- When an anomaly is focused, the second view shows who those rows are against the rest of the data (Profiler's coordinated views, SeeDB's reference). That is the existing Flow layout's "how they differ from who stays".
- The overview stays one keystroke away (Foresight).
- A flagged item becomes an answered finding, not a verdict.

**What studies found about overload and anchoring**

- **Anchoring.** Cho et al. (VAST 2017), the anchoring study: "Anchoring effect can be triggered by the order in which information is presented or the magnitude of information presented". Their interaction logs "reveal the impact of anchoring bias on the visual representation preferred and paths of analysis."
- **Trust.** Zehrung et al. (CHI 2021): "The relevance of presented information (e.g., the presence of certain data fields) was the most critical factor"; users were largely indifferent to whether a person or an algorithm made the recommendation.
- **Coverage awareness.**
  - Lumos (Narechania et al., TVCG 2022) targets cases where "users unknowingly overemphasize or underemphasize specific subsets of the data or attribute space". It "increases users' awareness".
  - Wall et al. (TVCG 2022) found effects on bias "inconclusive", with "mixed support that interaction traces, particularly in a summative format, can increase awareness."
- **Practitioners.** Alspaugh et al. (TVCG 2019) found "conflicting views about the role of intelligent tools in data exploration", and that some analysts treat "finding something interesting" as a goal while "others explicitly disavow this goal".
- **Caveat.** None of the recommender papers above measured anchoring directly. The 69% figure from Voyager shows that recommendations steer what people keep, which is not proof that they distort it.

What it means for TurboTab:
- Order is a strong lever, so rank by data quality and consequence, never by how striking a pattern looks or by association with the outcome.
- Keep the group order fixed.
- Show coverage quietly as a count and a tick in the overview strip.
- At Continue, add one summary sentence ("You looked at 14 of 52 columns; no outcome relationships"). This is calmer than Lumos-style in-place coloring and fits the finding that summaries work better.

---

### (b) Shneiderman's mantra and progressive disclosure

What the sources say:
- **Shneiderman (1996):** "Overview first, zoom and filter, then details-on-demand."
  - "Overview strategies include zoomed out views of each data type to see the entire collection plus an adjoining detail view."
  - "Smooth zooming helps users preserve their sense of position and context."
  - History: "keeping the history of actions and allowing users to retrace their steps is important. However, most prototypes fail to deal with this requirement."
- **Keim et al. (2008)** amend the mantra for large, complex data: "Analyze First - Show the Important - Zoom, Filter and Analyze Further - Details on Demand", because data sets are "too large to be visualized straightforward".
- **Baillie et al. (STRATOS, Rule 6)** adopt the original: "Well-designed visual reports provide an 'overview first, zoom and filter, then details-on-demand'."
- **NN/g on progressive disclosure:**
  - "Initially, show users only a few of the most important options."
  - "Offer a larger set of specialized options upon request."
  - The path to the secondary options needs strong "information scent".
  - Designs with more than two levels of disclosure tend to have poor usability. (The two-level point is the fetch tool's paraphrase.)

What it means for TurboTab, following Keim:
- **Analyze first.** The engine's readings, detectors and Explore stack compute the findings before anything is drawn.
- **Show the important.** The overview strip plus the top 3 items per group.
- **Zoom and filter.** Select a group, then an item, which opens the focal view and its structural-variable context.
- **Details on demand.** Hover, and "More angles".
- **Two levels, no more:**
  - level 1 is the group list with counts and top items;
  - level 2 is a group's full ranked list plus "More angles";
  - there is no third level.
- **History** is the looked-at ledger (see (d)).

---

### (c) STRATOS TG3: initial data analysis

What the sources say:
- **Huebner, le Cessie, Schmidt, Vach (Observational Studies 2018):**
  - "Key principles for IDA are to avoid analyzes that are part of the research question, and full documentation and transparency."
  - The six steps: metadata setup, data cleaning, data screening, initial data reporting, refining the analysis plan, and reporting IDA in papers.
- **Baillie, le Cessie, Schmidt, Lusa, Huebner ("Ten simple rules", PLOS Comput Biol 2022):**
  - "While EDA is a hypothesis-generating activity, IDA primarily ensures transparency and integrity of preconditions to conduct appropriate statistical analyzes… to answer predefined research questions."
  - Rule 5, "Avoid sneak peeks—IDA does not touch the research question": "Performing IDA to identify interesting patterns runs the risk of a data-driven selection of analyzes and methods, chance observations, which might lead to incorrect or inflated claims."
  - The same rule allows consequences: "skewed distributions may lead to applying a transformation, sparse multivariate distributions may identify the need to include or drop an interaction term, and certain observed missing data patterns may require more advanced methods… All changes to the SAP should be well motivated, and well documented."
  - Rule 1 warns that without a plan, IDA can "erode and turn into a cyclical investigation."
  - Rule 8 lists what an IDA report holds:
    - metadata;
    - a flow diagram;
    - a cleaning summary;
    - missingness;
    - univariate and multivariate distributions;
    - findings that affect interpretation;
    - findings that change the analysis plan.
- **Heinze, Baillie, Lusa, Sauerbrei, Schmidt, Harrell, Huebner (BMC Med Res Methodol 2024, TG2 and TG3):**
  - "a key principle of IDA is that it should – without good reason – not anticipate analysis directly related to the research question, implying that associations between outcome and predictors are not explored, neither numerically nor graphically. Nevertheless, the conduct of IDA is guided by the research question and the intended analyzes."
  - The checklist is grouped as: missing values M1–M4 (unit, item, complete cases, patterns "structured by structural variables"); univariate U1–U2 (categories; continuous variables with a "high-resolution histogram", key quantiles, the 5 highest and 5 lowest values, number of distinct values); and multivariate V1–V3 (each predictor against the structural variables; pairwise correlations in a matrix or heatmap; interaction pairs), with extensions for clustering and redundancy (VIF).
  - On scope: "we propose the following structured approach to limit the number of descriptions to be produced at this analysis stage. First, associations of each predictor with the structural variables should be evaluated graphically and numerically… Next, associations between all pairs predictors… could be restricted to numerical evaluation if the number of predictors is large."
  - "Revisions of the analysis strategy based on the results of IDA are justified if any predictor-outcome associations were strictly not evaluated during IDA."
- **Lusa et al. (PLOS ONE 2024, longitudinal):**
  - Structural variables "help to organize IDA results to provide a clear overview of data properties. In particular, this might reduce the number of conceivable multivariable explorations."
  - They "can be demographic variables, variables central to the research aim, or process variables (e.g., … centers…)."
  - It adds participation profiles and longitudinal aspects as domains.
  - "associations between the outcome variable and the explanatory variables are not evaluated. However, evaluating the changes of the outcome in time is part of the outcome assessment."
- **Huebner et al. ("Hidden analyzes", BMC Med Res Methodol 2020):**
  - IDA "is often 'hidden'… conducted in an unplanned and unstructured way."
  - Of 25 papers, 40% reported data cleaning, 44% item missingness, 60% unit missingness, and 44% changes to the plan.

What it means for TurboTab:
- **Grouping.** These are the canvas groups: Who is in the data (M1, flow), What is missing (M2–M4), Each variable (U1–U2), Variables together (V1–V3), and Over time when repeats are settled.
- **Organizing axis.** Structural variables bound the combinatorics. Ask once, proposing candidates from the readings: sex, age band, survey cycle or wave, center, batch or run order, stratum. Every multivariable view is then "X by structural variable": about p × s views instead of p². This answers "sensible grouping".
- **Outcome under inference.** Before the lock, the outcome's own distribution and its missingness are allowed (the U domain covers the outcome). Associations between the outcome and the predictors are not shown by default.
- **Plan changes.** Any change to the plan cites the finding that motivated it.
- **Reporting.** The export carries an IDA report with Rule 8's contents, closing the reporting gap Huebner 2020 measured.

**Tension to flag for Nolan.**
- MODELING_SEQUENCE §1 row 1 and `turbotab/core/stages/explore.py` include "the outcome's relationship with each continuous predictor" as a default Explore finding under both purposes ("recorded and disclosed… never blocked").
- STRATOS says not to explore these associations at all during IDA. Under inference, the exposure–outcome view is the research question itself.
- **Proposal: keep the leash ruling (never blocked) but change where these views sit.** Under inference before the lock, move them out of the ranked list into a collapsed, labeled group, "Looks at the outcome (recorded)", that records every opening. FOUNDATION §5 rule 6 would then extend in spirit from "no outcome-model estimate" to "no outcome–predictor association on the default canvas before the lock".
- Under prediction, the current design (training rows only, recorded, in-fold rules offered first) is sound EDA.
- Naming: the literature draws this line explicitly, so the chain label could say which activity is happening: "Look at your data" under inference, "Explore" under prediction.

---

### (d) Forking paths and how disclosure handles exploration

What the sources say:
- **Gelman and Loken (2013):**
  - "Researcher degrees of freedom can lead to a multiple comparisons problem, even in settings where researchers perform only a single analysis on their data… without the researcher having to perform any conscious procedure of fishing or examining multiple p-values."
  - "The researcher degrees of freedom do not feel like degrees of freedom because, conditional on the data, each choice appears to be deterministic."
  - Remedy: "analyze all relevant comparisons, not just focusing on whatever happens to be statistically significant", or run an exploratory study followed by a preregistered confirmatory one.
- **Zgraggen, Zhao, Zeleznik, Kraska (CHI 2018):**
  - "In our experiment, over 60% of user insights were false."
  - "Without any hypothesis testing, the false discovery rate averages over 73%."
  - Neither a background in statistics nor experience interpreting visualizations correlated with accuracy.
  - "without either confirming user insights on a validation dataset… or accounting for all comparisons made by users during exploration… we have no guarantees on the bounds of the expected number of false discoveries."
  - "burdening users to remember and code all of their insights during an exploration session is unfeasible. Could we create tools that automatically do this encoding while users explore a dataset?"
- **Pu and Kay (BELIV 2018):** "The vast majority of existing visualization systems have no correction for the forking paths problem." They lay out a design space with two axes: the type of correction (multiple-comparison correction, regularization, and so on) and how it is integrated visually (annotation versus data transformation).
- **Hullman and Gelman (HDSR 2021):**
  - "what is surprising is defined by the implicit or explicit model of our expectations."
  - "systems might enable specifying and explicitly comparing data to null and other reference distributions." In a lineup, if the analyst can pick the real plot out of N, that is a visual test with "type 1 error rate of 1/N".
  - The caution: "the additional cognitive load of interacting with reference distributions overwhelms some users."
- **Disclosure practice:**
  - Nosek et al. (PNAS 2018): "Presenting postdictions as predictions can increase the attractiveness and publishability of findings by falsely reducing uncertainty." For existing data, document "what was and was not known in advance about the dataset". "Preregistration with reported deviations provides substantially greater confidence."
  - Van den Akker et al. (Meta-Psychology 2021): "researchers' hypotheses and analyzes may be biased by their prior knowledge of the data."
  - Weston et al. (AMPPS 2019): secondary data can serve exploratory or confirmatory work if transparency practices are used.
  - Simmons, Nelson and Simonsohn's 21-word statement: "We report how we determined our sample size, all data exclusions (if any), all manipulations, and all measures in the study."
  - The epidemiology counterpoint, Lash and Vandenbroucke (Epidemiology 2012): "in secondary analyzes of existing data sets… hypothesis formulation can be virtually coincident with a brief initial data examination… These practices improve the yield from our science." They argue compulsory protocol preregistration "will inevitably dampen creativity" and propose registering data descriptions instead.
  - Steegen et al. (2016) propose multiverse analysis: run every reasonable processing variant and show which choices matter.

What it means for TurboTab:
- **The looked-at ledger.**
  - Record every view opened, automatically, under both purposes. This is Zgraggen's proposal and Shneiderman's History.
  - The manuscript gets one IDA sentence, for example: "Before the analysis plan was locked, distributions, missingness and relationships among predictors were examined; the outcome's relationship with predictors was [not examined | examined for a, b, c]."
  - The export gets the full list as "what was known in advance".
  - Do not block, which matches Lash and Vandenbroucke and the leash ruling; disclose.
- **Plan changes after looking.** A change motivated by a finding names the finding. A change after an outcome view is labeled as such, which the engine already does.
- **Optional angle (for the INBOX, not required now).** "Is this more than noise?" places a relationship the user pins among 8 permuted null plots (a lineup). It teaches calibration on outcome-free views. The cognitive-load caution applies, so it belongs behind "More angles" only.
- **Robustness after the lock.** Choices made from what IDA showed (exclusions, transforms) feed the existing "Which of my decisions mattered?" view, which is a multiverse.

---

### (e) Profiling tools and where they go wrong

What the sources say:
- **ydata-profiling:**
  - Six sections: Overview, Variables, Interactions, Correlations, Missing values, Sample.
  - Its alert list includes Constant, Zeros, High Correlation, High Cardinality, Imbalance, Skewness, Missing, Infinite, Unique, Seasonal, Non-stationary, Uniform, Duplicates and others.
  - Its own documentation concedes: "Although useful, the decision on whether an alert is in fact a data quality issue always requires domain validation."
- **Sweetviz:** "generates beautiful, high-density visualizations"; "The system is built around quickly visualizing target values and comparing datasets"; target analysis "Shows how a target value… relates to other features". It warns that its association measures "shouldn't be taken as gospel."
- **AutoProfiler (Epperson, Gorantla, Moritz, Perer; VIS 2023):**
  - "Previous profiling systems often require scrolling to look through multiple pages of charts [Lux; pandas-profiling], making it hard to find interesting problems or insights."
  - Such systems "often present an abundance of information that can be difficult… for users to parse quickly."
  - "Alerts must be customizable and designed to minimize alert fatigue, or else a user may totally ignore them." With inline profilers, alerts are "re-computed and displayed every time a user updates… quickly causing alert fatigue."
  - Its own fix is a column overview (name, small chart, % missing) with distributions opening on demand. 91% of insights came from the tool.
- **Batch and Elmqvist (TVCG 2018)** describe a "visualization gap": analysts often skip interactive visualization in initial exploration.

What these tools get wrong:
- They draw a wall of charts where a ranked list is needed.
- They raise alerts with no route to a decision.
- They recompute alerts every time and never remember the answer.
- Target analysis is a default, which is a sneak peek under inference.
- Their "high correlation" alerts carry no expectation behind them.

What it means for TurboTab:
- **Findings, not alerts.**
  - Each item is a sentence tied to a lever.
  - The user's answer persists through the readings ledger ("expected: fasting samples"), so the item never alerts again.
  - An item with no lever is a note in the details, not an entry in the list.
- **No scrolling wall.** Use the overview strip, top 12 then "and N more" (FOUNDATION §5's Strip rule), and at most three views.
- **No target analysis by default** under inference.

---

### How each lens fits the same groups

These checks come from detectors the engine already has. Their placement in the groups is my proposal and should be vetted in the per-domain review packets.

| Lens | Who is in the data | What is missing | Each variable | Variables together (by structural variable) |
|---|---|---|---|---|
| Dietary | recall days per person (repeats) | missing recall days | energy-intake screens; zero spike for foods many people never eat (`usual_intake`) | intakes by sex and age band; energy against body size |
| Clinical | fasting or visit flow | lab missingness by visit | plausibility tiers (NHANES 2017–18 central 98%, impossible limits, CDC paediatric z-scores); units | labs by sex and age band |
| Metabolomics | QC and pooled rows (`reference_rows`) | values below the detection limit | data type and scale | drift by run order and batch (`assay`); redundancy (VE3) |
| Genomics | sample QC | call or missing rate | data type (counts, TPM, log) (`genomics`) | batch or ancestry as structural variables |
| Survey | design: weights, strata, PSU (`survey`) | sentinel codes as missing (`codes`) | ordinal response blocks (`scales`) | items by stratum and cycle |

When repeated measures are settled, the Over time group appears (Lusa et al.): participation profile, dropout, and the outcome's change over time, which IDA permits.

---

### Consolidated design rules (a proposed §5 addendum, "Look" layout)

1. **Two modes.**
   - The Look card is a reading list.
   - Decide is the existing one-question card.
   - Continue stays the single primary action, in the same place as everywhere else.
2. **Overview first.** The canvas opens on an overview strip of every column; it is never blank.
3. **Fixed groups.** Groups follow the IDA domains in a fixed order. Lens checks slot into them with one quiet lens label.
4. **Structural variables organize every multivariable view.**
5. **Ranking.**
   - Rank by notability: deviation from a stated expectation × the leash stakes of the lever.
   - Three "worth a look" per group, then the rest in order.
   - Every item has a "Why is this here?" line.
6. **Outcome-blind by default under inference before the lock.**
   - Outcome–predictor views sit in a collapsed group, "Looks at the outcome (recorded)".
   - They are never blocked.
7. **Record everything opened.**
   - One IDA sentence in the manuscript.
   - An IDA report in the export.
   - Plan changes cite the finding that motivated them.
8. **Anti-anchoring.**
   - Stable group order.
   - Ranking by quality and consequence, never by striking patterns.
   - Overview one key away.
   - Suggestions one step at a time.
   - A quiet coverage count, with one summary sentence at Continue.
9. **Two disclosure levels at most.** Details on hover.
10. **Color.**
    - Look mode draws data in gray.
    - Indigo appears only once a finding is open as a decision ("what the choice touches"), which keeps the §4 color lesson intact.
    - "Worth a look" is a word, not a color; amber stays reserved for the coach.
11. **Pin to a tray, not a wall.** The tray feeds the IDA supplement.
12. **Continuity.** Moving between items morphs with stable axes, as Dziban and Shneiderman's smooth zooming recommend.

### Notes on sources

- Quotes from PDFs were extracted as text. Ligatures that extraction drops ("fi", "fl") were restored; the wording is otherwise verbatim.
- The NN/g two-level guidance, the Nosek "what was and was not known" phrase, and the Data Formulator 2 "data threads" wording came through a page-summarizing fetch. Check them against the originals before quoting them in the app.
- Data Formulator 1 and Steegen et al. are described from their abstracts and search summaries only; I did not read the full text.
- The repository was read only. Files consulted:
  - /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/calm/FOUNDATION.md
  - /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/MODELING_SEQUENCE.md
  - /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/BLUEPRINT.md
  - /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/explore.py
  - /Users/nhedglin/tabular-ml-lab/turbotab/core/explore_previews.py
  - /Users/nhedglin/tabular-ml-lab/turbotab/core/readings.py
  - /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/__init__.py
  - /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/lenses.py
  - /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/plausibility.py

## Sources
- https://idl.cs.washington.edu/files/2015-Voyager-InfoVis.pdf
- https://idl.uw.edu/papers/voyager
- https://faculty.washington.edu/billhowe/publications/pdfs/wongsuphasawat17voyager2.pdf
- https://idl.cs.washington.edu/files/2016-CompassQL-HILDA.pdf
- https://idl.cs.washington.edu/files/2019-Draco-InfoVis.pdf
- https://idl.cs.washington.edu/files/2020-Dziban-CHI.pdf
- https://arxiv.org/pdf/2105.00121
- https://info290.github.io/papers/show-me.pdf
- https://doi.org/10.1109/TVCG.2007.70594
- https://arxiv.org/abs/2408.16119
- https://arxiv.org/html/2408.16119v1
- https://arxiv.org/abs/2309.10094
- https://papers.rgrossman.com/proc-094.pdf
- http://www.vldb.org/pvldb/vol8/p2182-vartak.pdf
- https://arxiv.org/pdf/1707.03877
- https://idl.cs.washington.edu/files/2012-Profiler-AVI.pdf
- https://doi.org/10.1109/vast.2017.8585665
- https://doi.org/10.1145/3411764.3445195
- https://doi.org/10.1109/tvcg.2021.3114827
- https://doi.org/10.1109/tvcg.2021.3114862
- https://doi.org/10.1109/tvcg.2018.2865040
- https://doi.org/10.1109/tvcg.2017.2743990
- https://www.cs.umd.edu/~ben/papers/Shneiderman1996eyes.pdf
- https://bib.dbvis.de/uploadedFiles/55.pdf
- https://www.nngroup.com/articles/progressive-disclosure/
- https://doi.org/10.1353/obs.2018.0014
- https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009819
- https://www.stratos-initiative.org/wp-content/uploads/2021/10/Huebneretal-2020.pdf
- https://www.stratos-initiative.org/wp-content/uploads/2025/11/Heinze-2024.pdf
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0295726
- http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf
- http://emanuelzgraggen.com/assets/pdf/risk.pdf
- https://mucollective.northwestern.edu/files/2018-ForkingPaths-BELIV.pdf
- https://arxiv.org/pdf/2104.02015
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5856500/
- https://doi.org/10.15626/mp.2020.2625
- https://doi.org/10.1177/2515245919848684
- https://doi.org/10.1097/ede.0b013e318245c05b
- https://sites.stat.columbia.edu/gelman/research/published/multiverse_published.pdf
- https://arc.psych.wisc.edu/the-21-word-solution/
- https://docs.profiling.ydata.ai/latest/getting-started/concepts/
- https://github.com/fbdesignpro/sweetviz
- https://arxiv.org/html/2308.03964
- /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/calm/FOUNDATION.md
- /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/MODELING_SEQUENCE.md
- /Users/nhedglin/tabular-ml-lab/.worktrees/calm/docs/turbotab-next/BLUEPRINT.md
- /Users/nhedglin/tabular-ml-lab/turbotab/core/stages/explore.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/explore_previews.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/readings.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/__init__.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/lenses.py
- /Users/nhedglin/tabular-ml-lab/turbotab/core/detectors/plausibility.py