# Interaction pack — research specification

Same structure and same discipline as the four science packs: every recommendation carries
**SETTLED** · **CONVENTION** · **DISPUTED**, every claim names a source that resolves, and anything
not read in primary text is marked **[verify-at-build: legend]**. Proposed at `DESIGN_LANGUAGE.md` §05.2 and
scheduled by the product owner at L46 (`ROADMAP.md`, *"`research/INTERACTION_PACK.md` is
scheduled"*).

---

## How this was built, and the caveat that inverts the other four packs'

The four science packs record that *"the session's egress proxy blocked publisher domains… the
content below comes from search-surfaced excerpts."* **That did not happen here.** The anchor paper,
the sceptical review that opposes it, the W3C normative text, the CLS specification, the
change-blindness literature and both choice-overload meta-analyses were fetched and read.

**And then every claim was adversarially refuted**, by a second reader told to locate each source
independently, verify each quote and number against the text, downgrade badges, and — the part that
mattered most — **go looking for literature that contradicts it.** The numbers:

| | |
|---|---|
| Candidate claims | **105** |
| Read in primary text | **100** |
| Sources that could not be reached | **33** |
| Verdicts after refutation | **58 stand · 42 overstated · 3 wrong · 1 dropped** |
| Contradicting findings surfaced | **49** |
| Fabricated or misattributed citations found | **1** (an NN/g byline; corrected at §03.2) |

**Three of the corrections reversed a reading rather than softening one**, and they are in the text
below at §01.2, §04.1 and §05. The most important is §04.1: the first draft of this pack argued that
an *instant* change is more noticeable than an animated one and that this supported a shipped
ruling. The refutation found the nearest available analogue to the app's own case and it says the
opposite. **That correction is left visible rather than quietly absorbed**, because a pack whose
process is invisible is a pack you have to take on trust.

**The risk here is the opposite of the science packs' and it is worse.** In a biochemistry pack the
temptation is inventing a threshold, and a wrong threshold is checkable. Here it is **dressing taste
as evidence** — stating a design preference and attaching a citation that does not support it. The
evidence gate resolves that a source is *named and resolvable*; it cannot check that a claim is
*faithful to it*.

**Nothing here is a `SETTLED` design recommendation.** `SETTLED` appears only on documentary facts —
what a specification's normative text says, what an experiment did and did not manipulate, what a
measured effect was. Interaction design has almost no methodological consensus of the kind `SETTLED`
names in the science packs.

---

## 00 · What this pack is for

Three design questions were on the board when it was scheduled, and **two had already been ruled on
without it.** That is the better test of a pack than a greenfield question.

| | The question | The ruling it checks | Verdict | § |
|---|---|---|---|---|
| 1 | Does animating a transformation help a viewer follow it? | `PRODUCT_VISION.md` §04b; `DESIGN_LANGUAGE.md` §05.2 item 4 | **Supported, narrowly, and "which animation" matters more than "whether"** | §01.4 |
| 2 | Is a one-card affordance worse than the card? | L46-A2 — `attention.MIN_COLLAPSE = 2` | **Unsupported in either direction** | §03.1 |
| 3 | Is a promoted card disorienting? | L46-A3 — marked, not moved | **The literature's nearest analogue argues against the mechanism** | §04.1 |

---

## 01 · Animated transitions — does watching a change help?

### 01.1 · The anchor, read in primary

`DESIGN_LANGUAGE.md` §05.2 names **Heer & Robertson, *Animated Transitions in Statistical Data
Graphics*, IEEE TVCG 13(6):1240–1247, 2007** as the anchor it would start from and records that it
had not been read. It has now been read in full, twice, by two readers who independently
re-downloaded it (<https://idl.cs.washington.edu/files/2007-AnimatedTransitions-InfoVis.pdf>; the
second reader reports the PDF sha256 matched).

> **Animated transitions beat instantaneous ones on object tracking, in every condition tested.**
> Experiment 1: 3 (static / direct animation / staged animation) × 2 (element count) within-subjects,
> **24 subjects** (mean age 49.6), 288 trials each, 8 transition types. *"Repeated Measures ANOVA
> found significant differences at the .05 level for each transition type (F(2,286) ≥ 22.03,
> p < 0.001)."*
> **[CONVENTION]** — the direction is consistent across all eight types, but it is one experiment,
> one lab, 24 screened data-graphics professionals, never independently replicated. The paper itself
> reports mixed results from Bladh et al.'s StepTree study.

> **The dependent measure is memory, not tracking.** *"The display was masked 3 seconds after the
> transition onset, at which point subjects were to click the final locations of the targets"* —
> subjects had to keep the pointer away from the graphic until the mask. The authors' own reading,
> §6.1: the advantage *"may in part be due to improved transfer to memory."*
> **[SETTLED]** as a documentary fact. It matters for transfer: the experiment asks *after the
> change, can you still say which was which?*, not *during the change, can you keep your eyes on
> it?*

> **Animation reduced value-estimation error in 3 of 4 transition types** (Experiment 2, 2 s
> animations). **[CONVENTION]**.

> **Staging is preference-backed, not performance-backed — and Heer says so himself, later.**
> In Experiment 1 staged animation beat direct animation significantly in exactly one condition
> (Zoom & Filter, p = 0.026) and was significantly **worse** in one (Timestep Scatter, p = 0.002).
> In Experiment 2 it never significantly outperformed direct animation, and in the Donut Chart
> condition direct animation was significantly *more* accurate. What carries staging is preference:
> significant in 7 of 9 rated transitions, p < 0.003. Kim, Correll & Heer, *Designing Animated
> Transitions to Convey Aggregate Operations*, EuroVis 2019, §2, verbatim: *"the multi-stage
> animations evaluated by Heer & Robinson [HR07] were typically preferred by participants but did
> not significantly outperform single-stage animations."*
> **[CONVENTION]** — not a disagreement between papers. The anchor's own senior author records the
> null twelve years later.

> **Ten design principles, one of them tested.** §3.2 states the provenance verbatim: *"After
> reviewing literature in perception, visualization, and user interface design, we arrived at the
> following considerations."* The only factor either experiment manipulated was
> static / direct / staged. **[SETTLED]**.

> **The paper cannot be cited for any animation duration.** Easing was constant — *"all animations
> discussed below use slow-in slow-out timing"* — and duration was fixed at **1.25 s** (Exp 1) and
> **2 s** (Exp 2). No two durations were compared. The "around one second" figure it is usually
> cited for is imported from Robertson et al. 2002 and then revised **upward** on subject comments.
> **[SETTLED]**, and it is the guardrail most likely to be violated.

> **Animate a schema change only where a data dimension survives it.** §4.1.6, verbatim: *"In data
> schema changes, animation is only appropriate when there is a data dimension shared between the
> starting and ending states. Without a shared structure between graphics, animation may be
> ill-defined or misleadingly convey false relations. In such cases, we advocate… static or dissolve
> transitions."* And §3.2.1: *"marks representing specific data points should not be reused to
> depict different data points across a transition."*
> **[CONVENTION]** — a reasoned ruling with **zero** experimental support in this paper.

### 01.2 · The sceptical side, and the two places the first draft of this pack got it wrong

> **Tversky, Morrison & Bétrancourt (IJHCS 57(4), 2002) found no benefit for animation over
> informationally equivalent static graphics** — and the methodological standard they set is the
> right one: *"In order to know if animation per se is facilitatory, animated graphics must be
> compared to informationally equivalent static graphics."*
> **↳ CORRECTION.** That verdict was quantitatively reversed fourteen years later **by one of its
> own authors**. Berney & Bétrancourt, *Does animation enhance learning? A meta-analysis*,
> Computers & Education 101:150–167 (2016): 61 primary studies, **N = 7,036**, 140 pairwise
> comparisons, random effects — *"An overall positive effect of animation over static graphics was
> found, with a Hedges's g effect size of 0.226 (95% CI 0.12–0.33)."* Höffler & Leutner's earlier
> meta-analysis of 76 comparisons from 26 studies reports d = 0.37 `[verify-at-build: 0.37]` — read only
> as quoted inside Berney & Bétrancourt, not in primary.
> **[DISPUTED]**, and the scope caveat must travel with it: the corpus is **instructional
> multimedia**, not interface feedback. It does not license "UI motion aids comprehension." What it
> forbids is citing Tversky 2002 as the field's standing verdict.

> **Animation was slower than static small multiples for analysis, and the accuracy half did not
> replicate.** Robertson, Fernandez, Fisher, Lee & Stasko, InfoVis 2008 (n = 18): animated trend
> visualization **82% slower** and less accurate than small multiples for analysis, fastest and
> preferred for presentation.
> **↳ CORRECTION.** Brehmer, Lee, Isenberg & Choe, IEEE TVCG 2019 — an explicit sequel with **96**
> participants, using material shared by a 2008 co-author — replicates the time result and **fails
> to replicate the accuracy result**: *"With respect to accuracy, small multiples do not appear to
> have a substantial advantage, which is unlike what Robertson et al. found… in the absence of a
> clear accuracy advantage over small multiples, animation remains to be a viable design choice for
> some tasks."*
> **[DISPUTED]** — the presentation/analysis split stands on **time**; *"animation is less
> accurate"* does not, and must not be asserted.

> **Staggering has no demonstrated visual-tracking benefit — and the first draft of this pack
> described that wrongly.** It framed staggering as *"the technique Heer & Robertson recommend"*
> contradicted by Chevalier, Dragicevic & Franconeri (IEEE TVCG 20(12), 2014). **Neither primary
> text contains that disagreement.** Heer & Robertson used staggering in only **two of eight**
> transitions (*"Staging in the bar to donut and sorting cases involved staggering elements'
> animation with short delays to reduce occlusion"*), and those two are precisely the two whose
> staged-vs-direct comparison **failed** to reach p < .05 (p = 0.071 and p = 0.051). Chevalier et al.
> then found no effect on object tracking in two experiments (n = 20 each) built to favor it, and
> named Heer & Robertson as the source of the untested occlusion hypothesis.
> **[CONVENTION]** — a field practice with no demonstrated tracking benefit in either paper. Not a
> disagreement; a null nobody disputes.

> **What *is* supported on timing is the easing, not the staging.** Kim, Correll & Heer 2019 records:
> *"Dragicevic et al. compared time distortion methods, such as slow-in / slow-out and
> fast-in / fast-out, and found that slow-in / slow-out enabled users to better track visual
> objects."* **[CONVENTION]** `[verify-at-build: no number]` — the claim is that slow-in /
> slow-out helped tracking, which is a direction rather than a threshold. Read as quoted, not
> in primary.

### 01.3 · The modern re-test, and it changes the shape of the question

> **Which animation is used matters more than whether there is one — and for grouped elements
> nothing helped.** Rodrigues, Dennig, Brandt, Keim & Weiskopf, *Comparative Evaluation of Animated
> Scatter Plot Transitions*, IEEE TVCG 2024 (arXiv:2401.04692): **preregistered**, 170 participants,
> sample size fixed by an a-priori power analysis, six transition techniques, point-tracing and
> cluster-tracing. Verbatim: *"rotations with an orthographic camera or staged expansion of a depth
> axis significantly outperform all other animation techniques for the traceability of individual
> points… However, we could not find any significant differences for the traceability of
> clusters."*
> **[CONVENTION]** — the best-powered, most recent, most on-point study located, and its message is
> that **"animate the transition" is underspecified as a design rule**. The spread between the best
> and worst animation was significant; between animations, for clusters, nothing was.

> Its duration recommendation — *"We used an animation time of 1 second… We recommend this duration
> as a sensible default"* — is **not a measured optimum**: the main study held duration constant and
> the pilot that varied 0.5 / 1 / 2 s had five participants per condition and was deliberately not
> analyzed for accuracy. **[DISPUTED]**.

### 01.4 · **Question 1, answered** — what this means for §05.2 item 4

The assertion under test: *"the user is empowered to decide correctly only when they can watch their
working data morph"* under a join, merge or split — recorded as the one load-bearing design
assertion in `PRODUCT_VISION.md` resolving to no evidence.

**Supported at the identity level, narrowly, and with three constraints the design language does not
currently state.**

1. **Animate only where a shared dimension survives the reshape.** A join producing rows with no
   correspondence to the rows before it must **not** animate them into one another (§4.1.6, and
   §3.2.1's rule against reusing a mark for a different datum). The project's own governing rule
   makes it sharper: **an animation implying a row correspondence that does not exist is a false
   assertion in motion.**
2. **Expect it to carry one or two identities, not a table.** Chevalier, Dragicevic & Franconeri
   2014, §2.2/§4.6.2: tracking *which* object is which through a transition runs about **1–2
   identities** against **3–4 positions**. **[SETTLED]** as a measured result.
3. **Choosing to animate is not a design decision — choosing *which* animation is.** Rodrigues et
   al. is unambiguous on this, and it is the constraint most likely to be skipped.

**And the word *only* is unsupported.** No source says a static before/after fails. What the
literature supports is *a* morph beside a before/after, not a morph instead of one — which also puts
the reshape on the right side of the presentation/analysis split, since analysis is where animation
lost on time.

---

## 02 · Identity continuity across a state change

### 02.1 · Change blindness, read in primary — and the inference it does *not* license

> **A blank field between two versions of a scene makes large, repeated, expected changes take many
> seconds to find; remove the blank and identification is near-immediate.** Rensink, O'Regan &
> Clark, Psychological Science 8(5), 1997. **[SETTLED]**.

> **The transient does not have to cover the change.** O'Regan, Rensink & Clark 1999 — six small
> "mudsplashes" that never overlap the changed region suffice. **[SETTLED]**.

> **It survives a one-second real-world occlusion of the person you are talking to.** Simons & Levin
> 1998: roughly half of pedestrians failed to notice a swapped conversation partner. **[SETTLED]**.

> **But the blank is sufficient, not necessary — and in one experiment the disruption HELPED.**
> Simons, Franconeri & Reimer, *Change blindness in the absence of a visual disruption*, Perception
> 29(10), 2000 (n = 35 and n = 36): *"we demonstrate that change blindness can occur even in the
> absence of a visual disruption. In one experiment, subjects actually detected more changes with a
> disruption than without one."* Experiment 2: color changes detected 41% (SD 14) with a
> disruption versus 31% (SD 8) gradual, t(22) = 2.09, p = .049. **[SETTLED]**.
> → **The design inference that "removing the interruption restores noticing" does not follow.**

> **And none of it measures correspondence.** No condition in Rensink et al. asks observers *what
> became what*. **[CONVENTION]** — the honest limit on citing change blindness for an identity rule.

### 02.2 · What the design language may and may not claim

`DESIGN_LANGUAGE.md` §05.2 states two sentences with very different standing:

- *"Motion's job is to preserve identity across a state change, so the user never loses track of what
  became what."* — **[CONVENTION]**, on Heer & Robertson Experiment 1, bounded by the 1–2 identity
  capacity and by Rodrigues et al.'s finding that the technique matters more than the fact.
- *"An object that is destroyed and replaced teaches nothing, however smoothly it fades."* —
  **the anchor states the opposite.** §3.2.1 requires that some state changes be rendered as
  **removal and addition** of marks even when the graphic type is unchanged, because reusing a mark
  for a different datum establishes a false relation. **[CONVENTION]**, and the sentence is too
  strong: destroying and replacing is sometimes the *correct* rendering, precisely where identity
  does not survive.

**The closed list of four continuity slots survives and gains a harder argument.** §05.2 argues for
scarcity from attention economics; the measured capacity is better. At 1–2 identities, a vocabulary
spending continuity on more than a couple of objects at once is spending something the viewer does
not have.

> **A caution on borrowing the tracking numbers at all.** Pylyshyn & Storm (1988) did not test
> identity — subjects reported a probe flash on *any* target among identical distractors, in
> displays engineered so objects never came within 0.75° or occluded. **[SETTLED]** as a documentary
> fact. Alvarez & Franconeri 2007 show the "about four objects" figure trades continuously against
> speed (~8 slow, ~1 fast), and Franconeri, Jonathan & Scimeca 2010 later reverse that speed
> account in favor of **object spacing**. **[DISPUTED]**. Chevalier et al. name four specific
> mismatches between MOT displays and interface transitions. **So MOT capacity is not a limit on
> interface objects, and this pack does not lend it as one.**

---

## 03 · Bounding a list

### 03.1 · **Question 2, answered** — the one-card affordance is unsupported either way

L46-A2 ruled that a remainder of exactly one is shown rather than collapsed, because the affordance
costs one row to hide one row.

> **No study locates the break-even point at which the row cost of a disclosure control exceeds the
> cost of showing the hidden items.** The rule is unsupported by measurement **in either
> direction**. The only formal cost model in reach — Liu, Gori, Rioul, Beaudouin-Lafon & Guiard,
> *How Relevant is Hick's Law for HCI?*, CHI 2020 — addresses selection latency, not layout cost.
> **[DISPUTED]** — recorded as the finding rather than as a gap.

**A2 therefore stands as a design ruling and must not acquire a citation.**

**What the literature does say is that the usual argument for its parent — bounding at all — is
weak, and that the counter-argument is weaker than it first looked.**

> **Choice overload is a null effect at the mean.** Scheibehenne, Greifeneder & Todd, JCR 37(3),
> 2010: mean **D = 0.02** across 63 conditions, N = 5,036, no relationship between effect size and
> number of options; the Iyengar & Lepper jam study did not replicate at roughly double the sample.
> **[DISPUTED]** — Chernev, Böckenholt & Goodman reproduce the null unconditional effect and report
> an effect once four moderators are modeled. The dispute is older than the 2015 meta-analysis: it
> runs back through a 2010 comment and reply.

> **↳ AND THE NULL IS ITSELF CONTESTED ON METHOD.** Dean, Ravindran & Stoye, *A Better Test of Choice
> Overload* (arXiv:2212.03931v3, 2025) argue the mean-comparison tests producing the null are
> **underpowered**, and propose tests grounded in Random Utility Model characterisation theorems.
> **[DISPUTED]** — so *"choice overload is a null"* is not a fact this pack may lend either.

> **Longer lists increased engagement in the largest field test located.** Beierle, Aizawa, Collins
> & Beel, IJDL 2019: **41.3 million** deliveries, set size randomised 1–15, engagement rising
> monotonically, *"no strong evidence"* of choice overload. **[CONVENTION]**.

> **Menus of 30 options were used more readily than menus of 9.** Katz & Byrne, n = 32
> within-subjects; the breadth effect exceeded the scent effect. **[CONVENTION]**.

> **Hick's law cannot justify a shown/hidden split.** Liu et al., CHI 2020: for any logarithmic
> latency function an uncategorised split is strictly worse than showing everything, and a
> categorised split is at best equal. **[SETTLED]** as a result about the model. Whether Hick–Hyman
> applies to HCI menus at all is **[DISPUTED]** — Cockburn, Gutwin & Greenberg build it into a
> validated predictive model; Liu et al. call it largely irrelevant.

> **↳ THE ONE PIECE OF EVIDENCE THAT SUPPORTS HIDING.** Springer & Whittaker, *Progressive
> Disclosure: Designing for Effective Transparency* (arXiv:1811.02164; CHI/IUI), Study 1 n = 74
> within-subjects: before use, participants anticipated greater accuracy from the always-visible
> word-level system; **after** use, 37 of 74 preferred the document-level version and there were
> *"no overall differences in trust"* (t(73) = .910). Thirteen cited the distraction of the
> always-visible detail. Study 2 (n = 12, think-aloud) concludes users *"may benefit from initially
> simplified feedback."*
> **[CONVENTION]**, with two limits that must travel with it: it concerns **explanation detail in an
> affect-classification system**, not list truncation, and Study 2's n = 12 is qualitative. It is
> nonetheless the only located empirical work supporting an explicit-request disclosure, and it is
> the closest thing the Explore stack has to a friendly witness.

> **The three-to-five convention has no published rationale.** Beierle et al. document it as an
> industry norm and say so. Eye-tracking of search results (Granka, Joachims & Gay) shows attention
> dropping sharply after rank 2 and flattening from ranks 6–10, and the authors attribute the
> flattening **to the fold**. **[CONVENTION]** — and the same research group later showed a
> competing mechanism: Joachims et al.'s swapped/reversed-condition experiments demonstrate **trust
> bias**, users clicking high-ranked results because they are high-ranked. **[DISPUTED]** as an
> explanation of the gradient.

**The consequence for `attention.BOUND`, stated plainly.** The bound's recorded justification is a
**measurement about this repository's fixtures** — the median stack, so the collapse fires on the
tail rather than the typical table. Nothing above touches that, because it is an argument about
*when the rule fires* rather than about *whether short lists are easier*. **What this pack forbids
is reaching for the other argument.** A future loop must not justify the bound by choice overload,
by Hick's law, or by "three to five is the norm": the first is contested in both directions, the
second is a model saying the split does not help, and the third is a convention with nothing behind
it. **Springer & Whittaker is the citation that would be honest**, and it is about explanation
detail rather than about ranked findings.

### 03.2 · A note on practitioner sources

> **Nielsen Norman Group's progressive-disclosure and accordion articles state their central
> claims — that hidden content is missed, that expanding carries an interaction cost — with no
> participant-level data.** Progressive Disclosure is Jakob Nielsen, 3 December 2006; **Accordions on
> Desktop is Huei-Hsin Wang, 30 July 2023** — *not* Page Laubheimer, who wrote the three-click-rule
> piece. That misattribution was the one citation error the refutation pass found in 105 claims, and
> it is recorded here because a cross-contaminated byline is exactly how a fabricated citation gets
> built. **[SETTLED]** as a documentary fact about those pages.

---

## 04 · Content that arrives where the user acted

### 04.1 · **Question 3, answered — and the answer goes against the shipped mechanism**

L46-A3 ruled that clearing a card frees its slot, that the next collapsed finding is promoted, and
that the promoted card is **marked rather than animated**.

**The promotion is supported. The marker-instead-of-motion is not, and the nearest analogue in the
literature argues against it.**

> **An interaction-caused layout shift is acceptable when temporal proximity makes the causality
> legible.** Google / web.dev, *Cumulative Layout Shift*, §"Expected vs. unexpected layout shifts";
> Layout Instability API editor's draft: a shift within **500 ms** of user input sets
> `hadRecentInput` and is excluded from CLS entirely. **[SETTLED]** for the spec, **[CONVENTION]**
> for the guidance. Direct support for A3's shape — the promotion happens *because* the user acted,
> immediately, in the list they acted in.

> **Google's CLS thresholds of 0.1 and 0.25 are not derived from any published study, and Google
> says so** — and it is worse than that: the thresholds were derived in 2020 against a CLS defined
> as the **sum of every shift over the page's lifetime**, and the metric was redefined to session
> windows on 2 June 2021 without the thresholds being re-derived. **[CONVENTION]** — nobody may
> cite 0.1 as a research finding, or as a budget calibrated to the metric now measured.

> **↳ THE CORRECTION, AND IT REVERSES THIS PACK'S FIRST DRAFT.** That draft argued from Simons,
> Franconeri & Reimer (2000) — instant changes detected at 97% by observers hunting for them,
> gradual ones far less — that a marked, instant arrival is *more* detectable than a fade, and that
> A3 was therefore right on the evidence rather than merely cheap. **The inference does not hold.**
> Simons' gradual condition is a **twelve-second, transient-free dissolve engineered to suppress the
> motion signal**; a sub-second UI transition manufactures a large transient, which is the thing
> that draws attention. The two literatures are not in conflict; the inference was.

> **And the closest located analogue found the opposite.** Huhtala, Mäntyjärvi, Ahtinen, Ventä &
> Isomursu, *Animated Transitions for Adaptive Small Size Mobile Menus*, INTERACT 2009: four
> transition types for icons **appearing and disappearing in a twelve-icon grid menu** — the nearest
> published analogue to a card list reflowing after a dismissal — **40 subjects**, half in India and
> half in Europe. Abstract, verbatim: *"Statistical analysis of the results indicates that animated
> transition effects have a clear positive effect on perception and conception of change."*
> Concluding remarks: *"The animated transitions resulted in significantly better user performance
> compared with non-animated transitions."* And change blindness was reported **in the instant
> condition**: subjects *"thought they blinked just when change occurred… With animated transitions,
> this kind of problem was not mentioned."*
> **[CONVENTION]** — one study, 2009, a 12-icon phone menu rather than a finding stack, and the
> geography split is a confound the paper explores rather than controls. But it is on point and it
> points the other way.

**So the honest statement of A3's status, which is what the product owner asked for:**

- **Freeing the slot and promoting** — untouched by this. Nothing here bears on it.
- **Marking the promoted card** — a reasonable answer to an unmeasured risk. §04.2's numbers say
  even instant changes are missed 42% of the time, which is the argument for the marker existing at
  all rather than trusting the arrival to be seen.
- **Marking *instead of* animating** — **the literature's nearest case prefers the animation.** The
  app cannot do that today: `DESIGN_LANGUAGE.md` §05.2 measured 92 `innerHTML` assignments against
  22 node-owning writes and zero animation machinery, and `GUIDED-073` is deliberately unbuilt. So
  the ruling is **correct as a constraint and unsupported as a preference**, and the design language
  should say that rather than claiming the instant change is better.
- **What nobody has measured** is whether a promoted card reads as *"my dismissal did not work."*
  Every result located on undo/dismiss feedback was practitioner blog content or design-system
  documentation. **[DISPUTED]**, and it is the honest headline of §04.

### 04.2 · Detection rates, for the marker's own justification

> **Attention to the location is necessary but not sufficient.** Simons & Rensink, TiCS 2005:
> changes to attended objects frequently go unnoticed. **[SETTLED]**.

> **On a tablet interface with no visual disruption, participants missed 64% of gradual changes and
> 42% of instant ones**, against a 6% false-alarm rate. Brock, Quigley & Kristensson 2018, n = 16
> within-subjects. **[CONVENTION]** — 42% is the number that justifies a marker at all.

> **WCAG 2.2 SC 2.3.3 *Animation from Interactions* is Level AAA**, and requires that motion
> animation triggered by interaction can be disabled unless essential. **[SETTLED]**, verbatim
> normative text. **A state change is not motion animation**, so a marked-not-moved promotion
> carries no 2.3.3 obligation.

---

## 05 · Interruption budgets

`PRODUCT_VISION.md` §08's open question — *how many noticings may a step raise at once?* — and
`DESIGN_LANGUAGE.md` §09's blocker budget.

> **No source reached gives a defensible maximum number of advisories per decision point.**
> **[DISPUTED]**, and this is the answer rather than a gap. The closest is a **slope with no knee**:
> Ancker et al., BMC Med Inform Decis Mak 17:36 (2017) — each additional best-practice reminder in
> the same encounter multiplied the odds of acceptance by **0.70**, continuously, no threshold
> reported and none identifiable. **The prototype's "two" is contradicted by nothing and supported
> by nothing.**

> **A dismissal predicts further dismissal — but less decisively than it first reads.** Same paper:
> having overridden the first instance of a repeated reminder, the chance of overriding later
> instances was **87.9%**.
> **↳ CORRECTION, from the next sentence of the same paper:** *"Conversely, if the first instance was
> accepted, the chance of subsequent instances being overridden was 51.9%."* For drug alerts the
> pair is 99.9% against 58.4%. **So the measured effect of a first dismissal is roughly a 36-point
> increase over an already-high floor, not near-certainty** — and the study is observational, so the
> same latent cause (the alert being irrelevant to this patient) plausibly drives both.
> **[CONVENTION]**. It still supports the project's existing rule — *the record holds what was
> dismissed, because nothing else will bring it back* — which is the shelf rule applied to attention.

> **False-alarm rates in the clinical exemplar are extreme and measured.** Drew et al., PLoS ONE
> 9(10): e110274 (2014): 31 days, five adult ICUs, **2,558,760** alarms, **187 audible alarms per
> bed per day**, **88.8%** of 12,671 hand-annotated arrhythmia alarms false positive. **[SETTLED]**.

> **↳ AND DREW ET AL. REJECT THE SUPPRESSION RULE A DESIGNER WOULD DRAW FROM IT.** The same paper
> reports that 93% of correctly-detected ventricular tachycardia alarms were not sustained long
> enough to warrant treatment — and then explicitly declines the inference that a transient true
> positive should therefore be suppressed. **[DISPUTED]** — so *"a detector that is right is not
> thereby entitled to interrupt"* is a design position this pack may state as a position, **not** as
> Drew's finding.

> **Override rates for drug-safety alerts run 49%–96% across nine studies.** van der Sijs, Aarts,
> Vulto & Berg, JAMIA 13(2), 2006. **[SETTLED]**. **Severity does not monotonically predict action** —
> medium-severity alerts were overridden *more* often (96%) than high-severity ones (89%) in one
> hospital. **[DISPUTED]**.

> **Fatigue over time is measured, and it is real.** van der Sijs reports one study where override
> rates rose from ~50% to ~75% over five years; Embi & Leonard's randomised controlled study (178
> physicians, 907 eligible alerts, 36 weeks, Poisson regression) found the decay directly.
> **[CONVENTION]** `[verify-at-build: 50% and 75%]` — the override rates are the numbers this
> claim rests on. Read as reported by the refutation pass, not in primary here.

> **The two most-repeated numbers in this field have no basis.** The Joint Commission's *"85 to 99
> percent of alarm signals do not require clinical intervention"* is sourced in Sentinel Event Alert
> 50 to a trade-association magazine; the *"maximum acceptable override rate of 40%"* was proposed
> with no stated basis, and the systematic review reporting it says so in the same breath.
> **[DISPUTED]**.

**Consequence for the blocker budget.** §09 says *"one false-positive blocker that cannot be cleanly
resolved costs more trust than ten missed advisories"* and that the number is untracked. Nothing
above supplies the ratio. §09's existing instruction — *track blockers-per-session, alarm on trend,
not threshold* — is what this literature actually supports: there is a slope, there is no knee, and
there is decay over time.

---

## 06 · Motion, accessibility, and duration

### 06.1 · The normative text, read in primary

> **SC 2.3.3 is AAA.** *"Motion animation triggered by interaction can be disabled, unless the
> animation is essential to the functionality or the information being conveyed."* **[SETTLED]** —
> **a tool asserting that WCAG *requires* reduced-motion support at the AA level almost every
> project targets would be stating something false.**

> **SC 2.2.2 Pause, Stop, Hide (Level A) does not reach this app's motions**: it applies to content
> that moves/blinks/scrolls **and** starts automatically **and** lasts more than five seconds **and**
> is presented in parallel with other content. **[SETTLED]** — the refutation pass notes the
> criterion's full text also covers auto-updating information, which does not change the conclusion
> here.

> **`prefers-reduced-motion: reduce` asks for an interface that "removes *or replaces*" motion-based
> animation.** CSS Media Queries Level 5, read from the CSSWG source of record. It does **not**
> mandate collapsing to instant state changes — a cross-fade is equally conformant. The normative
> rationale names **two** populations: *"vestibular motion sensitivity"* **and** *"distraction for
> those with attention deficits."* **[SETTLED]**.
> → `DESIGN_LANGUAGE.md` §05 rule 5's collapse-to-instant is **a conforming choice, not the specified
> one**, and its justification names one of the two populations the spec does.

> **There is no controlled experiment demonstrating that UI animation causes vestibular symptoms.**
> W3C's own justification for 2.3.3 traces to a practitioner article citing first-person accounts.
> **[CONVENTION]** — precautionary guidance, which is a legitimate basis and is not an empirical
> finding.

> **↳ AND THE CLINICAL STANDARD OF CARE IS EXPOSURE, NOT AVOIDANCE.** Staab's 2023 review of
> persistent postural-perceptual dizziness lists, as treatment for the visually induced component,
> *"habituation exercises"* — graded **exposure** to moving visual stimuli, including screens.
> **[CONVENTION]** `[verify-at-build: no number]` — the claim is that graded exposure is the
> listed treatment, which carries no threshold. Reached by the refutation pass, not by the
> first reader.
> This does not argue against honoring the preference: a preference is a preference. It argues
> against justifying it as harm-prevention, which the clinical literature does not support in this
> form. The nearest controlled work located — Chaudhary et al., 40 adults, 20 motion-sensitive — is
> a protocol/exploratory study.

> **The "35% of adults have vestibular dysfunction" figure measures something else** — a modified
> Romberg balance test in adults 40+. **[SETTLED]** as a fact about the figure
`[verify-at-build: 35]`.

### 06.2 · Duration — the correction this pack owes the design language

> **The 200–500 ms UI-animation band is practitioner convention with no located empirical source.**
> NN/g's duration recommendations cite no study for the numbers. **[CONVENTION]** — the refutation
> pass notes an adversarial search does turn up *some* empirical work in the vicinity, so "zero
> sources exist" would be too strong; "NN/g cites none" is the claim.

> **Where duration has been studied on this task family the figure is around one second** — three to
> four times the practitioner band — and in neither case is it a measured optimum: Heer & Robertson
> imported it and revised it upward on subject comments; Rodrigues et al. held it constant and did
> not analyze their pilot. **[DISPUTED]**.

**`DESIGN_LANGUAGE.md` §05 specifies 250 ms (Settle), 300 ms with a 6 px rise (Arrive), and ≤150 ms
per section (Propagate). None of those numbers has a source, and this pack could not find one.**
That is not a finding that they are wrong — a 250 ms collapse of one element is not a 1.25 s
eight-element chart morph. It is a finding that **they are unsourced and should be recorded as
such**, and specifically that they **must not be attributed to Heer & Robertson**, which tested no
duration at all.

**And Propagate is a stagger.** *"Staleness sweeps downstream in document order, ≤150 ms per
section"* is the technique Chevalier, Dragicevic & Franconeri measured and found no effect from, in
two experiments built to favor it — and which Heer's own 2019 paper records as a null.
**[CONVENTION]** → §05 rule 3's *form* has no support. Its *purpose* — *the user watches their
edit's blast radius draw itself* — is a comprehension claim the offsetting has not been shown to
buy. This is a candidate for the next loop that touches motion, not a defect.

> **Respecting `prefers-reduced-motion` is majority practice**: roughly half of crawled pages include
> the media query (HTTP Archive, Web Almanac 2025). **[CONVENTION]** — site adoption, not user
> uptake.

---

## 07 · What this pack does not support

The most valuable output of a pack is the citation it refuses to supply.

1. **Any specific UI animation duration.** Not 250 ms, 300 ms, 150 ms, nor 200–500 ms.
2. **Staging or staggering as a performance improvement.** Preference, yes; performance, no, and
   the anchor's own senior author records the null.
3. **"Animation helps" as a general claim** — and equally **"animation is less accurate"**, whose
   source failed to replicate at n = 96.
4. **Tversky et al. 2002 as the field's standing verdict** — reversed by meta-analysis co-authored
   by one of its own authors.
5. **Choice overload, Hick's law, or "three to five" as a reason to bound a list** — and equally,
   *"choice overload is a null"*, which is contested on method.
6. **A number for how many advisories may fire at one decision point.**
7. **A break-even point for a disclosure control**, in either direction.
8. **WCAG as a requirement for reduced-motion support at AA.**
9. **Harm-prevention as the justification for a reduced-motion rule** — the clinical standard of
   care for the named population is graded exposure.
10. **MOT capacity as a limit on interface objects** — the paradigm did not test identity.
11. **Any claim about whether a promoted card reads as a failed dismissal.**
12. **The claim that an instant change is more noticeable than an animated one** — this pack's own
    first draft asserted it and the refutation pass reversed it.

---

## 08 · Routing inventory — deliberately not built

`DOMAIN_SCIENCE.md` §03b's routing inventory, detectors, advisories and any wiring are **not** in
this pack. L46 scoped this loop to content and the loop after this one reads §03b. Two reasons
beyond the instruction:

- **These claims mostly do not route to a *finding*.** They route to the design language and to loop
  prompts — *"do not cite this for a duration"* is not a card the app shows a user.
- **The two that might** — §01.4's constraint that a reshape animation must not imply a
  correspondence that does not exist, and §04.1's marker — **are design rules whose consumer is a
  renderer**, and `DOMAIN_PACKS.md` guard #1 forbids a pack adding interview components.

**If this research changes what §05.2's closed list should contain, this pack says so and does not
edit the list.** It does not: the four slots are all *decision and consequence*, and §01.4's capacity
result argues for keeping the list short rather than for changing its membership. **What it does
change is a sentence inside §05.2 and one inside §05** — see §01.2, §02.2, §04.1 and §06.2 — and
those are for a loop prompt, not for this file to apply.

---

## 09 · Sources, and what could not be reached

**Read in primary** — full text or the cited section fetched and quoted: Heer & Robertson 2007 ·
Tversky, Morrison & Bétrancourt 2002 · Berney & Bétrancourt 2016 · Berney & Bétrancourt 2009 ·
Robertson et al. 2008 · Brehmer, Lee, Isenberg & Choe 2019 · Rodrigues et al. 2024 · Kim, Correll &
Heer 2019 · Chevalier, Dragicevic & Franconeri 2014 · Huhtala et al. 2009 · Ondov et al. 2019 ·
Rensink, O'Regan & Clark 1997 · O'Regan, Rensink & Clark 1999 · Simons & Levin 1998 · Simons,
Franconeri & Reimer 2000 · Simons & Rensink 2005 · Brock, Quigley & Kristensson 2018 · Pylyshyn &
Storm 1988 · Alvarez & Franconeri 2007 · Franconeri, Jonathan & Scimeca 2010 · Scheibehenne,
Greifeneder & Todd 2010 and their 2010 reply · Chernev, Böckenholt & Goodman · Dean, Ravindran &
Stoye 2025 · Beierle et al. 2019 · Katz & Byrne · Springer & Whittaker 2018 · Liu et al. CHI 2020 ·
Cockburn, Gutwin & Greenberg 2007 · Pirolli & Card 1999 · Pirolli, Card & Van Der Wege · Granka,
Joachims & Gay · Joachims et al. · Drew et al. 2014 · van der Sijs et al. 2006 · Ancker et al. 2017 ·
Bonafide et al. · Mark, Gonzalez & Harris 2005 · Joint Commission SEA 50 · WCAG 2.2 SC 2.2.2 / 2.3.1
/ 2.3.3 and their Understanding documents · W3C Technique C39 · CSS Media Queries Level 5 (CSSWG
source) · Layout Instability API editor's draft · web.dev CLS, *Evolving the CLS metric*, and the
threshold-derivation page · MDN `prefers-reduced-motion` · HTTP Archive Web Almanac 2025 · NN/g
progressive disclosure, accordions and scrolling-and-attention.

**Could not be reached — 33 sources.** The largest classes: paywalled clinical and vestibular
literature (Staab, Bronstein, Agrawal, reached only in part by the refutation pass), ACM DL entries
with no author copy, Höffler & Leutner 2007 in primary, and Kelly & Azzopardi (SIGIR 2015), which
was **dropped rather than badged** — the bibliographic details verify and no one in the chain saw
the Methods or Results, so a badge would have been a claim about a paper nobody read.

**Not covered at all**, named because a pack reporting only what it read has not reported its
coverage:

- **Expertise reversal.** `DESIGN_LANGUAGE.md` §10 asserts *"explanations that help novices actively
  slow experts"* with no source. Same shape as the three questions above — a shipped design rule
  with a psychology-flavoured justification and no citation — and the obvious next question.
- **Split attention and worked examples**, cited in the same §10 list.
- **Typography and the three-voice rule**, which §11's open items already record as untested.
- **Attestation polymorphism**, §09's other recorded open item.
- **Intentional binding / sense of agency**, which came up as a candidate mechanism for §04 and was
  dropped: Grünbaum & Christensen argue the measure is not established.
