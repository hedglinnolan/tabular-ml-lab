# The quest log on Models and Results: a critique

Written 2026-10-10 by an Opus critic at max effort, at Nolan's request ("an art critique about the whole situation"), on the screens of `design/quest-models-stage` @ 4ea11554. Shown to Nolan as https://claude.ai/artifact/V1ztXKWyTSRtEZJBuAikA7. His rulings on it are FOUNDATION §0. The sketch paths below were session scratch files and are not kept.

## 1. The verdict

What you're feeling is the gap between a correct interface and a meaningful one. Every rule you set is followed: one action in one place, one indigo, plain names, no chips, seven stages, the bar, the sweep, the lock. The result is a calm, polite form that walks you through the engine's checklist in the engine's order. It is calm because things were taken away, not because things are clear. Nothing on these screens shows you your own question back:

- The picture beside "How should the analysis account for how much people eat overall?" is a histogram of fat, which you aren't studying.
- The recommended answer previews as "No nutrient changes".
- Two of the five available options give exactly the same estimate.
- The first number of the whole journey arrives as a list of radio buttons with sentence templates.

The app knows the things that would feel like magic: that your earlier answer already settled the energy model, that your result runs the wrong way, and that a 2% confounder would erase it. It hides them in a greyed-out option's footnote and the last line of the panel. The magic you described lives in two places: the "return" ("because you said X, here is Y") and the interpretation after Fit. Neither is on screen. That is "not quite there."

## 2. What it is trying to be, and what it is

**Trying to be:** an apprenticeship. The researcher's understanding becomes the visible spine of the analysis. Each piece of modeling "art" becomes a consequence you can see on your own data. A long journey stays bearable because of the bar, and becomes meaningful because what you understood keeps coming back and ends as a paper you can defend.

**What it is:** a faithful, nicely styled rendering of the engine's decision log.
- The card lists the engine's lines in the Router's order. Its three headings are the engine's tiers renamed: asked, stated and silent became Decide, Confirm and For the record.
- The tapestry draws the engine's own preview views.
- The triage shows the engine's recommended dispositions word for word.
- The numbers are the engine's raw per-gram coefficients.

It reads like a literal translation: every sentence is accurate, and the argument is gone. Squint at the seven screens and you can barely tell Models from Results. Each has the same beige column of label/value pairs, the same blue-grey box and one indigo button at the bottom.

**Walked as a researcher:**
- **At rest:** examined. It feels like multiple choice with eight near-synonyms and a crib note marked "Recommended".
- **Hover:** briefly curious, then let down. The picture moves; the meaning doesn't.
- **Sweep:** processed. It is quick, while the right half of the screen says "changes nothing here".
- **Triage:** either alarmed ("misread zeros in my exposure, and you want me to sign them off?") or trained to rubber-stamp.
- **Flowchart:** the first feeling of ownership.
- **Results:** anticlimax. The answer to the whole quest arrives as paperwork.

## 3. What works (keep these)

- **The color link between an option and its effect.** Pointing at an option tints both the option and what it touches: grey is your data now, indigo is this choice. You learn it in one hover. This is the core mechanic you named, and the visual system is right even where the content isn't.
- **The quiet technical name on the option's top edge** ("Known as Willett residual model, total energy kept"). It is the best two-language device in the project: plain words to choose by, and the colleague's term arriving exactly when you look.
- **The greyed-out option that remembers you:** "Not possible: you asked for sugar in place of other calories, a swap this cannot estimate." It is the only "because you said" on any screen. It is the seed of the whole fix, and it is hidden in a disabled option.
- **The analysis flowchart (screen 5).** It is the best screen: your whole analysis before you commit, "Nothing has been estimated yet", and "Fit · under a second" at the end of the pipe.
- **The Confirm sweep and the lock line.** The sweep respects your time. "Plan fixed at 03:30 UTC" makes commitment visible, which is what makes the results believable.
- **Restraint in light mode:** warm paper, a cool lightbox for data, one typeface family. It is a real advance on the 10-05 prototypes. Keep the palette and change what is drawn with it.

## 4. What fails, ranked

### Deep problems

**1. It asks what it already knows. The screens follow the engine's dependency graph, not the researcher's reasoning.**
- A few lines above the energy question, the card states what you study: "sugar, all of its effect, in place of other carbohydrate, with total calories unchanged." The answer is printed above the question.
- The question is then posed fresh, with eight options. "Why does this matter?" ends "Choose by the question.", but the question was already chosen.
- FOUNDATION §2's own example of a plain question is "Should sugar's calories replace other calories, or add to them?" The screen instead offers a grid:
  - "Keep total calories in the model"
  - "Calorie-adjusted, total calories kept"
  - "Per calorie, total calories kept"
  - "Calorie-adjusted, total calories dropped"
  - "Per calorie, total calories dropped"
- Two of the five available options are the same model written two ways. I fitted both on the journey's own table: the residual method with calories kept gives the standard model's sugar coefficient exactly (−0.019934 both; they differ by 2e-14).
- Why it matters: the magic beat in the understanding layer is "Return", where later cards visibly descend from earlier answers. Here every card starts from zero. A researcher feels examined, not understood.

**2. The tapestry draws mechanics, not meaning.**
- At rest it shows a table of nutrient means plus a histogram and scatter of fat_total ("the views follow fat_total, the nutrient closest to total calories"). You are studying sugar.
- Point at the Recommended option and it says "No nutrient changes" and stays grey. The choice that defines what your coefficient means looks like it does nothing.
- Point at the option that gives the identical answer and you get the most dramatic picture on the screen: seven bars, and "r 0.88 → 0.00" on every row, which is zero by construction.
- On the sweep, the panel says "One effect for everyone changes nothing here." Hovering the alternative lights up a fan of columns, while its caption says what the models read "stays the same".
- The routing diagram's middle column is headed "Roles · 11" and repeats the same eleven column names; no role is ever named.
- Why: the closed view vocabulary can only draw changes to data. The decisions that matter most in Models change the question (what is compared with what) and leave the data alone, so the tapestry draws side effects.
- The lesson sits behind the "Why" disclosure ("swaps calories at a fixed total; splitting calories by source adds calories"), and the picture shows something else. The tapestry earns its 62% of the screen on exactly one screen: the flowchart.

**3. Results doesn't interpret. The climax is paperwork.**
- The largest word on the card is "Results"; the finding is never a headline. It arrives as three wording radio buttons.
- Then comes a Placement control whose only enabled segment is the one already chosen: "Table 2 holds the plan's main estimate, so it always stays in Results."
- The estimate is given per gram: "−0.0199 (−0.0327 to −0.00718)". Per 25 g of sugar (100 kcal, the engine's own swap step) it is 0.50 mg/dL lower (0.18 to 0.82), a number a person can feel.
- More sugar with lower glucose runs against expectation, and nothing says so. In this very table, people with fasting glucose in the diabetic range get less of their energy from sugar (19% against 22%). That is the pattern you'd see if people cut sugar after a diagnosis, the usual suspect being reverse causation. I computed this outside the engine; it reads the outcome, so it belongs after the lock.
- The fragility is the panel's last line: E-value 1.02, robustness value 2.3%. Even so, the Recommended wording is the causal one.
- "Which of my decisions mattered?" is promised in FOUNDATION §5 and backed by the engine's materiality instrument. It is absent.
- Your brief was: "help them interpret with the tapestry". The screen curates but does not interpret.

**4. Trust leaks on screen.**
- **Triage defaults.** "152 cells in bp_di and 8 more hold 5.4e-79: a zero in a SAS transport file, misread." The default is "kept as limitations", and the button says "Confirm all 3". Point at that row and the tapestry lights sugar, kcal and every nutrient: the picture tells the truth the card won't.
- **The engine's own to-do list as a finding.** "gender is two-level text (female, male); which level counts as 1 is not asked yet."
- **A blocking question hidden in the wrong drawer.** "For the record · 2" (the drawer for things that don't matter) holds a question the fit waits on: "Tell me about these columns: age: an amount? … (9 whole-number values from 2,001 to 2,017…)". The years are even printed with thousands separators.
- **Two different swaps on one card.** Screen 5's card says what you study is sugar "in place of other carbohydrate". Six lines down it says "The swap: sugar to protein, 100 kcal at a time". Nothing says the second is an extra comparison, and Results never shows it. The new "one phrasing" test checks the rest and Results screens, not this one.
- A methods person sees each of these within seconds. They are mostly engine and fixture issues, but they sit on the screens you judged and mask whatever the design got right.

**5. The size hierarchy puts the app's structure above your content, and the copy reads like a telegram.**
- The biggest type on every card is the stage name, which the quest line already shows. Your question is 22 px. Your own words are 16-px body text under a grey label.
- The two stylesheets use fourteen font sizes, eight of them between 13 and 17 px. Hierarchy is carried by grey, not by size.
- The copy, quoted: "Same swap, on what calories don't explain." "Differs when age and such rise with calories." "Read as one day's intake, 501 of 21,849 rows fall outside 500–5,000: record kcal's days first."
- "age, gender and 7 more adjusted for": the "7 more" hides the nutrients, and they are what define the swap.
- Raw column tokens are everywhere (fat_mon, bp_di), even though Your data reads a codebook.
- The word budget cut 296 words to 238 by compression, and compression costs decoding. The plain names invented a third vocabulary that neither the epidemiologist ("residual method") nor the outsider ("swap or add?") speaks.

**6. No arc, and the quest exists only in the top line.**
- Hovering a stage name does nothing. The "Manuscript · 14" rail does nothing.
- Those are the two things you liked about the quest log, and they are the two things not built. You were shown the parts you didn't ask for.
- Nothing builds up on screen: not your understanding, not your paper.
- Screens 3 and 4 show the same diagram. The lock is a 15-px line, Fit jumps straight to Results, and the estimate is a bold table row.
- The bar gives progress; nothing gives meaning. The pace never builds before Fit or releases after it. Calm has become monotone.

### Cosmetic (briefly)
- At 1440×900 you see two and a half of the eight options; at 1280×800, two.
- The nutrient table leaves about 450 px of empty space between each name and its numbers.
- Table 2's interval plot is squeezed into a 160-px column, and the decimal places are uneven.
- Dark mode has three color temperatures: warm ground, a cold blue-black tapestry and warm boxes inside it. The tapestry reads as a hole.
- Indigo does three jobs: the button, the selected option and "what this choice touches" (#3B4AB9 vs #4954C2). On the hover screen, the biggest indigo shape is a histogram you can't click while Continue is grey.
- The "Your data now / With this choice" switch is disabled on most screens.
- The SHA-256 sits on Results' first line.
- The flowchart card and the flowchart list the same decisions twice.
- At 1024 px, words break mid-word ("recorde d").

## 5. The situation

**The process produced an interface shaped like the engine.** The pipeline was:
1. The crosswalk (708 items).
2. FOUNDATION (568 lines of rulings).
3. A designer rendering the engine's replayed quest log, "with the engine's sentence where it is plain enough to stand".
4. A reviewer auditing compliance: the five-second test, word counts, device counts.
5. A reviser fixing the audit.

The designer's freedom was whatever the rules left over: 23 small decisions about label sizes and line order. Nobody in the loop owned your question: what should a researcher understand on this screen, and how should they feel? The tests confirm one primary action per state. The design passes every test you wrote down and fails the one you didn't: is it magic?

**The fairness rule froze the thing you're reacting to.** On 10-05 the four structures shared one kit so that only the structure differed. That made the comparison fair, but it meant the core interaction (question card, radio options, preview panel) was never varied. You picked a navigation shell. The screen you're judging is the 10-05 kit with a new header.

**The fixture shows engine flaws without interpreting them:** misread zeros, a Model 2 that adjusts fat_total alongside its own parts, a swap pair at odds with the stated comparison, and a result shaped like reverse causation. The app catching exactly these things *is* the magic. The demo should show it catching them.

**The two pending rulings are symptoms.**
1. *"Do a question's own options count toward the 120-word budget?"* The right question: when your earlier answers already settle a question, should the card say so ("Because you said…, so…") and put the alternatives behind "I meant a different comparison"? Under that rule the energy card drops below 100 words and the budget question disappears. Better still, budget cards in decisions (one real decision per screen), not words.
2. *"Which wording is Recommended for an observational design?"* Association is the right floor for cross-sectional NHANES. The right question is whether Results should interpret first (what's surprising, what's fragile, which choices moved the answer). It also asks whether "Recommended" should be computed from that evidence rather than set by policy.

The ruling nobody has asked you for, and the one that unblocks the most: **may the tapestry draw what a choice means (the comparison it makes) even when no column changes?** FOUNDATION §5 rule 9 already allows new kinds of view by a recorded decision.

## 6. Directions

**A · The question, drawn (meaning first).** The tapestry draws the comparison each choice makes, on your own data and without reading the outcome. The card speaks in returns.
- **Models at rest:**
  - The card says "Your question already decides how calories are handled. You said: sugar, in place of other carbohydrate, at the same total calories. So total calories stay in the model." It has one button, "Yes, that's my question", and a link, "I meant a different comparison".
  - The tapestry draws an average day in your data as one calorie bar (sugar 116 g, other carbohydrate 142 g, protein 81 g, fat 81 g). Below it is the day your analysis imagines: 25 g more sugar, 25 g less other carbohydrate, the same total.
  - The alternatives are other pictures: "add" makes the bar longer; "share" shrinks every slice a little. The residual method is "the same swap, estimated another way".
  - Underneath is the Methods sentence this choice will write.
- **Results:**
  - The headline is the finding in your own comparison and human units: 0.50 mg/dL lower per 25 g (95% CI 0.18 to 0.82).
  - Then come three things a reviewer will ask about, each with its quiet technical name: it runs against expectation (reverse causation), it is fragile (E-value 1.02), and it is diluted (one day of recall, regression dilution).
  - Then a Recommended wording computed from them, and one button: "Put this in my paper".
  - The tapestry's main picture is how the estimate moved as you adjusted: −0.86 → −0.66 → −0.50 → −0.56 per 25 g, all real. Below it, "which of your other choices mattered", filled by the materiality instrument.
- **Gives up:** the data-change preview as the main picture (it moves to "More angles"). Each family of questions needs its comparison drawn once.
- **Risk:** authoring cost, though a dozen question families cover most of Models. The illustrative picture can be mistaken for an estimate. A "because you said" that is ever wrong feels presumptuous.

**B · The paper is the quest log.** Methods and Results are the main surface. Each open sentence is the question, the options are alternative phrasings of it, and the seven stages are the paper's outline.
- **Models at rest:** a typeset page. The open sentence reads "Total energy intake was handled with [the standard multivariate model], so that…", and the evidence sits beside it.
- **Results:** the paragraph and Table 2 appear in place on the page, and placement means moving exhibits within the outline.
- **Gives up:** the plain-language card as the main surface, and much of the tapestry's width. It also bends your quest-log ruling.
- **Risk:** sliding back toward the busy 10-05 document prototype. The reward is the return beat made literal: you watch your paper write itself.

**C · The map is home.** Screen 5 becomes the persistent surface, filling in node by node from the first stage. The open question is an empty slot on the map.
- **Models at rest:** People → What the model reads → [Calories: open] → Shape → Models across the top, with the card docked below. Pointing at an option re-routes kcal on the map.
- **Results:** the four estimates sit at the map's right edge, and each choice's line is drawn as thick as it moved the answer.
- **Gives up:** previewing each option on the data as the main picture.
- **Risk:** it makes plumbing the hero, the engine's shape again in a nicer frame. It is the strongest of the three for teaching where results come from, and the weakest for meaning.

**Sketches** (rough, not builds; every number is from the fixture, with grey placeholders where the engine would compute):
- Direction A, Models at rest and Results, light and dark: `/private/tmp/claude-501/-Users-nhedglin-tabular-ml-lab/a12607bd-3f0c-4a23-8d52-bda4a2c1292e/scratchpad/critique/sketch-A-meaning-first.html` (PNGs: `sketch-A-meaning-first-{1,2}-{light,dark}.png`)
- Directions B and C as four wireframes: `/private/tmp/claude-501/-Users-nhedglin-tabular-ml-lab/a12607bd-3f0c-4a23-8d52-bda4a2c1292e/scratchpad/critique/sketch-B-C-alternatives.html` (PNGs: `sketch-B-C-alternatives-{1..4}-{light,dark}.png`)
- The current screens 1 and 6 with 16 numbered marks: `/private/tmp/claude-501/-Users-nhedglin-tabular-ml-lab/a12607bd-3f0c-4a23-8d52-bda4a2c1292e/scratchpad/critique/sketch-annotated-current.html` (PNGs: `sketch-annotated-current-{1,2}.png`)

**Recommendation: A.** Keep C's flowchart as the end of Models, as it is today, and make B's manuscript the rail, actually opening and updating with each "because you said" sentence. A is the only direction that attacks the root problem (meaning) while keeping every binding ruling: the quest line, the bar, one action, the tapestry's width and the hover.

Most of what A needs already exists in the engine: the comparison each option implies, the exact-equivalence result (the materiality instrument's new theorem field), the 100-kcal swap step, and the named failure patterns from the understanding layer.

Before any of it, fix the four trust leaks. They are cheap, and each one hides any design gain.

## 7. What to show Nolan next, and what to ask

Show these in order:
1. The annotated current screens.
2. Direction A's two frames, side by side with the current screens 1 and 6.
3. The B and C wireframes, for half a minute.

Hold the two pending rulings until he has picked a direction; under A, both disappear.

Ask him:

> "When you said 'not quite there', is this it: the screens ask you to operate the engine, when you wanted them to hand your own question back — what you're comparing, what could fool you, and what you found? If so: may the tapestry draw what a choice means even when no column changes, and may a question your earlier answers already settle become a 'because you said' confirmation?"

---
Both checkouts are untouched (`git status` is clean). My interaction screenshots are in `/private/tmp/claude-501/-Users-nhedglin-tabular-ml-lab/a12607bd-3f0c-4a23-8d52-bda4a2c1292e/scratchpad/critique/shots/`. I rebuilt the earlier calm and living-methods prototypes into `/private/tmp/claude-501/-Users-nhedglin-tabular-ml-lab/a12607bd-3f0c-4a23-8d52-bda4a2c1292e/scratchpad/critique/calm-dist/index.html` and `.../critique/protos-dist/index.html`.