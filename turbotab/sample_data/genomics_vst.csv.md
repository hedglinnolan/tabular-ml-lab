# `genomics_vst.csv` — VST-shaped

**Derived** by size-factor-normalizing the counts and taking
`log2(x + 4)` — which reproduces VST's two defining features without pretending
to be DESeq2's estimator.

- rows are **samples** (60), gene columns 495
- continuous, max well under 25, **no negatives**
- a **repeated floor at 2.0**, shared by every zero count

**Signature** — §02 row 6: *continuous, max ~15–25, repeated floor, roughly
homoscedastic → VST.*

**The coaching this fixture must trigger is the load-bearing part.** §02:
VST and rlog are for visualization, clustering and PCA, and are **never** the
input to a DE test — `SETTLED`. A classifier that recognizes the shape and does
not say that has produced a label rather than a diagnosis.

**Not covered**: rlog, §02 row 7, which is *as above but small negatives
permitted*. `wide_assay.csv` is symmetric around zero and exercises row 9
instead; a true rlog fixture would need a negative floor and is not shipped.
