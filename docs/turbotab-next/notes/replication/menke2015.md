# Replication feasibility: Menke et al., JAMA 2015 (diabetes prevalence and trends, 1988–2012)

Prepared 2026-10-15. Every fact is tagged:
- **[V: url]** means verified at that URL in this session.
- **[U]** means unverified: inferred or recalled.

Numbers are taken from the paper's own text, tables and supplement, which were read in a browser this session. None come from memory.

---

## 0. Verdict

**Approximate replication: feasible. Exact replication: not guaranteed.**

All the data the paper used are public and still downloadable:
- NHANES III (1988–1994), as fixed-width ASCII;
- continuous NHANES 1999–2012, as SAS XPORT.

Nothing needs the RDC.

The definitions, thresholds and age-standardization weights are stated exactly. The published sample sizes give a hard first gate.

The differences that stand in the way of a byte-exact match are:
1. **A composite weighting scheme that is not fully specified.** The paper says only that interview, MEC and FPG (or OGTT) weights were combined, and points to Cowie 2009 (ref 2). [V: JAMA full text]
2. **The FPG calibration equations are not named.** The paper says it calibrated FPG in 2005–2012 to the earlier surveys "as recommended by the NCHS". The equations NCHS published are listed in §2.6, and the chain we would apply is an inference. The paper does not say whether 2-hour glucose was calibrated too.
3. **Unspecified handling details:** missing HbA1c or FPG, DIQ010 "borderline" or don't-know answers, which pregnancy variable was used, NHANES III fasting-time edits, education coding in NHANES III, and the cut points of the poverty income ratio (PIR) tertiles.
4. **Files revised after the analysis.** NCHS replaced six of the needed XPT files after early 2015. Archived 2015 copies exist in the Wayback Machine (§3.3).
5. **Variance and CI conventions.** SUDAAN 10.0.1 was used, with a mix of logit-type and Wald CIs (§4.6). TurboTab's design degrees of freedom differ slightly from SUDAAN's default.

---

## 1. Citation

| Item | Value | Status |
|---|---|---|
| Title | Prevalence of and Trends in Diabetes Among Adults in the United States, 1988-2012 | [V: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=26348752&retmode=xml] |
| Authors | Menke A, Casagrande S, Geiss L, Cowie CC | [V: same] |
| Journal | JAMA 2015;314(10):1021-1029; published 8 Sep 2015 | [V: same; Unpaywall gives published_date 2015-09-08] |
| PMID | 26348752 | [V: esearch https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=Menke+A[au]+AND+Cowie+CC[au]+AND+2015[dp]+AND+diabetes+prevalence+trends] |
| DOI | 10.1001/jama.2015.10029 | [V: efetch] |
| PMCID | **None.** The paper is not in PMC: idconv returns "Identifier not found in PMC", and elink has no pubmed_pmc link. | [V: https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=26348752&format=json] |
| Open access | Unpaywall says "closed", but the JAMA page carries a FREE badge, and the full text and supplement loaded without login. | [V: https://api.unpaywall.org/v2/10.1001/jama.2015.10029 ; https://jamanetwork.com/journals/jama/fullarticle/2434682] |
| Full text | https://jamanetwork.com/journals/jama/fullarticle/2434682 (article id 2434682; internal file id joi150096) | [V, read in Chrome. curl gets a Cloudflare 403 challenge, so use a browser.] |
| Supplement | eTables 1–9 PDF, 15 pages, 150,070 bytes: https://cdn.jamanetwork.com/ama/content_public/journal/jama/934405/joi150096supp1_prod.pdf. The link needs a signed query string taken from the article page. | [V, rendered in the browser and text extracted with pdf.js; not saved to disk] |
| Funding | NIDDK contract GS10F0381L | [V: efetch and full text] |
| Comments/letters | Editorial (PMID 26348749); letters (PMIDs 26881376, 26881377); Am J Kidney Dis comment (PMID 26725312). No erratum is listed. | [V: efetch CommentsCorrections] |
| Weighting reference (ref 2) | Cowie CC et al. Diabetes Care 2009;32(2):287-294. PMID 19017771, PMC2628695, DOI 10.2337/dc08-1296 | [V: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=2628695] |

Corrections to the "believed" premises in the brief:
- **Periods.** The trend periods are NHANES III plus all seven 2-year cycles from 1999-2000 to 2011-2012. The 1999-2012 cycles are not pooled. [V: Table 3]
- **Definition used for trends.** The primary trend definition is **HbA1c or FPG**, without 2-hour glucose. The definition that includes 2-hour glucose is used only for 2011-2012 (Table 1) and for ages 40–74 in NHANES III and 2005-2012 (eTables 7–9). [V: full text]
- **Standard population.** It is **the 2011-2012 NHANES interview population**, not the US 2000 Census. [V: full text]

---

## 2. What data the authors used

### 2.1 Cycles and periods [V: full text, Table 3]

| Period | Survey | Trend analytic n | n with diabetes |
|---|---|---|---|
| 1988-1994 | NHANES III | 8478 | 2136 |
| 1999-2000 | continuous | 2168 | 606 |
| 2001-2002 | continuous | 2479 | 669 |
| 2003-2004 | continuous | 2299 | 676 |
| 2005-2006 | continuous | 2191 | 635 |
| 2007-2008 | continuous | 2901 | 991 |
| 2009-2010 | continuous | 3118 | 955 |
| 2011-2012 | continuous | 2781 | 910 |

- The 1988–2010 sizes sum to 23,634, which matches the abstract's 23,634 adults. The 2011-2012 sample is 2,781 for the HbA1c-or-FPG definition. [V]
- With the 2-hour definition, 2011-2012 has n = 2623 (971 with diabetes). [V: Table 1]
- For ages 40–74 with the 2-hour definition, n = 4201 (1988-94), 1217 (2005-06), 1671 (2007-08), 1801 (2009-10) and 1559 (2011-12). [V: eTable 7]
- Population: adults aged 20 or older, morning examination session only, pregnant women excluded. [V: full text]

### 2.2 Components, files and variables

The paper names the measures but not the files or variable names. The mapping below comes from NCHS documentation. Each file-to-measure link is **[U]** unless marked otherwise; each variable's existence is [V] at the documentation URL.

| Measure | NHANES III (1988-94) | 1999-2000 | 2001-2002 | 2003-2004 | 2005-2012 (D/E/F/G) |
|---|---|---|---|---|---|
| Self-reported diagnosis | adult.dat **HAD1** (ever told you have sugar/diabetes). HAD3 and HAD4 cover the pregnancy follow-ups. [V: adult.sas] | DIQ: **DIQ010** | DIQ_B: DIQ010 | DIQ_C: DIQ010 | DIQ_D..G: DIQ010 [V: DIQ_G.htm] |
| Fasting plasma glucose | lab.dat **G1P** (mg/dL) with **PHPFAST** (hours fasted) [V: lab.sas] | LAB10AM: **LBXGLU**, WTSAF2YR, WTSAF4YR | L10AM_B: same | L10AM_C: LBXGLU, WTSAF2YR | GLU_D..G: LBXGLU, WTSAF2YR |
| 2-hour OGTT glucose | lab.dat **G2P**; ages 40–74 only [V: lab.sas; age limit from Cowie 2009 and Menke] | not done | not done | not done | OGTT_D..G: **LBXGLT**, **WTSOG2YR** [V] |
| HbA1c | lab.dat **GHP** (glycated hemoglobin %) and GHPMETH [V: lab.sas] | LAB10: **LBXGH** | L10_B: LBXGH | L10_C: LBXGH | GHB_D..G: LBXGH |
| Demographics and design | lab.dat or adult.dat: HSAGEIR, HSSEX, DMARETHN, DMPPIR, SDPPSU6, SDPSTRA6, WTPFQX6, WTPFEX6, **WTPFSD6** (total morning subsample weight), WTPFHSD6 [V: lab.sas] | DEMO: RIDAGEYR, RIAGENDR, RIDRETH1, DMDEDUC2, INDFMPIR, RIDEXPRG, WTINT2YR/4YR, WTMEC2YR/4YR, SDMVPSU, SDMVSTRA | DEMO_B | DEMO_C | DEMO_D..G. 2011-12 adds **RIDRETH3**, which has a non-Hispanic Asian category [V: DEMO_G.htm] |
| Education | adult.dat **HFA8R** (highest grade completed) [V: adult.sas] | DMDEDUC2 | same | same | same [V: DEMO_G.htm] |
| Pregnancy | exam.dat **MAPF12R** (pregnancy status recode) [V: exam.sas]. Which variable the authors used: [U] | RIDEXPRG in DEMO [V: DEMO_G.htm]. The authors' choice: [U] | | | |
| BMI (Fig 2, BMI columns) | exam.dat BMPBMI [V: exam.sas] | BMX: BMXBMI [U on name] | BMX_B | BMX_C | BMX_D..G |

- The question wording changed between surveys. NHANES III asked whether a "doctor" had ever told the respondent they had diabetes; 1999-2012 asked about a "doctor or other health professional". [V: full text]
- DIQ010 asks women 20 or older about diabetes "other than during pregnancy". [V: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DIQ_G.htm]
- Not stated in the paper: how DIQ010 = 3 (borderline), 7 or 9 were coded, and how NHANES III HAD3/HAD4 (told only during pregnancy) were handled. [U]

### 2.3 Definitions [V: full text; Tables 1–3 footnotes; eTables 1–9 footnotes]

- **Diagnosed diabetes:** a self-reported previous diagnosis.
- **Definition A** (HbA1c, FPG or 2-hour glucose):
  - undiagnosed means no diagnosis and HbA1c ≥ 6.5%, FPG ≥ 126 mg/dL, or 2-hour glucose ≥ 200 mg/dL;
  - total means diagnosed or undiagnosed.
- **Definition B** (HbA1c or FPG): undiagnosed means no diagnosis and HbA1c ≥ 6.5% or FPG ≥ 126 mg/dL. Definition B is used for every trend in the main text.
- **Prediabetes** applies only to people without diabetes. Under A it is HbA1c 5.7–6.4%, FPG 100–125 mg/dL, or 2-hour glucose 140–199 mg/dL. Under B the 2-hour criterion is dropped.
- **Tests required for "undiagnosed":** the paper does not say whether every test must be present. Logically, a person with no diagnosis needs at least FPG and HbA1c to be classed as not undiagnosed. The rule for missing values is **[U]**.
- **Wording slip in the supplement:** the footnotes of eTable 3 and eTable 5 say "A1c ≥ 6.5% **and** FPG ≥ 126". The main text and the other footnotes say "or", so "or" is the definition. [V]

### 2.4 Subsamples and weights

What the paper states [V: full text]:
- Morning-session data (8- to 24-hour fast) are used.
- For definition B, a combination of the interview, MEC and FPG weights was used. For definition A, the OGTT weights were added to that combination.
- The weights were combined so that they sum to the civilian, noninstitutionalized US population. Ref 2 is cited for this.
- The 2-hour-glucose estimates rest on a different subset (the OGTT sample) from the FPG estimates, so the totals and the diagnosed prevalence can differ between Tables 1 and 2: 9.1% versus 9.2% diagnosed.

What ref 2 (Cowie 2009) specifies [V: PMC2628695 efetch]:
- **2005-2006:** people with diagnosed diabetes **from the interviewed sample** were combined with people **without** diagnosed diabetes from the FPG subsample (or the OGTT subsample). Weights were set so the two groups sum to the US population.
- **1988-1994:** prevalences in the subsamples of people without diagnosed diabetes were each **adjusted for the interview-sample prevalence of diagnosed diabetes**. In effect, P(total) = P(dx) + (1 − P(dx)) × P(undx | no dx).
  - The reason given: the NHANES III FPG and OGTT subsample weights do not account for invalid or unknown fasting times or for missing glucose.

Inference: Menke 2015 used this same composite **[U]**. The published sizes support it:
- 2011-2012 definition B: 910 of 2781 (33%) have diabetes, unweighted.
- NHANES III: 2136 of 8478 (25%).
- A fasting-only adult sample would be far less diabetic. A count of diagnosed people from the interview or MEC sample plus non-diagnosed people from the fasting subsample fits these totals.
- First replication gate: reproduce n = 2781 / 910 and n = 8478 / 2136 under both candidate constructions.

NCHS weight facts [V]:
- **WTSAF2YR:** the fasting subsample weight, nonzero for people fasting 8 to under 24 hours.
  - It also covers diabetics who fasted but lack glucose. Diabetics who did not fast get zero.
  - Use WTSAF4YR for 1999–2002 combined, and WTSAF2YR when a single cycle is analyzed, as here.
  - Sources: [LAB10AM.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/LAB10AM.htm), [L10AM_B.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L10AM_B.htm), [L10AM_C.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L10AM_C.htm).
- **WTSOG2YR:** use it whenever OGTT glucose enters the analysis.
  - OGTT exclusions are hemophilia, chemotherapy, fasting under 9 hours, insulin or oral diabetes medicines, refusing phlebotomy, and not finishing the drink.
  - In 2009-2010, 9 people who were pregnant by lab test have OGTT weights, and 17 diagnosed diabetics with OGTT values were weighted in the diagnosed post-stratum.
  - Sources: [OGTT_D.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/OGTT_D.htm), [OGTT_E.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/OGTT_E.htm), [OGTT_F.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/OGTT_F.htm).
- **NHANES III:** the public lab file carries WTPFSD6 (total morning subsample) and WTPFHSD6 (morning MEC + home), and no separate OGTT weight. [V: https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.sas]
  - Exactly how Menke weighted NHANES III, presumably the Cowie ratio approach with WTPFQX6 and WTPFSD6: **[U]**.

### 2.5 Age standardization [V: full text, Table 1 footnote b, Table 3 footnote a]

- **Standard:** the overall 2011-2012 NHANES **interview** population.
  - 20–44 y: 0.4561; 45–64 y: 0.3673; ≥65 y: 0.1766 (sum 1.0000).
  - Applied to every stratified 2011-12 estimate and to all trend estimates.
- **What is crude in Table 1 and Table 2:** the "overall" and age-group rows are unadjusted. Sex and race/ethnicity rows are age-standardized.
- **Ages 40–74 (eTables 7 and 9):** these use groups 40–44, 45–64 and 65–74 against the same standard. **The weights for these groups are not published** [V: eTable 7 footnote]. They would need to be re-derived from DEMO_G WTINT2YR **[U]**.
- The ratio "% undiagnosed" is age-standardized as well (eTable 6). How a standardized ratio was formed is not stated. It is likely a ratio of standardized prevalences or a standardized domain proportion **[U]**.
- Check after downloading: the three weights should be reproducible from DEMO_G (sum of WTINT2YR by age band, ages ≥ 20). This is a cheap first known-answer test **[U]**.

### 2.6 Cross-cycle calibration

What the authors did [V: full text]:
- **FPG:** 2005–2012 was calibrated to the earlier surveys "as recommended by the NCHS". No equations are printed.
- **HbA1c:** no calibration, because NHANES recommends none.
- **2-hour glucose:** calibration is not mentioned.
- **NHANES III versus 1999–2004:** no calibration is mentioned.

What NCHS publishes [V]:
- **2005-06 versus 2003-04** (Hitachi 911 versus Cobas Mira):
  - Y(Cobas Mira) = 0.9835 × X(Hitachi 911), with no significant intercept.
  - The reverse is Y(Hitachi 911) = 0.9815 × X(Cobas Mira) + 3.5707.
  - Source: [GLU_D.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/GLU_D.htm). The same equation is printed in [OGTT_D.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/OGTT_D.htm).
  - **It was already present in February 2015**, before the paper. [V: https://web.archive.org/web/20150221154732/http://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/GLU_D.htm; that page was then marked last revised September 2008.]
- **2007-08 versus 2005-06** (Roche ModP versus Hitachi 911, Deming regression):
  - Y(Roche 911) = X(ModP) − 1.139; Y(ModP) = X(911) + 1.148.
  - Source: [GLU_E.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/GLU_E.htm); also printed in OGTT_E.htm.
  - Present in May 2015. [V: http://web.archive.org/web/20150519022333/http://wwwn.cdc.gov:80/nchs/nhanes/2007-2008/GLU_E.htm]
- **2009-10 and 2011-12:** no change in glucose equipment, method or lab from 2007-08. [V: GLU_F.htm, GLU_G.htm]
- **Implied chain to the 1999–2004 scale [U]:**
  - 2005-06: FPG × 0.9835.
  - 2007-12: (FPG − 1.139) × 0.9835.
- **HbA1c:** NCHS **withdrew** the 2005-06 → 2003-04 HbA1c crossover regression in November 2011 and advises using the original data, including for trends. This is consistent with the paper.
  - GHB_E and GHB_F were withdrawn and then re-released unchanged in March 2012. The analytic note describes an unexplained right shift in 2007-2010 HbA1c: ≥ 6.5% rose from 5.8% to 7.3% between 2005-06 and 2007-08.
  - Source: [GHB_E.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/GHB_E.htm), [LAB10.htm](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/LAB10.htm). [V]
- **Today's glucose documentation** still carries the same equations. GLU_D and GLU_G are marked "Last Revised: August 2016", and the visible additions concern insulin, not glucose. [V]

### 2.7 Exclusions and sample sizes

- Stated [V: full text]:
  - pregnant women;
  - people not in the morning session;
  - fasts outside 8–24 hours, implied by the morning-session text;
  - people under 20.
- Not stated [U]:
  - missing HbA1c or FPG;
  - missing self-report;
  - borderline, don't-know or refused answers;
  - people with diagnosed diabetes who were not fasting.
- Race in Tables 1 and 2: 69 people with "other" race appear only in overall and non-race analyses. In Table 3, 2,740 do. [V: Table 1 footnote j; Table 3 footnote d]
- Analytic sizes are listed in §2.1. [V]

---

## 3. Public availability today (HEAD checks on 2026-10-15)

### 3.1 Continuous NHANES 1999–2012

- The URL pattern is `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{begin-year}/DataFiles/{FILE}.xpt`, with documentation at `.htm`.
- The legacy pattern `https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT` now returns a 20,905-byte HTML page, not data. [V]
- All 39 files below returned HTTP 200, with matching .htm documentation also returning 200. [V]

| Begin year | File | HTTP | Bytes | Last-Modified |
|---|---|---|---|---|
| 1999 | DEMO.xpt | 200 | 11500560 | Thu, 22 May 2014 |
| 1999 | DIQ.xpt | 200 | 1294240 | Thu, 22 May 2014 |
| 1999 | LAB10AM.xpt | 200 | 210960 | Thu, 10 Mar 2016 **changed after 2015** |
| 1999 | LAB10.xpt | 200 | 109200 | Thu, 22 May 2014 |
| 1999 | BMX.xpt | 200 | 2827840 | Thu, 22 May 2014 |
| 2001 | DEMO_B.xpt | 200 | 3273520 | Thu, 22 May 2014 |
| 2001 | DIQ_B.xpt | 200 | 1427040 | Thu, 22 May 2014 |
| 2001 | L10AM_B.xpt | 200 | 236480 | Thu, 10 Mar 2016 **changed after 2015** |
| 2001 | L10_B.xpt | 200 | 120160 | Thu, 22 May 2014 |
| 2001 | BMX_B.xpt | 200 | 2183680 | Thu, 22 May 2014 |
| 2003 | DEMO_C.xpt | 200 | 3569840 | Thu, 22 May 2014 |
| 2003 | DIQ_C.xpt | 200 | 1314880 | Thu, 22 May 2014 |
| 2003 | L10AM_C.xpt | 200 | 189760 | Thu, 22 May 2014 |
| 2003 | L10_C.xpt | 200 | 112880 | Thu, 22 May 2014 |
| 2003 | BMX_C.xpt | 200 | 2551120 | Thu, 22 May 2014 |
| 2005 | DEMO_D.xpt | 200 | 3566560 | Thu, 22 May 2014 |
| 2005 | DIQ_D.xpt | 200 | 2834560 | Thu, 22 May 2014 |
| 2005 | GLU_D.xpt | 200 | 216400 | Thu, 22 May 2014 |
| 2005 | GHB_D.xpt | 200 | 112720 | Thu, 22 May 2014 |
| 2005 | OGTT_D.xpt | 200 | 162560 | Thu, 22 May 2014 |
| 2005 | BMX_D.xpt | 200 | 2153760 | Thu, 22 May 2014 |
| 2007 | DEMO_E.xpt | 200 | 3498080 | Thu, 22 May 2014 |
| 2007 | DIQ_E.xpt | 200 | 2789600 | Thu, 22 May 2014 |
| 2007 | GLU_E.xpt | 200 | 214000 | Thu, 22 May 2014 |
| 2007 | GHB_E.xpt | 200 | 111760 | Thu, 22 May 2014 |
| 2007 | OGTT_E.xpt | 200 | 320640 | Tue, 28 Nov 2017 **changed after 2015** |
| 2007 | BMX_E.xpt | 200 | 1800240 | Thu, 22 May 2014 |
| 2009 | DEMO_F.xpt | 200 | 3631600 | Thu, 22 May 2014 |
| 2009 | DIQ_F.xpt | 200 | 1620960 | Thu, 22 May 2014 |
| 2009 | GLU_F.xpt | 200 | 231040 | Thu, 22 May 2014 |
| 2009 | GHB_F.xpt | 200 | 118960 | Thu, 22 May 2014 |
| 2009 | OGTT_F.xpt | 200 | 306080 | Mon, 20 Nov 2017 **changed after 2015** |
| 2009 | BMX_F.xpt | 200 | 1890560 | Thu, 22 May 2014 |
| 2011 | DEMO_G.xpt | 200 | 3753760 | Mon, 10 Nov 2014 |
| 2011 | DIQ_G.xpt | 200 | 3978560 | Thu, 22 May 2014 |
| 2011 | GLU_G.xpt | 200 | 209200 | Thu, 22 May 2014 |
| 2011 | GHB_G.xpt | 200 | 105840 | Thu, 22 May 2014 |
| 2011 | OGTT_G.xpt | 200 | 272640 | Wed, 06 Dec 2017 **changed after 2015** |
| 2011 | BMX_G.xpt | 200 | 1946720 | Tue, 08 Sep 2015 **changed after 2015** |

The 22 May 2014 timestamps probably mark a server migration, not the original release dates **[U]**.

### 3.2 NHANES III (1988–1994): fixed-width ASCII plus SAS layouts [V: HEAD]

| URL | Bytes | Last-Modified |
|---|---|---|
| https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat | 67,107,314 | 19 Jul 2010 |
| https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat | 57,945,628 | 19 Jul 2010 |
| https://wwwn.cdc.gov/nchs/data/nhanes3/1a/exam.dat | 195,090,426 | 19 Jul 2010 |
| https://wwwn.cdc.gov/nchs/data/nhanes3/1a/{adult,lab,exam}.sas | 130,935 / 42,079 / 258,558 | 19 Jul 2010 |
| https://wwwn.cdc.gov/nchs/data/nhanes3/1a/{adult,lab,exam}-acc.pdf | 2.9 MB / 0.68 MB / 4.5 MB | 19 Jul 2010 |
| https://wwwn.cdc.gov/nchs/nhanes/nhanes3/datafiles.aspx | 200 | |

- All of these predate the paper.
- The old FTP path `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/nhanes/nhanes3/1A/adult.dat` returns 404.
- The SAS layouts give exact columns. Examples from lab.sas: SEQN 1-5, SDPPSU6 41, SDPSTRA6 42-43, WTPFQX6 50-58, WTPFSD6 95-103, PHPFAST 1263-1267, GHP 1861-1864, G1P 1866-1870, G2P 1885-1889. [V]

### 3.3 Later NCHS changes compared with what the authors had

- **Glucose and HbA1c calibration guidance:** unchanged in substance since before 2015 (§2.6). [V]
- **Six XPT files replaced after the analysis:** LAB10AM and L10AM_B (March 2016), OGTT_E, OGTT_F and OGTT_G (November/December 2017), and BMX_G (8 September 2015). [V: Last-Modified]
  - The Wayback CDX shows different content digests before and after for each one, for example OGTT_G in March 2015 versus September 2019, and L10AM_B in June 2015 versus October 2016. [V: https://web.archive.org/cdx/search/cdx?url=wwwn.cdc.gov/Nchs/Nhanes/2011-2012/OGTT_G.XPT]
  - The documentation pages do not say what changed. The OGTT pages still read "Last Revised: NA".
  - The record counts, the nonmissing counts of WTSOG2YR and LBXGLT (OGTT_E, OGTT_G), and the variable lists match between the 2015 archived codebooks and today's. [V]
  - The total record counts of LAB10AM (3,267) and L10AM_B (3,666) also match. [V]
  - Value-level changes are **[U]**.
- **How to settle it:** download both copies and diff them on SEQN. 2015 captures exist for LAB10AM, L10AM_B, OGTT_E/F/G and BMX_G. [V: CDX]
- **Restricted (RDC) data:** none is needed. Every variable above is in public files. Public files mask the design (SDMVPSU/SDMVSTRA; NHANES III pseudo-PSU/strata). Whether the authors (NIDDK/CDC) used unmasked design variables is **[U]**; the paper only says SUDAAN with the NHANES design.

---

## 4. Known answers (from the paper and its supplement only)

### 4.1 2011-2012, definition A, unadjusted overall (Table 1, n = 2623) [V]

| Measure | % (95% CI) |
|---|---|
| Total | 14.3 (12.2–16.8) |
| Diagnosed | 9.1 (7.8–10.6) |
| Undiagnosed | 5.2 (4.0–6.9) |
| % of cases undiagnosed | 36.4 (30.5–42.7) |
| Prediabetes | 38.0 (34.7–41.3) |

**Age-standardized total, Table 1:**

| Group | % (95% CI) |
|---|---|
| Men | 15.4 (13.2–17.9) |
| Women | 13.8 (11.4–16.6) |
| Non-Hispanic White | 11.3 (9.0–14.1) |
| Non-Hispanic Black | 21.8 (17.7–26.7) |
| Non-Hispanic Asian | 20.6 (15.0–27.6) |
| All Hispanic | 22.6 (18.4–27.5) |
| Mexican American | 23.8 (19.1–29.2) |

**Age-standardized % undiagnosed:** Asian 50.9 (38.3–63.4); Hispanic 49.0 (40.8–57.2). [V]

**Crude age groups, total:**

| Age | % (95% CI) |
|---|---|
| 20–44 | 5.0 (3.8–6.7) |
| 45–64 | 17.5 (14.4–21.0) |
| ≥65 | 33.0 (27.1–39.4) |

### 4.2 2011-2012, definition B, unadjusted overall (Table 2, n = 2781) [V]

| Measure | % (95% CI) |
|---|---|
| Total | 12.3 (10.8–14.1) |
| Diagnosed | 9.2 (8.0–10.7) |
| Undiagnosed | 3.1 (2.5–3.9) |
| % of cases undiagnosed | 25.2 (21.1–29.8) |
| Prediabetes | 36.5 (33.2–40.0) |

**Age-standardized total, Table 2:**

| Group | % (95% CI) |
|---|---|
| Men | 13.6 (12.0–15.3) |
| Women | 11.4 (9.6–13.6) |
| Non-Hispanic White | 9.5 (8.0–11.2) |
| Non-Hispanic Black | 20.6 (16.9–24.8) |
| Non-Hispanic Asian | 16.5 (13.0–20.6) |
| All Hispanic | 18.7 (15.1–23.0) |
| Mexican American | 20.5 (16.3–25.4) |

### 4.3 Trends, definition B, age-standardized, adults ≥ 20 [V: Table 3, eTables 4–6]

| Period | Total (Table 3) | Diagnosed (eTable 4) | Undiagnosed (eTable 5) | % undiagnosed (eTable 6) |
|---|---|---|---|---|
| 1988-1994 | 9.8 (8.9–10.6) | 5.9 (5.5–6.3) | 3.9 (3.3–4.4) | 40.3 (34.9–45.7) |
| 1999-2000 | 9.8 (8.5–11.0) | 6.7 (5.7–7.7) | 3.1 (2.4–3.7) | 35.4 (21.8–48.9) |
| 2001-2002 | 10.8 (9.5–12.0) | 7.2 (6.3–8.0) | 3.5 (2.8–4.3) | 30.0 (24.8–35.3) |
| 2003-2004 | 11.7 (10.3–13.0) | 8.2 (7.1–9.3) | 3.5 (2.7–4.2) | 32.9 (24.9–40.9) |
| 2005-2006 | 10.8 (9.5–12.2) | 8.0 (7.1–8.9) | 2.8 (1.8–3.9) | 26.3 (16.7–35.9) |
| 2007-2008 | 12.5 (10.9–14.2) | 8.8 (7.3–10.3) | 3.8 (3.3–4.4) | 31.7 (25.8–37.6) |
| 2009-2010 | 12.1 (10.3–13.9) | 8.6 (7.6–9.5) | 3.6 (2.6–4.5) | 30.7 (27.1–34.4) |
| 2011-2012 | 12.4 (10.8–14.2) | 9.2 (8.2–10.1) | 3.1 (2.4–3.8) | 31.0 (25.2–37.4) |
| P for trend | <.001 | <0.001 | 0.76 | <0.001 |

**Crude total (eTable 3), by period:** 8.6, 8.8, 10.0, 11.2, 10.8, 12.4, 12.1, 12.3. [V]

**Selected Table 3 subgroups, 1988-94 → 2011-12 (P for trend):**

| Subgroup | 1988-94 | 2011-12 | P |
|---|---|---|---|
| Non-Hispanic White | 8.6 (7.7–9.5) | 9.5 (8.0–11.2) | .04 |
| Non-Hispanic Black | 16.3 (14.6–18.0) | 20.6 (16.9–24.8) | .003 |
| Mexican American | 17.5 (15.6–19.3) | 20.5 (16.3–25.4) | <.001 |
| Age 20–44 | 2.7 (2.1–3.4) | 4.5 (3.5–5.8) | .01 |
| Age ≥ 65 | 20.5 (18.3–22.6) | 24.7 (21.6–28.1) | <.001 |
| Lowest PIR tertile | 13.0 (11.7–14.4) | 17.8 (15.1–20.5) | <.001 |
| BMI ≥ 30 (text) | 18.0 (15.9–20.0) | 20.1 (17.8–22.4) | .003 |

### 4.4 Ages 40–74, definition A, age-standardized (eTable 7) [V]

| Period | Total (n) | Undiagnosed (eTable 9) |
|---|---|---|
| 1988-94 | 15.9 (14.5–17.3), n = 4201 | 8.0 (6.9–9.1) |
| 2005-06 | 18.1 (15.1–21.0), n = 1217 | 7.4 (5.2–9.7) |
| 2007-08 | 19.0 (16.1–22.0), n = 1671 | 7.7 (6.4–9.1) |
| 2009-10 | 19.4 (16.5–22.3), n = 1801 | 8.0 (6.4–9.5) |
| 2011-12 | 18.0 (15.6–20.4), n = 1559 | 5.9 (4.3–7.4) |
| P for trend | 0.01 | 0.15 |

### 4.5 Internal consistency checks to build into the test [V arithmetic on the values above]

- Diagnosed plus undiagnosed is close to the total: 1988-94 gives 5.9 + 3.9 = 9.8, and 2011-12 gives 9.2 + 3.1 = 12.3 against 12.4 (rounding).
- Standardizing within a single age band changes nothing. The age-band rows of Table 3 equal the crude age-band rows of eTable 3.
- The 2011-12 column of Table 3 equals the age-standardized Table 2 values. For example, Mexican American is 20.5 (16.3–25.4) in both.

### 4.6 CI conventions visible in the tables [V by inspection; method [U]]

- The 2011-2012 tables and the 2011-12 column of Table 3 have **asymmetric** CIs, for example 12.4 (10.8–14.2), consistent with logit-transformed CIs.
- The 1988–2010 trend columns have **symmetric** Wald CIs, and eTable 5 even shows a negative lower bound, (−0.2–2.5).
- Replicating both conventions will be needed.

---

## 5. What TurboTab would need

From a read-only survey of `/Users/nhedglin/tabular-ml-lab/turbotab`. File and line references come from the subagent and were spot-checked here.

### 5.1 What already exists

- **XPT v5 and v8 reader:** `turbotab/core/xport.py` (`read_header` l.177, `read_frame` l.453), wired into `turbotab/core/datastore.py` (`_ingest_xpt`). It is tested against R haven and foreign.
- **Joins on SEQN:** `datastore.JoinInput` and `turbotab/core/assembly.py` `plan()` (DuckDB).
- **Knowledge of NHANES weights:**
  - `turbotab/core/survey.py` `NHANES_WEIGHTS` (l.43–47) includes WTSAF2YR, WTSAF4YR and WTSOG2YR.
  - `turbotab/core/recognizers.py` has tiered least-common-denominator logic that picks WTSAF* for LBXGLU and WTSOG* for LBXGLT.
- **Taylor-linearized variance with correct domain estimation:** `turbotab/core/models/survey.py` (`total_variance` l.239, `domain_of` l.397). Lonely PSUs are handled the way R's "adjust" option does.
- **Stacking cycles with unique strata and PSUs:** `turbotab/core/methods/cycles.py` `stack_cycles` (l.304) and `_assemble` (l.758).
- **Per-cycle prevalence and logistic trend tests on period midpoints:** `turbotab/core/methods/cycle_trends.py` (`cycle_trend` l.185, `_cycle_estimates` l.377, `_korn_graubard` l.411). The trend test is checked against R svyglm.
- **Adjusted Wald F test:** `adjusted_wald` (l.597).
- **Engine only:** none of this is reachable from the UI.

### 5.2 Gaps

1. **No NHANES III ASCII reader (missing).** There is no fixed-width reader and no SAS INPUT-layout parser; `_source_kind` refuses `.dat`. The layout parser is small, since the `.sas` files give `VAR start-end` columns. The files are large: 58–195 MB of ASCII.
2. **NHANES III design and weight names are not recognized (partial).**
   - SDPSTRA6/SDPPSU6 and WTPF* are unknown to `recognizers.py` and `cycles.WEIGHT_KINDS`.
   - G1P, G2P and GHP are not mapped to the FPG, OGTT or HbA1c analytes.
   - Workaround: `stack_cycles(renames=...)` and an explicit SetSurvey.
3. **Composite weighting is missing.** TurboTab has no way to estimate prevalence from diagnosed cases in the interview or MEC sample plus non-diagnosed people in a subsample, rescaled to the population (the Cowie 2009 approach). Nor does it have the NHANES III ratio adjustment, P(dx) + (1 − P(dx)) × P(undx | no dx), with its delta-method variance. This is the central methodological gap.
4. **Direct age standardization is missing.** The code has nothing beyond g-computation. Needed:
   - Σ w_a p̂_a with variance w′Vw from the joint covariance, which `_cycle_estimates` provides;
   - user-supplied standard weights (0.4561 / 0.3673 / 0.1766);
   - standardized ratios for "% undiagnosed".
5. **Per-cycle affine calibration is missing.** `cycles._convert` (l.654) allows only a multiplicative factor or a value map. 2005-06 (× 0.9835) fits that, but 2007-12, (x − 1.139) × 0.9835, does not.
6. **Single-cycle subgroup prevalence tables are partial.**
   - `cycle_trend` refuses fewer than two cycles, and `_cycle_estimates` is private.
   - There is no svyby-style multi-group table.
   - Logit CIs are not wired in; a helper exists at `turbotab/core/methods/usual_intake.py` `logit_interval` (l.1025). Wald and Korn-Graubard intervals exist.
7. **SUDAAN parity details.**
   - Degrees of freedom: TurboTab uses PSUs minus strata over domain-bearing strata (`design_df` l.587), while SUDAAN by default uses the full design's df.
   - There is no ready-made F test for differences across strata.
8. **Stacking across NHANES III and continuous NHANES has friction.**
   - NHANES III and continuous SEQNs overlap, so `_refuse_repeated_ids` refuses; use `id_column=None` or prefix the IDs.
   - `cycle_of` needs `years={key: 6}` for NHANES III (midpoint 1991).
   - `stack_cycles` rescales weights by years/T, which pools populations. For period-specific estimates and a trend logistic, each period should keep its own weights. Check that the rescaling does not change the relative weight NHANES III (6 years) gets in the trend regression compared with SUDAAN's run. Whether SUDAAN rescaled is **[U]**.
9. **No published-NHANES known-answer fixture exists in the repo.** `docs/turbotab-next/understanding/catalogs/CLINICAL.md` cites Menke 2015, but there is no test. These numbers would be the first.

---

## 6. Suggested replication order (not executed)

1. Download DEMO_G, DIQ_G, GLU_G, GHB_G and OGTT_G, about 8.3 MB in total. This needs the user's permission.
2. Check the standard weights against WTINT2YR sums by age band.
3. Reproduce n = 2781 / 910 (definition B) and n = 2623 / 971 (definition A) under each candidate composite construction.
4. Reproduce Table 2's overall row, then Table 1's.
5. Add 1999–2010, with the chained FPG calibration from 2005 on, and match the eTable 3 and 4 sizes and the crude values.
6. Add NHANES III through a fixed-width reader. Target n = 8478 / 2136 and 9.8% age-standardized.
7. If any period misses its target, rerun it on the archived 2015 XPT copies to separate file-revision effects from method effects.

---

## 7. URLs verified this session

- **PubMed and PMC:**
  - https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi (PMID 26348752)
  - https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=26348752&retmode=xml
  - https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=26348752&format=json (not in PMC)
  - https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=2628695 (Cowie 2009)
- **Unpaywall:** https://api.unpaywall.org/v2/10.1001/jama.2015.10029
- **JAMA:**
  - https://jamanetwork.com/journals/jama/fullarticle/2434682 (full text, via browser)
  - https://cdn.jamanetwork.com/ama/content_public/journal/jama/934405/joi150096t1.png and joi150096t3.png (table footnotes; signed)
  - https://cdn.jamanetwork.com/ama/content_public/journal/jama/934405/joi150096supp1_prod.pdf (signed)
- **CDC continuous NHANES:** the 39 XPT and 39 HTM URLs under https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{1999..2011}/DataFiles/ (§3.1). The documentation text was read for LAB10AM, L10AM_B, L10AM_C, GLU_D–G, OGTT_D–G, LAB10, L10_B, L10_C, GHB_D–G, DIQ, DIQ_G, DEMO_G and BMX_G.
- **CDC NHANES III:**
  - https://wwwn.cdc.gov/nchs/nhanes/nhanes3/datafiles.aspx
  - https://wwwn.cdc.gov/nchs/data/nhanes3/1a/{adult,lab,exam}.{dat,sas} and {adult,lab,exam}-acc.pdf
- **Wayback Machine:**
  - https://web.archive.org/web/20150221154732/http://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/GLU_D.htm
  - http://web.archive.org/web/20150519022333/http://wwwn.cdc.gov:80/nchs/nhanes/2007-2008/GLU_E.htm
  - 2015 OGTT_E, OGTT_G, LAB10AM and L10AM_B codebooks
  - CDX listings for LAB10AM, L10AM_B, OGTT_E/F/G and BMX_G XPT
