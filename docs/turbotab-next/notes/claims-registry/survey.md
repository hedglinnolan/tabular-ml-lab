# Survey: the engine's existing claim-like structures

This is a read-only survey of feat/s1-int (before fix/s1-inference merged), 2026-10-15, condensed. Line numbers drift; re-locate them before editing.

## The backtest inventory
- docs/turbotab-next/notes/claims-backtest/inventory/inventory_{A,B,C,D}.jsonl holds 991 claims in the middle layer's schema, keyed by file:line.
- verdicts/basis_*.jsonl holds 117 basis audits with grades (proved 6, certified 21, sourced 29, judgment 56, unsupported 5). The grade definitions are in PLAN.md:41-48.
- No code reads any of it.

## Citations
- export/data/citations.json holds 303 records; the `Citation` dataclass is at export/citations.py:79-107.
- How a citation resolves:
  - `mentions()` (:279) finds author-year spans;
  - `resolve_at` (:284) matches each record's `cited_as` regexes;
  - prose must resolve to exactly one record, and a contract's `sources` break ties (:324-330).
- `cited_texts()` (:267) scans:
  - contract fields (:185-192);
  - `models/sources.SOURCES` (:232);
  - teaching/content.py's upper-case constants (:239-245);
  - the noticing catalogs (:251-264).
- It does not scan voice.py, or the f-string concerns such as methods/trials.py:1386-1389.
- refs.bib must equal `C.bibtex()` (test_citations.py:372).
- The grade here is verification status only.

## Method contracts (the pattern for the registry)
- In core/contracts.py:
  - `ContractOption` (:69-84) has customary, sound[purpose] and rung[purpose].
  - `Relation` (:87-109) has kind, target, says, purposes, rung, when, exits, enforced_by, condition and id.
  - `MethodContract` (:112-169) has sources (free text) and a sentence callable.
- Registration is `register_contract` (:186-229), over `DECLARING_MODULES` (:242-275).
- The readers are `fired` (:338), `run_order` (:303), `paragraph` (:436) and `quest.contract_tier`.
- Relations are declared inline: methods/trials.py:2046-2069, and causal.RELATIONS at causal.py:139-209.
- `RELATION_TESTS` dicts sit in 8 test files: test_estimand_chain:410, test_leash:1370, test_e2_trials:1014, test_form_contracts:25, test_explore_chain:34, test_export:844, test_c1b:339 and test_d6:365. Causal relations have no map (test_causal_lane.py:257-260).
- reference/methods.py `render()` (:446) produces METHODS_REFERENCE.md; `--check` is at :541-558, and test_reference.py:39 enforces it.
- reference/packet.py writes the review packets.
- reference/defaults.py is the closest existing "sentence from the parameter, with a falsification check": `Case.when` is built from constants (:143-174), and `_check` raises `PremiseFailed` (:59-66).
- custom_sound.py holds labels outside contracts (:39-84).
- catalog.GAPS (reference/catalog.py:203+) lists methods with no contract.

## Model families
- `ModelFamily` (models/base.py:210-276) has about 40 members, including `Source(key, where)` (:100-106) and `defaults_version` (:249). It is registered by `register_family` (:351) and checked by `contract_problems` (:380).
- `TuningDecl` (models/tuning.py:155-200) carries standard and standard_source, small_standard and small_standard_source (`SMALL_N = 300`, :94), space_version, and a reason of at most 22 words.
- `defaults_version` and `space_version` enter the TuningPlan (:422-423, :998-999) and seed the candidates.
- acceptance/test_mc1_family_declarations.py pins each family's declaration (EXPECTED :53-69). The MC-2b tests pin the declaration-reading predicates.

## Readings
- The design is BLUEPRINT.md:564-640 (§14.1-14.3); the code is core/readings.py.
  - `Reading` (:102-128).
  - `KindRule` (:374-397), with alternatives and rejects. This is already a falsification shape.
  - `KIND_RULES` (:1103), `Consumer` (:3239-3258), `CONSUMERS` (:3269) and `EVIDENCE_ONLY` (:3496).
- Asks go through `confirm_exit` (:1357) and `ask_exits`; the sentences are in voice.py:724 and :730.
- acceptance/test_readings_ledger.py checks a census.

## Materiality
- In core/materiality.py:
  - `INSTRUMENTS` (:111-130), `Movement` (:168), `band_of` (:189), `tier` (:567);
  - `recommend` (:1024), `verify` (:1096), `Ledger` (:1137);
  - `ledger()` (:1190), which is derived and never stored.
- materiality_calibration.json holds, per instrument: the convention bands, their reason and `floored`; the fitted bands; `calibrated`; the cases; a confusion matrix; and `theorem`.
- It is rebuilt by `calibrate` (:1271) and `write_calibration` (:1326), through tests/materiality_cases.py --write.
- This is the only structure with an empirical grade and a falsification record.

## Sentences
- **Hand-written:**
  - teaching/content.py `section(...)` (:58), with an evidence status of SETTLED, CONVENTION or DISPUTED (:54);
  - `Relation.says`;
  - family reasons;
  - `packs.Evidence` (turbotab/packs.py:242).
- **Generated per decision kind:** voice.register_sentence and sentence_for (voice.py:250-296), for example `_set_split` (:1151). Each is stored once on `DecisionRecord.sentence` (decisions.py:2131).
- **Mixed f-strings.**
- **Gates:**
  - test_word_budgets.py;
  - test_plain_words.py, an AST scan (:35-77);
  - acceptance/test_plain_words_served.py;
  - test_trust_leaks_and_sentences.py;
  - tests/test_american_spelling.py;
  - acceptance/test_wp15_claims.py.
- **Numbers written in both code and prose:**
  - `estimand.FIXED_EFFECTS_LOGIT_ROWS = 16` (:1889) against teaching/content.py:1344 and :1349-1350;
  - `trials.FEW_CLUSTERS = 10` (:121) and `CV_RULE = 0.6` (:120) against the relation text at :2047, :2051, :2054-2055 and :1965;
  - `tuning.SMALL_N = 300` against boosted_trees.py:93 and :303, and xgboost_family.py:527 and :632. `boosted_trees.SMALL_N = 500` (:52) is a different rule.

## Provenance and methods text
- provenance.py: `in_force` (:82), `APPLIES` (:39), `seen_from` (:118), `restated` (:132) and `methods_text` (:166).
- export/methods.py orders sentences by guideline item through `SECTION_OF` (:59). export/checklists.py maps the TRIPOD and STROBE items.
- Sentences carry no link back to the rule behind them.

## Registries of placement
- purposes.py: `QUESTIONS` (:26) and `VIEW_PURPOSES` (:46-73), mirrored in the frontend.
- quest.py:
  - `Place` (:133), `QUESTIONS` (:145), `OTHER_KINDS` (:207);
  - `DECLARATIONS` (:491), `TIER_RULINGS` (:550), `EXPLORE_FINDINGS` (:576), `COMPUTE` (:673).
- quest_noticings.json is generated from the crosswalk.
- surfacing.py: `Surfaceable` (:57-70) and `crosswalk_rules` (:676).
- crosswalk.json holds 708 items, with `rule` wired in 106. Tests read it; the engine does not.
- understanding/catalogs/*.json holds 401 threads.
- tests/path_fuzzer.py: seeded journeys with the surfacing invariants I1-I11 checked at every state.

## Thresholds
- About 616 numeric upper-case module constants sit across 171 non-test files. config.py holds environment settings only.
- `FEW_CLUSTERS` is 10 at methods/trials.py:121 (Li & Redden) and 30 at models/inference.py:46 (Cameron & Miller).
- `COLLINEAR_TOL` is at models/variable_selection.py:64; `MIN_GAIN` at models/baseline.py:48; `ESTIMATED_COLUMNS` at methods/omics.py:75.
- r ≥ 0.3 appears four times: recognizers.CORROBORATE_R (:599), materiality.ENERGY_R (:107), stages/finding_words.TANGLED_R (:561), and a literal at coach.py:381.
- About 10 modules keep their own `SOURCES` dicts.

## Overlaps
- **Sources:** six places.
- **Grade vocabularies:** seven:
  - teaching's statuses;
  - packs.Evidence;
  - custom_sound's verdict;
  - materiality;
  - the audit's claims ledger;
  - the backtest's grades;
  - the rungs.
- **"When it fires", in words and code:** five places:
  - `Relation.condition` with `enforced_by`;
  - the crosswalk's `fires_when` with `rule`;
  - `Declaration.applies`;
  - `Consumer.path`;
  - `KindRule`.
- **Goal vocabularies:** three.
- **Placement:** stated four times.

## Riskiest migration points
1. Static scanners going blind to generated text.
2. Stored sentences and byte-equal artifacts.
3. Same-named constants with different values.
4. `space_version` and `defaults_version` feeding seeds.
5. Generated files with their own round trips.
6. Import order.
