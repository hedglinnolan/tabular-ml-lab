import Mathlib

/-!
# Stratification optimism, Tier A: exact algebra

Objects (all real-valued; an outcome is `0` or `1`, but the formulas are affine in the outcome, so
an event *rate* can be plugged in wherever an outcome can):

* `logLoss q y`  — log loss of predicting probability `q` for outcome `y`;
* `CE r q`       — mean log loss of predicting `q` on outcomes whose event rate is `r`;
* `KL r q`       — binary Kullback–Leibler divergence `KL(r ‖ q)` (with `0 log 0 = 0`);
* `binEntropy`   — Mathlib's binary entropy `H(p) = -p log p - (1-p) log (1-p)`;
* `freshLoss p q`— expected log loss of predicting `q` on one fresh outcome `Y ~ Bernoulli(p)`.

The sample is `K` folds of `s` rows: `y : Fin K → Fin s → ℝ` (`n = K s` rows; any partition of `n`
rows into `K` equal folds is this, up to relabelling). The base-rate predictor for fold `k` is the
training rate `trainRate y k` (the event rate of the other `K - 1` folds).
-/

open Real Finset

namespace Strat

noncomputable section

/-! ## Definitions -/

/-- Log loss of predicting probability `q` for outcome `y`. -/
def logLoss (q y : ℝ) : ℝ := -(y * log q) - (1 - y) * log (1 - q)

/-- Cross-entropy: the mean log loss of predicting `q` on outcomes whose event rate is `r`. -/
def CE (r q : ℝ) : ℝ := -(r * log q) - (1 - r) * log (1 - q)

/-- Binary Kullback–Leibler divergence `KL(r ‖ q)`. -/
def KL (r q : ℝ) : ℝ := r * log (r / q) + (1 - r) * log ((1 - r) / (1 - q))

/-- Expected log loss of predicting `q` on one fresh outcome `Y ~ Bernoulli(p)`. -/
def freshLoss (p q : ℝ) : ℝ := p * logLoss q 1 + (1 - p) * logLoss q 0

/-! ## A1. Cross-entropy decomposition -/

lemma mul_log_div (r : ℝ) {q : ℝ} (hq : q ≠ 0) : r * log (r / q) = r * log r - r * log q := by
  rcases eq_or_ne r 0 with rfl | hr
  · simp
  · rw [log_div hr hq]; ring

/-- **A1.** `CE(r, q) = H(r) + KL(r ‖ q)` for every `r` and every `q ∉ {0, 1}`. -/
theorem CE_eq_binEntropy_add_KL (r : ℝ) {q : ℝ} (hq0 : q ≠ 0) (hq1 : q ≠ 1) :
    CE r q = binEntropy r + KL r q := by
  have h1 : (1 - q) ≠ 0 := sub_ne_zero.mpr (Ne.symm hq1)
  rw [CE, KL, binEntropy, mul_log_div r hq0, mul_log_div (1 - r) h1, log_inv, log_inv]
  ring

/-- Predicting the rate itself costs exactly the entropy (no hypothesis on `r`). -/
theorem CE_self (r : ℝ) : CE r r = binEntropy r := by
  rw [CE, binEntropy, log_inv, log_inv]; ring

/-- **A1'.** The mean log loss of a constant prediction `q` over a finite nonempty set of outcomes
is the cross-entropy at their empirical rate. -/
theorem mean_logLoss {ι : Type*} (S : Finset ι) (hS : S.Nonempty) (y : ι → ℝ) (q : ℝ) :
    (∑ i ∈ S, logLoss q (y i)) / S.card = CE ((∑ i ∈ S, y i) / S.card) q := by
  have hc : (S.card : ℝ) ≠ 0 := by exact_mod_cast hS.card_pos.ne'
  have hsum : ∑ i ∈ S, logLoss q (y i)
      = -((∑ i ∈ S, y i) * log q) - ((S.card : ℝ) - ∑ i ∈ S, y i) * log (1 - q) := by
    simp only [logLoss, sum_sub_distrib, sum_neg_distrib, ← sum_mul, sum_const, nsmul_eq_mul,
      mul_one]
  rw [hsum, CE]
  field_simp

/-! ## A2. Fresh-data expected loss -/

theorem freshLoss_eq_CE (p q : ℝ) : freshLoss p q = CE p q := by
  simp only [freshLoss, logLoss, CE]; ring

/-- **A2.** The fresh-data expected log loss of predicting `q` under truth `p` is
`H(p) + KL(p ‖ q)`. -/
theorem freshLoss_eq (p : ℝ) {q : ℝ} (hq0 : q ≠ 0) (hq1 : q ≠ 1) :
    freshLoss p q = binEntropy p + KL p q := by
  rw [freshLoss_eq_CE, CE_eq_binEntropy_add_KL p hq0 hq1]

/-! ## A3. Gibbs' inequality for the binary KL -/

theorem KL_eq_klFun (r : ℝ) {q : ℝ} (hq0 : 0 < q) (hq1 : q < 1) :
    KL r q = q * InformationTheory.klFun (r / q)
      + (1 - q) * InformationTheory.klFun ((1 - r) / (1 - q)) := by
  have h1 : (1 - q) ≠ 0 := by linarith
  have e1 : q * (r / q) = r := mul_div_cancel₀ r hq0.ne'
  have e2 : (1 - q) * ((1 - r) / (1 - q)) = 1 - r := mul_div_cancel₀ (1 - r) h1
  simp only [InformationTheory.klFun, KL]
  calc r * log (r / q) + (1 - r) * log ((1 - r) / (1 - q))
      = (q * (r / q)) * log (r / q) + q - q * (r / q)
        + (((1 - q) * ((1 - r) / (1 - q))) * log ((1 - r) / (1 - q))
          + (1 - q) - (1 - q) * ((1 - r) / (1 - q))) := by rw [e1, e2]; ring
    _ = _ := by ring

theorem KL_nonneg {r q : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hq0 : 0 < q) (hq1 : q < 1) :
    0 ≤ KL r q := by
  rw [KL_eq_klFun r hq0 hq1]
  have h1 : (0 : ℝ) < 1 - q := by linarith
  have a := InformationTheory.klFun_nonneg (div_nonneg hr0 hq0.le)
  have b := InformationTheory.klFun_nonneg (div_nonneg (by linarith : (0 : ℝ) ≤ 1 - r) h1.le)
  exact add_nonneg (mul_nonneg hq0.le a) (mul_nonneg h1.le b)

/-- **A3.** `KL(r ‖ q) = 0` iff `r = q` (for `r ∈ [0,1]`, `q ∈ (0,1)`). -/
theorem KL_eq_zero_iff {r q : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hq0 : 0 < q) (hq1 : q < 1) :
    KL r q = 0 ↔ r = q := by
  have h1 : (0 : ℝ) < 1 - q := by linarith
  constructor
  · intro h
    rw [KL_eq_klFun r hq0 hq1] at h
    have a := InformationTheory.klFun_nonneg (div_nonneg hr0 hq0.le)
    have b := InformationTheory.klFun_nonneg (div_nonneg (by linarith : (0 : ℝ) ≤ 1 - r) h1.le)
    have A := mul_nonneg hq0.le a
    have B := mul_nonneg h1.le b
    have hA : q * InformationTheory.klFun (r / q) = 0 := by linarith
    have hk : InformationTheory.klFun (r / q) = 0 := by
      rcases mul_eq_zero.mp hA with h' | h'
      · exact absurd h' hq0.ne'
      · exact h'
    rw [InformationTheory.klFun_eq_zero_iff (div_nonneg hr0 hq0.le)] at hk
    exact (div_eq_one_iff_eq hq0.ne').mp hk
  · rintro rfl
    simp [KL, div_self hq0.ne', div_self h1.ne']

/-! ## A4. Folds, stratification, and the cross-validated log loss -/

variable {K s : ℕ}

/-- Event count of fold `k`. -/
def foldSum (y : Fin K → Fin s → ℝ) (k : Fin K) : ℝ := ∑ j, y k j

/-- Test rate of fold `k`. -/
def testRate (y : Fin K → Fin s → ℝ) (k : Fin K) : ℝ := foldSum y k / s

/-- Training rate for fold `k`: the event rate of the other `K - 1` folds. -/
def trainRate (y : Fin K → Fin s → ℝ) (k : Fin K) : ℝ :=
  (∑ k' ∈ univ.erase k, foldSum y k') / (((K : ℝ) - 1) * s)

/-- The sample's event rate `p̂`. -/
def sampleRate (y : Fin K → Fin s → ℝ) : ℝ := (∑ k, foldSum y k) / (K * s)

/-- Mean log loss on fold `k` of the base-rate predictor trained on the other folds. -/
def foldLoss (y : Fin K → Fin s → ℝ) (k : Fin K) : ℝ :=
  (∑ j, logLoss (trainRate y k) (y k j)) / s

/-- The `K`-fold cross-validated log loss of the base-rate predictor. -/
def cvLoss (y : Fin K → Fin s → ℝ) : ℝ := (∑ k, foldLoss y k) / K

/-- Exact stratification: every fold has the same event count. -/
def Stratified (y : Fin K → Fin s → ℝ) : Prop := ∀ k k', foldSum y k = foldSum y k'

theorem foldLoss_eq_CE (hs : 0 < s) (y : Fin K → Fin s → ℝ) (k : Fin K) :
    foldLoss y k = CE (testRate y k) (trainRate y k) := by
  have hne : (univ : Finset (Fin s)).Nonempty := ⟨⟨0, hs⟩, mem_univ _⟩
  have := mean_logLoss (univ : Finset (Fin s)) hne (y k) (trainRate y k)
  simpa [foldLoss, testRate, foldSum, card_univ, Fintype.card_fin] using this

lemma sum_testRate (hs : 0 < s) (hK : 0 < K) (y : Fin K → Fin s → ℝ) :
    ∑ k, testRate y k = K * sampleRate y := by
  have hs' : (s : ℝ) ≠ 0 := by exact_mod_cast hs.ne'
  have hK' : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  simp only [testRate, sampleRate, ← sum_div]
  field_simp

lemma trainRate_mul (hs : 0 < s) (hK : 2 ≤ K) (y : Fin K → Fin s → ℝ) (k : Fin K) :
    ((K : ℝ) - 1) * trainRate y k = K * sampleRate y - testRate y k := by
  have hs' : (s : ℝ) ≠ 0 := by exact_mod_cast hs.ne'
  have hK' : (K : ℝ) ≠ 0 := by have : (0 : ℝ) < K := by exact_mod_cast (by omega : 0 < K)
                               exact this.ne'
  have hK1 : (K : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ K := by exact_mod_cast hK
    linarith
  have hsplit := (add_sum_erase (univ : Finset (Fin K)) (foldSum y) (mem_univ k)).symm
  simp only [trainRate, sampleRate, testRate]
  rw [hsplit]
  field_simp
  ring

/-- **A4 (rates).** Under exact stratification, every test rate and every training rate equals
the sample rate `p̂`. -/
theorem stratified_rates (hK : 2 ≤ K) (hs : 0 < s) {y : Fin K → Fin s → ℝ} (h : Stratified y)
    (k : Fin K) : testRate y k = sampleRate y ∧ trainRate y k = sampleRate y := by
  have hs' : (s : ℝ) ≠ 0 := by exact_mod_cast hs.ne'
  have hK2 : (2 : ℝ) ≤ K := by exact_mod_cast hK
  have hK' : (K : ℝ) ≠ 0 := by linarith
  have hK1 : (K : ℝ) - 1 ≠ 0 := by linarith
  have hall : ∑ k', foldSum y k' = K * foldSum y k := by
    rw [sum_congr rfl (fun k' _ => h k' k), sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  have herase : ∑ k' ∈ univ.erase k, foldSum y k' = ((K : ℝ) - 1) * foldSum y k := by
    rw [sum_congr rfl (fun k' _ => h k' k), sum_const, card_erase_of_mem (mem_univ k), card_univ,
      Fintype.card_fin, nsmul_eq_mul, Nat.cast_sub (by omega : 1 ≤ K), Nat.cast_one]
  refine ⟨?_, ?_⟩
  · simp only [testRate, sampleRate, hall]; field_simp
  · simp only [trainRate, sampleRate, hall, herase]; field_simp

/-- **A4 (loss).** Under exact stratification the cross-validated log loss of the base-rate
predictor is exactly the plug-in entropy `H(p̂)`: the KL term vanishes in every fold. -/
theorem stratified_cvLoss (hK : 2 ≤ K) (hs : 0 < s) {y : Fin K → Fin s → ℝ} (h : Stratified y) :
    cvLoss y = binEntropy (sampleRate y) := by
  have hK' : (K : ℝ) ≠ 0 := by
    have : (2 : ℝ) ≤ K := by exact_mod_cast hK
    linarith
  have hfold : ∀ k, foldLoss y k = binEntropy (sampleRate y) := by
    intro k
    obtain ⟨h1, h2⟩ := stratified_rates hK hs h k
    rw [foldLoss_eq_CE hs, h1, h2, CE_self]
  simp only [cvLoss, hfold, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- **A4 (any fold).** For an arbitrary partition, a fold whose training rate is in `(0,1)` pays
`H(test rate) + KL(test rate ‖ training rate)`. -/
theorem foldLoss_decomp (hs : 0 < s) (y : Fin K → Fin s → ℝ) (k : Fin K)
    (hq0 : 0 < trainRate y k) (hq1 : trainRate y k < 1) :
    foldLoss y k = binEntropy (testRate y k) + KL (testRate y k) (trainRate y k) := by
  rw [foldLoss_eq_CE hs, CE_eq_binEntropy_add_KL _ hq0.ne' hq1.ne]

lemma testRate_mem (hs : 0 < s) {y : Fin K → Fin s → ℝ} (hy : ∀ k j, 0 ≤ y k j ∧ y k j ≤ 1)
    (k : Fin K) : 0 ≤ testRate y k ∧ testRate y k ≤ 1 := by
  have hs' : (0 : ℝ) < s := by exact_mod_cast hs
  refine ⟨div_nonneg (sum_nonneg fun j _ => (hy k j).1) hs'.le, ?_⟩
  rw [testRate, div_le_one hs', foldSum]
  calc ∑ j, y k j ≤ ∑ _j : Fin s, (1 : ℝ) := sum_le_sum fun j _ => (hy k j).2
    _ = s := by simp

/-- **A4 (KL term).** With outcomes in `[0,1]`, the extra KL term a fold pays is `≥ 0`, and it is
`0` exactly when the fold's test rate equals its training rate. -/
theorem fold_KL_nonneg_and_zero_iff (hs : 0 < s) {y : Fin K → Fin s → ℝ}
    (hy : ∀ k j, 0 ≤ y k j ∧ y k j ≤ 1) (k : Fin K)
    (hq0 : 0 < trainRate y k) (hq1 : trainRate y k < 1) :
    0 ≤ KL (testRate y k) (trainRate y k) ∧
      (KL (testRate y k) (trainRate y k) = 0 ↔ testRate y k = trainRate y k) := by
  obtain ⟨hr0, hr1⟩ := testRate_mem hs hy k
  exact ⟨KL_nonneg hr0 hr1 hq0 hq1, KL_eq_zero_iff hr0 hr1 hq0 hq1⟩

/-- A fold's test rate equals its training rate iff it equals the sample rate. -/
theorem test_eq_train_iff (hK : 2 ≤ K) (hs : 0 < s) (y : Fin K → Fin s → ℝ) (k : Fin K) :
    testRate y k = trainRate y k ↔ testRate y k = sampleRate y := by
  have hm := trainRate_mul hs hK y k
  have hK2 : (2 : ℝ) ≤ K := by exact_mod_cast hK
  constructor
  · intro h
    rw [← h] at hm
    have : (K : ℝ) * testRate y k = K * sampleRate y := by linarith
    exact mul_left_cancel₀ (by linarith) this
  · intro h
    rw [h] at hm
    have : ((K : ℝ) - 1) * trainRate y k = ((K : ℝ) - 1) * sampleRate y := by linarith
    rw [h]; exact (mul_left_cancel₀ (by linarith) this).symm

/-! ## A4+. Plain folds: a two-sided bound on how much any partition exceeds `H(p̂)` -/

lemma gap_lower {r q P : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hP0 : 0 < P) (hP1 : P < 1)
    (hq0 : 0 < q) (hq1 : q < 1) :
    (P - q) * (r - P) / (P * (1 - P)) ≤ CE r q - CE r P := by
  have hP1' : (0 : ℝ) < 1 - P := by linarith
  have hq1' : (0 : ℝ) < 1 - q := by linarith
  have h1 : log (q / P) ≤ q / P - 1 := log_le_sub_one_of_pos (div_pos hq0 hP0)
  have h2 : log ((1 - q) / (1 - P)) ≤ (1 - q) / (1 - P) - 1 :=
    log_le_sub_one_of_pos (div_pos hq1' hP1')
  rw [log_div hq0.ne' hP0.ne'] at h1
  rw [log_div hq1'.ne' hP1'.ne'] at h2
  have e : (P - q) * (r - P) / (P * (1 - P))
      = r * (1 - q / P) + (1 - r) * (1 - (1 - q) / (1 - P)) := by
    field_simp; ring
  rw [e, CE, CE]
  nlinarith [mul_le_mul_of_nonneg_left h1 hr0,
    mul_le_mul_of_nonneg_left h2 (by linarith : (0 : ℝ) ≤ 1 - r)]

lemma gap_upper {r q P : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hP0 : 0 < P) (hP1 : P < 1)
    (hq0 : 0 < q) (hq1 : q < 1) :
    CE r q - CE r P ≤ (P - q) * (r - q) / (q * (1 - q)) := by
  have hP1' : (0 : ℝ) < 1 - P := by linarith
  have hq1' : (0 : ℝ) < 1 - q := by linarith
  have h1 : log (P / q) ≤ P / q - 1 := log_le_sub_one_of_pos (div_pos hP0 hq0)
  have h2 : log ((1 - P) / (1 - q)) ≤ (1 - P) / (1 - q) - 1 :=
    log_le_sub_one_of_pos (div_pos hP1' hq1')
  rw [log_div hP0.ne' hq0.ne'] at h1
  rw [log_div hP1'.ne' hq1'.ne'] at h2
  have e : (P - q) * (r - q) / (q * (1 - q))
      = r * (P / q - 1) + (1 - r) * ((1 - P) / (1 - q) - 1) := by
    field_simp; ring
  rw [e, CE, CE]
  nlinarith [mul_le_mul_of_nonneg_left h1 hr0,
    mul_le_mul_of_nonneg_left h2 (by linarith : (0 : ℝ) ≤ 1 - r)]

lemma sum_CE_sampleRate (hs : 0 < s) (hK : 0 < K) (y : Fin K → Fin s → ℝ) :
    ∑ k, CE (testRate y k) (sampleRate y) = K * binEntropy (sampleRate y) := by
  set P := sampleRate y
  have : ∀ k, CE (testRate y k) P = (log (1 - P) - log P) * testRate y k - log (1 - P) := by
    intro k; simp only [CE]; ring
  rw [sum_congr rfl (fun k _ => this k), sum_sub_distrib, ← mul_sum, sum_testRate hs hK y,
    sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, ← CE_self P, CE]
  ring

/-- **A4+ (sandwich).** For *any* partition into `K ≥ 2` equal folds, with outcomes in `[0,1]`,
`0 < p̂ < 1` and every training rate in `(0,1)`, the base-rate predictor's cross-validated log loss
exceeds the stratified value `H(p̂)` by at least `Σₖ (rₖ - p̂)² / (K (K-1) p̂ (1-p̂))` and at most
`Σₖ (rₖ - p̂)² / ((K-1)² qₖ (1-qₖ))`, where `rₖ, qₖ` are fold `k`'s test and training rates.
In particular stratification minimises it, sample by sample. -/
theorem cv_sandwich (hK : 2 ≤ K) (hs : 0 < s) {y : Fin K → Fin s → ℝ}
    (hy : ∀ k j, 0 ≤ y k j ∧ y k j ≤ 1) (hP0 : 0 < sampleRate y) (hP1 : sampleRate y < 1)
    (hq : ∀ k, 0 < trainRate y k ∧ trainRate y k < 1) :
    (∑ k, (testRate y k - sampleRate y) ^ 2)
        / (K * ((K : ℝ) - 1) * sampleRate y * (1 - sampleRate y))
      ≤ cvLoss y - binEntropy (sampleRate y) ∧
    cvLoss y - binEntropy (sampleRate y)
      ≤ ∑ k, (testRate y k - sampleRate y) ^ 2
          / (((K : ℝ) - 1) ^ 2 * trainRate y k * (1 - trainRate y k)) := by
  set P := sampleRate y with hPdef
  have hK2 : (2 : ℝ) ≤ K := by exact_mod_cast hK
  have hKpos : (0 : ℝ) < K := by linarith
  have hK1 : (0 : ℝ) < (K : ℝ) - 1 := by linarith
  have hP1' : (0 : ℝ) < 1 - P := by linarith
  -- cvLoss - H(P) as a mean of per-fold gaps
  have hcv : cvLoss y - binEntropy P
      = (∑ k, (CE (testRate y k) (trainRate y k) - CE (testRate y k) P)) / K := by
    rw [sum_sub_distrib, sum_CE_sampleRate hs (by omega) y, cvLoss]
    simp only [foldLoss_eq_CE hs]
    field_simp
    rfl
  -- per-fold algebra: P - q = (r - P)/(K-1), r - q = K (r - P)/(K-1)
  have hPq : ∀ k, P - trainRate y k = (testRate y k - P) / ((K : ℝ) - 1) := by
    intro k
    have := trainRate_mul hs hK y k
    field_simp
    linarith
  have hrq : ∀ k, testRate y k - trainRate y k = K * (testRate y k - P) / ((K : ℝ) - 1) := by
    intro k
    have := trainRate_mul hs hK y k
    field_simp
    linarith
  constructor
  · rw [hcv]
    have hlow : ∀ k, (testRate y k - P) ^ 2 / (((K : ℝ) - 1) * P * (1 - P))
        ≤ CE (testRate y k) (trainRate y k) - CE (testRate y k) P := by
      intro k
      obtain ⟨hr0, hr1⟩ := testRate_mem hs hy k
      have g := gap_lower hr0 hr1 hP0 hP1 (hq k).1 (hq k).2
      have e : (P - trainRate y k) * (testRate y k - P) / (P * (1 - P))
          = (testRate y k - P) ^ 2 / (((K : ℝ) - 1) * P * (1 - P)) := by
        rw [hPq k]; field_simp
      linarith
    have hsum := sum_le_sum fun k (_ : k ∈ univ) => hlow k
    rw [← sum_div] at hsum
    rw [div_le_div_iff₀ (by positivity) hKpos]
    calc (∑ k, (testRate y k - P) ^ 2) * K
        = (∑ k, (testRate y k - P) ^ 2) / (((K : ℝ) - 1) * P * (1 - P))
            * (K * ((K : ℝ) - 1) * P * (1 - P)) := by field_simp
      _ ≤ (∑ k, (CE (testRate y k) (trainRate y k) - CE (testRate y k) P))
            * (K * ((K : ℝ) - 1) * P * (1 - P)) := by
          apply mul_le_mul_of_nonneg_right hsum; positivity
  · rw [hcv, div_le_iff₀ hKpos]
    rw [sum_mul]
    apply sum_le_sum
    intro k _
    obtain ⟨hr0, hr1⟩ := testRate_mem hs hy k
    obtain ⟨hq0, hq1⟩ := hq k
    have hq1' : (0 : ℝ) < 1 - trainRate y k := by linarith
    have g := gap_upper hr0 hr1 hP0 hP1 hq0 hq1
    have e : (P - trainRate y k) * (testRate y k - trainRate y k)
          / (trainRate y k * (1 - trainRate y k))
        = (testRate y k - P) ^ 2 / (((K : ℝ) - 1) ^ 2 * trainRate y k * (1 - trainRate y k))
          * K := by
      rw [hPq k, hrq k]; field_simp
    linarith

/-! ## A5. The exact optimism decomposition -/

/-- **A5 (pointwise).** For any truth-model rate `qT ∉ {0,1}` and any estimate equal to `H(p̂)`,
`truth − estimate = KL(p ‖ qT) + (H(p) − H(p̂))`. -/
theorem optimism_pointwise (p P : ℝ) {qT : ℝ} (hq0 : qT ≠ 0) (hq1 : qT ≠ 1) :
    freshLoss p qT - binEntropy P = KL p qT + (binEntropy p - binEntropy P) := by
  rw [freshLoss_eq p hq0 hq1]; ring

/-- **A5 (stratified CV).** Under exact stratification, the optimism of the base-rate predictor's
cross-validated log loss against the fresh-data loss of any truth model with rate `qT` is
`KL(p ‖ qT) + (H(p) − H(p̂))`, sample by sample. -/
theorem stratified_optimism (hK : 2 ≤ K) (hs : 0 < s) {y : Fin K → Fin s → ℝ} (h : Stratified y)
    (p : ℝ) {qT : ℝ} (hq0 : qT ≠ 0) (hq1 : qT ≠ 1) :
    freshLoss p qT - cvLoss y = KL p qT + (binEntropy p - binEntropy (sampleRate y)) := by
  rw [stratified_cvLoss hK hs h, optimism_pointwise p _ hq0 hq1]

/-- The fold-model truth: the mean fresh-data loss of the `K` fold models. -/
def foldTruth (p : ℝ) (y : Fin K → Fin s → ℝ) : ℝ := (∑ k, freshLoss p (trainRate y k)) / K

theorem stratified_foldTruth (hK : 2 ≤ K) (hs : 0 < s) {y : Fin K → Fin s → ℝ}
    (h : Stratified y) (p : ℝ) : foldTruth p y = freshLoss p (sampleRate y) := by
  have hK' : (K : ℝ) ≠ 0 := by
    have : (2 : ℝ) ≤ K := by exact_mod_cast hK
    linarith
  simp only [foldTruth, fun k => (stratified_rates hK hs h k).2, sum_const, card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- **A5 (expectation).** Over any finite probability space of exactly stratified samples and
truth-model rates in `(0,1)`, the expected optimism is
`E[KL(p ‖ qT)] + (H(p) − E[H(p̂)])`. -/
theorem expected_optimism {Ω : Type*} (S : Finset Ω) (w : Ω → ℝ) (hw : ∑ ω ∈ S, w ω = 1)
    (y : Ω → Fin K → Fin s → ℝ) (qT : Ω → ℝ) (p : ℝ) (hK : 2 ≤ K) (hs : 0 < s)
    (hstrat : ∀ ω ∈ S, Stratified (y ω)) (hq : ∀ ω ∈ S, qT ω ≠ 0 ∧ qT ω ≠ 1) :
    ∑ ω ∈ S, w ω * (freshLoss p (qT ω) - cvLoss (y ω))
      = ∑ ω ∈ S, w ω * KL p (qT ω)
        + (binEntropy p - ∑ ω ∈ S, w ω * binEntropy (sampleRate (y ω))) := by
  rw [sum_congr rfl fun ω hω =>
    congrArg (w ω * ·) (stratified_optimism hK hs (hstrat ω hω) p (hq ω hω).1 (hq ω hω).2)]
  simp only [mul_add, mul_sub, sum_add_distrib, sum_sub_distrib, ← sum_mul, hw, one_mul]

/-- **A5 (entropy bias).** If `p̂` has mean `p ∈ (0,1)`, the plug-in entropy's bias is exactly the
expected reverse KL: `H(p) − E[H(p̂)] = E[KL(p̂ ‖ p)] (≥ 0)`. -/
theorem entropy_bias_eq_KL {Ω : Type*} (S : Finset Ω) (w : Ω → ℝ) (hw : ∑ ω ∈ S, w ω = 1)
    (P : Ω → ℝ) {p : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hmean : ∑ ω ∈ S, w ω * P ω = p) :
    binEntropy p - ∑ ω ∈ S, w ω * binEntropy (P ω) = ∑ ω ∈ S, w ω * KL (P ω) p := by
  have hKL : ∀ ω, KL (P ω) p = (log (1 - p) - log p) * P ω - log (1 - p) - binEntropy (P ω) := by
    intro ω
    have := CE_eq_binEntropy_add_KL (P ω) hp0.ne' hp1.ne
    simp only [CE] at this
    linarith
  simp only [hKL, mul_sub, sum_sub_distrib]
  rw [show ∑ ω ∈ S, w ω * ((log (1 - p) - log p) * P ω)
      = (log (1 - p) - log p) * ∑ ω ∈ S, w ω * P ω by
        rw [mul_sum]; exact sum_congr rfl fun ω _ => by ring,
    hmean, ← sum_mul, hw, ← CE_self p, CE]
  ring

/-- **A5 (headline).** Under exact stratification, with `E[p̂] = p ∈ (0,1)`, the expected optimism
of the base-rate predictor is a sum of two expected KL divergences, each `≥ 0`:
`E[truth − estimate] = E[KL(p ‖ qT)] + E[KL(p̂ ‖ p)]`. -/
theorem expected_optimism_two_KL {Ω : Type*} (S : Finset Ω) (w : Ω → ℝ)
    (hw : ∑ ω ∈ S, w ω = 1) (y : Ω → Fin K → Fin s → ℝ) (qT : Ω → ℝ) {p : ℝ}
    (hp0 : 0 < p) (hp1 : p < 1) (hK : 2 ≤ K) (hs : 0 < s)
    (hstrat : ∀ ω ∈ S, Stratified (y ω)) (hq : ∀ ω ∈ S, qT ω ≠ 0 ∧ qT ω ≠ 1)
    (hmean : ∑ ω ∈ S, w ω * sampleRate (y ω) = p) :
    ∑ ω ∈ S, w ω * (freshLoss p (qT ω) - cvLoss (y ω))
      = ∑ ω ∈ S, w ω * KL p (qT ω) + ∑ ω ∈ S, w ω * KL (sampleRate (y ω)) p := by
  rw [expected_optimism S w hw y qT p hK hs hstrat hq,
    entropy_bias_eq_KL S w hw (fun ω => sampleRate (y ω)) hp0 hp1 hmean]

/-! ## A6. Fold-model truth: the optimism is an expected Jeffreys divergence -/

/-- The symmetrised KL (Jeffreys divergence) of two Bernoulli laws is
`(p - q)(logit p - logit q)`. -/
theorem KL_add_KL_swap {p q : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hq0 : 0 < q) (hq1 : q < 1) :
    KL p q + KL q p = (p - q) * (log (p / (1 - p)) - log (q / (1 - q))) := by
  have hp1' : (1 - p) ≠ 0 := by linarith
  have hq1' : (1 - q) ≠ 0 := by linarith
  rw [KL, KL, log_div hp0.ne' hq0.ne', log_div hq0.ne' hp0.ne', log_div hp1' hq1',
    log_div hq1' hp1', log_div hp0.ne' hp1', log_div hq0.ne' hq1']
  ring

/-- **A6.** Under exact stratification, against the fold-model truth (the mean fresh-data loss of
the `K` fold models), with every sample rate in `(0,1)` and `E[p̂] = p`, the expected optimism is
the expected Jeffreys divergence `E[(p̂ - p)(logit p̂ - logit p)]`. -/
theorem expected_foldTruth_optimism {Ω : Type*} (S : Finset Ω) (w : Ω → ℝ)
    (hw : ∑ ω ∈ S, w ω = 1) (y : Ω → Fin K → Fin s → ℝ) {p : ℝ}
    (hp0 : 0 < p) (hp1 : p < 1) (hK : 2 ≤ K) (hs : 0 < s)
    (hstrat : ∀ ω ∈ S, Stratified (y ω))
    (hP : ∀ ω ∈ S, 0 < sampleRate (y ω) ∧ sampleRate (y ω) < 1)
    (hmean : ∑ ω ∈ S, w ω * sampleRate (y ω) = p) :
    ∑ ω ∈ S, w ω * (foldTruth p (y ω) - cvLoss (y ω))
      = ∑ ω ∈ S, w ω * ((sampleRate (y ω) - p)
          * (log (sampleRate (y ω) / (1 - sampleRate (y ω))) - log (p / (1 - p)))) := by
  rw [sum_congr rfl fun ω hω => by rw [stratified_foldTruth hK hs (hstrat ω hω) p]]
  rw [expected_optimism_two_KL S w hw y (fun ω => sampleRate (y ω)) hp0 hp1 hK hs hstrat
    (fun ω hω => ⟨(hP ω hω).1.ne', (hP ω hω).2.ne⟩) hmean, ← sum_add_distrib]
  refine sum_congr rfl fun ω hω => ?_
  rw [← mul_add, KL_add_KL_swap hp0 hp1 (hP ω hω).1 (hP ω hω).2]
  ring

end

end Strat
