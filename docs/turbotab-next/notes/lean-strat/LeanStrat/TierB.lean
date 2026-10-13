import LeanStrat.TierA

/-!
# Stratification optimism, Tier B: second order, with explicit constants

* `binom m k p` — the Binomial(m, p) probability of `k` events (Mathlib's Bernstein polynomial);
* B1: `E[(p̂ - p)²] = p(1-p)/m` for a binomial proportion, as a finite sum;
* B2: `KL(p ‖ q) = (q-p)²/(2p(1-p)) + R`, `|R| ≤ |q-p|³ · Ccub p |q-p|`, for `|q-p| < min(p, 1-p)`;
* B3: the expected forward KL `E[KL(p ‖ p̂ₘ)]`, truncated to `|p̂ₘ - p| ≤ ε`, lies between explicit
  bounds around `1/(2m)`;
* B4: the plug-in entropy bias `H(p) - E[H(p̂ₙ)] = E[KL(p̂ₙ ‖ p)]` lies in `[0, 1/n]`.
-/

open Real Finset

namespace Strat

noncomputable section

/-! ## B1. Binomial moments -/

/-- Binomial(m, p) probability of `k` events. -/
def binom (m k : ℕ) (p : ℝ) : ℝ := (m.choose k : ℝ) * p ^ k * (1 - p) ^ (m - k)

lemma binom_eq_eval (m k : ℕ) (p : ℝ) : binom m k p = (bernsteinPolynomial ℝ m k).eval p := by
  simp [binom, bernsteinPolynomial]

lemma binom_nonneg {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (m k : ℕ) : 0 ≤ binom m k p := by
  have : 0 ≤ 1 - p := by linarith
  unfold binom; positivity

theorem binom_sum (m : ℕ) (p : ℝ) : ∑ k ∈ range (m + 1), binom m k p = 1 := by
  have := congrArg (Polynomial.eval p) (bernsteinPolynomial.sum ℝ m)
  simpa [Polynomial.eval_finsetSum, binom_eq_eval] using this

theorem binom_mean {m : ℕ} (hm : m ≠ 0) (p : ℝ) :
    ∑ k ∈ range (m + 1), binom m k p * ((k : ℝ) / m) = p := by
  have h := congrArg (Polynomial.eval p) (bernsteinPolynomial.sum_smul ℝ m)
  simp only [Polynomial.eval_finsetSum, nsmul_eq_mul, Polynomial.eval_mul, Polynomial.eval_natCast,
    Polynomial.eval_X, ← binom_eq_eval] at h
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  calc ∑ k ∈ range (m + 1), binom m k p * ((k : ℝ) / m)
      = (∑ k ∈ range (m + 1), (k : ℝ) * binom m k p) / m := by
        rw [sum_div]; exact sum_congr rfl fun k _ => by ring
    _ = p := by rw [h]; field_simp

/-- **B1.** `E[(p̂ - p)²] = p(1-p)/m` for `p̂ = k/m`, `k ~ Binomial(m, p)` (any real `p`). -/
theorem binom_var {m : ℕ} (hm : m ≠ 0) (p : ℝ) :
    ∑ k ∈ range (m + 1), binom m k p * ((k : ℝ) / m - p) ^ 2 = p * (1 - p) / m := by
  have h := congrArg (Polynomial.eval p) (bernsteinPolynomial.variance ℝ m)
  simp only [Polynomial.eval_finsetSum, Polynomial.eval_mul, Polynomial.eval_pow,
    Polynomial.eval_sub, nsmul_eq_mul, Polynomial.eval_X,
    Polynomial.eval_natCast, Polynomial.eval_one, ← binom_eq_eval] at h
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  calc ∑ k ∈ range (m + 1), binom m k p * ((k : ℝ) / m - p) ^ 2
      = (∑ k ∈ range (m + 1), ((m : ℝ) * p - k) ^ 2 * binom m k p) / (m : ℝ) ^ 2 := by
        rw [sum_div]; refine sum_congr rfl fun k _ => ?_; field_simp; ring
    _ = p * (1 - p) / m := by rw [h]; field_simp

/-! ## B2. Second-order expansion of the binary KL with an explicit cubic remainder -/

/-- The cubic-remainder constant `1/(p(p-d)) + 1/((1-p)(1-p-d))`. -/
def Ccub (p d : ℝ) : ℝ := 1 / (p * (p - d)) + 1 / ((1 - p) * (1 - p - d))

lemma log_series2 {x : ℝ} (hx : |x| < 1) :
    |x + x ^ 2 / 2 + log (1 - x)| ≤ |x| ^ 3 / (1 - |x|) := by
  have := Real.abs_log_sub_add_sum_range_le hx 2
  norm_num [Finset.sum_range_succ] at this
  exact this

/-- **B2.** For `0 < p < 1` and `|q - p| < min(p, 1-p)`:
`|KL(p ‖ q) - (q-p)²/(2p(1-p))| ≤ |q-p|³ (1/(p(p-|q-p|)) + 1/((1-p)(1-p-|q-p|)))`. -/
theorem KL_quadratic_cubic {p q : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hd : |q - p| < min p (1 - p)) :
    |KL p q - (q - p) ^ 2 / (2 * (p * (1 - p)))| ≤ |q - p| ^ 3 * Ccub p |q - p| := by
  have hdp : |q - p| < p := lt_of_lt_of_le hd (min_le_left _ _)
  have hdq : |q - p| < 1 - p := lt_of_lt_of_le hd (min_le_right _ _)
  have hp1' : 0 < 1 - p := by linarith
  have hq0 : 0 < q := by have := neg_abs_le (q - p); linarith
  have hq1 : q < 1 := by have := le_abs_self (q - p); linarith
  have hq1' : 0 < 1 - q := by linarith
  have hpd : 0 < p - |q - p| := by linarith
  have hpd' : 0 < 1 - p - |q - p| := by linarith
  have hx1 : |(p - q) / p| = |q - p| / p := by rw [abs_div, abs_of_pos hp0, abs_sub_comm]
  have hx2 : |(q - p) / (1 - p)| = |q - p| / (1 - p) := by rw [abs_div, abs_of_pos hp1']
  have hx1lt : |(p - q) / p| < 1 := by rw [hx1, div_lt_one hp0]; exact hdp
  have hx2lt : |(q - p) / (1 - p)| < 1 := by rw [hx2, div_lt_one hp1']; exact hdq
  have L1 := log_series2 hx1lt
  have L2 := log_series2 hx2lt
  have e1 : 1 - (p - q) / p = q / p := by field_simp; ring
  have e2 : 1 - (q - p) / (1 - p) = (1 - q) / (1 - p) := by field_simp; ring
  rw [e1, hx1] at L1
  rw [e2, hx2] at L2
  have l1 : log (p / q) = -log (q / p) := by rw [← log_inv, inv_div]
  have l2 : log ((1 - p) / (1 - q)) = -log ((1 - q) / (1 - p)) := by rw [← log_inv, inv_div]
  have hKL : KL p q - (q - p) ^ 2 / (2 * (p * (1 - p)))
      = -(p * ((p - q) / p + ((p - q) / p) ^ 2 / 2 + log (q / p)))
        - (1 - p) * ((q - p) / (1 - p) + ((q - p) / (1 - p)) ^ 2 / 2
            + log ((1 - q) / (1 - p))) := by
    rw [KL, l1, l2]
    generalize log (q / p) = A1
    generalize log ((1 - q) / (1 - p)) = A2
    field_simp
    ring
  rw [hKL]
  have b1 : p * ((|q - p| / p) ^ 3 / (1 - |q - p| / p)) = |q - p| ^ 3 / (p * (p - |q - p|)) := by
    field_simp
  have b2 : (1 - p) * ((|q - p| / (1 - p)) ^ 3 / (1 - |q - p| / (1 - p)))
      = |q - p| ^ 3 / ((1 - p) * (1 - p - |q - p|)) := by
    field_simp
  have hC : |q - p| ^ 3 * Ccub p |q - p|
      = |q - p| ^ 3 / (p * (p - |q - p|)) + |q - p| ^ 3 / ((1 - p) * (1 - p - |q - p|)) := by
    simp only [Ccub]; ring
  rw [hC, ← b1, ← b2]
  obtain ⟨L1a, L1b⟩ := abs_le.mp L1
  obtain ⟨L2a, L2b⟩ := abs_le.mp L2
  rw [abs_le]
  constructor <;> nlinarith [mul_le_mul_of_nonneg_left L1a hp0.le,
    mul_le_mul_of_nonneg_left L1b hp0.le, mul_le_mul_of_nonneg_left L2a hp1'.le,
    mul_le_mul_of_nonneg_left L2b hp1'.le]

lemma Ccub_nonneg {p d : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hd : d < min p (1 - p)) :
    0 ≤ Ccub p d := by
  have h1 : 0 < p - d := by have := min_le_left p (1 - p); linarith
  have h2 : 0 < 1 - p - d := by have := min_le_right p (1 - p); linarith
  have hp1' : 0 < 1 - p := by linarith
  unfold Ccub; positivity

lemma Ccub_mono {p d ε : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hd : d ≤ ε) (hε : ε < min p (1 - p)) :
    Ccub p d ≤ Ccub p ε := by
  have h1 : 0 < p - ε := by have := min_le_left p (1 - p); linarith
  have h2 : 0 < 1 - p - ε := by have := min_le_right p (1 - p); linarith
  have hp1' : 0 < 1 - p := by linarith
  unfold Ccub
  gcongr

/-- **B2'.** On `|q - p| ≤ ε < min(p, 1-p)`, the KL is sandwiched by quadratics:
`(q-p)² (1/(2p(1-p)) - ε C) ≤ KL(p ‖ q) ≤ (q-p)² (1/(2p(1-p)) + ε C)`, `C = Ccub p ε`. -/
theorem KL_quad_sandwich {p q ε : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hε : ε < min p (1 - p))
    (hqε : |q - p| ≤ ε) :
    (q - p) ^ 2 * (1 / (2 * (p * (1 - p))) - ε * Ccub p ε) ≤ KL p q ∧
    KL p q ≤ (q - p) ^ 2 * (1 / (2 * (p * (1 - p))) + ε * Ccub p ε) := by
  have hd : |q - p| < min p (1 - p) := lt_of_le_of_lt hqε hε
  have h := KL_quadratic_cubic hp0 hp1 hd
  have hC0 := Ccub_nonneg hp0 hp1 hd
  have hCm := Ccub_mono hp0 hp1 hqε hε
  have hsq : |q - p| ^ 2 = (q - p) ^ 2 := sq_abs _
  have hR : |q - p| ^ 3 * Ccub p |q - p| ≤ (q - p) ^ 2 * (ε * Ccub p ε) := by
    rw [← hsq]
    have ha : 0 ≤ |q - p| := abs_nonneg _
    calc |q - p| ^ 3 * Ccub p |q - p| = |q - p| ^ 2 * (|q - p| * Ccub p |q - p|) := by ring
      _ ≤ |q - p| ^ 2 * (ε * Ccub p ε) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        exact mul_le_mul hqε hCm hC0 (le_trans ha hqε)
  obtain ⟨ha, hb⟩ := abs_le.mp h
  constructor
  · have : (q - p) ^ 2 * (1 / (2 * (p * (1 - p))) - ε * Ccub p ε)
        = (q - p) ^ 2 / (2 * (p * (1 - p))) - (q - p) ^ 2 * (ε * Ccub p ε) := by ring
    linarith
  · have : (q - p) ^ 2 * (1 / (2 * (p * (1 - p))) + ε * Ccub p ε)
        = (q - p) ^ 2 / (2 * (p * (1 - p))) + (q - p) ^ 2 * (ε * Ccub p ε) := by ring
    linarith

/-! ## B3. The expected forward KL `E[KL(p ‖ p̂ₘ)]` near `p`, with explicit constants -/

/-- **B3 (upper).** `E[KL(p ‖ p̂ₘ); |p̂ₘ - p| ≤ ε] ≤ (1/(2m)) (1 + 2 p(1-p) ε C)`. -/
theorem trunc_KL_upper {m : ℕ} (hm : m ≠ 0) {p ε : ℝ} (hp0 : 0 < p) (hp1 : p < 1)
    (hε0 : 0 ≤ ε) (hε : ε < min p (1 - p)) :
    ∑ k ∈ (range (m + 1)).filter (fun k : ℕ => |(k : ℝ) / m - p| ≤ ε), binom m k p * KL p (k / m)
      ≤ 1 / (2 * m) * (1 + 2 * (p * (1 - p)) * ε * Ccub p ε) := by
  have hp1' : 0 < 1 - p := by linarith
  have hm' : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  have hC0 := Ccub_nonneg hp0 hp1 hε
  have hA : 0 ≤ 1 / (2 * (p * (1 - p))) + ε * Ccub p ε := by positivity
  calc ∑ k ∈ (range (m + 1)).filter (fun k : ℕ => |(k : ℝ) / m - p| ≤ ε), binom m k p * KL p (k / m)
      ≤ ∑ k ∈ (range (m + 1)).filter (fun k : ℕ => |(k : ℝ) / m - p| ≤ ε),
          binom m k p * (((k : ℝ) / m - p) ^ 2 * (1 / (2 * (p * (1 - p))) + ε * Ccub p ε)) := by
        apply sum_le_sum
        intro k hk
        exact mul_le_mul_of_nonneg_left (KL_quad_sandwich hp0 hp1 hε (mem_filter.mp hk).2).2
          (binom_nonneg hp0.le hp1.le m k)
    _ ≤ ∑ k ∈ range (m + 1),
          binom m k p * (((k : ℝ) / m - p) ^ 2 * (1 / (2 * (p * (1 - p))) + ε * Ccub p ε)) := by
        apply sum_le_sum_of_subset_of_nonneg (filter_subset _ _)
        intro k _ _
        exact mul_nonneg (binom_nonneg hp0.le hp1.le m k) (mul_nonneg (sq_nonneg _) hA)
    _ = (∑ k ∈ range (m + 1), binom m k p * ((k : ℝ) / m - p) ^ 2)
          * (1 / (2 * (p * (1 - p))) + ε * Ccub p ε) := by
        rw [sum_mul]; exact sum_congr rfl fun k _ => by ring
    _ = 1 / (2 * m) * (1 + 2 * (p * (1 - p)) * ε * Ccub p ε) := by
        rw [binom_var hm]; field_simp

/-- **B3 (lower).** `E[KL(p ‖ p̂ₘ); |p̂ₘ - p| ≤ ε] ≥ (1/(2p(1-p)) - ε C) (p(1-p)/m - T)`, where
`T = E[(p̂ₘ - p)²; |p̂ₘ - p| > ε]` is the (explicit, finite-sum) tail second moment. -/
theorem trunc_KL_lower {m : ℕ} (hm : m ≠ 0) {p ε : ℝ} (hp0 : 0 < p) (hp1 : p < 1)
    (hε : ε < min p (1 - p)) :
    (1 / (2 * (p * (1 - p))) - ε * Ccub p ε) *
        (p * (1 - p) / m - ∑ k ∈ (range (m + 1)).filter (fun k : ℕ => ¬ |(k : ℝ) / m - p| ≤ ε),
          binom m k p * ((k : ℝ) / m - p) ^ 2)
      ≤ ∑ k ∈ (range (m + 1)).filter (fun k : ℕ => |(k : ℝ) / m - p| ≤ ε),
          binom m k p * KL p (k / m) := by
  have hsplit := sum_filter_add_sum_filter_not (range (m + 1))
    (fun k : ℕ => |(k : ℝ) / m - p| ≤ ε) (fun k => binom m k p * ((k : ℝ) / m - p) ^ 2)
  rw [binom_var hm] at hsplit
  rw [show p * (1 - p) / m - ∑ k ∈ (range (m + 1)).filter (fun k : ℕ => ¬ |(k : ℝ) / m - p| ≤ ε),
        binom m k p * ((k : ℝ) / m - p) ^ 2
      = ∑ k ∈ (range (m + 1)).filter (fun k : ℕ => |(k : ℝ) / m - p| ≤ ε),
        binom m k p * ((k : ℝ) / m - p) ^ 2 by linarith, mul_sum]
  apply sum_le_sum
  intro k hk
  have h := (KL_quad_sandwich (q := (k : ℝ) / m) hp0 hp1 hε (mem_filter.mp hk).2).1
  have hb := binom_nonneg hp0.le hp1.le m k
  calc (1 / (2 * (p * (1 - p))) - ε * Ccub p ε) * (binom m k p * ((k : ℝ) / m - p) ^ 2)
      = binom m k p * (((k : ℝ) / m - p) ^ 2 * (1 / (2 * (p * (1 - p))) - ε * Ccub p ε)) := by
        ring
    _ ≤ binom m k p * KL p (k / m) := mul_le_mul_of_nonneg_left h hb

/-! ## B4. The plug-in entropy bias `E[KL(p̂ₙ ‖ p)]` lies in `[0, 1/n]` -/

/-- χ² bound: `KL(r ‖ q) ≤ (r-q)²/(q(1-q))` for `r ∈ [0,1]`, `q ∈ (0,1)`. -/
theorem KL_le_chi2 {r q : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hq0 : 0 < q) (hq1 : q < 1) :
    KL r q ≤ (r - q) ^ 2 / (q * (1 - q)) := by
  have hq1' : 0 < 1 - q := by linarith
  have t1 : r * log (r / q) ≤ r * (r / q - 1) := by
    rcases hr0.eq_or_lt with h | h
    · rw [← h]; simp
    · exact mul_le_mul_of_nonneg_left (log_le_sub_one_of_pos (div_pos h hq0)) h.le
  have t2 : (1 - r) * log ((1 - r) / (1 - q)) ≤ (1 - r) * ((1 - r) / (1 - q) - 1) := by
    rcases (sub_nonneg.mpr hr1).eq_or_lt with h | h
    · rw [← h]; simp
    · exact mul_le_mul_of_nonneg_left (log_le_sub_one_of_pos (div_pos h hq1')) h.le
  have e : (r - q) ^ 2 / (q * (1 - q))
      = r * (r / q - 1) + (1 - r) * ((1 - r) / (1 - q) - 1) := by
    field_simp; ring
  rw [KL, e]; linarith

/-- **B4.** For `p̂ₙ = k/n`, `k ~ Binomial(n, p)`, `0 < p < 1`: the plug-in entropy's bias
`H(p) − E[H(p̂ₙ)]` equals `E[KL(p̂ₙ ‖ p)]` and lies in `[0, 1/n]`. -/
theorem entropy_bias_binom {n : ℕ} (hn : n ≠ 0) {p : ℝ} (hp0 : 0 < p) (hp1 : p < 1) :
    binEntropy p - ∑ k ∈ range (n + 1), binom n k p * binEntropy ((k : ℝ) / n)
        = ∑ k ∈ range (n + 1), binom n k p * KL ((k : ℝ) / n) p ∧
    0 ≤ binEntropy p - ∑ k ∈ range (n + 1), binom n k p * binEntropy ((k : ℝ) / n) ∧
    binEntropy p - ∑ k ∈ range (n + 1), binom n k p * binEntropy ((k : ℝ) / n) ≤ 1 / n := by
  have hp1' : 0 < 1 - p := by linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero hn
  have heq := entropy_bias_eq_KL (range (n + 1)) (fun k => binom n k p) (binom_sum n p)
    (fun k => (k : ℝ) / n) hp0 hp1 (binom_mean hn p)
  have hk : ∀ k ∈ range (n + 1), 0 ≤ (k : ℝ) / n ∧ (k : ℝ) / n ≤ 1 := by
    intro k hk
    refine ⟨by positivity, ?_⟩
    rw [div_le_one hn']
    exact_mod_cast Nat.lt_succ_iff.mp (mem_range.mp hk)
  refine ⟨heq, ?_, ?_⟩
  · rw [heq]
    exact sum_nonneg fun k hk' =>
      mul_nonneg (binom_nonneg hp0.le hp1.le n k) (KL_nonneg (hk k hk').1 (hk k hk').2 hp0 hp1)
  · rw [heq]
    calc ∑ k ∈ range (n + 1), binom n k p * KL ((k : ℝ) / n) p
        ≤ ∑ k ∈ range (n + 1), binom n k p * (((k : ℝ) / n - p) ^ 2 / (p * (1 - p))) := by
          apply sum_le_sum
          intro k hk'
          exact mul_le_mul_of_nonneg_left (KL_le_chi2 (hk k hk').1 (hk k hk').2 hp0 hp1)
            (binom_nonneg hp0.le hp1.le n k)
      _ = (∑ k ∈ range (n + 1), binom n k p * ((k : ℝ) / n - p) ^ 2) / (p * (1 - p)) := by
          rw [sum_div]; exact sum_congr rfl fun k _ => by ring
      _ = 1 / n := by rw [binom_var hn]; field_simp

end

end Strat
