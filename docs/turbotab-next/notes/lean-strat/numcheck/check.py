"""Close the loop: instantiate each Lean theorem on simulated / exactly enumerated data (numpy only).
Setting of strat_check.py: y ~ Bernoulli(0.3), n = 200, K = 5 (s = 40), truth model on m = 160 rows."""
import math, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

p, n, K = 0.3, 200, 5
s, m = n // K, n * (K - 1) // K
R = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
CLIP = 1e-6
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 20261013)

def xlogy(a, b):  # a*log(b) with 0*log(0) = 0
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(np.broadcast(a, b).shape)
    nz = np.broadcast_to(a != 0, out.shape)
    out[nz] = (np.broadcast_to(a, out.shape) * np.log(np.broadcast_to(b, out.shape)))[nz]
    return out
H = lambda r: -xlogy(r, r) - xlogy(1 - r, 1 - r)
CE = lambda r, q: -xlogy(r, q) - xlogy(1 - r, 1 - q)
KL = lambda r, q: CE(r, q) - H(r)            # A1 used as the definition here; checked below
def KL_direct(r, q):
    r, q = np.asarray(r, float), np.asarray(q, float)
    return xlogy(r, r / q) + xlogy(1 - r, (1 - r) / (1 - q))
def binom_pmf(N, pp):
    k = np.arange(N + 1)
    lg = np.array([math.lgamma(N + 1) - math.lgamma(i + 1) - math.lgamma(N - i + 1) for i in k])
    with np.errstate(divide="ignore"):
        return np.exp(lg + k * np.log(pp) + (N - k) * np.log(1 - pp))
clip = lambda q: np.clip(q, CLIP, 1 - CLIP)
worst = {}
def rec(name, val):
    worst[name] = max(worst.get(name, 0.0), float(np.max(np.abs(val))))

# ---------- simulate ----------
Y = (rng.random((R, n)) < p).astype(float)
c = Y.sum(1).astype(int)
phat = c / n

def fold_rates(counts):  # counts: (R, K) events per fold of size s
    r = counts / s
    q = (c[:, None] - counts) / (m)
    return r, q
def cv_from_counts(counts):
    r, q = fold_rates(counts)
    return CE(r, clip(q)).mean(1), r, q

# sklearn-style stratification (sorted round-robin): fold i gets #{j>=n-c : j%K==i} events
def skl_counts(cc):
    j = np.arange(n)
    return np.stack([((j[None, :] >= n - cc[:, None]) & (j[None, :] % K == i)).sum(1) for i in range(K)], 1)
cnt_skl = skl_counts(c)
# plain K-fold: random permutation into K folds of s
perm = np.argsort(rng.random((R, n)), axis=1)
Yp = np.take_along_axis(Y, perm, 1).reshape(R, K, s)
cnt_plain = Yp.sum(2)
# exact stratification where K | c
exact = (c % K == 0)
cnt_exact = np.repeat((c // K)[:, None], K, 1)

cv_skl, r_skl, q_skl = cv_from_counts(cnt_skl)
cv_pl, r_pl, q_pl = cv_from_counts(cnt_plain)
cv_ex, r_ex, q_ex = cv_from_counts(cnt_exact)

# A1': mean log loss over a fold == CE(test rate, q) (row level, first 2000 datasets, plain folds)
qk = clip(q_pl[:2000])
rowloss = -(Yp[:2000] * np.log(qk[:, :, None]) + (1 - Yp[:2000]) * np.log(1 - qk[:, :, None])).mean(2)
rec("A1' mean row log loss - CE(r,q)", rowloss - CE(r_pl[:2000], qk))
# A1: CE = H + KL with KL computed directly
rec("A1 CE - (H + KL_direct)", CE(r_pl, qk := clip(q_pl)) - (H(r_pl) + KL_direct(r_pl, qk)))
# A2: fresh loss by Monte Carlo vs H(p)+KL(p||q)
q_test = np.array([0.1, 0.25, 0.3, 0.42, 0.8])
Yf = rng.random((2_000_000, 1)) < p
mc = -(Yf * np.log(q_test) + (~Yf) * np.log(1 - q_test)).mean(0)
a2 = H(p) + KL_direct(p, q_test)
se = (-(Yf * np.log(q_test) + (~Yf) * np.log(1 - q_test))).std(0) / np.sqrt(len(Yf))
print(f"A2 fresh loss: Monte Carlo (2e6 fresh draws, their event rate {Yf.mean():.5f}) vs H(p)+KL(p||q):")
for qq, a, b, e in zip(q_test, mc, a2, se):
    print(f"   q={qq:.2f}  MC={a:.5f}  formula={b:.5f}  diff={a-b:+.5f} ({(a-b)/e:+.1f} SE)")
# A3: KL >= 0, =0 iff r = q (grid)
g = np.linspace(0, 1, 401); gq = np.linspace(0.0025, 0.9975, 399)
Kg = KL_direct(g[:, None], gq[None, :])
print(f"A3 min KL on grid = {Kg.min():.2e}; KL at r=q: max {np.max(np.abs(KL_direct(gq, gq))):.1e}; "
      f"min KL off-diagonal (|r-q|>=0.0025) = {Kg[np.abs(g[:,None]-gq[None,:])>=0.0024].min():.2e}")
# A4: exact stratification -> all rates = phat and CV = H(phat)
rec("A4 exact-strat test/train rate - phat", np.concatenate([(r_ex - phat[:, None])[exact].ravel(), (q_ex - phat[:, None])[exact].ravel()]))
rec("A4 exact-strat CV - H(phat)", (cv_ex - H(phat))[exact])
print(f"A4 exactly stratified datasets (K | events): {exact.sum()} of {R}")
# A4+: sandwich for sklearn rounding and for plain folds
def sandwich(cv, r, q):
    gap = cv - H(phat)
    lo = ((r - phat[:, None]) ** 2).sum(1) / (K * (K - 1) * phat * (1 - phat))
    hi = ((r - phat[:, None]) ** 2 / ((K - 1) ** 2 * q * (1 - q))).sum(1)
    return gap, lo, hi
for name, (cv, r, q) in {"sklearn-strat": (cv_skl, r_skl, q_skl), "plain": (cv_pl, r_pl, q_pl)}.items():
    gap, lo, hi = sandwich(cv, r, q)
    ok = (lo <= gap + 1e-12) & (gap <= hi + 1e-12)
    print(f"A4+ sandwich [{name}]: holds in {ok.sum()}/{R}; mean lo={lo.mean():.5f} gap={gap.mean():.5f} hi={hi.mean():.5f}"
          f"  (1/n={1/n:.5f})")
# A5 pointwise decomposition: truth(random 4/5) - H(phat) = KL(p||qT) + H(p) - H(phat)
idx = np.argsort(rng.random((R, n)), axis=1)[:, :m]
qT = clip(np.take_along_axis(Y, idx, 1).mean(1))
truth = CE(p, qT)
rec("A5 truth - H(phat) - [KL(p||qT) + H(p) - H(phat)]", truth - H(phat) - (KL_direct(p, qT) + H(p) - H(phat)))
foldtruth_skl = CE(p, clip(q_skl)).mean(1)
foldtruth_ex = CE(p, clip(q_ex)).mean(1)
# A6 Jeffreys identity pointwise
J = (phat - p) * (np.log(phat / (1 - phat)) - np.log(p / (1 - p)))
rec("A6 KL(p||phat)+KL(phat||p) - Jeffreys", KL_direct(p, phat) + KL_direct(phat, p) - J)
print("\nMachine-precision identities (max |difference| over all simulated datasets):")
for k_, v in worst.items():
    print(f"   {k_:55s} {v:.2e}")

# ---------- exact expectations by finite sums ----------
w_n, w_m = binom_pmf(n, p), binom_pmf(m, p)
k_n, k_m = np.arange(n + 1), np.arange(m + 1)
EKL_fwd_clip = (w_m * KL_direct(p, clip(k_m / m))).sum()                # what the sim measures
inner = (k_m > 0) & (k_m < m)
EKL_fwd_inner = (w_m * np.where(inner, KL_direct(p, clip(k_m / m)), 0)).sum()
EKL_rev = (w_n * KL_direct(k_n / n, p)).sum()
ent_bias = H(p) - (w_n * H(k_n / n)).sum()
cnt_all = skl_counts(k_n)
cv_all = CE(cnt_all / s, clip((k_n[:, None] - cnt_all) / m)).mean(1)
E_gap = (w_n * (cv_all - H(k_n / n))).sum()
E_cv_skl = (w_n * cv_all).sum()
E_truth = (w_m * CE(p, clip(k_m / m))).sum()
ft_all = CE(p, clip((k_n[:, None] - cnt_all) / m)).mean(1)
E_ft_skl = (w_n * ft_all).sum()
EJ = (w_n * np.where((k_n > 0) & (k_n < n), (k_n / n - p) * (np.log(np.clip(k_n / n, 1e-300, 1) / np.clip(1 - k_n / n, 1e-300, 1)) - np.log(p / (1 - p))), 0)).sum()
print("\nB1: E[(phat_m - p)^2]: exact sum = %.7f, p(1-p)/m = %.7f, simulated = %.7f"
      % ((w_m * (k_m / m - p) ** 2).sum(), p * (1 - p) / m, ((np.take_along_axis(Y, idx, 1).mean(1) - p) ** 2).mean()))
print(f"     P(phat_m in {{0,1}}) = {w_m[0] + w_m[-1]:.2e}  (forward KL is +inf there; the sim clips at {CLIP})")
print(f"A5/B: E[KL(p||phat_m)] exact (clipped) = {EKL_fwd_clip:.6f}; on 0<k<m = {EKL_fwd_inner:.6f}; 1/(2m) = {1/(2*m):.6f}")
print(f"      E[KL(phat_n||p)] exact = {EKL_rev:.6f};  H(p)-E[H(phat_n)] = {ent_bias:.6f}; 1/(2n) = {1/(2*n):.6f}; B4 bound [0, {1/n:.4f}]")
print(f"      exact-stratification optimism (random-4/5 truth) = E KL_fwd + E KL_rev = {EKL_fwd_clip + EKL_rev:.6f}"
      f"   leading 1/(2m)+1/(2n) = {1/(2*m)+1/(2*n):.6f}")
print(f"      sklearn rounding gap E[cv_skl - H(phat)] (>=0 by A4+) = {E_gap:.6f}")
print(f"      => exact E[strat_skl CV - truth] = {E_cv_skl - E_truth:+.6f}   (decomposition: {-(EKL_fwd_clip + EKL_rev) + E_gap:+.6f})")
opt_sim = cv_skl - truth
print(f"      simulated here (R={R})  E[strat_skl CV - truth] = {opt_sim.mean():+.6f} +- {opt_sim.std(ddof=1)/np.sqrt(R):.6f};"
      f"  STRATIFICATION_OPTIMISM.md: -0.00536 +- 0.00020")
print(f"      simulated plain CV - truth = {(cv_pl - truth).mean():+.6f} +- {(cv_pl - truth).std(ddof=1)/np.sqrt(R):.6f}  (note: +0.0002 +- 0.0002 [sign: CV-truth])")
print(f"A6: fold-model truth, exact strat: optimism = E[Jeffreys] exact = {EJ:.6f} (1/n = {1/n:.6f});"
      f"  with sklearn rounding exact = {E_ft_skl - E_cv_skl:.6f}; simulated = {(foldtruth_skl - cv_skl).mean():.6f} +- {(foldtruth_skl - cv_skl).std(ddof=1)/np.sqrt(R):.6f} (note: 0.00458 +- 0.00020)")
gap_pl = cv_pl - H(phat)
foldtruth_pl = CE(p, clip(q_pl)).mean(1)
pd_ = (foldtruth_skl - cv_skl) - (foldtruth_pl - cv_pl)
print(f"     paired (strat optimism - plain optimism), fold-model truths: {pd_.mean():.6f} +- {pd_.std(ddof=1)/np.sqrt(R):.6f}  (note: 0.00493 +- 0.00003; exact {E_ft_skl - E_cv_skl:.6f})")
pc = cv_pl - cv_skl
print(f"A4+: plain CV - sklearn-strat CV = {pc.mean():.6f} +- {pc.std(ddof=1)/np.sqrt(R):.6f}; exact E = E[truth]-E[cv_skl] = {E_truth - E_cv_skl:.6f}")
print(f"sample mean of phat - p in this simulation: {phat.mean()-p:+.6f} (SE {np.sqrt(p*(1-p)/n/R):.6f})")
# control variate: KL(phat||p) = H(p) - H(phat) + H'(p)(phat - p) has the same mean as H(p)-H(phat), lower variance
Hp = np.log((1 - p) / p)
cv_opt = (truth - cv_skl) + Hp * (phat - p)
print(f"control-variate optimism (random-4/5 truth): {-cv_opt.mean():+.6f} +- {cv_opt.std(ddof=1)/np.sqrt(R):.6f} (exact {E_cv_skl - E_truth:+.6f})")

# B2: grid check of the cubic remainder bound
pg = np.linspace(0.05, 0.95, 37)
viol, maxratio = 0, 0
for pp in pg:
    dmax = min(pp, 1 - pp)
    d = np.linspace(-0.999 * dmax, 0.999 * dmax, 2001)
    q = pp + d
    err = np.abs(KL_direct(pp, q) - d ** 2 / (2 * pp * (1 - pp)))
    bound = np.abs(d) ** 3 * (1 / (pp * (pp - np.abs(d))) + 1 / ((1 - pp) * (1 - pp - np.abs(d))))
    viol += (err > bound + 1e-15).sum()
    nz = bound > 1e-14
    maxratio = max(maxratio, (err[nz] / bound[nz]).max())
print(f"\nB2: cubic remainder bound on a 37 x 2001 grid of (p, q): violations = {viol}; max err/bound = {maxratio:.3f}")
# B3: truncated forward KL vs bounds
print("B3: E[KL(p||phat_m); |phat_m - p| <= eps] (exact) vs proved bounds, m = 160, p = 0.3")
for eps in [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]:
    C = 1 / (p * (p - eps)) + 1 / ((1 - p) * (1 - p - eps))
    sel = np.abs(k_m / m - p) <= eps
    val = (w_m * np.where(sel, KL_direct(p, np.clip(k_m / m, 1e-300, 1)), 0)).sum()
    T = (w_m * np.where(~sel, (k_m / m - p) ** 2, 0)).sum()
    up = 1 / (2 * m) * (1 + 2 * p * (1 - p) * eps * C)
    lo = (1 / (2 * p * (1 - p)) - eps * C) * (p * (1 - p) / m - T)
    print(f"   eps={eps:.2f}  lower={lo:.6f}  exact={val:.6f}  upper={up:.6f}  ok={lo <= val <= up}  tail T={T:.2e}")
