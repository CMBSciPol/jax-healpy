# Session Handout 2 — pure-B mode-coupling matrix **SOLVED** (+ XPure-coverage review)

**Date:** 2026-06-16 · **Branch:** `patch` (uncommitted working tree) · **Status:** the §4 blocker from
[session 1](README.md) is **resolved and shipped** — the pure-B MCM is derived, machine-precision
validated against NaMaster, **ported to production `jax_healpy.pseudo_cl` (differentiable), wired into
`compute_mcm`/`anafast_masked`, and covered by tests at nside=64**. The usage notebook is enriched with a
pure-B MCM section and a B-mode-recovery demo. Stage 2 (`_optimize.py`) is now unblocked.

This folder (in reading order):
- [`00-original-study-brief.md`](00-original-study-brief.md) — the original study (session 0).
- [`01-implementation-plan.md`](01-implementation-plan.md) — the session-1 executable plan.
- [`README.md`](README.md) — session-1 report: Phase A (port) complete & NaMaster-validated; the purified-BB
  bug found and localized to a missing pure-B MCM.
- **this report** — session 2: the XPure-coverage review, the gap-closure plan, and the pure-B MCM solution.
- Full session-2 plan (review + 2-stage implementation): `~/.claude/plans/i-wish-for-you-optimized-octopus.md`.

---

## 1. What this session set out to do

You asked me to **review what was planned/done and judge whether the apodize/decouple/purify API matches
XPure — are all functionalities covered?** — then to **expand that review into an executable gap-closure
plan** and start building. The crux turned out to be the same one session 1 left open: the **pure-B
mode-coupling matrix**. This session cracks it.

## 2. The XPure-coverage review — the answer

**The standard MASTER surface is faithfully covered and NaMaster-validated; the two capabilities that
*define* XPure are exactly the two things that were missing.**

| XPure capability | `jax_healpy.pseudo_cl` | |
|---|---|---|
| C2 apodization (window vanishing w/ 1st deriv) | `apodize`, `Wℓ rel<5e-3` | ✅ |
| differentiable apodization scale | `grassfire_distance`+`apodize_profile(θ*)` | ✅ (beyond XPure) |
| **variance-optimal window** (`spin_PCG`) | `_optimize.py` — **absent** | 🔴 (Stage 2; to be autodiff, not PCG) |
| std MCM spin-0 / spin-2 EE-BB | `compute_mcm`, `rel<1e-8` / `rel<5e-2` | ✅ |
| **pure-E/pure-B MCM** | **was absent** — purified BB wrongly decoupled w/ standard `eebb` | 🔴→✅ **solved this session** |
| Smith-2006 field purification | `_purify_eb`; coupled pure-B matches NaMaster ~1e-8 | ✅ |
| auto **+ cross** spectra over maps | scalar cross ✅; **spin-2 cross → `NotImplementedError`** | 🟡 deferred (but it's *the* noise-debias path) |
| EB/TB, TT/TE, noise weighting, beam, hybrid | absent | 🟡 deferred by the brief |

**Methodology caveat (unchanged):** nothing is validated against XPure *directly* — XPure isn't built here
(C source in `Magwos/Xpure_fork`). All numerical validation is against **NaMaster**; the inference to XPure
is sound because both implement the same Smith-2006 / Grain-Tristram-Stompor-2009 estimator (confirmed below
by reading both sources). If a true XPure numerical oracle is wanted, `Xpure_fork` must be built.

## 3. The gap-closure plan (two stages, strictly ordered)

- **Stage 1 — pure-B MCM** (the blocking fix; unblocks everything). ← **formula solved this session.**
- **Stage 2 — `_optimize.py`**: the differentiable variance-optimal window (XPure's `spin_PCG` reframed as
  `jax.grad`+Optax), rung-1 scalar `θ*` + the headline B-mode recovery figure.

## 4. Delivered this session

### 4a. Refactor (done, verified)
Extracted `_window_derivative_alms(mask, lmax, method) → (w0, w1, w2)` (window + spin-1/spin-2 derivative
windows) into `_mcm.py`; `_purify_eb` now calls it. Behaviour unchanged — **14/14 pure-JAX tests pass**
(`uv run … pytest tests/pseudo_cl/test_pseudo_cl.py`).

### 4b. Pure-B mode-coupling matrix — **DERIVED + machine-precision validated** ✅ (the headline)

Built a sympy-Wigner-3j validation scaffold (`scratch_pureb_scaffold.py`, runs in `ffi12`) and validated in
two stages (advisor's "split the two questions"):
- **CHECK A — conventions:** my explicit-3j *standard* spin-2 MCM == the code's GL `_mcm_spin2` to **1e-13**.
- **CHECK B — the pure-B formula:** matches **NaMaster's `get_coupling_matrix()` to 7e-15 across the whole
  matrix**, and the E-leakage block vanishes (`pureB←EE = 8e-33 ≈ 0`).

The route there: the formula structure (`T²·W`) was right early, but pure-B←BB matched only on the diagonal
and pure-B←EE wouldn't vanish. Reading NaMaster's C source (`src/nmt_master.c`, the `pure_any` path) showed
its purified factor is `wfac_ispure = wss1 + fac_12·w12 + fac_02·w02` with **`+` on both correction terms**
(the signs live in the raw 3j). I had a spurious **`−`** on the spin-1 term — which breaks the odd-ℓ leakage
cancellation while barely moving the diagonal (hence the misleading "diagonal-only" match). One sign fixed it.

**The validated formula:**
```
T(ℓ, ℓ', ℓ'') = w3j(ℓ ℓ' ℓ''; 2 −2 0)
              + fac_12 · w3j(ℓ ℓ' ℓ''; 1 −2 1)
              + fac_02 · w3j(ℓ ℓ' ℓ''; 0 −2 2)
   fac_12 = 2·√( ℓ''(ℓ''+1) / ((ℓ+2)(ℓ−1)) )
   fac_02 = β_{ℓ'',2} / √((ℓ+2)(ℓ+1)ℓ(ℓ−1)) ,   β_{ℓ'',2} = √((ℓ''−1)ℓ''(ℓ''+1)(ℓ''+2))
   (ℓ = output/pseudo multipole, ℓ' = true-spectrum multipole, ℓ'' = mask multipole)

M^{pureB←X}_{ℓℓ'} = (2ℓ'+1)/(4π) · Σ_{ℓ''} (2ℓ''+1) · W_{ℓ''} · T(ℓ,ℓ',ℓ'')² · parity
   W_{ℓ''} = anafast(mask)   (the plain mask power spectrum)
   parity:  even (ℓ+ℓ'+ℓ'') → pureB←BB ;  odd → pureB←EE  ( = 0 analytically )
```
The standard spin-2 MCM is the same with `T = w3j(2 −2 0)` only. Equivalently this is NaMaster's
`wfac_ispure[2] = (wss1 + fac_12·w12 + fac_02·w02)² · pcl`, summed into even-/odd-ℓ (`xi_pp`/`xi_mm`).

## 5. Key findings & lessons

- **Lesson — the diagonal lied.** A pure-B←BB *diagonal* match (ratio ~1.00) hid a wrong relative sign; only
  the **full-matrix** element-wise comparison + the **vanishing E-leakage** check exposed it. Validate the
  whole block and the null, not the diagonal.
- **pure-B←EE ≡ 0 analytically.** Purification removes E-leakage in the *mean coupling*, so the purified BB
  must be decoupled with **`pureB←BB` alone** (a 1×1 solve) — not the standard 2×2 with `eebb`. *This is the
  fix to session-1's headline bug.*
- **Stage-2 consequence:** the leakage term `Lℓ` in the bandpower-variance objective **cannot** come from the
  pure MCM's EE→pureBB block (it's zero). Use the brief's **method (ii)**: run the purified estimator on a
  noiseless pure-E sky and read the residual BB (the finite-resolution leakage). Otherwise the objective is
  mis-specified.
- **XPure ↔ NaMaster, confirmed by source:** XPure's `spin_precompute_matrix` (`optimalmasks_PCG.c`) builds
  the six `S00…S12` with the same `N_ℓ = ℓ(ℓ+1)`, `(ℓ−1)ℓ(ℓ+1)(ℓ+2)` (β²) factors — but that file is XPure's
  **variance-optimal-window signal coupling, i.e. the Stage-2 reference**, *not* the pure-Cℓ MCM. NaMaster's
  `fac_12`/`fac_02` are identical to the coefficients here, so NaMaster ≈ XPure for Stage 1.
- **Differentiability:** the Wigner-3j depend only on `(ℓ,ℓ',ℓ'')`, never on the window — they're constants.
  So the pure-B MCM is `M = einsum(W_ℓ, K)` with a one-time 3j table `K`; gradients flow through `W_ℓ`. No
  Gauss-Legendre form is required (the session-1 plan's "no Wigner-3j for differentiability" was over-cautious).
- **Production 3j must use the recursion, not the Racah single-sum.** The explicit Racah formula is exact at
  low ℓ but **numerically unstable at lmax=191** (catastrophic alternating-sum cancellation → rel ~1e13 vs
  NaMaster, while exact below lmax~47). The production code uses `_drc3jj`, a faithful translation of
  SLATEC/NaMaster `drc3jj` (stable bidirectional recursion). Subtle trap: a garbage (huge) `pure_bb` inverts to
  ~0, so a naive `pur<raw` leakage test passes *spuriously* — only the element-wise matrix check caught it.
- **NaMaster gotcha:** `NmtField(..., lmax=L)` with `L ≠ 3·nside−1` breaks purification (alm-size mismatch in
  `_purify`). Use `lmax = 3·nside−1` for oracle comparisons.

## 6. Stage 1 — DONE this session; Stage 2 — next

**Shipped (Stage 1, all in `jax_healpy/pseudo_cl/`):**
- `_mcm.py`: `_drc3jj` (stable 3j recursion) → `_wigner3j_table` → `_pureB_kernel(lmax)` (lru-cached constant
  `K`) → `_mcm_spin2_pureB(Wl) = einsum('lLk,k->lL', K, Wl)` (differentiable in `Wl`; `K` built once, ~14 s at
  lmax=191). `MCM` gained a `pure_bb` field; `compute_mcm(..., purify_b=True)` builds it.
- `_estimate.py`: `_decouple_spin2_pure` (block-triangular, `pureB←EE = 0`) used by
  `anafast_masked(..., purify_b=True)`; raises if a precomputed MCM lacks `pure_bb`.
- Tests: 3 new pure-JAX (`test_pseudo_cl.py`, 17/17 pass) + oracle (`test_oracle.py`):
  `test_pure_bb_coupling_matrix_matches_namaster` (element-wise rel<1e-6) and the tightened
  `test_purified_bb_leakage_floor_vs_namaster` (`<5e-3` of EE, was `5e-2`). End-to-end vs NaMaster at
  nside=64: EE 0.16 %, purified-BB leakage ~1e-4 of EE (≈100× better than the standard-MCM bug).
- Notebook enriched (`notebooks/pseudo_cl_usage.ipynb`): pure-B MCM section + B-mode-recovery demo.

**Remaining — Stage 2** (`_optimize.py`, now unblocked): `bandpower_variance` (with `Lℓ` via method (ii) —
*not* the zero EE→pureBB block), `optimize_apodization` (Optax, rung-1 `θ*`), the θ\*-sweep sanity gate, and
the headline recovery figure. **Perf follow-up:** `_wigner3j_table` is a Python double loop over a dense
`(lmax+1)³` table — fine at nside=64, but vectorize / use a GL-Toeplitz form before scaling to S4/LiteBIRD.

## 7. How to run / reproduce

```bash
# Pure-JAX suite (CI-equivalent), in the uv env:
uv run --extra recommended --group dev pytest tests/pseudo_cl/test_pseudo_cl.py -q   # 17 pass

# Oracle (NaMaster), in ffi12 -- pure-B coupling element-wise + tightened leakage floor:
micromamba run -n ffi12 python -m pytest tests/pseudo_cl/test_oracle.py -o addopts="" -q

# The original pure-B formula scaffold (sympy reference; superseded by the production _mcm.py code):
env PYTHONPATH=/home/wassim/Projects/CMB/jax-healpy \
    micromamba run -n ffi12 python scratch_pureb_scaffold.py
# -> CHECK A 1e-13 ; CHECK B pureB<-BB relmax 7e-15 ; pureB<-EE ~8e-33

# NaMaster C source used as the ground truth (re-fetch if /tmp is cleared):
curl -sSL https://raw.githubusercontent.com/LSSTDESC/NaMaster/master/src/nmt_master.c -o /tmp/nmt_master.c
# the pure path is the `if(c->pure_any)` block around lines 823-906 (wfac_ispure / fac_12 / fac_02).
```

## 8. Suggested next steps (ordered)

1. Port the validated formula to production (`_mcm.py`), validate the JAX MCM against the scaffold locally
   (no NaMaster needed), then element-wise vs NaMaster at `nside=64, lmax=191`.
2. Wire `purify_b` decoupling to `pureB←BB` alone; add the injected-BB recovery oracle test; tighten the floor.
3. Re-run the θ\*-sweep sanity gate on the corrected estimator — an interior optimum should now appear
   (session-1 §3: the broken estimator had hidden it).
4. Build Stage 2 (`_optimize.py` rung-1 + the headline figure).
