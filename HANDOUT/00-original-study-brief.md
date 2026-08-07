# Study & Plan — Differentiable B-mode Purification (Session 1: apodization · decoupling · purification · Cℓ estimation)

> **Status:** Study / "plan to make a plan." No code is written this session. This document is the
> brief for the *next* Claude session, which will implement `jax_healpy.pseudo_cl`.
>
> **Out of scope for now:** component separation. We only build the masked-sky power-spectrum
> estimator (apodization → purification → MCM decoupling → Cℓ), plus the *optimizable apodization*
> research layer. Component separation chains behind this later.

---

## Context — why this exists

The paper goal is a **differentiable B-mode purification pipeline**: from multi-frequency data →
cleaned CMB → purified B-modes → `Cℓᴮᴮ`, all in one differentiable JAX pass, so gradients can drive
both component separation *and* the apodization from the science metric (recovered `Cℓᴮᴮ`). Today
this is two disjoint, separately-tuned steps (FGBuster, then NaMaster/Xpure); the apodization is
fixed in advance and never informed by the residuals it is meant to suppress.

**This session designs the power-spectrum half**, and specifically the novelty: an **optimizable
apodization** — porting Xpure's *variance-optimal* window, but as a **differentiable** optimization
(autodiff + gradient descent) instead of Xpure's hand-rolled PCG solve.

**The strategic shortcut:** a fully validated MCM + Smith-2006 purification already exists in the
author's other package, `jax-fli`, written in pure `jax_healpy` and tested against NaMaster. Session 1
is therefore: **(a)** port that into a new `jax_healpy.pseudo_cl` subpackage (low-risk, fast), then
**(b)** build the differentiable optimizable-apodization layer on top (the research).

### Decisions locked with the user
| Decision | Choice |
|---|---|
| Code location | New **`jax_healpy.pseudo_cl`** subpackage (mirrors the existing `clustering/` subpackage) |
| Apodization research target | **Free per-pixel, Xpure-equivalent window** (variance-optimal), reached via a staged ladder |
| Optimization objective | **Analytic BB-bandpower variance** (Knox / mask-moment form — Xpure's target) |
| Session-1 scope | spin-2 `(Q,U) → EE/BB`; **float64 mandatory**; `nside=64`, `lmax≈191`; inject a known `r`; noise/beam-free first, then white-noise + Gaussian beam |
| NaMaster | **test-only oracle**, hard-imported (see Risks §7) — never a runtime dependency |

---

## 1. Physics & math the next session must hold

### 1.1 Pseudo-Cℓ / MASTER decoupling (Hivon 2002)
A mask `W` couples multipoles: the measured ("pseudo") spectrum is `C̃ℓ = Σℓ' Mℓℓ' Cℓ'`. The
**mode-coupling matrix (MCM)** `Mℓℓ'` depends only on the mask power spectrum `Wℓ = anafast(W)`.
Recovering an unbiased spectrum = **decouple** = invert the (binned) MCM:
`Ĉ_b = (B·M·S)⁻¹ · (B·C̃)`, with `B`/`S` the bandpower binning/spreading operators.

For **spin-2**, the mask also mixes E and B. The coupling is a 2×2 block:
```
[C̃ᴱᴱ]   [ M₊  M₋ ] [Cᴱᴱ]            M₊ = (EE←EE) coupling
[C̃ᴮᴮ] = [ M₋  M₊ ] [Cᴮᴮ]            M₋ = (EE←BB) leakage block
```
Decoupling inverts this stacked block. (EB / TB and TT/TE are deferred — §7.)

**How the MCM is computed (the differentiable trick).** Instead of explicit Wigner-3j sums, the
jax-fli port evaluates the MASTER coupling integral in *position space* on Gauss–Legendre nodes:
`G(x) = Σℓ (2ℓ+1) Wℓ Pℓ(x)`, then `Mℓ₁ℓ₂ ∝ (2ℓ₂+1) · Σ_q wq Pℓ₁(xq) G(xq) Pℓ₂(xq)` — pure matmuls
and recursions, **fully differentiable** and matching NaMaster to `<1e-8` (spin-0). Spin-2 uses the
`d²₂,±₂` Wigner-d kernels: `EEEE=(M₊₊+M₋₋)/2`, `EEBB=(M₊₊−M₋₋)/2`.

### 1.2 E→B leakage & purification (Smith 2006)
On a cut sky, E and B are not cleanly separable ("ambiguous modes"); the bright E (here `Cℓᴱᴱ` ≫
`Cℓᴮᴮ`) leaks into the faint B and swamps it. The **pure-B estimator** removes the leakage by adding
correction terms built from the **first (spin-1) and second (spin-0) covariant derivatives of the
window** applied to the field. The jax-fli `_purify_eb` implements the standard 3-term construction:
the naive spin-2 transform of `(W·Q, W·U)`, plus a spin-1 correction with window derivative
`−√(ℓ(ℓ+1))·alm_mask` scaled by `2/√((ℓ+2)(ℓ−1))`, plus a spin-0 correction scaled by
`1/√((ℓ+2)(ℓ+1)ℓ(ℓ−1))`.

> **THE BINDING CONSTRAINT (organizing principle of the whole design):** purification *differentiates
> the window*, so the window must reach zero **smoothly** — value **and** first derivative (C1) and
> second derivative (C2) vanishing at the footprint boundary. A window that is merely continuous but
> kinked at the edge silently breaks purification. Every apodization parametrization below is ranked
> by how it honours this.

### 1.3 Apodization
A hard 0/1 mask rings in harmonic space (Gibbs), which is what drives both the multipole coupling and
the E→B leakage. Apodization tapers the mask to 0 over an edge band so the ringing decays. NaMaster's
**C2** window is `f=(1−cos(πx))/2`, `x=√((1−cos d)/(1−cos θ*))`, with `d` = great-circle distance to
the nearest masked pixel, capped at the apodization scale `θ*`.

### 1.4 Xpure's *optimal* apodization — and our differentiable reframing
Xpure does not use a fixed taper; it solves for the **variance-minimizing window**, per multipole
bin, separately for the **scalar / vector / tensor** (spin-0/1/2) components — and those three
windows are exactly the derivative windows purification needs (§1.2). Reference file the user
supplied: **`Xpure_fork/src/apodizations/optimalmasks_PCG.c`**.

What that file does (cite these symbols in code comments when we port the *objective*):
- **`spin_PCG()`** — the preconditioned conjugate-gradient solver. Solves `A[W] = B` for the window,
  where the operator is `A[W] = M·T[W] + N[W]` (`compute_spin_window` applies the mask `M`;
  `spin_wlmXsignal` applies the signal coupling `T`; `spin_windowXnoise` adds noise weighting `N`),
  RHS `B = local_mask` (`local_B_scal`).
- **`spin_precompute_matrix()`** — builds the signal coupling matrices `S00,S11,S22,S01,S02,S12` from
  **Wigner-3j** symbols (`wig3j()`) × signal `local_Cl` × beam, summed over the bin `[llow,lhigh]`.
- **`compute_spin_weight()` / `spin_windowXinverse()`** — the **preconditioner**: divide by
  `weightᵢ + factorᵢ·local_noise[pix]`, where `weight_scal/vect/tens` are the bin-integrated
  `(2ℓ+1)·wignerᵢ²·Cℓ·beam /(16π·4π)` and `factor₀ = Σℓ ℓ(ℓ+1)/(8π²)`.
- Final normalization: each window `×= √(f_sky · N_pix / ⟨W_scal|W_scal⟩)` (`scalar_product`).
- Outputs: real-space `local_W_scal`, `local_W_vect`, `local_W_tens`.

**Our reframing (the paper's contribution).** We do **not** re-implement the PCG. We make the window a
differentiable function of parameters and minimize the **same kind of variance objective** with an
autodiff optimizer. Correspondence to cite:

| Xpure (`optimalmasks_PCG.c`) | This work (`jax_healpy.pseudo_cl`) |
|---|---|
| `spin_PCG()` linear solve `A[W]=M` | `jax.grad` of the variance loss + Optax step |
| Preconditioner `1/(weight+factor·noise)` (`spin_windowXinverse`) | Smoothness + boundary **regularizer** on `W` (well-posedness) |
| Signal coupling `spin_precompute_matrix` (Wigner-3j) | Differentiable MCM via Gauss–Legendre quadrature (§1.1) |
| `weight_scal/vect/tens` bin integrals | Analytic Knox / mask-moment BB variance (§4) |
| separate scalar/vector/tensor windows | one window `W`; its derivatives are taken inside `_purify_eb` |
| `√(fsky·Npix/⟨W|W⟩)` normalization | optional output normalization (does not affect decoupled Cℓ) |

---

## 2. Existing assets to reuse — do **not** reinvent

### 2.1 Port target: `jax-fli/src/jax_fli/power/decouple.py` (≈276 lines, validated)
| Symbol | Role | Action |
|---|---|---|
| `_legendre_all`, `_wigner_d2_all` | GL-node Legendre / Wigner-d recursions | copy verbatim |
| `_mcm_spin0`, `_mcm_spin2` | differentiable MCM blocks | copy verbatim |
| `_linear_bins` | binning `B`,`S`,`ell_eff` (matches `NmtBin.from_nside_linear`) | copy; also expose bandpower windows (§3) |
| `_decouple_spin0`, `_decouple_spin2` | `jnp.linalg.solve` decoupling | copy verbatim |
| `_purify_eb` | Smith-2006 E/B purification | copy verbatim |
| `MCM` (eqx.Module), `compute_mcm`, `anafast_masked` | container + public API | port; CMB-rename (§3) |

### 2.2 Port target: `jax-fli/src/jax_fli/data/apodize.py` (C2, differentiable)
Grassfire distance transform (`lax.scan` of `min` over true-neighbour separations) + cosine taper.
Already differentiable **w.r.t. mask values**. **Refactor needed** to make the *scale* `θ*`
differentiable: split the (static) grassfire `niter` from `θ*` — see §4 / §7.

### 2.3 Validation oracle: `jax-fli/tests/power/test_decouple.py`
Reuse the test pattern: scalar decoupling vs `compute_full_master` (`rel<1e-8`), spin-2 EE vs NaMaster
(`rel<5e-2`, the iter=0 floor), `purify_b` reduces leakage, differentiability through the map.
**Correction:** it uses `pytest.importorskip("pymaster")`; per the user's global CLAUDE.md this is a
**required** validation dep → hard-import at module top, let CI install pymaster (§7).

### 2.4 `jax_healpy` primitives — all present & differentiable (confirmed)
`map2alm`, `map2alm_spin(maps, spin, lmax, iter, method, healpy_ordering)`,
`alm2map_spin(alms, nside, spin, …)`, `almxfl`, `alm2cl`, `anafast`, `synfast`/`synalm`,
`gauss_beam`, and the pixel tools the grassfire needs (`get_all_neighbours`, `pix2vec`,
`npix2nside`). The spin-transform signatures are **drop-in** for the jax-fli code.

### 2.5 NaMaster 2.6 oracle API (installed at `…/ffi12/.../pymaster`)
`NmtField(mask, [maps], spin=, purify_e=, purify_b=, beam=)`,
`NmtBin.from_nside_linear(nside, nlb)`, `NmtWorkspace.compute_coupling_matrix / decouple_cell /
get_bandpower_windows`, `compute_full_master(f_a, f_b, bins)`, `mask_apodization(mask, aposize,
apotype='C1'|'C2'|'Smooth')`.

---

## 3. Proposed API — `jax_healpy.pseudo_cl`

Subpackage layout (mirror `jax_healpy/clustering/`):
```
jax_healpy/pseudo_cl/
  __init__.py        # exports below
  _apodize.py        # apodize, grassfire_distance, apodize_profile
  _mcm.py            # MCM, compute_mcm, _mcm_spin0/_mcm_spin2, _linear_bins, bandpower_windows
  _estimate.py       # anafast_masked, _decouple_*, _purify_eb
  _optimize.py       # bandpower_variance (Knox), optimize_apodization, window parametrizations
```

### Ported (stable) surface
```python
apodize(binary_mask, aposize_deg=1.0, *, apotype="C2") -> Array
    # differentiable C2 apodized mask (RING). Exactly 0 outside the footprint.

compute_mcm(mask, *, lmax=None, nlb=16, pol=False, method="jax") -> MCM
    # lmax default 3*nside-1; nlb=16 matches NmtBin.from_nside_linear; pol builds spin-2 blocks.

anafast_masked(map1, map2=None, *, mask=None, lmax=None, pol=False,
               purify_e=False, purify_b=False, mcm=None, nlb=16, method="jax") -> (ell, cl)
    # mask=None -> plain anafast / coupled pseudo (premasked). mask given -> decoupled bandpowers.
    # pol=True takes map1=(2,npix)=(Q,U) -> (EE,EB,BB). purify_* require a mask.
```
`MCM` stays an `eqx.Module` (precompute once, reuse / freeze; differentiable leaves). Defaults chosen
to match NaMaster so the oracle comparison is apples-to-apples.

### New (research) surface — optimizable apodization
```python
grassfire_distance(binary_mask, *, max_aposize_deg) -> Array     # geometry, computed ONCE (static niter)
apodize_profile(dist, theta_star_deg, *, apotype="C2") -> Array  # DIFFERENTIABLE in theta_star_deg

bandpower_variance(cl_bb_total, mcm, mask, *, fsky=None) -> Array # Knox/mask-moment BB variance (§4)

optimize_apodization(binary_mask, q_maps, *, window="scalar"|"profile"|"perpixel",
                     init, n_steps, lr, lmax=None, nlb=16,
                     cl_bb_inject, cl_ee, noise=None, beam=None,
                     reg_smooth=0.0, reg_boundary=0.0, method="jax") -> (window, history)
    # minimizes bandpower_variance via autodiff + Optax. window= selects the §4 ladder rung.
```

**Differentiability contract** (state in the module docstring):
- `loss = Σ_b Var(Ĉ_b^BB)` ← `Ĉ^BB` ← `decouple(MCM(W), pseudo_cl(purify(W, Q, U)))` ← `W = apodize(params)`.
  Everything is differentiable in `params`. **float64 mandatory.**
- Use `iter=0` spin transforms by default (no fixed-point). If accuracy demands `iter>0` (§7), use a
  **fixed, unrolled** count so the graph stays differentiable.
- Scalar/profile path: freeze the grassfire distance field, differentiate only the taper profile +
  MCM + Cℓ. Per-pixel path: `params` **is** the window (no grassfire); needs the regularizers.

---

## 4. The optimizable apodization — staged ladder (research core)

Three rungs, increasing power and risk. **Session 1 targets rungs 0–2 solidly and *prototypes* rung 3.**

| Rung | Parametrization | DoF | Boundary safety | Notes |
|---|---|---|---|---|
| 0 | **Fixed** C2 (`apodize`) | 0 | safe by construction | reproduces NaMaster; baseline |
| 1 | **Scalar `θ*`** (`apodize_profile`) | 1 | safe | freeze distance field; differentiate profile → MCM → Cℓ |
| 2 | **Parametric monotone profile** | ~3–5 | safe by construction | scale + shape of the edge ramp |
| 3 | **Free per-pixel window** (Xpure-equiv.) | `npix` | **must regularize** | smoothness + boundary penalty replace Xpure's preconditioner |

### The objective (analytic BB variance — Xpure's target)
Knox / "mask-moment" approximation of the bandpower error, fully differentiable in `W`:
```
Var(Ĉ_b^BB) ≈ [ ⟨W⁴⟩ / ⟨W²⟩² ] · 2 / [ (2ℓ_b+1) Δℓ ] · ( C_ℓ^{BB,total} )²
   C_ℓ^{BB,total} = C_ℓ^{BB,signal} + N_ℓ/⟨W²⟩ + L_ℓ      (L = residual E→B leakage AFTER purify)
   ⟨Wⁿ⟩ = mean(W**n)   (mask moments; ⟨W⁴⟩/⟨W²⟩² ≥ 1, = 1 only for a binary mask)
```
**The leakage term `L_ℓ` is load-bearing — do not leave it schematic.** The two mask-moment factors
are *each* minimized by **less** apodization: `⟨W⁴⟩/⟨W²⟩²` grows as the window concentrates (a binary
mask gives the smallest ratio for a fixed footprint) and `N_ℓ/⟨W²⟩` grows too. So with `L_ℓ` dropped,
the optimizer drives `W → binary` — straight to the one geometry where purification breaks. The
interior optimum lives **entirely** in `L_ℓ` (more apodization suppresses leakage, less inflates it),
and `L_ℓ` is **not** a mask moment. Compute it differentiably one of two ways: **(i)** contract the
spin-2 leakage block `mcm.eebb` with the fiducial `C_ℓ^EE`; or **(ii)** run the purified estimator
forward on a noiseless **pure-E** sky and read off the residual BB. Feed the result into `C_ℓ^{BB,total}`.

> **Sanity gate before any gradient work (§6 Step 4):** sweep `Var(Ĉ_b^BB)` vs `θ*` **with
> `purify_b=True`** and inspect the shape. Two *legitimate* outcomes:
> - **Interior U-shape** → leakage-vs-mode-loss trade-off is active; gradient optimization has a real
>   target. Proceed.
> - **Monotone toward small `θ*`** → purification has already removed most leakage, so the optimum sits
>   at *the least apodization purification tolerates*. This is **itself a finding** ("purification
>   dominates; optimized apodization buys only the variance tail / matters more at higher noise") —
>   **not** a mis-specified loss. Then: add noise (raises `N_ℓ/⟨W²⟩`), report the variance-tail gain,
>   and lean the paper on the per-pixel rung, where the exact signal+noise covariance (not the moment
>   proxy) can still find structure the scalar `θ*` cannot.

The per-pixel rung (3) replaces this analytic proxy's `⟨W⁴⟩/⟨W²⟩²` mode-count penalty with the
exact signal+noise covariance contraction (closer to Xpure's `spin_precompute_matrix` weights), and
adds `reg_smooth·‖∇W‖² + reg_boundary·(boundary penalty)` — the differentiable stand-in for Xpure's
preconditioner — to keep the solution C1/C2 and well-posed.

---

## 5. First case study — CMB B-mode recovery

**Setup:** `nside=64`, `lmax=191`, **float64**. Mask: use the jax-fli `_quadrant` mask **only** for the
exact-decoupling oracle test (a) — its 90° corners are *pessimal* for purification (C1/C2 hardest at
corners) and would inflate leakage in a way that is about the mask, not the method. Make a **smooth
SO/SAT-like cap the primary mask** for the purification + recovery story (b). Signal: a known fiducial — lensing-like
`C_ℓ^EE` plus `C_ℓ^BB(r)` for an injected `r` (e.g. `r=0.01`), drawn with `synfast`/`synalm` into
`(Q,U)`. Noise/beam: off first (exact NaMaster comparison), then white noise + `gauss_beam`.

**(a) Validation / null + oracle.** Pure-E sky (`r=0` → true BB≈0): show `purify_b` collapses the
spurious BB, and that decoupled EE (and purified BB) match NaMaster (`compute_full_master` with
`purify_b=True`). State tolerances explicitly: EE to the spin-2 `~1%` iter=0 floor (§7); for **BB**,
because it is a small residual, express the bound as an **absolute leakage level (fraction of
`C_ℓ^EE`)**, not a percentage of BB — and tie it to that same iter=0 floor, so the next session can
tell a real regression from the known transform floor.

**(b) Headline recovery.** Inject known `r`; for each ladder rung optimize the apodization to minimize
`Σ_b Var(Ĉ_b^BB)`; show the recovered BB bandpowers track the injected truth with **error bars that
shrink** vs the fixed-C2 baseline and vs the non-purified estimator. Headline figure: recovered
`C_ℓ^BB ± σ` over bandpowers for {no-purify, fixed-apo+purify, optimized-apo+purify} against the
injected theory line.

**Metrics:** decoupled-BB / truth ratio over the reliable band; `Σ_b σ_b` (the optimized objective);
residual E→B leakage on the pure-E null.

---

## 6. Staged implementation plan (session-2 work items)

0. **Scaffold** `jax_healpy/pseudo_cl/` + exports; add the float64 guardrail/warning. Set up the test
   module with a **hard** `import pymaster` and pytest **fixtures** (per CLAUDE.md; ref
   `tests/sphtfunc/test_map_alm.py`).
1. **Port decoupling** (`_mcm.py`, `_estimate.py` scalar+spin-2) → validate vs NaMaster: scalar `<1e-8`,
   spin-2 EE `~1%`.
2. **Port apodization** (`_apodize.py`); refactor grassfire/profile split → validate `Wℓ` vs NaMaster
   C2 (`<2e-3`); confirm `apodize_profile` is grad-stable in `θ*`.
3. **Port purification** (`_purify_eb`) → validate `purify_b` reduces leakage and matches NaMaster
   purified BB.
4. **Objective** (`bandpower_variance`) → run the 1-D `θ*` sweep sanity gate (§4).
5. **Rung 1 (scalar `θ*`)** gradient optimization (Optax) → show it recovers the sweep optimum;
   produce the headline figure (§5b).
6. **Rung 3 prototype (per-pixel + regularizers)** → may slip to a later session; rung 1–2 is the
   committed session-1 deliverable.

---

## 7. Risks & gotchas (load-bearing)

- **EE ≫ BB dynamic range (the CMB-specific risk the shear port never faced).** At small `r`, BB sits
  10²–10³× below EE, so the `~1%` spin-2 transform floor (iter=0) and any `~1%` residual leakage can
  swamp the signal. **Decide the target `r`/dynamic range up front;** if `iter=0` is insufficient, use
  a **fixed unrolled `iter>0`** (stays differentiable). Purification accuracy is load-bearing here in a
  way it was not for weak-lensing shear.
- **float64 mandatory.** The spin-2 decoupling solve is ill-conditioned; float32 silently returns
  **all-NaN**. Enable `jax_enable_x64` before importing. (Already in jax-fli's memory + docs.)
- **Grassfire `niter` must be decoupled from `θ*`.** In `apodize.py`, `niter = ceil(2.5·θ*/resol)+6`
  ties the (static) sweep count to `θ*`; if `θ*` is traced this re-traces and makes the loss landscape
  jump. Fix: set `niter` once from a `max_aposize_deg` bound; let `θ*` enter only via the differentiable
  `clip` + cosine normalization.
- **Boundary smoothness is the binding constraint** for the per-pixel rung — without
  smoothness/boundary regularization, purification breaks (§1.2). This *is* Xpure's preconditioner,
  re-expressed.
- **MCM recompute is the per-step cost bottleneck.** Because `W` (hence `Wℓ`, hence the MCM) changes
  every optimization step, the MCM is rebuilt each step: `O(lmax²·n_quad)`. Fine at `nside=64`; plan a
  cheaper path (cache, Toeplitz approx à la NaMaster's `l_toeplitz`) before scaling to SO/S4/LiteBIRD.
- **Test deps fail loudly.** NaMaster is a required *validation* dep → hard `import pymaster`, install
  in CI; **no `importorskip`** for it (overrides the jax-fli test's current pattern).
- **Deferred:** EB/TB decoupling block, temperature TT and TE cross-spectra, spin-0×spin-2 MCM. The EB
  row stays the binned coupled pseudo for now.

---

## 8. References
- **Hivon et al. 2002** — MASTER pseudo-Cℓ. **Smith 2006 (astro-ph/0511629)** — E/B purification.
  **Grain, Tristram, Stompor 2009 (arXiv:0903.2350)** — pure pseudo-cross-spectrum (Xpure method).
- **Xpure source (user-provided):** `https://github.com/Magwos/Xpure_fork/blob/main/src/apodizations/optimalmasks_PCG.c`
  — `spin_PCG`, `spin_precompute_matrix`, `compute_spin_weight`, `spin_windowXinverse`,
  `compute_spin_window`, `spin_wlmXsignal`, `spin_windowXnoise`; windows `local_W_scal/vect/tens`.
- **NaMaster** (oracle): `NmtField`/`NmtBin`/`NmtWorkspace`, `mask_apodization`, `compute_full_master`.
- **Internal:** `jax-fli/src/jax_fli/power/decouple.py`, `…/data/apodize.py`,
  `jax-fli/tests/power/test_decouple.py`, `jax-fli/docs/5-experiments/08-masked-shear/README.md`.
