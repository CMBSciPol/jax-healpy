# Session Handout — `jax_healpy.pseudo_cl` (differentiable masked-sky Cℓ)

**Date:** 2026-06-15/16 · **Branch:** `patch` (uncommitted working tree) · **Status:** Phase A complete & NaMaster-validated; a real purification bug found and localized; fix (pure-B mode-coupling matrix) in progress.

This folder:
- [`00-original-study-brief.md`](00-original-study-brief.md) — the original plan I was asked to implement (the "plan to make a plan").
- [`01-implementation-plan.md`](01-implementation-plan.md) — the executable implementation plan derived from it (approved this session).
- this report — what actually happened (session 1).
- [`02-session-2-pureb-mcm.md`](02-session-2-pureb-mcm.md) — **session 2** follow-up: XPure-coverage review + the pure-B MCM **solved** (the §4 blocker below, resolved).

---

## 1. Goal

Build `jax_healpy.pseudo_cl`: a differentiable masked-sky power-spectrum estimator (apodize → purify → MCM decouple → Cℓ) by porting the NaMaster-validated MCM + Smith-2006 purification from the author's `jax-fli` package, then add the **optimizable-apodization** research layer (Xpure's variance-optimal window reframed as autodiff + gradient descent). Target case: spin-2 `(Q,U)→EE/BB`, `nside=64`, `lmax=191`, float64, inject a known `r`.

## 2. Delivered — Phase A (port), validated against NaMaster ✅

New subpackage `jax_healpy/pseudo_cl/` (mirrors `clustering/`):

| File | Contents |
|---|---|
| `_apodize.py` | `apodize` (C2, NaMaster-equivalent) + the **differentiability split** `grassfire_distance` (static geometry) / `apodize_profile` (differentiable in `θ*`) |
| `_mcm.py` | GL-quadrature mode-coupling: `_legendre_all`, `_wigner_d2_all`, `_mcm_spin0/2`, `_linear_bins`, `MCM` (eqx.Module), `compute_mcm`, `bandpower_windows` |
| `_estimate.py` | `_decouple_spin0/2`, `_purify_eb` (Smith-2006), `anafast_masked` (public entry), `iter` param threaded through purification |
| `_utils.py` | `require_x64` float64 guardrail (float32 → silent all-NaN) |
| `__init__.py` | public exports |

**Validation (`tests/pseudo_cl/`):**
- `test_pseudo_cl.py` — **14/14 pure-JAX tests pass** (run in CI + locally).
- `test_oracle.py` — **4/4 NaMaster oracle tests pass**: `Wℓ` vs NaMaster C2 (`rel<5e-3`), scalar decouple (`rel<1e-8`, exact), spin-2 EE (`rel<5e-2`, iter=0 floor), purified-BB vs NaMaster.

**Infra:** `equinox`+`optax` added to `[project.dependencies]`, `uv.lock` regenerated; the wheel ships all `pseudo_cl` modules (hatchling auto-discovers — the old `packages=['jax_healpy']` concern is obsolete); `ci.yml` excludes the NaMaster oracle module with a documented `--ignore` (pymaster is conda-only, not pip/uv-installable — run oracle tests in the `ffi12` env).

**Bug caught & fixed during Phase A:** `apodize_profile` produced **NaN gradients** w.r.t. `θ*` — the classic `√0` autodiff trap at masked pixels (`dist=0`). Fixed with a double-`where` safe-sqrt; values unchanged (oracle + equality tests still hold).

## 3. ⚠️ Headline finding — the §7 purification risk is real

The study brief §7 flagged: *"Purification accuracy is load-bearing here in a way it was not for weak-lensing shear."* It is. Validating the **purified BB** against NaMaster in the EE≫BB (CMB) regime exposed a genuine bug the shear port never caught:

**We decouple the *purified* B-pseudo with the *standard* (non-pure) mode-coupling matrix.** `compute_mcm` builds the `eebb` (EE→BB) coupling block from the mask alone; purification removes E-leakage from pure-B, so feeding the purified pseudo through the standard 2×2 block **re-injects the bright E-power into BB**.

Direct comparison on a pure-E sky (smooth 40° cap, `nside=64`):

| θ\* | ours (full `eebb`) | ours (`eebb=0` diag) | NaMaster |
|---|---|---|---|
| 2° | 1.90e-5 (−, ~1% of EE) | 1.77e-7 (107× better) | 4.9e-8 |
| 6° | 2.00e-5 (−) | 5.26e-8 (380× better) | 1.9e-9 |

Diagnostics that pinned it down:
- Residual is **bias-dominated** (bias/fluct ≈ 5–8.5), **sign-definite negative**, **flat in θ\***, and **iter-insensitive** (iter=3 identical to iter=0) → it is *not* a transform-accuracy floor.
- Full-sky BB of the "pure-E" sky is tiny (BB/EE ~1e-4 at iter=0) → the residual is *real mask leakage*, not a contaminated input.
- Zeroing `eebb` collapses it 100–380× **and restores the physical θ\*-decreasing trend** (NaMaster's residual *falls* with apodization — the real leakage-vs-mode-loss trade-off).

This had masked the science: the analytic θ\*-sweep looked monotone ("no interior optimum / purification dominates"), but that was the **artifact of the broken estimator**. The prior oracle test only passed because its tolerance (`5e-2·EE ≈ 5e-5`) was looser than the error (`2e-5`).

> **Lesson recorded:** the same latent bug exists in `jax-fli` (its purification is field-level only, decoupled with the standard MCM; shear EE~BB hid it). Any purified-BB decoupling needs a pure-B MCM.

## 4. Decision & current frontier — pure-B mode-coupling matrix

**You chose:** build the correct **pure-B MCM** (Grain-Tristram-Stompor 2009 / NaMaster Alonso 2019) for NaMaster-level (~1e-8) purified BB, tighten the oracle test to NaMaster's *actual* residual, then resume the research layer (rung-1 optimization + headline figure).

Located the math: Alonso 2019 **eq. 57** (pure-B field = the 3 derivative-window terms our `_purify_eb` already builds) and **eqs 63-66** (the coupling replaces the single Wigner-3j `(ℓ ℓ' ℓ''; 2 -2 0)` with a 3-term spin-mixing sum, equivalently the **six window-derivative cross-spectra** `W^{00,01,02,11,12,22}` — the same `S00,S11,S22,S01,S02,S12` Xpure's `spin_precompute_matrix` uses), with `β_{ℓ,s}=√((ℓ+s)!/(ℓ-s)!)`.

**Blocking sub-question — answered:** is this *one* bug (coupling only) or *two* (coupling **and** the `_purify_eb` field purification)? Compared the **coupled** pseudo (before decoupling): our `alm2cl(Bb)` vs NaMaster `compute_coupled_cell`. Result — **mostly one bug (the coupling).** The coupled pure-BB matches NaMaster in magnitude at θ\*=2° (ours/nmt ≈ 0.95–1.28, absolutes ~1e-8) — so the field purification is essentially correct and the −1%-of-EE decoupled residual is the MCM amplifying tiny coupled leakage. There is a **secondary** θ\*-dependent gap (ours ≈ 3× NaMaster at θ\*=6°, both sub-1e-9, only marginally helped by `iter=3`) → revisit the field-purification accuracy *after* the MCM fix, but the pure-B MCM is correctly the dominant fix.

**Implementation plan for the fix (GL-quadrature, differentiable):**
1. Build the six window cross-spectra as `anafast` of the derivative-window maps `_purify_eb` already synthesizes; **validate each vs healpy `anafast`** before assembling (a wrong derivative-window normalization is the likeliest error).
2. Assemble the pure-B coupling against NaMaster's `NmtWorkspace.get_coupling_matrix()` **element-wise**: EE→pureB block should collapse to ~0 first, then fix the BB→pureB diagonal.
3. **Acceptance test:** mean decoupled BB over ≥4 seeds recovers a *known nonzero injected* BB within the error bar (not just the pure-E null). Then tighten `test_oracle.py` to NaMaster's real residual level.

## 5. Not yet done (behind a correct estimator)

- `_optimize.py`: `bandpower_variance` (Knox + mask-moment + leakage), `optimize_apodization` (optax), window parametrizations. **Deliberately not written** — its objective design depends on the corrected leakage, which depends on the pure-B MCM.
- Rung-1 (scalar `θ*`) gradient optimization + the headline recovery figure (`notebooks/bmode_purification_demo.ipynb`).
- A usage-showcase notebook is included now (`notebooks/pseudo_cl_usage.ipynb`) covering the **validated** Phase-A surface.

## 6. How to run

```bash
# Pure-JAX tests (CI-equivalent; uv env has the editable repo install + s2fft + equinox + optax)
uv run --extra recommended --group dev pytest tests/pseudo_cl/test_pseudo_cl.py -q

# NaMaster oracle tests — only in the conda env that has pymaster (run from repo so pytest picks up live code)
micromamba run -n ffi12 python -m pytest tests/pseudo_cl/test_oracle.py -o addopts="" -q

# Usage notebook
papermill notebooks/pseudo_cl_usage.ipynb notebooks/pseudo_cl_usage.ipynb --log-output
```

**Environment gotcha (important):** `ffi12` has a **stale non-editable `jax_healpy`** in its site-packages. `pytest` (rootdir insertion) and `cd repo && python -c …` pick up the live repo, but a `/tmp/script.py` does **not** — run scripts in `ffi12` with `env PYTHONPATH=/home/wassim/Projects/CMB/jax-healpy`. The **uv** env has the proper editable install, so prefer uv for scripts/notebooks; reserve `ffi12` for the pymaster oracle.

## 7. Suggested next steps

1. Resolve the one-bug-vs-two coupled-pseudo test (§4), fix `_purify_eb` first if needed.
2. Build + validate the pure-B MCM; tighten `test_oracle.py`.
3. Re-run the θ\*-sweep sanity gate on the corrected estimator — an interior optimum should now appear (NaMaster's leakage falls with θ\*).
4. Implement `_optimize.py` rung-1 + the headline figure.
