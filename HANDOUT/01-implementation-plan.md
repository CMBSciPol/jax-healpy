# Implementation Plan — `jax_healpy.pseudo_cl` (differentiable masked-sky Cℓ + optimizable apodization)

> Executable plan derived from the study brief `~/.claude/plans/i-wish-to-start-elegant-pancake.md`.
> Scope: build the masked-sky power-spectrum estimator (apodize → purify → MCM decouple → Cℓ) and the
> differentiable optimizable-apodization research layer. Component separation is out of scope.

## Context — why

The paper goal is a fully differentiable B-mode pipeline (data → cleaned CMB → purified B → `Cℓᴮᴮ`)
so gradients drive both component separation and the apodization from the science metric. This plan
builds the **power-spectrum half**. The strategy is a low-risk **verbatim port** of an already
NaMaster-validated MCM + Smith-2006 purification from the author's `jax-fli` package, then a new
**differentiable optimizable-apodization** layer on top (the research contribution — Xpure's
variance-optimal window, reframed as autodiff + gradient descent instead of Xpure's PCG solve).

Source files confirmed to exist and read:
- `/home/wassim/Projects/NBody/jax-fli/src/jax_fli/power/decouple.py` (276 lines) — MCM + decouple + purify + `anafast_masked`.
- `/home/wassim/Projects/NBody/jax-fli/src/jax_fli/data/apodize.py` (80 lines) — differentiable C2 apodization.
- `/home/wassim/Projects/NBody/jax-fli/tests/power/test_decouple.py` (175 lines) — the test patterns / NaMaster oracle.

## Decisions

**Locked by the study brief:** new `jax_healpy.pseudo_cl` subpackage (mirrors `clustering/`); `MCM` stays an
`eqx.Module`; spin-2 `(Q,U)→EE/BB`; **float64 mandatory**; `nside=64`, `lmax=191`, `nlb=16`; inject known `r`
(use `r=0.01`); NaMaster is a test-only oracle; objective = analytic BB-bandpower variance.

**Settled this session:**
- **Oracle tests run locally only** (in the micromamba `ffi12` env, which has `pymaster 2.6` + `equinox 0.13.8`).
  Oracle tests live in a dedicated module that **hard-imports `pymaster`** (no `importorskip`, per global CLAUDE.md).
  CI stays uv-only and runs just the pure-JAX tests; the oracle module is excluded from CI by an explicit,
  commented `--ignore` flag in `ci.yml`. This is a **knowing, documented deviation** from the no-skip rule until
  NaMaster is wired into CI in a follow-up — it is *not* a silent per-test skip.
- **New core dependencies:** `equinox` (MCM container — needed by the whole estimation path) and `optax`
  (gradient optimizer for the apodization). Both go in `[project.dependencies]`; `uv.lock` is regenerated.
  `pymaster` is **not** added to uv (stays conda-only).

## Files

**New subpackage** `jax_healpy/pseudo_cl/` (mirror `jax_healpy/clustering/` layout; not re-exported at top level —
import as `from jax_healpy.pseudo_cl import ...`, matching `clustering`):
```
jax_healpy/pseudo_cl/
  __init__.py     # exports: apodize, grassfire_distance, apodize_profile,
                  #          MCM, compute_mcm, bandpower_windows, anafast_masked,
                  #          bandpower_variance, optimize_apodization
  _apodize.py     # apodize (port) + grassfire_distance / apodize_profile (refactor split)
  _mcm.py         # _legendre_all, _wigner_d2_all, _mcm_spin0/2, _linear_bins, MCM, compute_mcm, bandpower_windows
  _estimate.py    # _decouple_spin0/2, _purify_eb, anafast_masked, _require_x64 guardrail
  _optimize.py    # bandpower_variance (Knox + leakage), window parametrizations, optimize_apodization
```
**New tests** `tests/pseudo_cl/`:
```
  conftest.py        # autouse x64 + jax.clear_caches() (mirror tests/sphtfunc/conftest.py); shared fixtures
  test_pseudo_cl.py  # all pure-JAX tests (no pymaster) — runs in CI + local
  test_oracle.py     # hard `import pymaster` — NaMaster comparisons — runs locally (ffi12) only
```
**New example** `notebooks/bmode_purification_demo.ipynb` (mirror `notebooks/spin_transforms_demo.ipynb`; the §5 headline figure; executed with papermill).

**Modified:** `pyproject.toml` (add `equinox`,`optax` deps; `--ignore` not here — see CI), `uv.lock` (regenerate),
`.github/workflows/ci.yml` (add commented `--ignore=tests/pseudo_cl/test_oracle.py` to the pytest step),
`jax_healpy/pseudo_cl/__init__.py` exports.

---

## Phase A — scaffold + verbatim port + validate (study steps 0–3)

**A0. Scaffold.** Create the subpackage + `__init__.py` exports. Add `equinox`,`optax` to `[project.dependencies]`
and run `uv lock`. Add `_require_x64()` helper (raises a clear error if `jax.config.read('jax_enable_x64')` is
False) and call it at the top of `compute_mcm`, `anafast_masked`, `optimize_apodization` — float32 silently returns
**all-NaN** from the spin-2 solve (study §7), so guard loudly inside the subpackage, not only via the `__init__` warning.

**A1. Port MCM + decoupling** → `_mcm.py`, `_estimate.py`. Copy **verbatim** from `decouple.py`:
`_legendre_all`, `_wigner_d2_all`, `_mcm_spin0`, `_mcm_spin2`, `_linear_bins`, `MCM`, `compute_mcm`,
`_decouple_spin0`, `_decouple_spin2`, `anafast_masked`. **Preserve `healpy_ordering=True` and the `iter` choices
exactly** (mask `map2alm` uses `iter=3`; spin/correction transforms use `iter=0`) — these are load-bearing.
Reuses `jhp.anafast`, `jhp.map2alm_spin`, `jhp.alm2cl`, `jhp.npix2nside` (all confirmed present & JAX-native).
Add `bandpower_windows(mcm)` returning `W = (B·M·S)⁻¹·B·M` (for the §3 oracle comparison vs NaMaster
`get_bandpower_windows`).

**A2. Port apodization + refactor the grassfire/θ\* split** → `_apodize.py`. Copy `apodize` from `apodize.py`, then
split it so the apodization scale becomes differentiable (study §4 rung 1, §7):
- `grassfire_distance(binary_mask, *, max_aposize_deg)` — geometry only; computes the **static** `niter` from the
  *max* bound: `niter = ceil(2.5·deg2rad(max_aposize_deg)/resol)+6`, `resol = sqrt(4π/npix)`; returns the distance field.
- `apodize_profile(dist, theta_star_deg, *, apotype="C2")` — the cosine taper, **differentiable in `theta_star_deg`**
  (it enters only via `clip(dist,0,θ*)` and the `x=sqrt((1-cos d)/(1-cos θ*))` normalization, never via `niter`).
- `apodize(binary_mask, aposize_deg=1.0, *, apotype="C2")` becomes the wrapper
  `apodize_profile(grassfire_distance(binary_mask, max_aposize_deg=aposize_deg), aposize_deg, apotype=apotype)`,
  preserving the original public behaviour. Reuses `jhp.get_all_neighbours`, `jhp.pix2vec`, `jhp.npix2nside`.

**A3. Port purification** → `_estimate.py`. Copy `_purify_eb` verbatim (3-term Smith-2006: spin-2 of `(W·Q,W·U)` +
spin-1 window-derivative correction + spin-0 correction; `healpy_ordering=True` throughout).

**A-validate** (oracle, run in `ffi12`): scalar decouple vs NaMaster `compute_full_master` `rel<1e-8`; spin-2 EE
`rel<5e-2` (iter=0 floor); `Wℓ` vs NaMaster C2 `rel<5e-3`; purified BB on a pure-E sky vs NaMaster purified BB,
bound expressed as an **absolute leakage level (fraction of `Cℓᴱᴱ`)** tied to the iter=0 floor (study §5a).

---

## Phase B — research layer: objective + sanity gate + rung-1 optimization + figure (study steps 4–5)

**B1. Objective** → `_optimize.py`: `bandpower_variance(cl_bb_total, mcm, mask, *, fsky=None)` implementing the
Knox / mask-moment BB variance (study §4), **fully differentiable in the window**:
```
Var(Ĉ_b^BB) ≈ [⟨W⁴⟩/⟨W²⟩²] · 2/[(2ℓ_b+1)Δℓ] · (Cℓ^{BB,total})²
Cℓ^{BB,total} = Cℓ^{BB,sig} + Nℓ/⟨W²⟩ + Lℓ
```
**The leakage term `Lℓ` is load-bearing and must NOT be schematic** (study §4 bold warning): without it the two
mask-moment factors both shrink toward a binary mask, driving the optimizer to `W→binary` — exactly where
purification breaks. Implement `Lℓ` by **method (i): contract the spin-2 leakage block `mcm.eebb` with the fiducial
`Cℓᴱᴱ`** (cheap, differentiable). Calibrate/cross-check its normalization once against **method (ii)** (run the
purified estimator forward on a noiseless pure-E sky, read off residual BB).

**B2. Sanity gate (BEFORE any gradient work, study §4/§6 step 4):** sweep `Σ_b Var(Ĉ_b^BB)` vs `θ*` **with
`purify_b=True`** and inspect the shape. Interior U-shape → trade-off active, proceed to gradients. Monotone toward
small `θ*` → *itself a finding* (purification dominates); then add white noise (raises `Nℓ/⟨W²⟩`), report the
variance-tail gain, and lean on the per-pixel rung. Capture this as a test (`bandpower_variance` finite, decreasing
then increasing — or monotone — across the swept `θ*`) and as the first notebook cell.

**B3. Window parametrizations + optimizer** → `_optimize.py`: `optimize_apodization(binary_mask, q_maps, *,
window="scalar"|"profile"|"perpixel", init, n_steps, lr, lmax=None, nlb=16, cl_bb_inject, cl_ee, noise=None,
beam=None, reg_smooth=0.0, reg_boundary=0.0, method="jax") -> (window, history)`. Differentiability contract (state
in the module docstring): `loss = Σ_b Var(Ĉ_b^BB) ← decouple(MCM(W), pseudo_cl(purify(W,Q,U))) ← W = apodize(params)`,
all differentiable in `params`; **float64**; iter=0 spin transforms (fixed unrolled iter>0 only if accuracy demands).
Rung 0 = fixed C2 baseline; **rung 1 = scalar `θ*`** (freeze grassfire distance, differentiate profile→MCM→Cℓ) is the
committed target; rung 2 = parametric monotone profile. Use **`optax`** (Adam) for the gradient loop.

**B4. Headline figure** → `notebooks/bmode_purification_demo.ipynb` (study §5b): `nside=64`, `lmax=191`, float64;
primary mask = smooth SO/SAT-like cap (the `_quadrant` mask only for the exact-decoupling oracle test, since its 90°
corners are pessimal for purification); inject `r=0.01` via `synfast`/`synalm` into `(Q,U)`. Show recovered
`Cℓᴮᴮ ± σ` over bandpowers for {no-purify, fixed-apo+purify, optimized-apo+purify} against the injected theory line,
with error bars **shrinking** for the optimized window. Execute with
`papermill nb.ipynb nb.ipynb --log-output --progress-bar --log-level INFO`.

---

## Phase C — rung-3 per-pixel prototype (study step 6, **stretch / may slip**)

Per-pixel free window: `params` *is* the window (no grassfire). Replace the analytic `⟨W⁴⟩/⟨W²⟩²` mode-count proxy
with the exact signal+noise covariance contraction, and add `reg_smooth·‖∇W‖² + reg_boundary·(boundary penalty)`
(the differentiable stand-in for Xpure's preconditioner) to keep `W` C1/C2 and well-posed. Prototype only;
rungs 0–2 + the figure are the committed deliverable.

---

## Dependencies & CI

- `pyproject.toml`: add `'equinox'` and `'optax'` to `[project.dependencies]`; then `uv lock` (CI runs
  `uv sync --locked`, so a stale lockfile fails the build). Do **not** add `pymaster`.
- `.github/workflows/ci.yml`: change the test step to
  `uv run pytest -vv -m "not slow" --ignore=tests/pseudo_cl/test_oracle.py`, with a comment explaining the oracle
  tests require the conda `namaster` env and are run locally pending CI wiring.
- **Markers:** oracle + port-validation tests are **not** `slow` (must run when collected). The optimization-loop
  tests (`optimize_apodization` over many steps) are marked `@pytest.mark.slow`. Spin-2 oracle comparisons may use
  `@pytest.mark.flaky(reruns=...)` (pytest-rerunfailures is already a dev dep) for s2fft nondeterminism.

## Tests (per global CLAUDE.md — fixtures, never closures; fail loudly on missing deps)

- `tests/pseudo_cl/conftest.py`: autouse fixture enabling x64 + `jax.clear_caches()` per test (mirror
  `tests/sphtfunc/conftest.py`); module-scoped fixtures `quadrant_mask`, `apo` (apodized quadrant), `smooth_cap_mask`,
  `cmb_qu` (synfast Q/U at injected `r`), `pure_e_qu`.
- `test_pseudo_cl.py` (pure-JAX, no pymaster): port tests that need no oracle — `apodize` range/zero-outside,
  jittability, `mask=None` scalar==`anafast`, `mask=None` pol→3 spectra, `purify` requires mask,
  pol cross-spectrum raises `NotImplementedError`, coupled-pseudo via premask, MCM-precomputed-identical,
  EB-is-binned-pseudo, decoupled-Cℓ differentiable-in-map, **purify_b reduces leakage** (uses only `anafast_masked`);
  plus new: `apodize_profile` grad-stable in `θ*`, `grassfire_distance`/`apodize_profile` reproduce `apodize`,
  `bandpower_variance` finite+differentiable + the §B2 sweep-shape assertion.
- `test_oracle.py` (hard `import pymaster`): `Wℓ` vs NaMaster C2 `rel<5e-3`; scalar decouple `rel<1e-8`;
  spin-2 EE `rel<5e-2`; purified BB vs NaMaster purified BB (absolute-leakage bound). Module docstring documents
  the `ffi12`/conda requirement and the CI exclusion.

## Verification (end-to-end)

1. **Local full suite (in `ffi12`, has pymaster+equinox+optax):**
   `micromamba run -n ffi12 python -m pytest tests/pseudo_cl -vv` → all pass, including oracle tolerances above.
2. **CI-equivalent (uv, no pymaster):**
   `uv run pytest tests/pseudo_cl -vv -m "not slow" --ignore=tests/pseudo_cl/test_oracle.py` → all pure-JAX pass;
   confirm the package imports under uv (equinox+optax resolved from regenerated `uv.lock`).
3. **Wheel ships the subpackage** (closes the stale `packages=['jax_healpy']` memory note — repo now uses hatchling,
   which should auto-discover): `uv build` then `unzip -l dist/*.whl | grep -E 'pseudo_cl|clustering'` → both present.
4. **Notebook:** `papermill notebooks/bmode_purification_demo.ipynb notebooks/bmode_purification_demo.ipynb
   --log-output --progress-bar --log-level INFO` → runs clean; headline figure shows optimized-apo error bars
   below the fixed-C2 baseline (or the documented "purification dominates" finding from §B2).

## Risks (load-bearing — from study §7)

- **EE ≫ BB dynamic range.** At `r=0.01`, BB is 10²–10³× below EE; the ~1% iter=0 spin-2 floor / residual leakage can
  swamp BB. Validate the leakage floor against the target `r` first; if iter=0 is insufficient, switch the purify
  transforms to a **fixed unrolled `iter>0`** (stays differentiable).
- **float64 mandatory** — enforced by `_require_x64()` (raise) at the public entry points.
- **MCM rebuilt every optimization step** (`W`→`Wℓ`→MCM): `O(lmax²·n_quad)`. Fine at `nside=64`; note a cheaper path
  (cache / Toeplitz approx) before scaling. 
- **Boundary smoothness is the binding constraint** for rung 3 — without smoothness/boundary regularizers
  purification breaks; these *are* Xpure's preconditioner re-expressed.
- **Deferred:** EB/TB decoupling block, TT/TE, spin-0×spin-2 MCM (EB row stays the binned coupled pseudo).
- **Memory housekeeping (post-merge):** correct `pr4-packaging-clustering-dropped.md` — that `packages=['jax_healpy']`
  drop was the setuptools era; the repo now builds with hatchling (verified in step 3 above).
