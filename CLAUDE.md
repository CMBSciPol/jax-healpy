# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**jax-healpy** is a JAX-native **reimplementation** of HEALPix — pixel functions, spherical harmonic transforms, pseudo-Cl estimation, and spherical K-means clustering, all differentiable and jittable.

**Reimplementation is the word that matters.** What upstream `healpy` does is a *hypothesis* about what this code does, never a substitute for reading it. Numerical agreement is tested where it holds and deliberately absent where it does not, so an answer of the form "healpy does X, therefore this does X" is a guess wearing a citation.

**This is not solely Wassim's package.** `pyproject.toml` names Pierre Chanial, Simon Biquard and Wassim Kabalan as authors, with Chanial as maintainer — it is a CMB SciPol project. **Who wrote which part is a `git log` / `git blame` question, never an assumption.**

## Read the source before you quote an API

This is a working tree on a feature branch with uncommitted work, and both move. **A citation needs a state** — pin it at the top of every answer:

```bash
git rev-parse --abbrev-ref HEAD; git log -1 --format='%h %ad %s' --date=short; git status --porcelain | wc -l
```

**`grep` is the index.** Locate with `grep -rn "def ang2pix" jax_healpy/`, then read the function and quote what you see. Never describe a signature, a default or a return type you did not open — a plausible-looking API with a `file:line` attached is the worst thing to produce, because the line number makes it look verified.

**Tracked beats untracked.** `git ls-files <path>` (empty output means not tracked) is the authority test. **This file is untracked**, so the source outranks everything below it.

**What a docstring claims, what the code does, and what a test verifies are three separate facts.** Say which you have; the disagreement is usually the finding.

## Module map

Symbols move within their files as the package grows, so this table names them rather than pinning line numbers: **grep for the `def`, then cite the line you actually landed on.**

| module | what it is |
|---|---|
| `jax_healpy/pixelfunc.py` | the core, and the biggest file by far — `ang2pix`, `pix2ang`, `ang2vec`, nest/ring conversions, `ud_grade`, `get_nside`/`isnpixok`, `get_interp_val`, `get_interp_weights`, `get_all_neighbours`, `UNSEEN` |
| `jax_healpy/sphtfunc.py` | spherical harmonic transforms — `alm2map`/`map2alm`, `anafast`, `synalm`/`synfast`, `smoothing`, `pixwin`, and the spin-2 pair `map2alm_spin`/`alm2map_spin` that an E/B decomposition needs |
| `jax_healpy/_query_disc.py` | `query_disc` plus the disc-size estimators `estimate_disc_pixel_count` and `estimate_disc_radius`, and the ring / brute-force implementations behind them |
| `jax_healpy/pseudo_cl/` | pseudo-Cl estimation — `_apodize.py`, `_mcm.py` (mode-coupling matrix), `_estimate.py`, `_utils.py` |
| `jax_healpy/clustering/` | `_clustering.py` `find_kmeans_clusters`; `_kmeans.py` `KMeans` / `KMeansState` and the `radec2xyz`/`xyz2radec` pair; plus the mask machinery — `get_cutout_from_mask`, `combine_masks`, `get_fullmap_from_cutout`, `normalize_by_first_occurrence`, `shuffle_labels` |

`jax_healpy/__init__.py` is the authoritative export list — **read it rather than assuming a name is public.** In particular `pseudo_cl/` is **not** exported from it, so it is reached by its full path. Check whether that is still true before saying so.

## Three things that are easy to get wrong

**`s2fft` is an optional dependency and it fails late.** `sphtfunc.py` builds on `s2fft`, imported behind a `requires_s2fft` decorator (`sphtfunc.py:59`, applied at `:75`, `:122`, `:319`). A missing install therefore fails **at call time, not at import time** — so "the module imported fine" proves nothing about whether a transform will run.

**`Healpix_3.83/` is not part of this package.** It is a local copy of upstream C/C++ HEALPix, and `git ls-files Healpix_3.83` is empty. It is reference material for reading the reference implementation. Do not cite it as if it were this package, and remember it will be absent from a fresh clone.

**Packaging has dropped the clustering subpackage before.** A `packages = ['jax_healpy']` declaration excludes `clustering/` from built wheels — which matters because clustering is precisely what the downstream science uses. Check `pyproject.toml` before assuming a wheel contains it.

## Tests

`tests/`, including `tests/clustering/test_kmeans.py`, with real fixture data (`tests/data/GAL_PlanckMasks_64.npz`, WMAP FITS maps).

Prefer the test to your reading of a function — it has actually run. **Absence of a test is itself a finding.** Two known fragilities worth knowing before trusting a green run: the spin/scalar transform tests need a cache-clear fixture, `nside=16` was dropped from the `sphtfunc` fixture (healpy returns NaNs there and the error floor is larger), and CI runs unmasked.

## The clustering path is the one downstream work depends on

`clustering/` is what the component-separation side uses to build patch configurations: spherical K-means run inside predefined Planck `GAL` Galactic masks. The lineage is `kmeans_radec` (Sheldon), adapted for spherical coordinates — positions are given as right ascension and declination, distances are angular separations, and each position is mapped to a unit vector so that a centroid is the mean 3D direction converted back to (RA, Dec).

**Every clause in that paragraph is a checkable claim about `_kmeans.py`.** When a write-up makes one — a thesis chapter, a paper, a README or a talk — read the centroid update and the `radec2xyz`/`xyz2radec` pair and check it, rather than agreeing because it sounds right. Write-ups drift: an earlier description of this code quoted a `.tex` line number that has since moved and a citation key that has since changed, and anyone following the old anchor would have been sent somewhere wrong with full confidence. **Grep the document for the claim; never trust a line number written down elsewhere.**

## Scope

The physics — CMB, lensing, what a pseudo-Cl estimator is for — is not here. Whether an algorithm is standard maths is a textbook question, and the `/books` skill has those on disk. The N-body side of the stack is `jax-fli` / `JaxPM` / `jaxDecomp`, each with its own `CLAUDE.md`.

**This repository is edited by other sessions.** If its mid-edit state breaks an import for something that depends on it, pause and say so rather than fixing it from the outside.
