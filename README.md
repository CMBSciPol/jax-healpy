# Healpy with JAX

This project intends to assess the interest of implementing healpy functions using JAX.

_WARNING: BETA STAGE!!!_

## Installation

First install JAX following the [documentation](https://jax.readthedocs.io/en/latest/installation.html).

Then install the package with:

```bash
pip install jax-healpy
```

To use the spherical harmonics functions,
you will need [s2fft](https://astro-informatics.github.io/s2fft/),
part of the recommended dependencies:

```bash
pip install jax-healpy[recommended]
```

## Benchmarks

Execution time measured on the [Jean Zay supercomputer](http://www.idris.fr/jean-zay/cpu/jean-zay-cpu-hw.html).

- CPU: Intel(R) Xeon(R) Gold 2648 @ 2.50GHz
- GPU: NVIDIA Tesla V100-SXM2-16GB

![Benchmark](/docs/benchmarks/chart-darkbackground-n10000000.png)
