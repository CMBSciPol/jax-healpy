Welcome to jax-healpy's documentation!
========================================

**jax-healpy** is a JAX-based implementation of HEALPix (Hierarchical Equal Area isoLatitude Pixelization) functions, designed for high-performance scientific computing with GPU acceleration support.

This project provides a JAX-native implementation of popular HEALPix functions, enabling automatic differentiation, just-in-time compilation, and seamless integration with modern machine learning and scientific computing workflows.

.. note::

   This project is in beta stage. APIs may change and some features are still under development.

Key Features
------------

* **GPU Acceleration**: Leverage JAX's XLA compilation for high-performance computing on CPUs and GPUs
* **Automatic Differentiation**: Full support for forward and reverse-mode automatic differentiation
* **Vectorized Operations**: Efficient batch processing of HEALPix operations
* **HEALPix Compatibility**: Drop-in replacement for many healpy functions
* **Spherical Harmonics**: Integration with s2fft for spherical harmonic transforms
* **Clustering Tools**: Advanced clustering algorithms for astronomical data analysis

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   installation
   user_guide/pixelization
   user_guide/spherical_harmonics
   user_guide/clustering

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_usage
   examples/benchmarks
   examples/clustering_analysis

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/pixelfunc
   api/sphtfunc
   api/clustering
   api/query_disc

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/contributing
   development/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
