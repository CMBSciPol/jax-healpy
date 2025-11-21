from typing import Any, Callable

import numpy as np
import pytest

s2fft = pytest.importorskip('s2fft')

from s2fft.sampling.s2_samples import flm_2d_to_hp  # noqa: E402
from s2fft.utils import signal_generator  # noqa: E402


@pytest.fixture
def flm_generator(numpy_rng) -> Callable[[...], np.ndarray]:
    # Import s2fft (and indirectly numpy) locally to avoid
    # `RuntimeWarning: numpy.ndarray size changed` when importing at module level
    # import s2fft as s2f
    # from s2fft.utils import signal_generator
    def generate_flm(L: int, healpy_ordering: bool = False, **keywords: Any) -> np.ndarray:
        flm = signal_generator.generate_flm(numpy_rng, L, **keywords)
        if healpy_ordering:
            flm = flm_2d_to_hp(flm, L)
        return flm

    return generate_flm

@pytest.fixture(scope='session')
def numpy_rng() -> np.random.RandomState:
    seed = 0
    return np.random.RandomState(seed)


@pytest.fixture
def flm_generator_batched() -> Callable[[...], np.ndarray]:
    # Import s2fft (and indirectly numpy) locally to avoid
    # `RuntimeWarning: numpy.ndarray size changed` when importing at module level
    # import s2fft as s2f
    # from s2fft.utils import signal_generator
    numpy_rng_1 = np.random.RandomState(1)
    numpy_rng_2 = np.random.RandomState(2)
    def generate_flm_1(L: int, healpy_ordering: bool = False, **keywords: Any) -> np.ndarray:
        flm = signal_generator.generate_flm(numpy_rng_1, L, **keywords)
        if healpy_ordering:
            flm = flm_2d_to_hp(flm, L)
        return  flm
    
    def generate_flm_2(L: int, healpy_ordering: bool = False, **keywords: Any) -> np.ndarray:
        flm = signal_generator.generate_flm(numpy_rng_2, L, **keywords)
        if healpy_ordering:
            flm = flm_2d_to_hp(flm, L)
        return flm

    return generate_flm_1 , generate_flm_2