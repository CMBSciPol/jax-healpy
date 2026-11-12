import subprocess
import sys
import warnings

import jax
import pytest
from pytest_mock import MockerFixture

import jax_healpy as hp
from jax_healpy import _precision


@pytest.fixture
def not_warned_yet(mocker: MockerFixture) -> None:
    """Reset the warn-once flag of the precision check."""
    mocker.patch.object(_precision, '_warned', False)


def test_import_does_not_warn() -> None:
    """Importing jax-healpy (and s2fft) in 32-bit precision emits no warning."""
    result = subprocess.run(
        [sys.executable, '-W', 'always', '-c', 'import jax_healpy'],
        capture_output=True,
        text=True,
        env={'JAX_ENABLE_X64': '0', 'JAX_PLATFORMS': 'cpu'},
        check=True,
    )
    assert result.stderr == ''


@pytest.mark.usefixtures('not_warned_yet')
def test_warns_once_in_32_bit() -> None:
    with jax.enable_x64(False):
        with pytest.warns(UserWarning, match='64-bit precision'):
            hp.pix2ang(16, 0)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            hp.pix2ang(16, 0)


@pytest.mark.usefixtures('not_warned_yet')
def test_does_not_warn_in_64_bit() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        hp.pix2ang(16, 0)
