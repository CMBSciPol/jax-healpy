import subprocess
import sys


def test_no_eager_backend_init():
    """Verify that importing jax_healpy does not trigger JAX backend initialisation."""
    script = (
        'from jax._src.xla_bridge import backends_are_initialized; '
        'import jax_healpy; '
        "assert not backends_are_initialized(), 'JAX backend initialised during import'"
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
