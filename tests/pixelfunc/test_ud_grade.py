"""Tests for ud_grade function."""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jax_healpy as jhp


@pytest.mark.parametrize('nside_in', [4, 8, 16, 32])
@pytest.mark.parametrize('nside_out', [2, 8, 32, 64])
@pytest.mark.parametrize('order_in', ['RING', 'NESTED'])
def test_ud_grade_basic_compatibility(nside_in, nside_out, order_in):
    """Test basic ud_grade functionality against healpy for upgrade/degrade scenarios.

    This test verifies that our ud_grade implementation matches healpy exactly
    for various resolution changes and pixel ordering schemes.
    """
    # Skip if nside values are the same (no change needed)
    if nside_in == nside_out:
        pytest.skip(f'No resolution change needed: nside_in={nside_in}, nside_out={nside_out}')

    # Create a test map with simple linear values
    npix_in = jhp.nside2npix(nside_in)
    map_in = np.arange(npix_in, dtype=np.float64)

    # Test our implementation against healpy
    jax_result = jhp.ud_grade(map_in, nside_out, order_in=order_in, order_out=order_in)
    hp_result = hp.ud_grade(map_in, nside_out, order_in=order_in, order_out=order_in)

    # Verify exact match
    assert_array_equal(jax_result, hp_result, f'ud_grade mismatch: nside {nside_in}->{nside_out}, order {order_in}')


@pytest.mark.parametrize('nside', [4, 8, 16])
def test_ud_grade_round_trip_consistency(nside):
    """Test that upgrade followed by degrade returns to original resolution.

    This tests the mathematical consistency of the ud_grade operation:
    degrade(upgrade(map, 2*nside), nside) should be close to the original map.
    """
    # Create test map
    npix = jhp.nside2npix(nside)
    original_map = np.arange(npix, dtype=np.float64)

    # Test round-trip: upgrade then degrade
    upgraded = jhp.ud_grade(original_map, nside * 2)
    round_trip = jhp.ud_grade(upgraded, nside)

    # Should be very close to original (within numerical precision)
    assert_allclose(round_trip, original_map, rtol=1e-10, atol=1e-10, err_msg=f'Round-trip failed for nside={nside}')


def test_ud_grade_interface_validation():
    """Test that ud_grade function has correct interface and parameter validation."""
    nside_in = 4
    npix_in = jhp.nside2npix(nside_in)
    test_map = np.arange(npix_in, dtype=np.float64)

    # Test invalid nside_out parameter
    with pytest.raises(ValueError):
        # Invalid nside (not power of 2)
        jhp.ud_grade(test_map, 7)

    with pytest.raises(ValueError):
        # Invalid nside (too large)
        jhp.ud_grade(test_map, 2**31)

    # Test that valid calls work correctly
    result = jhp.ud_grade(test_map, 8)
    assert len(result) == jhp.nside2npix(8), 'Should return correct output size'

    # Test parameter types are preserved
    result = jhp.ud_grade(test_map, 8, pess=True, order_in='RING', order_out='NESTED', power=2.0, dtype=np.float32)
    assert result.dtype == np.float32, 'Should respect dtype parameter'


@pytest.mark.parametrize('map_type', ['single', 'multiple'])
def test_ud_grade_map_formats(map_type):
    """Test ud_grade with different map input formats (single map vs multiple maps)."""
    nside_in = 4
    npix_in = jhp.nside2npix(nside_in)
    nside_out = 8

    if map_type == 'single':
        # Single map
        test_map = np.arange(npix_in, dtype=np.float64)
    else:
        # Multiple maps (3 maps)
        test_map = np.array(
            [
                np.arange(npix_in, dtype=np.float64),
                2 * np.arange(npix_in, dtype=np.float64),
                np.sin(np.arange(npix_in, dtype=np.float64)),
            ]
        )

    result = jhp.ud_grade(test_map, nside_out)

    # Verify output shape
    if map_type == 'single':
        assert result.ndim == 1, 'Single map should return 1D array'
        assert len(result) == jhp.nside2npix(nside_out), 'Wrong output size'
    else:
        assert result.ndim == 2, 'Multiple maps should return 2D array'
        assert result.shape[0] == 3, 'Should preserve number of maps'
        assert result.shape[1] == jhp.nside2npix(nside_out), 'Wrong output size'

    # Compare with healpy
    hp_result = hp.ud_grade(test_map, nside_out)
    assert_array_equal(result, hp_result, f'Should match healpy for {map_type} map(s)')


def test_ud_grade_unseen_handling():
    """Test handling of UNSEEN pixels during upgrade/degrade operations."""
    nside_in = 4
    npix_in = jhp.nside2npix(nside_in)
    nside_out = 2

    # Create map with some UNSEEN pixels
    test_map = np.arange(npix_in, dtype=np.float64)
    test_map[0] = jhp.UNSEEN
    test_map[10] = jhp.UNSEEN
    test_map[npix_in - 1] = jhp.UNSEEN

    # Test pessimistic vs optimistic handling
    result_optimistic = jhp.ud_grade(test_map, nside_out, pess=False)
    result_pessimistic = jhp.ud_grade(test_map, nside_out, pess=True)

    # Results should be different when UNSEEN pixels are present
    # Pessimistic should have more UNSEEN pixels
    unseen_optimistic = np.sum(result_optimistic == jhp.UNSEEN)
    unseen_pessimistic = np.sum(result_pessimistic == jhp.UNSEEN)
    assert unseen_pessimistic >= unseen_optimistic, 'Pessimistic should have more UNSEEN pixels'

    # Compare with healpy
    hp_optimistic = hp.ud_grade(test_map, nside_out, pess=False)
    hp_pessimistic = hp.ud_grade(test_map, nside_out, pess=True)

    assert_array_equal(result_optimistic, hp_optimistic, 'Optimistic mode should match healpy')
    assert_array_equal(result_pessimistic, hp_pessimistic, 'Pessimistic mode should match healpy')


@pytest.mark.parametrize('power', [0.5, 1.0, 2.0, -1.0])
@pytest.mark.parametrize('nside_in,nside_out', [(4, 8), (8, 4), (4, 16), (16, 2)])
def test_ud_grade_power_parameter(power, nside_in, nside_out):
    """Test power parameter functionality with various scaling factors."""
    npix_in = jhp.nside2npix(nside_in)
    test_map = np.arange(npix_in, dtype=np.float64) + 1.0  # Avoid zeros for power tests

    # Test power parameter
    jax_result = jhp.ud_grade(test_map, nside_out, power=power)
    hp_result = hp.ud_grade(test_map, nside_out, power=power)

    # Use allclose for power tests due to floating point precision differences in power calculations
    assert_allclose(
        jax_result,
        hp_result,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f'Power={power} scaling should match healpy (nside {nside_in}→{nside_out})',
    )


@pytest.mark.parametrize('order_in', ['RING', 'NESTED'])
@pytest.mark.parametrize('order_out', ['RING', 'NESTED'])
@pytest.mark.parametrize('nside_in,nside_out', [(4, 8), (8, 2)])
def test_ud_grade_cross_ordering(order_in, order_out, nside_in, nside_out):
    """Test all combinations of input and output ordering schemes."""
    npix_in = jhp.nside2npix(nside_in)
    test_map = np.arange(npix_in, dtype=np.float64)

    # Test cross-ordering scenarios
    jax_result = jhp.ud_grade(test_map, nside_out, order_in=order_in, order_out=order_out)
    hp_result = hp.ud_grade(test_map, nside_out, order_in=order_in, order_out=order_out)

    assert_array_equal(jax_result, hp_result, f'Cross-ordering {order_in}→{order_out} should match healpy')


@pytest.mark.parametrize('pess', [False, True])
@pytest.mark.parametrize('nside_degradation', [(16, 4), (8, 2)])
def test_ud_grade_pessimistic_detailed(pess, nside_degradation):
    """Detailed test of pessimistic vs optimistic UNSEEN handling."""
    nside_in, nside_out = nside_degradation
    npix_in = jhp.nside2npix(nside_in)

    # Create map with scattered UNSEEN pixels to test averaging behavior
    test_map = np.ones(npix_in, dtype=np.float64)

    # Set every 4th pixel to UNSEEN to create mixed parent groups
    for i in range(0, npix_in, 4):
        test_map[i] = jhp.UNSEEN

    # Also set a large contiguous block to UNSEEN
    test_map[npix_in // 2 : npix_in // 2 + 50] = jhp.UNSEEN

    jax_result = jhp.ud_grade(test_map, nside_out, pess=pess)
    hp_result = hp.ud_grade(test_map, nside_out, pess=pess)

    # The main test: exact match with healpy
    assert_array_equal(jax_result, hp_result, f'Pessimistic={pess} detailed UNSEEN handling should match healpy')

    # Verify that pessimistic mode produces more UNSEEN pixels than optimistic
    if pess:
        jax_optimistic = jhp.ud_grade(test_map, nside_out, pess=False)
        hp_optimistic = hp.ud_grade(test_map, nside_out, pess=False)

        jax_unseen_pess = np.sum(jax_result == jhp.UNSEEN)
        jax_unseen_opt = np.sum(jax_optimistic == jhp.UNSEEN)
        hp_unseen_pess = np.sum(hp_result == hp.UNSEEN)
        hp_unseen_opt = np.sum(hp_optimistic == hp.UNSEEN)

        # Pessimistic should have at least as many UNSEEN pixels as optimistic
        assert jax_unseen_pess >= jax_unseen_opt, 'JAX pessimistic should have more UNSEEN pixels'
        assert hp_unseen_pess >= hp_unseen_opt, 'HP pessimistic should have more UNSEEN pixels'


@pytest.mark.parametrize('map_type', ['single', 'multiple'])
@pytest.mark.parametrize('order_combo', [('RING', 'NESTED'), ('NESTED', 'RING')])
def test_ud_grade_advanced_combinations(map_type, order_combo):
    """Test advanced parameter combinations with single and multiple maps."""
    order_in, order_out = order_combo
    nside_in, nside_out = 4, 8
    npix_in = jhp.nside2npix(nside_in)

    if map_type == 'single':
        test_map = np.arange(npix_in, dtype=np.float64)
    else:
        test_map = np.array(
            [
                np.arange(npix_in, dtype=np.float64),
                2 * np.arange(npix_in, dtype=np.float64),
                np.sin(np.arange(npix_in, dtype=np.float64)),
            ]
        )

    # Test with power scaling and cross-ordering
    jax_result = jhp.ud_grade(test_map, nside_out, order_in=order_in, order_out=order_out, power=2.0, dtype=np.float32)
    hp_result = hp.ud_grade(test_map, nside_out, order_in=order_in, order_out=order_out, power=2.0, dtype=np.float32)

    assert_array_equal(
        jax_result, hp_result, f'Advanced combo {map_type} maps {order_in}→{order_out} with power should match healpy'
    )
    assert jax_result.dtype == np.float32, 'Should respect dtype parameter'


@pytest.mark.parametrize('nside', [4, 8, 16])
def test_ud_grade_jit(nside):
    """Test ud_grade works correctly under JIT compilation."""
    npix_in = jhp.nside2npix(nside)
    test_map = np.arange(npix_in, dtype=np.float64)

    # Test single map with basic parameters
    nside_out = nside * 2  # Upgrade test

    # Direct call (already JIT compiled)
    result_direct = jhp.ud_grade(test_map, nside_out)

    # Explicit JIT compilation
    jit_ud_grade = jax.jit(
        jhp.ud_grade, static_argnames=['nside_out', 'pess', 'order_in', 'order_out', 'power', 'dtype']
    )
    result_jit = jit_ud_grade(test_map, nside_out)

    assert_array_equal(result_direct, result_jit, f'nside {nside}: JIT and direct results must match')

    # Test with multiple maps
    multi_maps = np.array([test_map, 2 * test_map])
    result_multi_direct = jhp.ud_grade(multi_maps, nside_out)
    result_multi_jit = jit_ud_grade(multi_maps, nside_out)

    assert_array_equal(
        result_multi_direct, result_multi_jit, f'nside {nside}: JIT multiple maps must match direct call'
    )

    # Test with advanced parameters
    if nside >= 8:  # Need sufficient resolution for degrade
        nside_degrade = nside // 2
        result_advanced_direct = jhp.ud_grade(test_map, nside_degrade, pess=True, power=2.0, dtype=np.float32)
        result_advanced_jit = jit_ud_grade(test_map, nside_degrade, pess=True, power=2.0, dtype=np.float32)

        assert_allclose(
            result_advanced_direct,
            result_advanced_jit,
            rtol=1e-6,
            err_msg=f'nside {nside}: JIT advanced parameters must match direct call',
        )

    # Test JIT caching - multiple calls should be consistent
    result_cached_1 = jit_ud_grade(test_map, nside_out)
    result_cached_2 = jit_ud_grade(test_map, nside_out)

    assert_array_equal(result_cached_1, result_cached_2, f'nside {nside}: JIT cached calls must be consistent')


@pytest.mark.parametrize('nside', [4, 8])
def test_ud_grade_gradient(nside):
    """Test gradient computation through ud_grade for map inputs."""
    npix_in = jhp.nside2npix(nside)

    def ud_grade_sum(map_vals):
        """Sum of ud_grade result (for gradient testing)."""
        # Upgrade the map and sum the result
        result = jhp.ud_grade(map_vals, nside * 2)
        return jnp.sum(result, dtype=jnp.float32)

    # Use a simple test map
    test_map = jnp.arange(npix_in, dtype=jnp.float32) + 1.0

    # Compute gradients with respect to input map values
    grad_fn = jax.grad(ud_grade_sum)
    gradients = grad_fn(test_map)

    # Gradients should be finite (not NaN or infinite)
    assert jnp.all(jnp.isfinite(gradients)), f'nside {nside}: all gradients must be finite'

    # Check gradient shape matches input
    assert gradients.shape == test_map.shape, f'nside {nside}: gradient shape must match input'

    # For upgrade, each input pixel contributes to multiple output pixels
    # So gradients should generally be positive (since we're summing the output)
    rat2 = (nside * 2) ** 2 // nside**2  # Number of children per parent
    expected_grad = rat2  # Each input pixel contributes to rat2 output pixels

    assert_allclose(
        gradients,
        expected_grad,
        rtol=1e-6,
        err_msg=f'nside {nside}: gradients should equal replication factor for upgrade',
    )


def test_ud_grade_gradient_degrade():
    """Test gradient computation for degradation case."""
    nside_in = 8
    nside_out = 4
    npix_in = jhp.nside2npix(nside_in)

    def ud_grade_degrade_sum(map_vals):
        """Sum of degraded map (for gradient testing)."""
        result = jhp.ud_grade(map_vals, nside_out)
        return jnp.sum(result, dtype=jnp.float32)

    # Use a test map with positive values
    test_map = jnp.ones(npix_in, dtype=jnp.float32)

    # Compute gradients
    grad_fn = jax.grad(ud_grade_degrade_sum)
    gradients = grad_fn(test_map)

    # Gradients should be finite
    assert jnp.all(jnp.isfinite(gradients)), 'Degrade gradients must be finite'

    # Check gradient shape
    assert gradients.shape == test_map.shape, 'Degrade gradient shape must match input'

    # For degradation, gradients should be positive fractions (averaging)
    rat2 = nside_in**2 // nside_out**2  # Number of children per parent
    expected_grad = 1.0 / rat2  # Each input contributes 1/rat2 to its parent

    assert_allclose(gradients, expected_grad, rtol=1e-6, err_msg='Degrade gradients should equal 1/averaging_factor')


def test_ud_grade_gradient_with_power():
    """Test gradient computation with power parameter."""
    nside = 4
    npix_in = jhp.nside2npix(nside)
    power = 2.0

    def ud_grade_power_sum(map_vals):
        """Sum of power-scaled ud_grade result."""
        result = jhp.ud_grade(map_vals, nside * 2, power=power)
        return jnp.sum(result, dtype=jnp.float32)

    # Use test map avoiding zeros (important for power calculations)
    test_map = jnp.arange(npix_in, dtype=jnp.float32) + 1.0

    # Compute gradients
    grad_fn = jax.grad(ud_grade_power_sum)
    gradients = grad_fn(test_map)

    # Gradients should be finite
    assert jnp.all(jnp.isfinite(gradients)), 'Power-scaled gradients must be finite'

    # With power scaling, gradients are scaled by the power factor
    power_factor = (nside * 2 / nside) ** power  # (nside_out/nside_in)^power
    rat2 = (nside * 2) ** 2 // nside**2  # Replication factor
    expected_grad = rat2 * power_factor  # Replication * power scaling

    assert_allclose(gradients, expected_grad, rtol=1e-5, err_msg='Power-scaled gradients should include power factor')


@pytest.mark.parametrize('order_combo', [('RING', 'RING'), ('RING', 'NESTED'), ('NESTED', 'RING')])
@pytest.mark.parametrize('pess', [False, True])
def test_ud_grade_jit_advanced_parameters(order_combo, pess):
    """Test JIT compilation with advanced parameter combinations."""
    order_in, order_out = order_combo
    nside = 8
    npix_in = jhp.nside2npix(nside)
    test_map = np.arange(npix_in, dtype=np.float64) + 1.0

    # Add some UNSEEN pixels for pessimistic testing
    if pess:
        test_map[::10] = jhp.UNSEEN  # Every 10th pixel

    # Create JIT compiled version
    jit_ud_grade = jax.jit(
        jhp.ud_grade, static_argnames=['nside_out', 'pess', 'order_in', 'order_out', 'power', 'dtype']
    )

    # Test degradation with advanced parameters
    nside_out = 4

    # Call 1: Test with power and cross-ordering
    result1_direct = jhp.ud_grade(
        test_map, nside_out, pess=pess, order_in=order_in, order_out=order_out, power=2.0, dtype=np.float32
    )
    result1_jit = jit_ud_grade(
        test_map, nside_out, pess=pess, order_in=order_in, order_out=order_out, power=2.0, dtype=np.float32
    )

    assert_allclose(
        result1_direct,
        result1_jit,
        rtol=1e-6,
        err_msg=f'JIT advanced params {order_combo} pess={pess} must match direct call',
    )

    # Call 2: Test JIT caching with same parameters
    result2_jit = jit_ud_grade(
        test_map, nside_out, pess=pess, order_in=order_in, order_out=order_out, power=2.0, dtype=np.float32
    )

    assert_allclose(
        result1_jit, result2_jit, rtol=1e-12, err_msg=f'JIT cached calls {order_combo} pess={pess} must be identical'
    )

    # Call 3: Test with different parameters to ensure no caching conflicts
    result3_jit = jit_ud_grade(
        test_map, nside_out, pess=not pess, order_in=order_in, order_out=order_out, power=1.0, dtype=np.float64
    )
    result3_direct = jhp.ud_grade(
        test_map, nside_out, pess=not pess, order_in=order_in, order_out=order_out, power=1.0, dtype=np.float64
    )

    assert_array_equal(result3_direct, result3_jit, f'JIT different params {order_combo} must work correctly')


def test_ud_grade_jit_multiple_maps():
    """Test JIT compilation with multiple maps and various dtypes."""
    nside = 4
    npix_in = jhp.nside2npix(nside)
    nside_out = 8

    # Test different map configurations
    single_map = np.arange(npix_in, dtype=np.float64)
    double_maps = np.array([single_map, 2 * single_map])
    triple_maps = np.array([single_map, 2 * single_map, np.sin(single_map)])

    # JIT compiled version
    jit_ud_grade = jax.jit(
        jhp.ud_grade, static_argnames=['nside_out', 'pess', 'order_in', 'order_out', 'power', 'dtype']
    )

    test_cases = [('single', single_map), ('double', double_maps), ('triple', triple_maps)]

    for case_name, test_data in test_cases:
        # Test direct vs JIT
        result_direct = jhp.ud_grade(test_data, nside_out)
        result_jit = jit_ud_grade(test_data, nside_out)

        assert_array_equal(result_direct, result_jit, f'JIT {case_name} maps must match direct call')

        # Test with dtype conversion
        result_f32_direct = jhp.ud_grade(test_data, nside_out, dtype=np.float32)
        result_f32_jit = jit_ud_grade(test_data, nside_out, dtype=np.float32)

        assert_array_equal(result_f32_direct, result_f32_jit, f'JIT {case_name} maps with dtype must match direct call')
        assert result_f32_jit.dtype == np.float32, f'{case_name} maps should respect dtype parameter'
