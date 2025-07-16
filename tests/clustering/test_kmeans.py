import chex
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.errors import TracerBoolConversionError
from jaxtyping import Array
from numpy.testing import assert_array_equal

import jax_healpy as jhp


@pytest.fixture(scope='module', params=['FULL_MAP', 'GAL020', 'GAL040', 'GAL060'])
def mask(request: pytest.FixtureRequest, data_path: str, nside: int) -> tuple[str, Array]:
    if request.param == 'FULL_MAP':
        return request.param, jnp.ones(jhp.nside2npix(nside))
    else:
        return request.param, np.load(f'{data_path}/GAL_PlanckMasks_64.npz')[request.param]


@pytest.fixture(scope='module', params=[64])
def nside(
    request: pytest.FixtureRequest,
) -> int:
    return request.param


def test_kmeans(mask: tuple[str, Array]) -> None:
    name, mask = mask

    (indices,) = jnp.where(mask == 1)

    n_regions = 10
    key = jax.random.key(0)

    clustered = jhp.get_clusters(mask, indices, n_regions, key)
    print(f'Got {n_regions} regions for mask {name}')

    cutout = jhp.get_cutout_from_mask(clustered, indices)
    # Shape must be the same as the mask
    assert cutout.shape == indices.shape

    labels, counts = jnp.unique(cutout, return_counts=True)

    # Check that all the regions are present
    assert_array_equal(labels, jnp.arange(n_regions))

    # Check that number of pixels in each region is close
    assert (counts.std() / counts.mean()) < 0.5

    print(f'all good for mask {name}')


def test_kmeans_jit(mask: tuple[str, Array]) -> None:
    name, mask = mask

    (indices,) = jnp.where(mask == 1)

    n_regions = 10
    key = jax.random.key(0)

    # number of regions cannot be a tracer if max_centroids is None
    jitted_clusters = jax.jit(jhp.get_clusters, static_argnums=(4, 5))
    with pytest.raises(TracerBoolConversionError):
        jitted_clusters(mask, indices, n_regions, key, max_centroids=None)

    #
    # If max_centroids is not None, n_regions can be a tracer and it is jitted once
    @jax.jit
    @chex.assert_max_traces(n=1)
    def jit_clusters(
        mask: Array,
        indices: Array,
        n_regions: Array,
    ) -> Array:
        return jhp.get_clusters(mask, indices, n_regions, key, max_centroids=10)

    _ = jit_clusters(mask, indices, 5)
    _ = jit_clusters(mask, indices, 10)

    chex.clear_trace_counter()

    # If requested number of regions is greater than max_centroids, raise a runtime error

    with pytest.raises(RuntimeError):
        jit_clusters(mask, indices, 20)


def test_cutout_and_reconstruct(mask: tuple[str, Array], nside: int) -> None:
    name, mask = mask

    (indices,) = jnp.where(mask == 1)
    (inv_indices,) = jnp.where(mask != 1)

    gaussian_map = jax.random.normal(jax.random.key(0), jhp.nside2npix(nside))
    # set to unseen everything outside the mask
    gaussian_map = gaussian_map.at[inv_indices].set(jhp.UNSEEN)

    cutout = jhp.get_cutout_from_mask(gaussian_map, indices)

    assert cutout.shape == indices.shape

    reconstruct = jhp.from_cutout_to_fullmap(cutout, indices, nside)

    assert_array_equal(reconstruct, gaussian_map)


def test_frequency_map_cutout(mask: tuple[str, Array], nside: int) -> None:
    # This is usually done to get a cutout out of d the Frequency landscape object from furax

    name, mask = mask

    (indices,) = jnp.where(mask == 1)
    (inv_indices,) = jnp.where(mask != 1)

    frequency_maps = jax.random.normal(jax.random.key(0), (10, jhp.nside2npix(nside)))
    # set to unseen everything outside the mask
    frequency_maps = frequency_maps.at[..., inv_indices].set(jhp.UNSEEN)

    cutout = jhp.get_cutout_from_mask(frequency_maps, indices, axis=1)

    assert cutout.shape == (10, *indices.shape)


def test_normalize_from_clusters(mask: tuple[str, Array], nside: int) -> None:
    name, mask = mask
    (indices,) = jnp.where(mask == 1)

    n_regions = 25
    max_centroids = 50
    key = jax.random.PRNGKey(0)

    # Get raw clusters
    raw_labels = jhp.get_clusters(mask, indices, n_regions, key, max_centroids=max_centroids)
    normalized = jhp.normalize_by_first_occurrence(raw_labels, n_regions, max_centroids)

    # Ensure UNSEEN positions are unchanged
    raw_unseen_mask = raw_labels == hp.UNSEEN
    normalized_unseen_mask = normalized == hp.UNSEEN
    assert_array_equal(raw_unseen_mask, normalized_unseen_mask), 'UNSEEN positions were modified'

    # Remove UNSEEN and validate cluster properties
    valid = np.array(normalized[~normalized_unseen_mask]).astype(np.int64)

    uniques, idx = np.unique(valid, return_index=True)
    assert len(uniques) == n_regions, f'Expected {n_regions} regions, got {len(uniques)}'
    assert (np.sort(idx) == idx).all(), 'First-occurrence indices are not sorted'


def test_shuffle_labels_randomizes_labels():
    arr = np.array([0, 0, 1, 1, 2, 2])
    shuffled = jhp.shuffle_labels(arr)

    # Should contain same elements (set equality), but likely different order
    assert set(shuffled) == {0, 1, 2}
    assert not np.array_equal(arr, shuffled)  # Not equal in order most times


def test_shuffle_labels_preserves_unseen():
    arr = np.array([0, 1, 2, hp.UNSEEN, 1, hp.UNSEEN])
    shuffled = jhp.shuffle_labels(arr)

    # UNSEEN stays where it is
    assert shuffled[3] == hp.UNSEEN
    assert shuffled[5] == hp.UNSEEN

    # All other entries are in the valid label range
    valid_labels = shuffled[shuffled != hp.UNSEEN]
    assert set(valid_labels).issubset({0, 1, 2})
