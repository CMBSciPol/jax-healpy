import sys
from typing import NamedTuple, Optional

import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jax.numpy import arccos, arctan2, cos, deg2rad, newaxis, pi, rad2deg, sin, sqrt
from jaxtyping import Array, PRNGKeyArray

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


_TOL_DEF = 1.0e-5
_MAXITER_DEF = 100
_PIOVER2 = pi * 0.5


class KMeansState(NamedTuple):
    """Holds the state of the KMeans clustering process.

    Attributes:
        ra_dec (Array): The array of RA and DEC coordinates.
        centroids (Array): The array of current centroid coordinates.
        labels (Array): Labels assigning each point to a centroid.
        mean_distance (Array): Current mean distance between points and centroids.
        previous_mean_distance (Array): Previous mean distance for convergence check.
        count (int): Iteration count.
    """

    ra_dec: Array
    centroids: Array
    labels: Array
    mean_distance: Array
    previous_mean_distance: Array
    count: int


class KMeans:
    """KMeans clustering for spherical coordinates using JAX."""

    def __init__(
        self: Self,
        ncenters: int,
        max_centroids: Optional[int] = None,
        tol: float = _TOL_DEF,
        maxiter: int = _MAXITER_DEF,
        initial_sample_size: int = 3,
    ) -> None:
        """Initialize the KMeans instance.

        Args:
            ncenters (int): Number of clusters.
            max_centroids (Optional[int]): Maximum number of centroids to consider.
            tol (float): Tolerance for convergence.
            maxiter (int): Maximum number of iterations.
            initial_sample_size (int): Initial sample size.
        """
        self.ncenters = ncenters
        self.max_centroids = max_centroids
        self.tol = tol
        self.maxiter = maxiter
        self.initial_sample_size = initial_sample_size

    def sample_initial(self: Self, ra_dec: Array, key: PRNGKeyArray) -> tuple[Array, Array]:
        """Sample initial data points and centroids.

        Args:
            ra_dec (Array): Array of RA and DEC coordinates.
            key (PPRNGKeyArrayRNGKey): JAX random key.

        Returns:
            tuple[Array, Array]: Sampled RA/DEC points and initial centroids.
        """
        if self.max_centroids is None:
            nsamples = int(max(2 * np.sqrt(ra_dec.shape[0]), self.initial_sample_size * self.ncenters))
        else:
            nsamples = int(max(2 * np.sqrt(ra_dec.shape[0]), self.initial_sample_size * self.max_centroids))

        sample_key, center_key = jr.split(key, 2)
        if nsamples > ra_dec.shape[0]:
            raise ValueError(
                'Requested centers size is too large for the number of samples. '
                'Consider increasing the nside or decreasing the number of centers (or max_centroids)'
            )

        ra_dec_samples = random_sample(sample_key, ra_dec, nsamples)
        if self.max_centroids is None:
            centroids_samples = random_sample(center_key, ra_dec, self.ncenters)
        else:
            centroids_samples = random_sample(center_key, ra_dec, self.max_centroids)

        return ra_dec_samples, centroids_samples

    def kmeans_init(self: Self, ra_dec: Array, centroids: Array) -> KMeansState:
        """Initialize the KMeans state.

        Args:
            ra_dec (Array): Array of RA and DEC coordinates.
            centroids (Array): Initial centroid coordinates.

        Returns:
            KMeansState: Initialized KMeans state.
        """
        return KMeansState(
            ra_dec=ra_dec,
            centroids=centroids,
            labels=jnp.zeros(ra_dec.shape[0], dtype=jnp.int32),
            mean_distance=jnp.inf,
            previous_mean_distance=0.0,
            count=0,
        )

    def fit(self: Self, ra_dec: Array, centroids: Array) -> KMeansState:
        """Run the KMeans clustering algorithm.

        Args:
            ra_dec (Array): Array of RA and DEC coordinates.
            centroids (Array): Initial centroid coordinates.

        Returns:
            KMeansState: Final state after clustering.
        """
        centroid_mask = jnp.arange(centroids.shape[0]) < self.ncenters

        def kmeans_step(
            carry: tuple[Array, Array, tuple[Array, Array, Array], KMeansState],
        ) -> tuple[Array, Array, tuple[Array, Array, Array], KMeansState]:
            """Perform a single step of the KMeans algorithm.

            Args:
                carry (tuple): Tuple containing current data and state.

            Returns:
                tuple: Updated data and state.
            """
            ra_dec, indices, XYZ, state = carry
            Xs, Ys, Zs = XYZ

            state = state._replace(previous_mean_distance=state.mean_distance)

            distances = cdist_radec(ra_dec, state.centroids)
            if self.max_centroids is not None:
                distances = jnp.where(centroid_mask[None, :], distances, jnp.inf)
            labels = distances.argmin(axis=1).astype(jnp.int32)

            distances = distances[indices, labels]
            mean_distance = distances.mean()

            def for_loop_body(center_indx: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
                """Update centroids in the for loop.

                Args:
                    center_indx (int): Index of the centroid.
                    carry (tuple): Current centroids and labels.

                Returns:
                    tuple: Updated centroids and labels.
                """
                centroids, labels = carry
                mask = jnp.where(labels == center_indx, 1, 0)
                masked_X = mask * Xs
                masked_Y = mask * Ys
                masked_Z = mask * Zs
                mean_X = masked_X.sum() / mask.sum()
                mean_Y = masked_Y.sum() / mask.sum()
                mean_Z = masked_Z.sum() / mask.sum()

                current_centroid = centroids[center_indx]
                cdistance = xyz2radec(mean_X, mean_Y, mean_Z)
                new_centroid = jnp.where(jnp.isfinite(cdistance), cdistance, current_centroid)

                return centroids.at[center_indx].set(new_centroid), labels

            new_centroids, _ = lax.fori_loop(0, self.ncenters, for_loop_body, (state.centroids, labels))
            new_state = KMeansState(
                ra_dec=ra_dec,
                centroids=new_centroids,
                labels=labels,
                mean_distance=mean_distance,
                previous_mean_distance=state.previous_mean_distance,
                count=state.count + 1,
            )

            return ra_dec, indices, XYZ, new_state

        def kmeans_continue_criteria(carry: tuple[Array, Array, tuple[Array, Array, Array], KMeansState]) -> Array:
            """Check convergence criteria for KMeans.

            Args:
                carry (tuple): Tuple containing current data and state.

            Returns:
                Array: Boolean array indicating whether to continue.
            """
            _, _, _, state = carry

            converged = ((1 - self.tol) * state.previous_mean_distance <= state.mean_distance) & (
                state.previous_mean_distance >= state.mean_distance
            )
            return (state.count < self.maxiter) & (~converged)

        XYZ = radec2xyz(ra_dec[:, 0], ra_dec[:, 1])
        indices = jnp.arange(ra_dec.shape[0])

        init_state = self.kmeans_init(ra_dec, centroids)

        _, _, _, final_state = lax.while_loop(kmeans_continue_criteria, kmeans_step, (ra_dec, indices, XYZ, init_state))

        return final_state


def cdist_radec(a1: Array, a2: Array) -> Array:
    """Compute pairwise spherical distances between two sets of points.

    Args:
        a1 (Array): First array of RA and DEC coordinates.
        a2 (Array): Second array of RA and DEC coordinates.

    Returns:
        Array: Pairwise spherical distances.
    """
    ra1 = a1[:, 0][:, newaxis]
    dec1 = a1[:, 1][:, newaxis]
    ra2 = a2[:, 0]
    dec2 = a2[:, 1]

    phi1, theta1 = deg2rad(ra1), _PIOVER2 - deg2rad(dec1)
    phi2, theta2 = deg2rad(ra2), _PIOVER2 - deg2rad(dec2)

    sintheta1, sintheta2 = sin(theta1), sin(theta2)
    x1, y1, z1 = sintheta1 * cos(phi1), sintheta1 * sin(phi1), cos(theta1)
    x2, y2, z2 = sintheta2 * cos(phi2), sintheta2 * sin(phi2), cos(theta2)

    costheta = x1 * x2 + y1 * y2 + z1 * z2
    costheta = jnp.clip(costheta, -1.0, 1.0)
    return arccos(costheta)


def random_sample(key: PRNGKeyArray, ra_dec: Array, nsamples: int) -> Array:
    """Randomly sample points from the RA/DEC array.

    Args:
        key (PRNGKeyArray): JAX random key.
        ra_dec (Array): Array of RA and DEC coordinates.
        nsamples (int): Number of samples to draw.

    Returns:
        Array: Randomly sampled RA/DEC points.
    """
    nra_dec = ra_dec.shape[0]
    indices = jr.choice(key, nra_dec, shape=(nsamples,), replace=False)
    return ra_dec[indices]


def xyz2radec(x: Array, y: Array, z: Array) -> Array:
    """Convert Cartesian coordinates to spherical (RA/DEC).

    Args:
        x (Array): X coordinate.
        y (Array): Y coordinate.
        z (Array): Z coordinate.

    Returns:
        Array: Array containing RA and DEC.
    """
    r = sqrt(x**2 + y**2 + z**2)
    theta = arccos(z / r)
    phi = arctan2(y, x)
    ra = rad2deg(phi)
    dec = rad2deg(_PIOVER2 - theta)
    ra = atbound1(ra, 0.0, 360.0)
    return jnp.array([ra, dec])


def radec2xyz(ra: Array, dec: Array) -> tuple[Array, Array, Array]:
    """Convert spherical coordinates (RA/DEC) to Cartesian.

    Args:
        ra (Array): Right Ascension.
        dec (Array): Declination.

    Returns:
        tuple[Array, Array, Array]: Cartesian coordinates (X, Y, Z).
    """
    phi, theta = deg2rad(ra), _PIOVER2 - deg2rad(dec)
    sintheta = sin(theta)
    return sintheta * cos(phi), sintheta * sin(phi), cos(theta)


def atbound1(longitude_in: Array, minval: float, maxval: float) -> Array:
    """Ensure longitude is within specified bounds.

    Args:
        longitude_in (Array): Input longitude values.
        minval (float): Minimum bound.
        maxval (float): Maximum bound.

    Returns:
        Array: Longitude values within specified bounds.
    """
    range_size = maxval - minval
    longitude = (longitude_in - minval) % range_size + minval
    return longitude


def kmeans_sample(
    key: PRNGKeyArray,
    ra_dec: Array,
    ncenters: int,
    max_centroids: Optional[int] = None,
    tol: float = _TOL_DEF,
    maxiter: int = _MAXITER_DEF,
    initial_sample_size: int = 3,
) -> KMeansState:
    """Perform KMeans clustering on RA/DEC data.

    Args:
        key (PRNGKeyArray): JAX random key.
        ra_dec (Array): Array of RA and DEC coordinates.
        ncenters (int): Number of clusters.
        max_centroids (Optional[int]): Maximum number of centroids.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        initial_sample_size (int): Initial sample size.

    Returns:
        KMeansState: Final state after clustering.
    """
    km = KMeans(ncenters, max_centroids, tol, maxiter, initial_sample_size=initial_sample_size)
    ra_dec_samples, centroids_samples = km.sample_initial(ra_dec, key)

    state = km.fit(ra_dec_samples, centroids_samples)
    state = km.fit(ra_dec, state.centroids)

    return state
