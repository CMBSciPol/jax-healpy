"""Benchmark of the spherical harmonic transforms.

The transforms of jax-healpy are timed for each transform backend (the ``method`` argument of the
functions), on the device that JAX uses, and those of healpy on the CPU, as a reference:

    python benchmarks/sht.py --nside 64 256 1024 --methods healpy jax ducc0

For each case, the script reports the time of the first call, which includes the compilation, the
minimum and median times of the following calls, the maximum difference with healpy and, on a GPU,
the peak memory of the device. The methods that the installed jax-healpy or the device do not
support are reported as failed.

The transforms can also be timed under ``jax.vmap``, for several batch sizes, and under ``jax.grad``:

    python benchmarks/sht.py --nside 64 256 --methods jax ducc0 --batch 4 16 --grad

The gradient is that of half the squared norm of the transform, with respect to its input. Its
error is relative: it is that of the product of the gradient by the input, which is the squared
norm of the transform, taken from healpy.

The peak memory is that of the device since the start of the process: the figure of a case is not
lower than those of the cases that ran before it. To measure the memory of a single case, run it
alone, with ``--nside`` and ``--cases``.

healpy is timed first, before JAX is imported: once JAX has run a computation, its threads compete
with those of healpy and inflate its timings.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

CASES = ['alm2map_T', 'alm2map_TEB', 'map2alm_T', 'map2alm_IQU']
HEADER = (
    f'{"nside":>6s}  {"case":<18s} {"method":<9s} {"first call":>10s} {"min":>10s} {"median":>10s} '
    f'{"max error":>10s} {"peak mem":>10s}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--nside', type=int, nargs='+', default=[64, 128, 256], help='nside values (default: 64 128 256)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['healpy', 'jax', 'ducc0'],
        help="transform backends: 'healpy' for healpy itself, and the methods of jax-healpy, 'jax', 'jax_cuda' "
        "and 'ducc0' (default: healpy jax ducc0)",
    )
    parser.add_argument('--cases', nargs='+', default=CASES, choices=CASES, help='transforms to time (default: all)')
    parser.add_argument('--precision', type=int, choices=[32, 64], default=64, help='precision of JAX (default: 64)')
    parser.add_argument('--iter', type=int, default=3, help='iterations of map2alm (default: 3)')
    parser.add_argument('--lmax-factor', type=int, default=3, help='lmax = factor * nside - 1 (default: 3)')
    parser.add_argument(
        '--batch',
        type=int,
        nargs='+',
        default=[],
        help='also time the transforms vmapped over batches of these sizes',
    )
    parser.add_argument(
        '--grad',
        action='store_true',
        help='also time the gradients, with respect to the input, of half the squared norm of the transforms',
    )
    parser.add_argument('--min-time', type=float, default=1.0, help='minimum time spent per case, in seconds')
    parser.add_argument('--max-repeat', type=int, default=50, help='maximum number of timed calls per case')
    parser.add_argument(
        '--max-call-time',
        type=float,
        default=120.0,
        help='a case is skipped when its calls are expected to take longer, in seconds, given the time at the '
        'previous nside and a cost growing as nside^3 (default: 120)',
    )
    parser.add_argument('--output', help='file in which the results are written in JSON')
    return parser.parse_args()


def time_calls(func: Callable[[], Any], min_time: float, max_repeat: int) -> tuple[Any, float, float, float]:
    """Return the output of ``func``, the time of its first call, and the min and median of the next ones."""
    start = time.perf_counter()
    out = func()
    first = time.perf_counter() - start
    times = []
    while len(times) < max_repeat and (not times or sum(times) < min_time):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return out, first, min(times), statistics.median(times)


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return '-'
    if seconds < 1e-3:
        return f'{seconds * 1e6:.0f} us'
    if seconds < 1:
        return f'{seconds * 1e3:.3g} ms'
    return f'{seconds:.3g} s'


def format_memory(size: int | None) -> str:
    if size is None:
        return '-'
    if size < 2**30:
        return f'{size / 2**20:.0f} MiB'
    return f'{size / 2**30:.3g} GiB'


def cpu_name() -> str:
    try:
        with open('/proc/cpuinfo') as file:
            for line in file:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'


class Report:
    def __init__(self) -> None:
        self.records = []

    def add(self, nside: int, case: str, method: str, **values: Any) -> None:
        record = {'nside': nside, 'case': case, 'method': method, **values}
        self.records.append(record)
        for status in ('skipped', 'failed'):
            if status in record:
                print(f'{nside:6d}  {case:<18s} {method:<9s} {status}: {record[status]}', flush=True)
                return
        error = '-' if record.get('max_error') is None else f'{record["max_error"]:.1e}'
        print(
            f'{nside:6d}  {case:<18s} {method:<9s} {format_time(record.get("first_call")):>10s} '
            f'{format_time(record["min"]):>10s} {format_time(record["median"]):>10s} {error:>10s} '
            f'{format_memory(record.get("peak_memory")):>10s}',
            flush=True,
        )


def healpy_transforms(maps: np.ndarray, alms: np.ndarray, nside: int, lmax: int, n_iter: int) -> dict[str, Callable]:
    import healpy

    return {
        'alm2map_T': partial(healpy.alm2map, alms[0], nside, lmax=lmax),
        'alm2map_TEB': partial(healpy.alm2map, alms, nside, lmax=lmax),
        'map2alm_T': partial(healpy.map2alm, maps[0], lmax=lmax, iter=n_iter),
        'map2alm_IQU': partial(healpy.map2alm, maps, lmax=lmax, iter=n_iter),
    }


def jax_healpy_transforms(
    maps: np.ndarray,
    alms: np.ndarray,
    nside: int,
    lmax: int,
    n_iter: int,
    method: str,
    precision: int,
    batch: list[int],
    grad: bool,
) -> dict[str, Callable[[], Callable]]:
    """The transforms of jax-healpy, each one as a function that returns the transform to time.

    The transform is bound to its input, which is on the device. The input of a batched transform is
    only allocated when the transform is requested, so that it does not weigh on the memory of the
    other cases.
    """
    import jax

    import jax_healpy as jhp

    real = np.float64 if precision == 64 else np.float32
    maps = jax.device_put(maps.astype(real))
    alms = jax.device_put(alms.astype(np.result_type(real, np.complex64)))
    alm2map = partial(jhp.alm2map, nside=nside, lmax=lmax, healpy_ordering=True, method=method)
    map2alm = partial(jhp.map2alm, lmax=lmax, iter=n_iter, healpy_ordering=True, method=method)
    cases = {
        'alm2map_T': (alm2map, alms[0]),
        'alm2map_TEB': (alm2map, alms),
        'map2alm_T': (map2alm, maps[0]),
        'map2alm_IQU': (map2alm, maps),
    }

    def bind(function: Callable, x: Any, size: int = 0) -> Callable:
        if size:
            x = jax.block_until_ready(jax.numpy.stack([x] * size))
        return partial(call_and_wait, partial(function, x))

    transforms = {case: partial(bind, function, x) for case, (function, x) in cases.items()}
    for size in batch:
        for case, (function, x) in cases.items():
            transforms[f'{case} x{size}'] = partial(bind, jax.jit(jax.vmap(function)), x, size)
    if grad:
        for case, (function, x) in cases.items():
            transforms[f'{case} grad'] = partial(bind, jax.jit(jax.grad(partial(half_squared_norm, function))), x)
    return transforms


def half_squared_norm(function: Callable, x: Any) -> Any:
    import jax.numpy as jnp

    y = function(x)
    return 0.5 * jnp.sum(jnp.real(y * jnp.conj(y)))


def call_and_wait(func: Callable[[], Any]) -> Any:
    import jax

    return jax.block_until_ready(func())


def gradient_error(gradient: np.ndarray, x: np.ndarray, expected: np.ndarray) -> float:
    """Return the relative error of the gradient of half the squared norm of a transform.

    The transform is linear, so that the gradient g at x satisfies Re <g, x> = |y|², where y is the
    transform of x, taken from healpy.
    """
    squared_norm = float(np.sum(np.abs(expected) ** 2))
    return abs(float(np.sum(gradient * x).real) - squared_norm) / squared_norm


def peak_memory() -> int | None:
    """Return the peak memory of the device since the start of the process, in bytes, if it is tracked."""
    import jax

    stats = jax.devices()[0].memory_stats()
    return stats.get('peak_bytes_in_use') if stats else None


def main() -> None:
    args = parse_args()
    import healpy

    report = Report()
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    print(f'{cpu_name()}, {cpu_count} CPUs available to the process')
    print(f'healpy {healpy.__version__}, numpy {np.__version__}, OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS")}')
    print(
        f'lmax = {args.lmax_factor} nside - 1, iter = {args.iter} for map2alm, precision of JAX: {args.precision} bits'
    )
    print()
    print(HEADER)

    # ----- healpy: the inputs, the reference results and the reference timings
    inputs = {}
    for nside in args.nside:
        lmax = args.lmax_factor * nside - 1
        maps = np.random.default_rng(nside).standard_normal((3, 12 * nside**2))
        alms = healpy.map2alm(maps, lmax=lmax, iter=0)
        transforms = healpy_transforms(maps, alms, nside, lmax, args.iter)
        expected = {}
        for case in args.cases:
            expected[case], _, tmin, tmedian = time_calls(transforms[case], args.min_time, args.max_repeat)
            if 'healpy' in args.methods:
                report.add(nside, case, 'healpy', min=tmin, median=tmedian)
        inputs[nside] = (maps, alms, expected)

    # ----- jax-healpy, for each method
    methods = [method for method in args.methods if method != 'healpy']
    environment = {}
    if methods:
        import jax

        jax.config.update('jax_enable_x64', args.precision == 64)
        device = jax.devices()[0]
        environment['device'] = f'{device.device_kind} ({device.platform})'
        print()
        print(
            f'jax-healpy {version("jax-healpy")}, jax {jax.__version__}, s2fft {version("s2fft")}, '
            f'ducc0 {version("ducc0")}, device: {environment["device"]}'
        )
        if 'ducc0' in methods and version('ducc0') != 'not installed':
            import ducc0

            print(f'ducc0 uses {ducc0.misc.thread_pool_size()} threads')
        print(HEADER)

    for method in methods:
        last = {}  # case -> (nside, time of a call)
        failed = {}  # case -> nside at which it failed
        unavailable = False
        for nside in args.nside:
            if unavailable:
                break
            lmax = args.lmax_factor * nside - 1
            maps, alms, expected = inputs[nside]

            # the cases whose calls would take too long, given their time at the previous nside
            too_long = {}
            for case, (last_nside, last_time) in last.items():
                expected_time = last_time * (nside / last_nside) ** 3
                if expected_time > args.max_call_time:
                    too_long[case] = f'a call would take about {expected_time:.0f} s'
            if last and len(too_long) == len(last):
                for case, reason in too_long.items():
                    report.add(nside, case, method, skipped=reason)
                continue

            transforms = jax_healpy_transforms(
                maps, alms, nside, lmax, args.iter, method, args.precision, args.batch, args.grad
            )
            for case, get_transform in transforms.items():
                if case.split()[0] not in args.cases:
                    continue
                if case in too_long:
                    report.add(nside, case, method, skipped=too_long[case])
                    continue
                if case in failed:
                    # a larger transform would fail as well, for instance for lack of memory
                    report.add(nside, case, method, skipped=f'it failed at nside={failed[case]}')
                    continue
                try:
                    out, first, tmin, tmedian = time_calls(get_transform(), args.min_time, args.max_repeat)
                except Exception as exc:  # noqa: BLE001 (whatever fails, the other cases are still worth timing)
                    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ''
                    report.add(nside, case, method, failed=f'{type(exc).__name__}: {message}'[:160])
                    if not last:
                        # the first transform of the method failed: the method is not available
                        print(f'{"":6s}  the method {method!r} is not timed further')
                        unavailable = True
                        break
                    failed[case] = nside
                    continue
                name = case.split()[0]
                if case.endswith(' grad'):
                    x = (alms if name.startswith('alm2map') else maps)[0 if name.endswith('_T') else ...]
                    error = gradient_error(np.asarray(out), x, expected[name])
                else:
                    if ' x' in case:
                        out = out[0]
                    error = float(np.max(np.abs(np.asarray(out) - expected[name])))
                report.add(
                    nside,
                    case,
                    method,
                    first_call=first,
                    min=tmin,
                    median=tmedian,
                    max_error=error,
                    peak_memory=peak_memory(),
                )
                last[case] = (nside, tmin)

    if args.output:
        environment |= {
            'precision': args.precision,
            'iter': args.iter,
            'lmax_factor': args.lmax_factor,
            'cpu_count': cpu_count,
            'processor': cpu_name(),
            'versions': {name: version(name) for name in ('jax-healpy', 'jax', 'jaxlib', 's2fft', 'ducc0', 'healpy')},
        }
        with open(args.output, 'w') as file:
            json.dump({'environment': environment, 'results': report.records}, file, indent=2)
        print(f'\nResults written in {args.output}')


if __name__ == '__main__':
    main()
