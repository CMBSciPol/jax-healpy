import os
import platform
import re
import subprocess
import timeit
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable, Literal

import healpy as hp
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import yaml
from jaxtyping import ArrayLike
from matplotlib.ticker import ScalarFormatter

BENCH_PATH = Path(__file__).parent / 'results'
BENCHMARKED_FUNCS = [
    'ang2vec',
    'vec2ang',
    'ang2pix',
    'pix2ang',
    'vec2pix',
    'pix2vec',
    'ring2nest',
    'nest2ring',
    'pix2xyf',
    'xyf2pix',
    'reorder',
]
CHART_PATH_NAME = 'chart-{style}-n{n}.png'

# TODO: use those when typer supports Literals
LibraryType = Literal['healpy', 'jax-healpy']
PrecisionType = Literal['32', '64']

app = typer.Typer()


@dataclass(frozen=True)
class BenchmarkResult:
    library: str
    version: str
    processor: str
    processor_type: str
    precision: int
    n: int
    execution_times_s: dict[str, float]


def bench_it(
    library: str,
    func_name: str,
    nside: int,
    n: int,
    precision: str,
    rng: np.random.Generator,
) -> float:
    if precision == '32':
        dtype = np.dtype('float32')
    elif precision == '64':
        dtype = np.dtype('float64')
    else:
        raise ValueError(f'Invalid precision {precision}')

    args = _get_args(library, func_name, nside, n, dtype, rng)
    func = _get_func(library, func_name, *args)
    with jax.experimental.enable_x64(precision == '64'):
        return time_it(func)


def _get_args(
    library: str,
    func_name: str,
    nside: int,
    n: int,
    dtype: np.dtype,
    rng: np.random.Generator,
) -> tuple[ArrayLike]:
    if func_name in {'ang2vec', 'ang2pix'}:
        theta = rng.uniform(0, np.pi, size=n).astype(dtype)
        phi = rng.uniform(0, 2 * np.pi, size=n).astype(dtype)
        if func_name == 'ang2vec':
            args = (theta, phi)
        else:
            args = (nside, theta, phi)

    elif func_name in {'vec2ang', 'vec2pix'}:
        vec = rng.uniform(-np.pi, np.pi, size=(3, n))
        vec /= np.sqrt(np.sum(vec**2, axis=0))
        vec = vec.astype(dtype)
        if func_name == 'vec2ang':
            args = (vec.T.copy(),)
        else:
            args = (nside, vec[0], vec[1], vec[2])

    elif func_name in {'pix2ang', 'pix2vec', 'pix2xyf', 'ring2nest', 'nest2ring'}:
        pixels = rng.uniform(0, hp.nside2npix(nside), size=n).astype(int)
        args = (nside, pixels)

    elif func_name == 'xyf2pix':
        x = rng.uniform(0, nside, size=n).astype(int)
        y = rng.uniform(0, nside, size=n).astype(int)
        f = rng.uniform(0, 12, size=n).astype(int)
        args = (nside, x, y, f)

    elif func_name == 'reorder':
        npix = hp.nside2npix(nside)
        map_in = rng.uniform(size=npix)
        args = (map_in,)

    else:
        raise NotImplementedError

    if library == 'jax-healpy':
        args = tuple(jax.device_put(_) if isinstance(_, np.ndarray) else _ for _ in args)

    return args


def _get_func(library: str, func_name: str, *args: Any):
    if library == 'jax-healpy':
        import jax_healpy as module
    else:
        import healpy as module

    func = getattr(module, func_name)
    if library == 'healpy':
        if func_name == 'reorder':
            func_ = lambda: func(*args, r2n=True)  # noqa: E731
        else:
            func_ = lambda: func(*args)  # noqa: E731
    else:
        if func_name in {'pix2ang', 'vec2ang'}:

            def func_() -> None:
                theta, phi = func(*args)
                theta.block_until_ready()
                phi.block_until_ready()

        elif func_name == 'pix2xyf':

            def func_() -> None:
                x, y, f = func(*args)
                x.block_until_ready()
                y.block_until_ready()
                f.block_until_ready()

        elif func_name == 'reorder':

            def func_() -> None:
                func(*args, r2n=True).block_until_ready()

        else:

            def func_() -> None:
                func(*args).block_until_ready()

        func_()  # discard first call, which includes compilation

    return func_


def time_it(func: Callable[[], None]) -> float:
    timer = timeit.Timer(func)
    number, _ = timer.autorange()
    execution_time = min(timer.repeat(number=number)) / number
    return execution_time


@app.command()
def run(
    library: str,
    nside: int = 512,
    n: int = 10_000_000,
    precision: str = '64',
) -> None:
    if library == 'jax-healpy':
        version = f'jax({jax.__version__})'
        device = jax.devices()[0]
        processor = device.device_kind
        if processor == 'cpu':
            processor = _get_cpu_name()
            processor_type = 'cpu'
        else:
            processor_type = 'gpu'
    elif library == 'healpy':
        version = hp.__version__
        processor = _get_cpu_name()
        processor_type = 'cpu'
    else:
        raise ValueError(f'Invalid library {library}')

    rng = np.random.default_rng(0)
    execution_times = {}

    print(f'Running {library}...')
    for func_name in BENCHMARKED_FUNCS:
        execution_time = bench_it(library, func_name, nside, n, precision, rng)
        execution_times[func_name] = execution_time

    result = BenchmarkResult(
        library=library,
        version=version,
        processor=processor,
        processor_type=processor_type,
        precision=int(precision),
        n=n,
        execution_times_s=execution_times,
    )
    pprint(result)

    chars = '[] '
    formatted_processor = processor.lower().rstrip(chars).replace('(c)', '').replace('(tm)', '')
    filename = f'{library}-{formatted_processor}-precision{precision}-n{n}.yaml'
    for char in chars:
        filename = filename.replace(char, '-')
    BENCH_PATH.mkdir(parents=True, exist_ok=True)
    with open(BENCH_PATH / filename, 'w') as f:
        yaml.dump(asdict(result), f)


def _get_cpu_name():
    if platform.system() == 'Darwin':
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = 'sysctl -n machdep.cpu.brand_string'
        return subprocess.check_output(command).strip()
    elif platform.system() == 'Linux':
        command = 'cat /proc/cpuinfo'
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split('\n'):
            if 'model name' in line:
                return re.sub('.*model name.*:', '', line, 1).strip()
    return platform.processor()


@app.command()
def collect():
    files = BENCH_PATH.glob('*-n*.yaml')
    ns = {}
    for file in files:
        result_as_dict = yaml.safe_load(file.read_text())
        result = BenchmarkResult(**result_as_dict)
        ns.setdefault(result.n, []).append(result)

    for n, results in ns.items():
        results = sorted(
            results,
            key=lambda _: sum(_.execution_times_s.values()) / len(BENCHMARKED_FUNCS),
        )
        export_chart(n, results, 'default')
        export_chart(n, results, 'dark_background')


def export_chart(n: int, results: Iterable[BenchmarkResult], style: str):
    chart_path = BENCH_PATH / CHART_PATH_NAME.format(style=style.replace('_', ''), n=n)
    plt.style.use(style)
    # I've used pandas because I was not able to do what I wanted with matplotlib alone
    df = pd.DataFrame(
        [
            [f'{res.library}[{res.processor_type}{res.precision}]']
            + [res.execution_times_s[_] for _ in BENCHMARKED_FUNCS]
            for res in results
        ],
        columns=['library'] + BENCHMARKED_FUNCS,
    )
    for func in BENCHMARKED_FUNCS:
        df[func] /= df[func][0]
    pprint(df)

    ax = df.plot.bar(x='library', title='Benchmark (lower is better)', rot=60)
    ax.legend(framealpha=0, loc='upper left')
    plt.yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(chart_path, transparent=True)


if __name__ == '__main__':
    app()
