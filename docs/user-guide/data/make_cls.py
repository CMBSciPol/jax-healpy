"""Compute the CMB power spectra used in the user guide with CAMB.

Planck 2018 best-fit LCDM parameters (TT,TE,EE+lowE+lensing, Planck 2018 VI, Table 2),
with a tensor-to-scalar ratio r = 0.01, lensed spectra.

Usage (from this directory): uv run --with camb python make_cls.py cmb_cls.txt
"""

import sys

import numpy as np

LMAX = 767


def main(filename: str) -> None:
    import camb

    params = camb.set_params(
        H0=67.36,
        ombh2=0.02237,
        omch2=0.1200,
        tau=0.0544,
        As=2.100e-9,
        ns=0.9649,
        r=0.01,
        lmax=LMAX + 500,
        WantTensors=True,
    )
    results = camb.get_results(params)
    cls = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True, lmax=LMAX)['total']  # TT, EE, BB, TE
    ell = np.arange(LMAX + 1)
    header = (
        'Lensed CMB power spectra C_ell [muK^2] computed with CAMB '
        f'{camb.__version__} for the Planck 2018 best-fit LCDM\n'
        'parameters (H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544, As=2.100e-9, ns=0.9649) and r=0.01.\n'
        'Columns: ell, TT, EE, BB, TE'
    )
    np.savetxt(filename, np.column_stack([ell, cls]), header=header, fmt=['%d'] + 4 * ['%.6e'])


if __name__ == '__main__':
    main(sys.argv[1])
