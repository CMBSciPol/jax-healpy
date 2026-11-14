"""Compute the CMB power spectra used in the user guide with CAMB.

Planck 2018 best-fit LCDM parameters (TT,TE,EE+lowE+lensing, Planck 2018 VI, Table 2),
with a tensor-to-scalar ratio r = 0.01, lensed spectra. Writes in this directory:

- cmb_cls.txt: the TT, EE, BB and TE spectra;
- bb_templates.txt: the two contributions to the BB spectrum, from lensing and from the
  primordial gravitational waves for r = 1.

Usage (from this directory): uv run --with camb python make_cls.py
"""

from pathlib import Path

import numpy as np

LMAX = 767
R = 0.01
PARAMETERS = 'H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544, As=2.100e-9, ns=0.9649'


def main(directory: Path) -> None:
    import camb

    params = camb.set_params(
        H0=67.36,
        ombh2=0.02237,
        omch2=0.1200,
        tau=0.0544,
        As=2.100e-9,
        ns=0.9649,
        r=R,
        lmax=LMAX + 500,
        WantTensors=True,
    )
    results = camb.get_results(params)
    spectra = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True, lmax=LMAX)
    ell = np.arange(LMAX + 1)
    origin = f'computed with CAMB {camb.__version__} for the Planck 2018 best-fit LCDM\nparameters ({PARAMETERS})'

    header = f'Lensed CMB power spectra C_ell [muK^2] {origin} and r={R}.\nColumns: ell, TT, EE, BB, TE'
    cls = spectra['total']  # TT, EE, BB, TE
    np.savetxt(directory / 'cmb_cls.txt', np.column_stack([ell, cls]), header=header, fmt=['%d'] + 4 * ['%.6e'])

    # The tensor spectrum is computed for r = 0.01 and rescaled, since the tensor spectral
    # index n_t = -r/8 depends on r: BB(r) = BB_lensing + r * BB_tensor holds for r << 1.
    header = (
        f'BB power spectrum templates C_ell [muK^2] {origin}.\n'
        f'BB_tensor is the tensor contribution for r={R} divided by {R}.\n'
        'Columns: ell, BB_lensing, BB_tensor'
    )
    templates = np.column_stack([spectra['lensed_scalar'][:, 2], spectra['tensor'][:, 2] / R])
    np.savetxt(
        directory / 'bb_templates.txt', np.column_stack([ell, templates]), header=header, fmt=['%d'] + 2 * ['%.6e']
    )


if __name__ == '__main__':
    main(Path(__file__).parent)
