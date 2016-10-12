from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from .. import poppy_core
from .. import optics
from .. import zernike
from .. import wfe

NWAVES = 0.5
WAVELENGTH = 1e-6
RADIUS = 1.0
NPIX = 101
DIAM = 3.0

def test_ZernikeAberration(display=False):
    # verify that we can reproduce the same behavior as ThinLens
    # using ZernikeAberration
    pupil = optics.CircularAperture(radius=RADIUS)
    lens = optics.ThinLens(nwaves=NWAVES, reference_wavelength=WAVELENGTH, radius=RADIUS)
    tl_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=WAVELENGTH)
    tl_wave *= lens
    # The ThinLens class is a subclass of CircularAperture so it automatically applies
    # pupil aperture shape as well as the wavefront


    zern_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=WAVELENGTH)
    # need a negative sign in the following b/c of different sign conventions for
    # zernikes vs "positive" and "negative" lenses.
    zernike_lens = wfe.ZernikeWFE(
        coefficients=[0, 0, 0, -NWAVES * WAVELENGTH / (2 * np.sqrt(3))],
        radius=RADIUS
    )
    zern_wave *= zernike_lens
    # But the ZernikeWFE class does not include any aperture transmission shape, so
    # we have to apply that here separately:
    zern_wave *= pupil

    if display:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(zern_wave.intensity-tl_wave.intensity)
        plt.title("Intensity difference")
        plt.figure()
        plt.imshow(zern_wave.phase-tl_wave.phase)
        plt.title("Phase difference")
        plt.draw()
        plt.figure()
        plt.plot(tl_wave.intensity[50,:],color='black', label='Thin Lens intensity')
        plt.plot(tl_wave.phase[50,:], color='black', linestyle=':', label='Thin Lens phase')
        plt.plot(zern_wave.intensity[50,:],color='red', linestyle='--', label='Zernike intensity')
        plt.plot(zern_wave.phase[50,:], color='red', linestyle=':', label='Zernike phase')
        plt.legend(loc='upper right', frameon=False)

    assert np.allclose(zern_wave.intensity, tl_wave.intensity)

    # The Zernike class right now only has nonzero phase in pixels that are strictly
    # radius < 1, and doesn't include partially illuminated pixels beyond that. So to
    # get this test passing for now, we need to only compute the std dev for inside
    # r < 1.

    # FIXME: enhance Zernike class to provide appropriate OPD values for
    # partially-illuminated pixels at the boundary (this is nontrivial given
    # current implementation)
    y, x = zern_wave.coordinates()
    r = np.sqrt(y**2 + x**2)
    stddev = np.std((zern_wave.phase - tl_wave.phase)[r < RADIUS])

    assert stddev < 1e-16, ("ZernikeAberration disagrees with ThinLens! stddev {}".format(stddev))

def test_wavefront_or_meters_decorator():
    zernike_lens = wfe.ZernikeWFE(
        coefficients=[0, 0, 0, NWAVES * WAVELENGTH / (2 * np.sqrt(3))],
        radius=RADIUS
    )
    opd_waves_a = zernike_lens.get_opd(WAVELENGTH)
    opd_waves_b = zernike_lens.get_opd(poppy_core.Wavefront(wavelength=WAVELENGTH))

    stddev = np.std(opd_waves_a - opd_waves_b)
    assert stddev < 1e-16, "OPD map disagreement based on form of argument to get_opd!"

def test_zernike_get_opd():
    zernike_optic = wfe.ZernikeWFE(coefficients=[NWAVES * WAVELENGTH,], radius=RADIUS)
    opd_map = zernike_optic.get_opd(WAVELENGTH, units='meters')
    assert np.max(opd_map) == NWAVES * WAVELENGTH

    opd_map_waves = zernike_optic.get_opd(WAVELENGTH, units='waves')
    assert np.max(opd_map_waves) == NWAVES

def test_ParameterizedAberration():
    # verify that we can reproduce the same behavior as ZernikeAberration
    # using ParameterizedAberration
    NWAVES = 0.5
    WAVELENGTH = 1e-6
    RADIUS = 1.0

    pupil = optics.CircularAperture(radius=RADIUS)

    zern_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=1e-6)
    zernike_wfe = wfe.ZernikeWFE(
        coefficients=[0, 0, 2e-7, NWAVES * WAVELENGTH / (2 * np.sqrt(3)), 0, 3e-8],
        radius=RADIUS
    )
    zern_wave *= pupil
    zern_wave *= zernike_wfe

    parameterized_distortion = wfe.ParameterizedWFE(
        coefficients=[0, 0, 2e-7, NWAVES * WAVELENGTH / (2 * np.sqrt(3)), 0, 3e-8],
        basis_factory=zernike.zernike_basis,
        radius=RADIUS
    )

    pd_wave = poppy_core.Wavefront(npix=NPIX, diam=3.0, wavelength=1e-6)
    pd_wave *= pupil
    pd_wave *= parameterized_distortion

    stddev = np.std(pd_wave.phase - zern_wave.phase)

    assert stddev < 1e-16, ("ParameterizedAberration disagrees with "
                            "ZernikeAberration! stddev {}".format(stddev))
