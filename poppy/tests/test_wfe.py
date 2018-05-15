from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import astropy.units as u

from .. import poppy_core
from .. import optics
from .. import zernike
from .. import wfe

NWAVES = 0.5
WAVELENGTH = 1e-6
RADIUS = 1.0
NPIX = 101
DIAM = 3.0
CN2 = 1e-14*u.m**(-2/3)
DZ = 50.0*u.m

def test_ZernikeAberration():
    # verify that we can reproduce the same behavior as ThinLens
    # using ZernikeAberration
    pupil = optics.CircularAperture(radius=RADIUS)
    lens = optics.ThinLens(nwaves=NWAVES, reference_wavelength=WAVELENGTH, radius=RADIUS)
    tl_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=WAVELENGTH)
    tl_wave *= pupil
    tl_wave *= lens

    zern_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=WAVELENGTH)
    # need a negative sign in the following b/c of different sign conventions for
    # zernikes vs "positive" and "negative" lenses.
    zernike_lens = wfe.ZernikeWFE(
        coefficients=[0, 0, 0, -NWAVES * WAVELENGTH / (2 * np.sqrt(3))],
        radius=RADIUS
    )
    zern_wave *= pupil
    zern_wave *= zernike_lens

    stddev = np.std(zern_wave.phase - tl_wave.phase)

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

def test_KolmogorovWFE_stats():
    # verify statistics of random numbers
    KolmogorovWFE = wfe.KolmogorovWFE(Cn2=CN2, dz=DZ)
    npix = 2048
    a = KolmogorovWFE.rand_turbulent(npix)
    b = KolmogorovWFE.rand_symmetrized(npix, 1)
    c = KolmogorovWFE.rand_symmetrized(npix, -1)
    
    assert(np.round(np.abs(np.mean(a)), 2) == np.round(0.0, 2))
    assert(np.round(np.var(a), 2) == np.round(1.0, 2))
    assert(np.round(np.mean(b), 2) == np.round(0.0, 2))
    assert(np.round(np.var(b), 2) == np.round(1.0, 2))
    assert(np.round(np.mean(c), 2) == np.round(0.0, 2))
    assert(np.round(np.var(c), 2) == np.round(1.0, 2))

def test_KolmogorovWFE_Cn2():
    # verifiy correct calculation of Cn2 from Fried parameter
    lam = WAVELENGTH*u.m
    dz = 50.0*u.m
    r0 = 0.185*(lam**2/CN2/dz)**(3.0/5.0) # analytical equation
    KolmogorovWFE = wfe.KolmogorovWFE(r0=r0, dz=dz)
    Cn2_test = KolmogorovWFE.get_Cn2(lam)
    
    assert(np.round(Cn2_test.value, 9) == np.round(CN2.value, 9))

def test_KolmogorovWFE_ps():
    # verifiy that first element of power spectrum is zero
    npix = 64
    wf = poppy_core.Wavefront(wavelength=WAVELENGTH*u.m,
                              npix=npix,
                              diam=3.0)
    
    KolmogorovWFE = wfe.KolmogorovWFE(Cn2=CN2, dz=DZ)
    
    ps1 = KolmogorovWFE.power_spectrum(wf, kind='Kolmogorov')
    ps2 = KolmogorovWFE.power_spectrum(wf, l0=1*u.cm, kind='Tatarski')
    ps3 = KolmogorovWFE.power_spectrum(wf, L0=10*u.m, l0=1*u.cm, kind='van Karman')
    ps4 = KolmogorovWFE.power_spectrum(wf, l0=1*u.cm, kind='Hill')
    
    assert(np.round(ps1[0,0].value, 9) == np.round(0.0, 9))
    assert(np.round(ps2[0,0].value, 9) == np.round(0.0, 9))
    assert(np.round(ps3[0,0].value, 9) == np.round(0.0, 9))
    assert(np.round(ps4[0,0].value, 9) == np.round(0.0, 9))

def test_KolmogorovWFE_correlation():
    # verify correlation of random numbers
    KolmogorovWFE = wfe.KolmogorovWFE(Cn2=CN2, dz=DZ)
    num_ensemble = 2000
    npix = 64
    
    average = np.zeros((npix, npix), dtype=complex)
    for j in range(num_ensemble):
        a = KolmogorovWFE.rand_turbulent(npix)
        for l in range(npix):
            for m in range(npix):
                average[l, m] += np.sum(a[:, l])*np.sum(np.conj(a[:, m]))/num_ensemble/npix
    
    for l in range(npix):
        for m in range(npix):
            if l == m:
                average[l, m] -= 1.0
    
    assert(np.max(np.abs(average.real)) < 0.1)
    assert(np.max(np.abs(average.imag)) < 0.1)