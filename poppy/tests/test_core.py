# Test functions for core poppy functionality
import os

import numpy as np
from astropy.io import fits
import pytest
try:
    import scipy
except ImportError:
    scipy = None

from .. import poppy_core
from .. import optics

####### Test Common Infrastructre #######

def check_wavefront(filename_or_hdulist, slice=0, ext=0, test='nearzero', comment=""):
    """ A helper routine to verify certain properties of a wavefront FITS file,
    as requested by some test routine. """
    if isinstance(filename_or_hdulist, str):
        hdulist = fits.open(filename_or_hdulist)
        filename = filename_or_hdulist
    elif isinstance(filename_or_hdulist, fits.HDUList):
        hdulist = filename_or_hdulist
        filename = 'input HDUlist'
    imstack = hdulist[ext].data
    im = imstack[slice,:,:]


    if test=='nearzero':
        return np.all(np.abs(im) < np.finfo(im.dtype).eps*10)
    elif test == 'is_real':
        #assumes output type = 'all'
        cplx_im = imstack[1,:,:] * np.exp(1j*imstack[2,:,:])
        return np.all( cplx_im.imag < np.finfo(im.dtype).eps*10)

wavelength=2e-6


######### Core tests functions #########

def test_basic_functionality():
    """ For one specific geometry, test that we get the expected value based on a prior reference
    calculation."""
    osys = poppy_core.OpticalSystem("test", oversample=1)
    pupil = optics.CircularAperture(radius=1)
    osys.addPupil(pupil) #function='Circle', radius=1)
    osys.addDetector(pixelscale=0.1, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux

    psf = osys.calcPSF(wavelength=1.0e-6)
    # we need to be a little careful here due to floating point math comparision equality issues... Can't just do a strict equality
    assert abs(psf[0].data.max() - 0.201) < 0.001


def test_input_wavefront_size():

    # if absolutely nothing is set then the default is 1024. 
    # the oversample parameter multiplies that *only* if padding
    # is applied during an FFT propagation; by default there's no effect
    # in the unpadded array.
    for oversamp in (1,2,4):
        osys = poppy_core.OpticalSystem("test", oversample=oversamp)
        #pupil = optics.CircularAperture(radius=1)
        wf = osys.inputWavefront()
        expected_shape = (1024,1024) if (wf.ispadded == False) else (1024*oversamp, 1024*oversamp)
        assert wf.shape == expected_shape, 'Wavefront is not the expected size: is {} expects {}'.format(wf.shape,  expected_shape)


    # test setting the size based on the npix parameter, with null optical system
    for size in [512, 1024, 2001]:
        osys = poppy_core.OpticalSystem("test", oversample=1, npix=size)
        #pupil = optics.CircularAperture(radius=1)
        wf = osys.inputWavefront()
        expected_shape = (size,size)
        assert wf.shape == expected_shape, 'Wavefront is not the expected size: is {} expects {}'.format(wf.shape,  expected_shape)

    # test setting the size based on the npix parameter, with a non-null optical system
    for size in [512, 1024, 2001]:
        osys = poppy_core.OpticalSystem("test", oversample=1, npix=size)
        osys.add_pupil(optics.CircularAperture(radius=1))
        wf = osys.inputWavefront()
        expected_shape = (size,size)
        assert wf.shape == expected_shape, 'Wavefront is not the expected size: is {} expects {}'.format(wf.shape,  expected_shape)


    # test setting the size based on an input optical element
    for npix in [512, 1024, 2001]:
        osys = poppy_core.OpticalSystem("test", oversample=1)
        pupil = optics.CircularAperture(radius=1)
        pupil_fits = pupil.toFITS(npix=npix)
        osys.add_pupil(transmission=pupil_fits)

        wf = osys.inputWavefront()
        expected_shape = (npix,npix)
        assert pupil_fits[0].data.shape == expected_shape, 'FITS array from optic element is not the expected size: is {} expects {}'.format(pupil_fits[0].data.shape,  expected_shape)
        assert wf.shape == expected_shape, 'Wavefront is not the expected size: is {} expects {}'.format(wf.shape,  expected_shape)



def test_CircularAperture_Airy(display=False):
    """ Compare analytic 2d Airy function with the results of a POPPY
    numerical calculation of the PSF for a circular aperture.
    
    Note that we expect very close but not precisely perfect agreement due to
    the quantization of the POPPY PSF relative to a perfect geometric circle.
    """
    
    from ..misc import airy_2d
    # Analytic PSF for 1 meter diameter aperture
    analytic = airy_2d(diameter=1)
    analytic /= analytic.sum() # for comparison with poppy outputs normalized to total=1


    # Numeric PSF for 1 meter diameter aperture
    osys = poppy_core.OpticalSystem()
    pupil = optics.CircularAperture(radius=0.5)
    osys.addPupil(pupil)
    osys.addDetector(pixelscale=0.010,fov_pixels=512, oversample=1)
    numeric = osys.calcPSF(wavelength=1.0e-6, display=False)

    # Comparison
    difference = numeric[0].data-analytic
    assert np.all(np.abs(difference) < 3e-5)

    if display:
        from .. import utils
        #comparison of the two
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=1e-6, vmax=1e-2)

        pl.figure(figsize=(15,5))
        pl.subplot(141)
        ax1=pl.imshow(analytic, norm=norm)
        pl.title("Analytic")
        pl.subplot(142)
        #ax2=pl.imshow(numeric[0].data, norm=norm)
        utils.display_PSF(numeric, vmin=1e-6, vmax=1e-2, colorbar=False)
        pl.title("Numeric")
        pl.subplot(143)
        ax2=pl.imshow(numeric[0].data-analytic, norm=norm)
        pl.title("Difference N-A")
        pl.subplot(144)
        ax2=pl.imshow(np.abs(numeric[0].data-analytic) < 3e-5)
        pl.title("Difference <1e-5")


def test_multiwavelength_opticalsystem():
    """
    Tests the ability to just provide wavelengths, weights directly
    """
    wavelengths = [2.0e-6, 2.1e-6, 2.2e-6]
    weights = [0.3, 0.5, 0.2]

    osys = poppy_core.OpticalSystem("test")
    pupil = optics.CircularAperture(radius=1)
    osys.addPupil(pupil) #function='Circle', radius=1)
    osys.addDetector(pixelscale=0.1, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux


    psf = osys.calcPSF(wavelength=wavelengths, weight=weights)
    assert psf[0].header['NWAVES'] == len(wavelengths), \
        "Number of wavelengths in PSF header does not match number requested"

    # Check weighted sum
    output = np.zeros_like(psf[0].data)
    for wavelength, weight in zip(wavelengths, weights):
        output += weight * osys.calcPSF(wavelength=wavelength)[0].data

    assert np.allclose(psf[0].data, output), \
        "Multi-wavelength PSF does not match weighted sum of individual wavelength PSFs"

    return psf


def test_normalization():
    """ Test that we can compute a PSF and get the desired flux, 
    depending on the normalization """
    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=6.5/2)
    osys.addPupil(pupil) #function='Circle', radius=6.5/2)
    osys.addDetector(pixelscale=0.01, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux

    from .. import conf
    conf.enable_flux_tests  = True

    # we need to be a little careful here due to floating point math comparision equality issues... Can't just do a strict equality

    # this should be very very close to one
    psf_last = osys.calcPSF(wavelength=1.0e-6, normalize='last')
    assert abs(psf_last[0].data.sum() - 1) < 0.01

    # this should be a little further but still pretty close
    psf_first = osys.calcPSF(wavelength=1.0e-6, normalize='first')
    assert abs(psf_first[0].data.sum() - 1) < 0.01
    assert abs(psf_first[0].data.sum() - 1) > 0.0001

    # for the simple optical system above, the 'first' and 'exit_pupil' options should be equivalent:
    psf_exit_pupil = osys.calcPSF(wavelength=1.0e-6, normalize='exit_pupil')
    assert (psf_exit_pupil[0].data.sum() - 1) < 1e-9
    assert np.abs( psf_exit_pupil[0].data - psf_first[0].data).max()  < 1e-10


    # and if we make an pupil stop with half the radius we should get 1/4 the light if normalized to 'first'
    # but normalized to 1 if normalized to last_pupil
    osys2 = poppy_core.OpticalSystem("test", oversample=2)
    osys2.addPupil(  optics.CircularAperture(radius=6.5/2) )
    osys2.addPupil(  optics.CircularAperture(radius=6.5/2/2) )
    osys2.addDetector(pixelscale=0.01, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux

    psf_small_pupil_first = osys2.calcPSF(wavelength=1.0e-6, normalize='first')
    psf_small_pupil_exit  = osys2.calcPSF(wavelength=1.0e-6, normalize='exit_pupil')
    psf_small_pupil_last  = osys2.calcPSF(wavelength=1.0e-6, normalize='last')
    # normalized for the output to 1 we should of course get 1
    assert abs(psf_small_pupil_last[0].data.sum() - 1) < 1e-9
    # normalized to the exit pupil we should get near but not exactly 1 (due to finite FOV)
    assert abs(psf_small_pupil_exit[0].data.sum() - 1) < 0.01
    assert abs(psf_small_pupil_exit[0].data.sum() - 1) > 0.0001
    # normalized to the entrance pupil we should get very close to 4x over the exit pupil one
    # (not totally sure why the agreement isn't closer - presumably due to finite sampling quantization
    #  of the discretized arrays)
    assert abs(psf_small_pupil_first[0].data.sum() *4 - psf_small_pupil_exit[0].data.sum()) < 1e-3


def test_fov_size_pixels():
    """ Test the PSF field of view size is as requested, in pixels for a square aperture"""

    # square FOV
    for size in (100, 137, 256):
        osys = poppy_core.OpticalSystem("test", oversample=2)
        pupil = optics.CircularAperture(radius=6.5/2)
        osys.addPupil(pupil)
        osys.addDetector(pixelscale=0.1, fov_pixels=size, oversample=1)

        psf = osys.calcPSF(wavelength=1e-6)

        assert psf[0].data.shape[0] == size
        assert psf[0].data.shape[1] == size


    # rectangular FOV
    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=6.5/2)
    osys.addPupil(pupil)
    osys.addDetector(pixelscale=0.1, fov_pixels=(100,200) , oversample=1)

    psf = osys.calcPSF(wavelength=1e-6)

    assert psf[0].data.shape[0] == 100
    assert psf[0].data.shape[1] == 200



###    EXPECTED TO FAIL RIGHT NOW - Offsets don't work yet.
###    See https://github.com/mperrin/poppy/issues/40
import pytest
@pytest.mark.xfail
def test_fov_offset(scale=1.0):
    """ Test offsetting the field of view of a Detector
    This is distinct from offsetting the source! """
    from ..utils import measure_centroid

    size=100
    pixscale = 0.1

    # A PSF created on-axis with no offset
    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=6.5/2)
    osys.addPupil(pupil)
    osys.addDetector(pixelscale=pixscale, fov_pixels=size, oversample=1)
    psf1 = osys.calcPSF()
    # The measured centroid should put it in the center of the array
    cent1 = measure_centroid(psf1, relativeto='center')
    poppy_core._log.info("On-axis PSF (no offset) centroid is:" + str(cent1))
    assert(abs(cent1[0]-0) < 1e-5)
    assert(abs(cent1[1]-0) < 1e-5)

    # Now create an equivalent PSF but offset the axes by 1 pixel in the first axis
    osys2 = poppy_core.OpticalSystem("test", oversample=2)
    osys2.addPupil(pupil)
    osys2.addDetector(pixelscale=pixscale, fov_pixels=size, oversample=1, offset=(pixscale*scale,0))
    psf2 = osys2.calcPSF()
    # Its centroid shouldbe offset by a pixel
    poppy_core._log.info("Offset PSF (by ({0},0) pixels ) centroid is: {1}".format(str(scale), str(cent1)))
    cent2 = measure_centroid(psf2, relativeto='center')
    assert(abs(cent2[0]-scale) < 1e-5)
    assert(abs(cent2[1]-0) < 1e-5)


    # and do the same thing in the second axis (after the above works)



def test_inverse_MFT():
    """
    Verify basic functionality of the Inverse MFT code. 
    """

    fov_arcsec  = 5.0

    test_ap = optics.ParityTestAperture(radius=6.5/2, pad_factor=1.5)

    osys = poppy_core.OpticalSystem("test", oversample=4)
    osys.addPupil(test_ap)
    osys.addDetector(pixelscale=0.010, fov_arcsec=fov_arcsec) # use a large FOV so we grab essentially all the light and conserve flux
    psf1 = osys.calcPSF(wavelength=wavelength, normalize='first', display_intermediates=False)

    #osys.addPupil(test_ap)
    osys.addPupil() # this will force an inverse MFT
    osys.addDetector(pixelscale=0.010, fov_arcsec=fov_arcsec) # use a large FOV so we grab essentially all the light and conserve flux
    #plt.clf()
    psf = osys.calcPSF(wavelength=wavelength, normalize='first', display_intermediates=False)

    # the intermediate PSF (after one MFT) should be essentially identical to the
    # final PSF (after an MFT, inverse MFT, and another MFT):
    assert(   np.abs(psf1[0].data - psf[0].data).max()  < 1e-7 )


@pytest.mark.skipif(
    (scipy is None),
    reason='No SciPy installed'
)
def test_optic_resizing():
    '''
    Tests the rescaling functionality of OpticalElement.getPhasor(),
    by first creating an optic with a small pixel scale and then
    creating an optic with a large pixel scale, and checking the returned
    phasor of each has the dimensions of the input wavefront.
    '''

    # diameter 1 meter, pixel scale 2 mm
    inputwf = poppy_core.Wavefront(diam=1.0, npix=500)

    # Test rescaling from finer scales: diameter 1 meter, pixel scale 1 mm 
    test_optic_small=fits.HDUList([fits.PrimaryHDU(np.zeros([1000,1000]))])
    test_optic_small[0].header["PUPLSCAL"]=.001
    test_optic_small_element=poppy_core.FITSOpticalElement(transmission=test_optic_small)
    assert(test_optic_small_element.getPhasor(inputwf).shape ==inputwf.shape )

    # Test rescaling from coarser scales: diameter 1 meter, pixel scale 10 mm
    test_optic_large=fits.HDUList([fits.PrimaryHDU(np.zeros([100,100]))])
    test_optic_large[0].header["PUPLSCAL"]=.01
    test_optic_large_element=poppy_core.FITSOpticalElement(transmission=test_optic_large)
    assert(test_optic_large_element.getPhasor(inputwf).shape ==inputwf.shape )

    # Test rescaling where we have to pad with extra zeros: 
    # diameter 0.8 mm, pixel scale 1 mm
    test_optic_pad=fits.HDUList([fits.PrimaryHDU(np.zeros([800,800]))])
    test_optic_pad[0].header["PUPLSCAL"]=.001
    test_optic_pad_element=poppy_core.FITSOpticalElement(transmission=test_optic_pad)
    assert(test_optic_pad_element.getPhasor(inputwf).shape ==inputwf.shape )

    # Test rescaling where we have to trim to a smaller size:
    # diameter 1.2 mm, pixel scale 1 mm
    test_optic_crop=fits.HDUList([fits.PrimaryHDU(np.zeros([1200,1200]))])
    test_optic_crop[0].header["PUPLSCAL"]=.001
    test_optic_crop_element=poppy_core.FITSOpticalElement(transmission=test_optic_crop)
    assert(test_optic_crop_element.getPhasor(inputwf).shape ==inputwf.shape )


def test_unit_conversions():
    """ Test the astropy.Quantity unit conversions
    This is a modified version of test_CircularAperture
    """
    from ..misc import airy_2d
    import astropy.units as u
    # Analytic PSF for 1 meter diameter aperture
    analytic = airy_2d(diameter=1)
    analytic /= analytic.sum() # for comparison with poppy outputs normalized to total=1


    # Numeric PSF for 1 meter diameter aperture
    osys = poppy_core.OpticalSystem()
    pupil = optics.CircularAperture(radius=0.5)
    osys.addPupil(pupil)
    osys.addDetector(pixelscale=0.010,fov_pixels=512, oversample=1)

    # test versions with 3 different ways of saying the wavelength:
    for wavelen in [1e-6, 1e-6*u.m, 1*u.micron]:
        numeric_psf = osys.calcPSF(wavelength=wavelen, display=False)

        # Comparison
        difference = numeric_psf[0].data-analytic
        assert np.all(np.abs(difference) < 3e-5)

def test_return_complex():
    osys =poppy_core.OpticalSystem()
    osys.add_pupil(optics.CircularAperture(radius=3))   
    osys.add_detector(pixelscale=0.010, fov_arcsec=5.0)
    psf = osys.calc_psf(2e-6,return_final=True) 
    assert len(psf[1])==1 #make sure only one element was returned
    #test that the wavefront returned is the final wavefront:
    assert np.allclose(psf[1][0].intensity,psf[0][0].data)
