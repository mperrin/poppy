"""
Analytic optical element classes to introduce a specified wavefront
error in an OpticalSystem

 * ZernikeWFE
 * ParameterizedWFE (for use with hexike or zernike basis functions)
 * SineWaveWFE
 * TODO: MultiSineWaveWFE ?
 * TODO: PowerSpectrumWFE
 * TODO: KolmogorovWFE

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
from functools import wraps
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from .optics import AnalyticOpticalElement, CircularAperture
from .poppy_core import Wavefront, OpticalElement, _PUPIL
from . import zernike
from . import utils

__all__ = ['WavefrontError', 'ParameterizedWFE', 'ZernikeWFE', 'SineWaveWFE']

def _accept_wavefront_or_meters(f):
    """Decorator that ensures the first positional method argument
    is a poppy.Wavefront or a floating point number of meters
    for a wavelength
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], Wavefront):
            wave = Wavefront(wavelength=args[1])
            new_args = (args[0],) + (wave,) + (args[2:])
            return f(*new_args, **kwargs)
        else:
            return f(*args, **kwargs)
    return wrapper

class WavefrontError(AnalyticOpticalElement):
    """A base class for different sources of wavefront error

    Analytic optical elements that represent wavefront error should
    derive from this class and override methods appropriately.
    Defined to be a pupil-plane optic.
    """
    def __init__(self, **kwargs):
        if 'planetype' not in kwargs:
            kwargs['planetype'] = _PUPIL
        super(WavefrontError, self).__init__(**kwargs)
        # in general we will want to see phase rather than intensity at this plane
        self.wavefront_display_hint='phase'

    @_accept_wavefront_or_meters
    def get_opd(self, wave, units='meters'):
        """Construct the optical path difference array for a wavefront error source
        as evaluated across the pupil for an input wavefront `wave`

        Parameters
        ----------
        wave : Wavefront
            Wavefront object with a `coordinates` method that returns (y, x)
            coordinate arrays in meters in the pupil plane
        units : 'meters' or 'waves'
            The units of optical path difference (Default: meters)
        """
        if not isinstance(wave, Wavefront):
            wave = Wavefront(wavelength=wave)

    @_accept_wavefront_or_meters
    def getPhasor(self, wave):
        """Construct the phasor array for an input wavefront `wave`
        that will apply the wavefront error model based on an OPD
        map from the `get_opd` method.

        Parameters
        ----------
        wave : Wavefront or float
            Wavefront object with a `coordinates` method that returns (y, x)
            coordinate arrays in meters in the pupil plane, or float with
            a wavefront wavelength in meters
        """
        opd_map = self.get_opd(wave, units='meters')
        opd_as_phase = 2 * np.pi * opd_map / (wave.wavelength.to(u.meter).value)
        wfe_phasor = np.exp(1.j * opd_as_phase)
        return wfe_phasor

    def rms(self):
        """RMS wavefront error induced by this surface"""
        raise NotImplementedError('Not implemented yet')

    def peaktovalley(self):
        """Peak-to-valley wavefront error induced by this surface"""
        raise NotImplementedError('Not implemented yet')

def _wave_to_rho_theta(wave, pupil_radius):
    """
    Return wave coordinates in (rho, theta) for a Wavefront object
    normalized such that rho == 1.0 at the pupil radius

    Parameters
    ----------
    wave : Wavefront
        Wavefront object with a `coordinates` method that returns (y, x)
        coordinate arrays in meters in the pupil plane
    pupil_radius : float
        Radius (in meters) of a circle circumscribing the pupil.
    """
    y, x = wave.coordinates()
    r = np.sqrt(x ** 2 + y ** 2)

    rho = r / pupil_radius
    theta = np.arctan2(y / pupil_radius, x / pupil_radius)

    return rho, theta

class ParameterizedWFE(WavefrontError):
    """
    Define an optical element in terms of its distortion as decomposed
    into a set of orthonormal basis functions (e.g. Zernikes,
    Hexikes, etc.). Included basis functions are normalized such that
    user-provided coefficients correspond to meters RMS wavefront
    aberration for that basis function.

    Parameters
    ----------
    coefficients : iterable of numbers
        The contribution of each term to the final distortion, in meters
        RMS wavefront error. The coefficients are interpreted as indices
        in the order of Noll et al. 1976: the first term corresponds to
        j=1, second to j=2, and so on.
    radius : float
        Pupil radius, in meters. Defines the region of the input
        wavefront array over which the distortion terms will be
        evaluated. For non-circular pupils, this should be the circle
        circumscribing the actual pupil shape.
    basis_factory : callable
        basis_factory will be called with the arguments `nterms`, `rho`,
        `theta`, and `outside`.

        `nterms` specifies how many terms to compute, starting with the
        j=1 term in the Noll indexing convention for `nterms` = 1 and
        counting up.

        `rho` and `theta` are square arrays holding the rho and theta
        coordinates at each pixel in the pupil plane. `rho` is
        normalized such that `rho` == 1.0 for pixels at `radius` meters
        from the center.

        `outside` contains the value to assign pixels outside the
        radius `rho` == 1.0. (Always 0.0, but provided for
        compatibility with `zernike.zernike_basis` and
        `zernike.hexike_basis`.)
    """
    def __init__(self, name="Parameterized Distortion", coefficients=None, radius=None,
                 basis_factory=None, **kwargs):
        if not isinstance(basis_factory, collections.Callable):
            raise ValueError("'basis_factory' must be a callable that can "
                             "calculate basis functions")
        try:
            self.radius = float(radius)
        except TypeError:
            raise ValueError("'radius' must be the radius of a circular aperture in meters"
                             "(optionally circumscribing a pupil of another shape)")
        self.coefficients = coefficients
        self.basis_factory = basis_factory
        super(ParameterizedWFE, self).__init__(name=name, **kwargs)

    @_accept_wavefront_or_meters
    def get_opd(self, wave, units='meters'):
        rho, theta = _wave_to_rho_theta(wave, self.radius)
        combined_distortion = np.zeros(rho.shape)

        nterms = len(self.coefficients)
        computed_terms = self.basis_factory(nterms=nterms, rho=rho, theta=theta, outside=0.0)

        for idx, coefficient in enumerate(self.coefficients):
            if coefficient == 0.0:
                continue  # save the trouble of a multiply-and-add of zeros
            combined_distortion += coefficient * computed_terms[idx]
        if units == 'meters':
            return combined_distortion
        elif units == 'waves':
            return combined_distortion / wave.wavelength
        else:
            raise ValueError("'units' argument must be 'meters' or 'waves'")

class ZernikeWFE(WavefrontError):
    """
    Define an optical element in terms of its Zernike components by
    providing coefficients for each Zernike term contributing to the
    analytic optical element.

    Parameters
    ----------
    coefficients : iterable of floats
        Specifies the coefficients for the Zernike terms, ordered
        according to the convention of Noll et al. JOSA 1976. The
        coefficient is in meters of optical path difference (not waves).
    radius : float
        Pupil radius, in meters, over which the Zernike terms should be
        computed such that rho = 1 at r = `radius`.
    """
    def __init__(self, name="Zernike WFE", coefficients=None, radius=None, **kwargs):
        try:
            self.radius = float(radius)
        except TypeError:
            raise ValueError("'radius' must be the radius of a circular aperture in meters"
                             "(optionally circumscribing a pupil of another shape)")

        self.coefficients = coefficients
        self.circular_aperture = CircularAperture(radius=self.radius, **kwargs)
        kwargs.update({'name': name})
        super(ZernikeWFE, self).__init__(**kwargs)

    @_accept_wavefront_or_meters
    def get_opd(self, wave, units='meters'):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        units : 'meters' or 'waves'
            Coefficients are supplied in `ZernikeWFE.coefficients` as
            meters of OPD, but the resulting OPD can be converted to
            waves based on the `Wavefront` wavelength or a supplied
            wavelength value.
        """
        rho, theta = _wave_to_rho_theta(wave, self.radius)

        # the Zernike optic, being normalized on a circle, is
        # implicitly also a circular aperture:
        aperture_intensity = self.circular_aperture.get_transmission(wave)

        pixelscale_m = wave.pixelscale.to(u.meter/u.pixel).value

        combined_zernikes = np.zeros(wave.shape, dtype=np.float64)
        for j, k in enumerate(self.coefficients, start=1):
            combined_zernikes += k * zernike.cached_zernike1(
                j,
                wave.shape,
                pixelscale_m,
                self.radius,
                outside=0.0,
                noll_normalize=True
            )

        combined_zernikes *= aperture_intensity
        if units == 'waves':
            combined_zernikes /= wave.wavelength
        return combined_zernikes

class SineWaveWFE(WavefrontError):
    """ A single sine wave ripple across the optic

    Specified as a a spatial frequency in cycles per meter, an optional phase offset in cycles,
    and an amplitude.

    By default the wave is oriented in the X direction.
    Like any AnalyticOpticalElement class, you can also specify a rotation parameter to
    rotate the direction of the sine wave.


    (N.b. we intentionally avoid letting users specify this in terms of a spatial wavelength
    because that would risk potential ambiguity with the wavelength of light.)
    """
    @utils.quantity_input(spatialfreq=1./u.meter, amplitude=u.meter)
    def  __init__(self,  name='Sine WFE', spatialfreq=1.0, amplitude=1e-6, phaseoffset=0, **kwargs):
        super(WavefrontError, self).__init__(name=name, **kwargs)

        self.sine_spatial_freq = spatialfreq
        self.sine_phase_offset = phaseoffset
        # note, can't call this next one 'amplitude' since that's already a property
        self.sine_amplitude = amplitude

    @_accept_wavefront_or_meters
    def get_opd(self, wave, units='meters'):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        units : 'meters' or 'waves'
            Coefficients are supplied as meters of OPD, but the
            resulting OPD can be converted to
            waves based on the `Wavefront` wavelength or a supplied
            wavelength value.
        """

        y, x = self.get_coordinates(wave)  # in meters

        opd = self.sine_amplitude.to(u.meter).value * \
                np.sin( 2*np.pi * (x * self.sine_spatial_freq.to(1/u.meter).value + self.sine_phase_offset))

        if units == 'waves':
            opd /= wave.wavelength.to(u.meter).value
        return opd


class StatisticalOpticalElement(OpticalElement):
    """
    A statistical realization of some wavefront error, computed on a fixed grid.

    This is in a sense like an AnalyticOpticalElement, in that it is computed on some arbitrary grid,
    but once computed it has a fixed sampling that cannot easily be changed.

    """

    @utils.quantity_input(grid_size=u.meter)
    def __init__(self, name="Statistical WFE model",  seed=None,  npix=512, grid_size=1*u.meter, **kwargs):
        OpticalElement.__init__(self,name=name,**kwargs)

        self.npix = npix
        self.grid_size=grid_size
        self.pixelscale= grid_size/(npix*u.pixel)

        self.randomize(seed=seed)
        self.generate_opd()
        self.generate_transmission()

    def randomize(self,seed=None):
        """ Create new random instance """
        self._RandomState = np.random.RandomState(seed=seed)


    def generate_opd(self):
        # This should be overridden by the subclass
        self.opd = np.zeros( (self.npix, self.npix) )

    def generate_transmission(self):
        # This may be overridden by the subclass, if desired
        self.amplitude = np.ones( (self.npix, self.npix) )


    def structure_fn(self,npoints=None):
        """ Calculate two-point correlation function for WFE

        simple, brute-force and slightly hacky calculation just using
		shifts in the 4 cardinal directions.
        """
        if npoints is None:
            npoints = int(self.npix/2)

        array = self.opd

        result=np.zeros((npoints+1))
        for i in range(1,npoints+1):
            for shift,axis in ((1,0),(-1,0),(1,1),(-1,1)):
                result[i]=((array-np.roll(array,shift*i,axis=axis))**2).mean()
            result[i]/4
        return result

    def plot_structure_fn(self,ax=None):

        sf = self.structure_fn()


        if ax is None:
            ax = plt.gca()
        ax.loglog(sf, label = "Structure function of OPD")

        x=np.arange(40)
        ax.plot(x,sf[1]*(x/x[1])**(5./3),ls=":", label="Theoretical Kolmogorov")


class KolmogorovWFE(StatisticalOpticalElement):
    """
    See

    http://www.opticsinfobase.org/view_article.cfm?gotourl=http%3A%2F%2Fwww%2Eopticsinfobase%2Eorg%2FDirectPDFAccess%2F8E2A4176%2DED0A%2D7994%2DFB0AC49CECB235DF%5F142887%2Epdf%3Fda%3D1%26id%3D142887%26seq%3D0%26mobile%3Dno&org=

    http://optics.nuigalway.ie/people/chris/chrispapers/Paper066.pdf

    """
    @utils.quantity_input(r0=u.meter)
    def __init__(self, name="Kolmogorov WFE",  r0=0.1*u.meter, **kwargs):
        self.r0 = r0
        StatisticalOpticalElement.__init__(self,name=name,**kwargs)


    def generate_opd(self):

        # Based on IDL code in fgui.pro by Tuan Do, used by permission
        r0= self.r0.to(u.meter).value
        shape= (self.npix, self.npix)
        pixelscale= self.pixelscale.to(u.meter/u.pixel).value

        # set up indices arrays
        dk = 1./(np.asarray(shape)*pixelscale)
        y,x = np.indices(shape)
        y=(y-shape[0]/2.)*dk[0]
        x=(x-shape[1]/2.)*dk[1]
        ksq=x**2+y**2
        ksq[shape[0]/2,shape[1]/2]=1

        # calculate Kolmogorov PSD
        psd=0.023*(2*np.pi)**(5./6) * r0**(-5./6) * ksq**(-11/6) * np.sqrt(dk[0]*dk[1])
        psd[shape[0]/2,shape[1]/2]=0  # set piston to 0

        # Generate complex independent, Gaussian, random numbers with zero mean and unit variance
        w_r=np.random.normal(size=shape)
        w_i=np.random.normal(size=shape)
        w=w_r + np.complex(0,1)*w_i

        #multiply by sqrt of PSD
        wf=w*np.sqrt(psd)
        self.opd = np.fft.fft2(np.fft.fftshift(wf)).real





class PowerSpectralDensityWFE(StatisticalOpticalElement):
    """ Compute WFE from a power spectral density. 

    Inspired by (and loosely derived from) prop_psd_errormap in John Krist's PROPER library.


    For some background on structure functions & why they are useful, see : http://www.optics.arizona.edu/optomech/Spr11/523L/Specifications%20final%20color.pdf

    """
    def __init__(self, name=None,  seed=None, low_freq_amp=1, correlation_length=1.0, powerlaw=1.0,  **kwargs):
        """ 

        Parameters
        -----------
        low_freq_amp : float
            RMS error per spatial frequency at low spatial frequencies. 
        correlation_length : float
            Correlation length parameter in cycles/meter. This indicates where the PSD transitions from
            the low frequency behavior (~ constant amplitude per spatial frequency) to the high
            frequency behavior (~decreasing amplitude per spatial frequency)
        powerlaw : float
            The power law exponent for the falloff in amplitude at high spatial frequencies.
        """
        if name is None: name = "Power Spectral Density WFE map "
        StatisticalOpticalElement.__init__(self,name=name,**kwargs)
        raise NotImplementedError('Not implemented yet')

        # compute X and Y coordinate grids 
        # compute wavenumber K in cycles/meter
        # compute 2D PSD
        # set piston to zero
        # scale RMS error as desired
        # create realization of the PSD using random phases
        # force realized map to have the desired RMS

    def saveto(self, filename):
        raise NotImplementedError('Not implemented yet')



