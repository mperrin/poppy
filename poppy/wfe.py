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

__all__ = ['WavefrontError', 'ParameterizedWFE', 'ZernikeWFE', 'SineWaveWFE', 'KolmogorovWFE']

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
    def __init__(self,
                 name="Statistical WFE model",
                 seed=None,
                 **kwargs):
        OpticalElement.__init__(self,name=name,**kwargs)

        self.randomize(seed=seed)

    def randomize(self,seed=None):
        """ Create new random instance """
        self._RandomState = np.random.RandomState(seed=seed)

    def structure_fn(self,npoints=None):
        """ Calculate two-point correlation function for WFE

        simple, brute-force and slightly hacky calculation just using
		shifts in the 4 cardinal directions.


        Returns a tuple containing 2 arrays:
            x = distance in meters for the structure function, and
            the structure function itself
        """
        if npoints is None:
            npoints = int(self.npix/2)

        array = self.opd

        result=np.zeros((npoints+1))
        for i in range(1,npoints+1):
            for shift,axis in ((1,0),(-1,0),(1,1),(-1,1)):
                result[i]=((array-np.roll(array,shift*i,axis=axis))**2).mean()
            result[i]/4

        x = np.arange(npoints+1) * self.pixelscale
        return x, result

    def plot_structure_fn(self,ax=None):
        """ Plot the structure function of the WFE OPD.

        For some background on structure functions & why they are useful, see :
            Parks (2010),
            https://www.osapublishing.org/abstract.cfm?uri=OFT-2010-OWE3

        """

        dist, sf = self.structure_fn()

        if ax is None:
            ax = plt.gca()
        ax.loglog(dist, sf, label = "Structure function of OPD")

        x=np.arange(40) * self.pixelscale
        ax.plot(x,sf[1]*(x/x[1])**(5./3),ls=":", label="Theoretical Kolmogorov")

        ax.set_xlabel("Separation [meters]")
        ax.set_ylabel("Structure Function [m^2]")

    def writeto(self, filename):
        raise NotImplementedError('Not implemented yet')


class KolmogorovWFE(StatisticalOpticalElement):
    """ A turbulent phase screen.
    
    This is an implementation of a turbulent phase screen as by the
    Kolmogorov theory of turbulence.
    
    Parameters
    -----------------
    wave : wavefront object
        Wavefront to calculate the phase screen for.

    References
    -------------------
    For a general overview of the Kolmogorov theory, read
    L. C. Andrews and R. L. Phillips, Laser Beam Propagation Through Random
    Media, 2nd ed. (Society of Photo Optical, 2005).
    
    Other relevant references are mentioned in the respective functions.
    """
    
    @utils.quantity_input(r0=u.meter, Cn2=u.meter**(-2/3), dz=u.meter, l0=u.meter, L0=u.meter)
    def __init__(self, name="Kolmogorov WFE", r0=None, Cn2=None,
                 dz=None, l0=None, L0=None, kind='Kolmogorov', **kwargs):
        
        if dz is None and not all(item is not None for item in [r0, Cn2]):
            raise AttributeError('To prepare a turbulent phase screen, \
                                 dz and either Cn2 or r0 must be given.')
        
        self.r0 = r0
        self.Cn2 = Cn2
        self.dz = dz
        self.l0 = l0
        self.L0 = L0
        self.kind = kind
        
        StatisticalOpticalElement.__init__(self, name=name, **kwargs)
    
    def get_opd(self, wave):
        """ Returns an optical path difference for a turbulent phase screen.
        
        Parameters
        -----------------
        wave : wavefront object
            Wavefront to calculate the phase screen for.
    
        References
        -------------------
        J. A. Fleck Jr, J. R. Morris, and M. D. Feit, Appl. Phys. 10, 129 (1976).
        
        E. M. Johansson and D. T. Gavel,
        in Proc. SPIE, edited by J. B. Breckinridge
        (International Society for Optics and Photonics, 1994), pp. 372–383.
        
        B. J. Herman and L. A. Strugala, in Proc. SPIE,
        edited by P. B. Ulrich and L. E. Wilson
        (International Society for Optics and Photonics, 1990), pp. 183–192.
        
        G. Gbur, J. Opt. Soc. Am. A 31, 2038 (2014).
        
        D. L. Knepp, Proc. IEEE 71, 722 (1983).
        """
        
        coordinates = wave.coordinates()
        npix = coordinates[0].shape[0]
        pixelscale = wave.pixelscale.to(u.m/u.pixel)
        dq = 2.0*np.pi/npix/pixelscale
        
        # create complex random numbers with required symmetry
        a = self.rand_turbulent(npix)
        
        # get phase spectrum
        Cn2 = self.get_Cn2(wave.wavelength)
        phi = self.power_spectrum(Cn2, npix, pixelscale,
                                  L0=self.L0, l0=self.l0, kind=self.kind)
        
        # calculate OPD
        # Note: Factor dq consequence of delta function having a unit
        opd_FFT = dq*a*np.sqrt(2.0*np.pi*self.dz*phi)
        opd = np.fft.ifft2(opd_FFT)
        self.opd = opd.real
        
        return opd.real
    
    @utils.quantity_input(wavelength=u.meter)
    def get_Cn2(self, wavelength):
        """ Returns the index-of-refraction structure constant (m^-2/3).
        
        Parameters
        -----------------
        wavelength : float
            The wavelength (m).
        
        References
        -------------------
        B. J. Herman and L. A. Strugala, in Proc. SPIE,
        edited by P. B. Ulrich and L. E. Wilson
        (International Society for Optics and Photonics, 1990), pp. 183–192.
        """
        
        if all(item is not None for item in [self.r0, self.dz]):
            r0 = self.r0.to(u.m)
            wavelength2 = wavelength.to(u.m)**2
            return wavelength2/self.dz * (r0/0.185)**(-5.0/3.0)
        elif self.Cn2 is not None:
            return self.Cn2.to(u.m**(-2/3))
    
    def rand_symmetrized(self, npix, sign):
        """ Returns a real-valued random number array of shape (npix, npix)
        with the symmetry required for a turbulent phase screen.
        
        Parameters
        -----------------
        npix : int
            Number of pixels.
        
        sign : int
            Sign of mirror symmetry. Must be either +1 or -1.
        
        References
        -------------------
        Eq. (65) in J. A. Fleck Jr, J. R. Morris, and M. D. Feit,
        Appl. Phys. 10, 129 (1976).
        """
        
        sign = float(sign)
        
        # create zero-mean, unit variance random numbers
        a = np.random.normal(size=(npix, npix))
        
        # apply required symmetry
        a[0, int(npix/2)+1:npix] = sign*a[0, 1:int(npix/2)][::-1]
        a[int(npix/2)+1:npix, 0] = sign*a[1:int(npix/2), 0][::-1]
        a[int(npix/2)+1:npix, int(npix/2)+1:npix] = sign*np.rot90(a[1:int(npix/2), 1:int(npix/2)], 2)
        a[int(npix/2)+1:npix, 1:int(npix/2)] = sign*np.rot90(a[1:int(npix/2), int(npix/2)+1:npix], 2)
        
        # remove any overall phase resulting from the zero-frequency component
        a[0, 0] = 0.0
        
        return a
    
    def rand_turbulent(self, npix):
        """ Returns a complex-valued random number array of shape (npix, npix)
        with the symmetry required for a turbulent phase screen.
        
        Parameters
        -----------------
        npix : int
            Number of pixels.
        
        References
        -------------------
        Eq. (63) in J. A. Fleck Jr, J. R. Morris, and M. D. Feit,
        Appl. Phys. 10, 129 (1976).
        """
        
        # create real-valued random numbers with required symmetry
        a = self.rand_symmetrized(npix, 1)
        b = self.rand_symmetrized(npix, -1)
        
        # create complex-valued random number with required variance
        c = (a + 1j*b)/np.sqrt(2.0)
        
        return c
    
    @utils.quantity_input(Cn2=u.meter**(-2/3), dz=u.meter, pixelscale=u.meter/u.pixel)
    def power_spectrum(self, Cn2, npix, pixelscale,
                       L0=None, l0=None, kind='Kolmogorov'):
        """ Returns the spatial power spectrum.
        
        Parameters
        -----------------
        Cn2 : float
            Index-of-refraction structure constant (m^-2/3).
        
        npix : int
            Number of pixels.
            
        pixelscale : astropy quantity
            The pixel scale (m/pixel).
        
        L0 : float
            The outer scale of the turbulent eddies (m).
        
        l0 : float
            The inner scale of the turbulent eddies (m).
        
        kind : string
            The type of the power spectrum, must be one of 'Kolmogorov',
            'Tatarski', 'van Karman', 'Hill'.
        
        References
        -------------------
        G. Gbur, J. Opt. Soc. Am. A 31, 2038 (2014).
        
        R. Frehlich, Appl. Opt. 39, 393 (2000).
        """
        
        q = np.fft.fftfreq(npix, d=pixelscale.to(u.m/u.pixel))*2.0*np.pi
        qx, qy = np.meshgrid(q, q)
        
        q2 = qx**2 + qy**2
        if kind is 'van Karman':
            k2 = qx**2 + qy**2
            if L0 is not None:
                q2 += 1.0/L0**2
            else:
                raise AttributeError('If van Karman type of turbulent phase \
                                     screen is chosen, the outer scale L_0 \
                                     must be provided.')
        else:
            q2[0, 0] = np.inf # this is to avoid a possible error message in the next line
        
        phi = 0.0330054*Cn2*q2**(-11.0/6.0)
        
        if kind is 'Tatarski' or kind is 'van Karman' or kind is 'Hill':
            if l0 is not None:
                if kind is 'Tatarski' or kind is 'van Karman':
                    m = (5.92/l0)**2
                    phi *= np.exp(-k2/m)
                elif kind is 'Hill':
                    m = np.sqrt(k2)*l0
                    phi *= (1.0 + 0.70937*m + 2.8235*m**2
                            - 0.28086*m**3 + 0.08277*m**4) * np.exp(-1.109*m)
            else:
                raise AttributeError('If van Karman, Hill, or Tatarski type \
                                     of turbulent phase screen is chosen, the \
                                     inner scale l_0 must be provided.')
        
        return phi


class PowerSpectralDensityWFE(StatisticalOpticalElement):
    """ Compute WFE from a power spectral density.

    Inspired by (and loosely derived from) prop_psd_errormap in John Krist's PROPER library.



    **** PLACEHOLDER NOT YET IMPLEMENTED ****

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


