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

import collections
from functools import wraps
import numpy as np
import astropy.units as u

from .optics import AnalyticOpticalElement, CircularAperture
from .poppy_core import Wavefront, PlaneType, BaseWavefront
from poppy.fresnel import FresnelWavefront
from .physical_wavefront import PhysicalFresnelWavefront

from . import zernike
from . import utils
from . import accel_math

__all__ = ['WavefrontError', 'ParameterizedWFE', 'ZernikeWFE', 'SineWaveWFE',
        'StatisticalPSDWFE', 'ThermalBloomingWFE']


def _check_wavefront_arg(f):
    """Decorator that ensures the first positional method argument
    is a poppy.Wavefront or FresnelWavefront
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], BaseWavefront):
            raise ValueError("The first argument must be a Wavefront or FresnelWavefront object.")
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
            kwargs['planetype'] = PlaneType.pupil
        super(WavefrontError, self).__init__(**kwargs)
        # in general we will want to see phase rather than intensity at this plane
        self.wavefront_display_hint = 'phase'

    @_check_wavefront_arg
    def get_opd(self, wave):
        """Construct the optical path difference array for a wavefront error source
        as evaluated across the pupil for an input wavefront `wave`

        Parameters
        ----------
        wave : Wavefront
            Wavefront object with a `coordinates` method that returns (y, x)
            coordinate arrays in meters in the pupil plane
        """
        raise NotImplementedError('Not implemented yet')

    def rms(self):
        """RMS wavefront error induced by this surface"""
        raise NotImplementedError('Not implemented yet')

    def peaktovalley(self):
        """Peak-to-valley wavefront error induced by this surface"""
        raise NotImplementedError('Not implemented yet')


def _wave_y_x_to_rho_theta(y, x, pupil_radius):
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

    if accel_math._USE_NUMEXPR:
        rho = accel_math.ne.evaluate("sqrt(x**2+y**2)/pupil_radius")
        theta = accel_math.ne.evaluate("arctan2(y / pupil_radius, x / pupil_radius)")
    else:
        rho = np.sqrt(x ** 2 + y ** 2) / pupil_radius
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

    @utils.quantity_input(coefficients=u.meter, radius=u.meter)
    def __init__(self, name="Parameterized Distortion", coefficients=None, radius=1*u.meter,
                 basis_factory=None, **kwargs):
        if not isinstance(basis_factory, collections.Callable):
            raise ValueError("'basis_factory' must be a callable that can "
                             "calculate basis functions")
        self.radius = radius
        self.coefficients = coefficients
        self.basis_factory = basis_factory
        self._default_display_size = radius * 3
        super(ParameterizedWFE, self).__init__(name=name, **kwargs)

    @_check_wavefront_arg
    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)
        rho, theta = _wave_y_x_to_rho_theta(y, x, self.radius.to(u.meter).value)

        combined_distortion = np.zeros(rho.shape)

        nterms = len(self.coefficients)
        computed_terms = self.basis_factory(nterms=nterms, rho=rho, theta=theta, outside=0.0)

        for idx, coefficient in enumerate(self.coefficients):
            if coefficient == 0.0:
                continue  # save the trouble of a multiply-and-add of zeros
            coefficient_in_m = coefficient.to(u.meter).value
            combined_distortion += coefficient_in_m * computed_terms[idx]
        return combined_distortion


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

    @utils.quantity_input(coefficients=u.meter, radius=u.meter)
    def __init__(self, name="Zernike WFE", coefficients=None, radius=None,
            aperture_stop=False, **kwargs):

        if radius is None:
            raise ValueError("You must specify a radius for the unit circle "
                             "over which the Zernike polynomials are normalized")
        self.radius = radius
        self.aperture_stop = aperture_stop
        self.coefficients = coefficients
        self.circular_aperture = CircularAperture(radius=self.radius, gray_pixel=False, **kwargs)
        self._default_display_size = radius * 3
        kwargs.update({'name': name})
        super(ZernikeWFE, self).__init__(**kwargs)

    @_check_wavefront_arg
    def get_opd(self, wave):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        """

        # the Zernike optic, being normalized on a circle, is
        # implicitly also a circular aperture:
        aperture_intensity = self.circular_aperture.get_transmission(wave)

        pixelscale_m = wave.pixelscale.to(u.meter / u.pixel).value

        # whether we can use pre-cached zernikes for speed depends on whether
        # there are any coord offsets. See #229
        has_offset_coords = (hasattr(self, "shift_x") or hasattr(self, "shift_y")
                             or hasattr(self, "rotation"))
        if has_offset_coords:
            y, x = self.get_coordinates(wave)
            rho, theta = _wave_y_x_to_rho_theta(y, x, self.radius.to(u.meter).value)

        combined_zernikes = np.zeros(wave.shape, dtype=np.float64)
        for j, k in enumerate(self.coefficients, start=1):
            k_in_m = k.to(u.meter).value

            if has_offset_coords:
                combined_zernikes += k_in_m * zernike.zernike1(
                    j,
                    rho=rho,
                    theta=theta,
                    outside=0.0,
                    noll_normalize=True
                )
            else:
                combined_zernikes += k_in_m * zernike.cached_zernike1(
                    j,
                    wave.shape,
                    pixelscale_m,
                    self.radius.to(u.meter).value,
                    outside=0.0,
                    noll_normalize=True
                )

        combined_zernikes[aperture_intensity==0] = 0
        return combined_zernikes


    def get_transmission(self, wave):
        if self.aperture_stop:
            return self.circular_aperture.get_transmission(wave)
        else:
            return np.ones(wave.shape)


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

    @utils.quantity_input(spatialfreq=1. / u.meter, amplitude=u.meter)
    def __init__(self, name='Sine WFE', spatialfreq=1.0, amplitude=1e-6, phaseoffset=0, **kwargs):
        super(WavefrontError, self).__init__(name=name, **kwargs)

        self.sine_spatial_freq = spatialfreq
        self.sine_phase_offset = phaseoffset
        # note, can't call this next one 'amplitude' since that's already a property
        self.sine_amplitude = amplitude

    @_check_wavefront_arg
    def get_opd(self, wave):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        """

        y, x = self.get_coordinates(wave)  # in meters

        opd = self.sine_amplitude.to(u.meter).value * \
              np.sin(2 * np.pi * (x * self.sine_spatial_freq.to(1 / u.meter).value + self.sine_phase_offset))

        return opd


class StatisticalPSDWFE(WavefrontError):
    """
    Statistical PSD WFE class from power law for optical noise.

    Parameters
    ----------
    name : string
        name of the optic
    index: float
        negative power law spectra index, defaults to 3
    wfe: astropy quantity
        wfe in linear astropy units, defaults to 50 nm
    radius: astropy quantity
        radius of optic in linear astropy units, defaults to 1 m
    seed : integer
        seed for the random phase screen generator
    """

    @utils.quantity_input(wfe=u.nm, radius=u.meter)
    def __init__(self, name='PSD WFE', index=3.0, wfe=50*u.nm, radius=1*u.meter, seed=None, **kwargs):

        super().__init__(name=name, **kwargs)
        self.index = index
        self.wfe = wfe
        self.radius = radius
        self.seed = seed

    @_check_wavefront_arg
    def get_opd(self, wave):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        """
        y, x = self.get_coordinates(wave)
        rho, theta = _wave_y_x_to_rho_theta(y, x, self.radius.to(u.meter).value)
        psd = np.power(rho, -self.index)   # generate power-law PSD

        np.random.seed(self.seed)   # if provided, set a seed for random number generator
        rndm_phase = np.random.normal(size=(len(y), len(x)))   # generate random phase screen
        rndm_psd = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rndm_phase)))   # FT of random phase screen to get random PSD
        scaled = np.sqrt(psd) * rndm_psd    # scale random PSD by power-law PSD
        phase_screen = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(scaled))).real   # FT of scaled random PSD makes phase screen

        phase_screen -= np.mean(phase_screen)  # force zero-mean
        opd = phase_screen / np.std(phase_screen) * self.wfe.to(u.m).value  # normalize to wanted input rms wfe

        return opd


class ThermalBloomingWFE(WavefrontError):
    """ A thermal blooming phase screen.
    
    Parameters
    -----------------
    abs_coeff : astropy.quantity
        Aerosol absorption coefficient (m^-1).
    
    dz : astropy.quantity
        Propagation distance (m).
    
    v0x : astropy.quantity
        x-component of ambient wind velocity (m.s^-1).
    
    v0y : astropy.quantity
        y-component of ambient wind velocity (m.s^-1).
    
    cp : astropy.quantity
        Specific isobaric heat capacity (J.kg^-1.K^-1).
    
    cV : astropy.quantity
        Specific isochore heat capacity (J.kg^-1.K^-1).
    
    rho0 : astropy.quantity
        Ambient mass density (kg.m^-3).
    
    eta : astropy.quantity
        Dynamic viscosity (Pa.s).
    
    p0 : astropy.quantity
        Ambient pressure (Pa).
    
    T0 : astropy.quantity
        Ambient temperature (K).
    
    direction : string
        Direction of wind velocity. Must be one of 'x' or 'y'. The direction
        affects the calculation results if isobaric=True.
    
    isobaric : bool
        Wether to use the isobaric approximation.
    
    Note
    -------------------
    Initial values are those for dry air at room temperature, taken from:
    https://www.engineeringtoolbox.com/dry-air-properties-d_973.html
    """
    
    @utils.quantity_input(abs_coeff=1/u.meter, dz=u.meter,
                          v0x=u.meter/u.second, v0y=u.meter/u.second,
                          cp=u.Joule/u.kg/u.Kelvin, cV=u.Joule/u.kg/u.Kelvin,
                          rho0=u.kg/u.meter**3, p0=u.Pascal,
                          eta=u.Pascal*u.second, T0=u.Kelvin)
    def __init__(self, abs_coeff, dz, name="Thermal Blooming WFE",
                 v0x=0.0*u.m/u.s, v0y=0.0*u.m/u.s,
                 cp=1.0049*u.kJ/u.kg/u.K, cV=0.7178*u.kJ/u.kg/u.K,
                 rho0=1.177*u.kg/u.m**3, eta=18.46*u.uPa*u.s,
                 p0=101.325*u.kPa, T0=300.0*u.K, direction='x',
                 isobaric=False, **kwargs):
        
        super(ThermalBloomingWFE, self).__init__(name=name, **kwargs)
        
        self.abs_coeff = abs_coeff.to(1/u.m).value
        self.dz = dz.to(u.m).value
        self.v0x = v0x.to(u.m/u.s).value
        self.v0y = v0y.to(u.m/u.s).value
        self.cp = cp.to(u.J/u.kg/u.K).value
        self.cV = cV.to(u.J/u.kg/u.K).value
        self.rho0 = rho0.to(u.kg/u.m**3).value
        self.eta = eta.to(u.Pa*u.s).value
        self.p0 = p0.to(u.Pa).value
        self.T0 = T0.to(u.Kelvin).value
        self.direction = direction
        self.isobaric = isobaric
        self.gamma = self.cp/self.cV
        self.cs2 = self.gamma*self.p0/self.rho0
        
    def nat_conv_vel(self, wave):
        """ Approximation for natural convection velocity (m.s^-1).
        
        Parameters
        -----------------
        wave : poppy.PhysicalFresnelWavefront
            Wavefront to calculate the natural convection velocity for.
        
        References
        -------------------
        Smith, D. C.
        High-power laser propagation: Thermal blooming.
        Proc. IEEE 65, 1679–1714 (1977).
        """
        
        g = 9.81 # Gravitational constant (m.s^-2)
        P = wave.power
        
        return (2.0*self.abs_coeff*P*g / (self.rho0*self.cp*self.T0))**(1.0/3.0)
    
    def get_opd(self, wave):
        """ Returns an optical path difference for a thermal blooming phase screen (m^-1).
        
        Parameters
        -----------------
        wave : poppy.PhysicalFresnelWavefront
            Wavefront to calculate the phase screen for.
    
        References
        -------------------
        Fleck, J. A., Jr, Morris, J. R. & Feit, M. D.
        Time-dependent propagation of high energy laser beams through the atmosphere.
        Appl. Phys. 10, 129–160 (1976).
        
        Fleck, J. A., Jr, Morris, J. R. & Feit, M. D.
        Time-dependent propagation of high-energy laser beams through the atmosphere: II.
        Appl. Phys. 14, 99–115 (1977).
        """
        
        # Check if correct wavefront object type
        if type(wave) is not PhysicalFresnelWavefront:
            raise AttributeError("The wavefront must be of type \
                                 'PhysicalFresnelWavefront' to calculate a \
                                 thermal blooming phase screen.")
        
        # Set velocity components according to input
        if self.v0x==0.0 and self.v0y==0.0:
            # If stagnation point, use approximation for natural convection velocity
            self.v0y = -self.nat_conv_vel(wave)
            self.isobaric = True
            self.direction ='y'
        elif self.isobaric:
            if self.direction=='x' and self.v0x!=0.0:
                self.v0y = 0.0
            elif self.direction=='y' and self.v0y!=0.0:
                self.v0x = 0.0
            else:
                raise ValueError("The direction must be either 'x' or 'y' \
                                 and the respective velocity non-vanishing.")
        else:
            # Use given values (defined in init)
            pass
        
        rho = self.rho(wave)
        opd = (wave.n0-1.0)*rho*self.dz/(wave.n0*self.rho0)
        self.opd = opd
        
        return opd
    
    def rho(self, wave):
        """ Top-level routine to calculate density changes (kg.m^-3).
        
        Parameters
        -----------------
        wave : poppy.PhysicalFresnelWavefront
            Wavefront to calculate the density changes for.
        """
        
        if (self.isobaric):
            rho = self.rho_isobaric(wave)
        else:
            rho = self.rho_nonisobaric(wave)
        
        return rho
    
    def rho_isobaric(self, wave):
        """ Isobaric density variation (kg.m^-3).
        
        Parameters
        -----------------
        wave : poppy.PhysicalFresnelWavefront
            Wavefront to calculate the density changes for.
        
        References
        -------------------
        Fleck, J. A., Jr, Morris, J. R. & Feit, M. D.
        Time-dependent propagation of high energy laser beams through the atmosphere.
        Appl. Phys. 10, 129–160 (1976).
        """
        
        gamma = self.gamma
        cs2 = self.cs2
        intens = wave.intensity
        npix = wave.npix
        dx = wave.dx
        rho = np.zeros((npix, npix))
        v0 = max(abs(self.v0x), abs(self.v0y))
        
        if (self.direction == 'x'):
            if v0 > 0.0:
                for idx_x in range(npix):
                    for idx_y in range(npix):
                        rho[idx_x, idx_y] = np.sum(intens[0:idx_x, idx_y])
            else:
                for idx_x in range(npix):
                    for idx_y in range(npix):
                        rho[idx_x, idx_y] = np.sum(intens[idx_x:-1, idx_y])
        elif (self.direction == 'y'):
            if v0 > 0.0:
                for idx_x in range(npix):
                    for idx_y in range(npix):
                        rho[idx_x, idx_y] = np.sum(intens[idx_x, 0:idx_y])
            else:
                for idx_x in range(npix):
                    for idx_y in range(npix):
                        rho[idx_x, idx_y] = np.sum(intens[idx_x, idx_y:-1])
        else:
            raise AttributeError('The direction must be either x or y.')
        
        rho *= -(gamma-1.0)*self.abs_coeff*dx/cs2/np.abs(v0)
        
        return rho
    
    def rho_dot_FT(self, wave):
        """ Fourier transform of the derivative of the non-isobaric density variation (unit?).
        
        Parameters
        -----------------
        wave : poppy.PhysicalFresnelWavefront
            Wavefront to calculate the density changes for.
        
        References
        -------------------
        Fleck, J. A., Jr, Morris, J. R. & Feit, M. D.
        Time-dependent propagation of high-energy laser beams through the atmosphere: II.
        Appl. Phys. 14, 99–115 (1977).
        """
        
        if (self.v0x == 0.0 and self.v0y == 0.0):
            raise ValueError('The velocity must be non-zero in at least one direction.')
        
        npix = wave.npix
        gamma = self.gamma
        cs2 = self.cs2
        rho0 = self.rho0
        intens = wave.intensity
        rho_dot_FT = np.fft.fft2(intens)
        rho_dot_FT *= -(gamma-1.0)*self.abs_coeff/cs2
        q = wave.q
        
        for idx_x in range(npix):
            for idx_y in range(npix):
                if idx_x == idx_y == 0:
                    rho_dot_FT[idx_x, idx_y] *= 0.0
                else:
                    rho_dot_FT[idx_x, idx_y] *= (q[idx_x]**2 + q[idx_y]**2)
                    rho_dot_FT[idx_x, idx_y] /= (q[idx_x]**2 + q[idx_y]**2
                              * (1.0 + 4.0j*self.eta*(self.v0x*q[idx_x] + self.v0y*q[idx_y])/rho0/cs2/3.0)
                              - (self.v0x*q[idx_x] + self.v0y*q[idx_y])**2/cs2)
        
        return rho_dot_FT
    
    def rho_nonisobaric(self, wave):
        """ Non-isobaric density variations (kg.m^-3).
        
        Parameters
        -----------------
        wave : poppy.PhysicalFresnelWavefront
            Wavefront to calculate the density changes for.
        
        References
        -------------------
        Fleck, J. A., Jr, Morris, J. R. & Feit, M. D.
        Time-dependent propagation of high-energy laser beams through the atmosphere: II.
        Appl. Phys. 14, 99–115 (1977).
        """
        
        npix = wave.npix
        rho = np.zeros((npix, npix), dtype=complex)
        
        eps = 1.0e-16 # this is to prevent numpy.sign to return 0
        dx = wave.dx
        beta = np.abs((self.v0y+eps)/(self.v0x+eps))
        i_prime = np.sign(self.v0x+eps)
        j_prime = np.sign(self.v0y+eps)
        
        rho_dot_FT = self.rho_dot_FT(wave)
        rho_dot = np.fft.ifft2(rho_dot_FT)
        
        if beta > 1.0:
            a = dx/2/abs(self.v0y)
            if i_prime == 1 and j_prime == 1:
                for idx_x in range(npix):
                    rho[idx_x, 0] = a*rho_dot[idx_x, 0]
                
                for idx_y in range(1, npix):
                    for idx_x in range(npix):
                        rho[idx_x, idx_y] = (1.0-1.0/beta)*rho[idx_x, idx_y-1]
                        rho[idx_x, idx_y] += a*rho_dot[idx_x, idx_y]
                        rho[idx_x, idx_y] += a*(1.0-1.0/beta)*rho_dot[idx_x, idx_y-1]
                        if idx_x != 0:
                            rho[idx_x, idx_y] += 1.0/beta*rho[idx_x-1, idx_y-1]
                            rho[idx_x, idx_y] += a/beta*rho_dot[idx_x-1, idx_y-1]
                
            elif i_prime == 1 and j_prime == -1:
                for idx_x in range(npix):
                    rho[idx_x, -1] = a*rho_dot[idx_x, -1]
                
                for idx_y in reversed(range(npix-1)):
                    for idx_x in range(npix):
                        rho[idx_x, idx_y] = (1.0-1.0/beta)*rho[idx_x, idx_y+1]
                        rho[idx_x, idx_y] += a*rho_dot[idx_x, idx_y]
                        rho[idx_x, idx_y] += a*(1.0-1.0/beta)*rho_dot[idx_x, idx_y+1]
                        if idx_x != 0:
                            rho[idx_x, idx_y] += 1.0/beta*rho[idx_x-1, idx_y+1]
                            rho[idx_x, idx_y] += a/beta*rho_dot[idx_x-1, idx_y+1]
            
            elif i_prime == -1 and j_prime == 1:
                for idx_x in range(npix):
                    rho[idx_x, 0] = a*rho_dot[idx_x, 0]
                
                for idx_y in range(1, npix):
                    for idx_x in range(npix):
                        rho[idx_x, idx_y] = (1.0-1.0/beta)*rho[idx_x, idx_y-1]
                        rho[idx_x, idx_y] += a*rho_dot[idx_x, idx_y]
                        rho[idx_x, idx_y] += a*(1.0-1.0/beta)*rho_dot[idx_x, idx_y-1]
                        if idx_x != npix-1:
                            rho[idx_x, idx_y] += 1.0/beta*rho[idx_x+1, idx_y-1]
                            rho[idx_x, idx_y] += a/beta*rho_dot[idx_x+1, idx_y-1]
                
            elif i_prime == -1 and j_prime == -1:
                for idx_x in range(npix):
                    rho[idx_x, -1] = a*rho_dot[idx_x, -1]
                
                for idx_y in reversed(range(npix-1)):
                    for idx_x in range(npix):
                        rho[idx_x, idx_y] = (1.0-1.0/beta)*rho[idx_x, idx_y+1]
                        rho[idx_x, idx_y] += a*rho_dot[idx_x, idx_y]
                        rho[idx_x, idx_y] += a*(1.0-1.0/beta)*rho_dot[idx_x, idx_y+1]
                        if idx_x != npix-1:
                            rho[idx_x, idx_y] += 1.0/beta*rho[idx_x+1, idx_y+1]
                            rho[idx_x, idx_y] += a/beta*rho_dot[idx_x+1, idx_y+1]
                
        else: # beta <= 1.0
            a = dx/2.0/abs(self.v0x)
            if i_prime == 1 and j_prime == 1:
                for idx_y in range(npix):
                    rho[0, idx_y] = a*rho_dot[0, idx_y]
                
                for idx_x in range(1, npix):
                    for idx_y in range(npix):
                        rho[idx_x, idx_y] = (1.0-beta)*rho[idx_x-1, idx_y]
                        rho[idx_x, idx_y] += a*rho_dot[idx_x, idx_y]
                        rho[idx_x, idx_y] += a*(1.0-beta)*rho_dot[idx_x-1, idx_y]
                        if idx_y != 0:
                            rho[idx_x, idx_y] += beta*rho[idx_x-1, idx_y-1]
                            rho[idx_x, idx_y] += a*beta*rho_dot[idx_x-1, idx_y-1]
            
            elif i_prime == 1 and j_prime == -1:
                for idx_y in range(npix):
                    rho[0, idx_y] = a*rho_dot[0, idx_y]
                
                for idx_x in range(1, npix):
                    for idx_y in range(npix):
                        rho[idx_x, idx_y] = (1.0-beta)*rho[idx_x-1, idx_y]
                        rho[idx_x, idx_y] += a*rho_dot[idx_x, idx_y]
                        rho[idx_x, idx_y] += a*(1.0-beta)*rho_dot[idx_x-1, idx_y]
                        if idx_y != npix-1:
                            rho[idx_x, idx_y] += beta*rho[idx_x-1, idx_y+1]
                            rho[idx_x, idx_y] += a*beta*rho_dot[idx_x-1, idx_y+1]
            
            elif i_prime == -1 and j_prime == 1:
                for idx_y in range(npix):
                    rho[-1, idx_y] = a*rho_dot[-1, idx_y]
                
                for idx_x in reversed(range(0, npix-1)):
                    for idx_y in range(npix):
                        rho[idx_x, idx_y] = (1.0-beta)*rho[idx_x+1, idx_y]
                        rho[idx_x, idx_y] += a*rho_dot[idx_x, idx_y]
                        rho[idx_x, idx_y] += a*(1.0-beta)*rho_dot[idx_x+1, idx_y]
                        if idx_y != 0:
                            rho[idx_x, idx_y] += beta*rho[idx_x+1, idx_y-1]
                            rho[idx_x, idx_y] += a*beta*rho_dot[idx_x+1, idx_y-1]
            
            elif i_prime == -1 and j_prime == -1:
                for idx_y in range(npix):
                    rho[-1, idx_y] = a*rho_dot[-1, idx_y]
                
                for idx_x in reversed(range(0, npix-1)):
                    for idx_y in range(npix):
                        rho[idx_x, idx_y] = (1.0-beta)*rho[idx_x+1, idx_y]
                        rho[idx_x, idx_y] += a*rho_dot[idx_x, idx_y]
                        rho[idx_x, idx_y] += a*(1.0-beta)*rho_dot[idx_x+1, idx_y]
                        if idx_y != npix-1:
                            rho[idx_x, idx_y] += beta*rho[idx_x+1, idx_y+1]
                            rho[idx_x, idx_y] += a*beta*rho_dot[idx_x+1, idx_y+1]
        
        return rho.real
