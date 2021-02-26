import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import logging
import time

import poppy
from poppy.poppy_core import PlaneType, Wavefront, BaseWavefront, BaseOpticalSystem
from . import utils
from . import accel_math
if accel_math._USE_NUMEXPR:
    import numexpr as ne
    pi = np.pi  # needed for evaluation inside numexpr strings.

from .accel_math import _float, _complex
from . import conf

_log = logging.getLogger('poppy')

__all__ = ['QuadPhase', 'QuadraticLens', 'FresnelWavefront', 'FresnelOpticalSystem']


class QuadPhase(poppy.optics.AnalyticOpticalElement):
    """
    Quadratic phase factor,  q(z)
    suitable for representing a radially-dependent wavefront curvature.

    Parameters
    -----------------
    z : float or astropy.Quantity of type length
        radius of curvature
    planetype : poppy.PlaneType constant
        plane type
    name : string
        Descriptive string name

    References
    -------------------
    Lawrence eq. 88

    """

    @utils.quantity_input(z=u.m)
    def __init__(self,
                 z=1.0 * u.m,  # FIXME consider renaming fl? z seems ambiguous with distance.
                 planetype=PlaneType.intermediate,
                 name='Quadratic Wavefront Curvature Operator',
                 **kwargs):
        poppy.AnalyticOpticalElement.__init__(self,
                                              name=name,
                                              planetype=planetype,
                                              **kwargs)
        self.z = z
        self._z_m = z.to(u.m).value

    def get_opd(self, wave):
        """ Return OPD representing a quadratic phase

        Parameters
        ----------
        wave : obj
            a Fresnel Wavefront object
        """

        y, x = wave.coordinates()
        _log.debug("Applying spherical phase curvature ={0:0.2e}".format(self.z))
        _log.debug("Applying spherical lens phase ={0:0.2e}".format(1.0 / self.z))
        z = self._z_m  # numexpr can't evaluate self.
        if (z == np.inf) | (z == -np.inf):
            # Phasor should be 1
            # OPD should be flat
            _log.debug("infinite radius of curvature -> quad phase becomes 0")
            return 0
        if accel_math._USE_NUMEXPR:
            opd = ne.evaluate("(x ** 2 + y ** 2) / (2.0 * z)")
        else:
            opd = (x ** 2 + y ** 2)  / (2.0 *z)

        return opd

class _QuadPhaseShifted(QuadPhase):
    """
    Identical to class 'QuadPhase' except for array origin.
    This class provides a quadratic phase factor for application to FFT shifted wavefronts,
    with the origin in the corner.
    For centered "physical" coordinate system optics with an origin at the wavefront center use  `QuadPhase`.
    """

    def __init__(self, z, **kwargs):
        QuadPhase.__init__(self, z, **kwargs)

    def get_phasor(self, wave):
        """ Return complex phasor, for FFT shifted array

        Parameters
        -----------
        wave : object
            FresnelWavefront instance
        """
        return accel_math._fftshift(super(_QuadPhaseShifted, self).get_phasor(wave))


class QuadraticLens(QuadPhase):
    """
    Gaussian Lens

    Thin wrapper for QuadPhase

    Parameters
    -----------------
    f_lens : float or astropy.Quantity of type length
        Focal length of this lens
    name : string
        Descriptive string name
    planetype : poppy.PlaneType constant
        plane type

    """

    @utils.quantity_input(f_lens=u.m)
    def __init__(self,
                 f_lens=1.0 * u.m,
                 planetype=PlaneType.unspecified,
                 name='Quadratic Lens',
                 **kwargs):
        QuadPhase.__init__(self,
                           f_lens,
                           planetype=planetype,
                           name=name,
                           **kwargs)
        self.fl = f_lens.to(u.m)
        _log.debug("Initialized: " + self.name + ", fl ={0:0.2e}".format(self.fl))

    def __str__(self):
        return "Lens: {0}, with focal length {1}".format(self.name, self.fl)


class ConicLens(poppy.optics.CircularAperture):
    @u.quantity_input(f_lens=u.m, radius=u.m)
    def __init__(self,
                 f_lens=1.0 * u.m,
                 K=1.0,
                 radius=1.0 * u.m,
                 planetype=PlaneType.unspecified,
                 name="Conic lens",
                 **kwargs):
        """Conic Lens/Mirror
        Parabolic, elliptical, hyperbolic, or spherical powered optic.

        Parameters
        ----------------
        f_lens : astropy.quantities.Quantity of dimension length
            Focal length of the optic
        K : float
            Conic constant
        radius: astropy.quantities.Quantity of dimension length
            Radius of the clear aperture of the optic as seen on axis.
        name : string
            Descriptive name
        planetype : poppy.PlaneType, optional
            Optional optical plane type specifier
        """
        super(ConicLens, self).__init__(name=name, radius=radius.to(u.m).value, planetype=planetype, **kwargs)
        self.f_lens = f_lens
        self.K = K


class FresnelWavefront(BaseWavefront):
    angular_coordinates = False
    """Should coordinates be expressed in arcseconds instead of meters at the current plane? """
    
    @u.quantity_input(beam_radius=u.m)
    def __init__(self,
                 beam_radius,
                 units=u.m,
                 rayleigh_factor=2.0,
                 oversample=2,
                 **kwargs):
        """
        Wavefront for Fresnel diffraction calculation.

        This class inherits from and extends the Fraunhofer-domain
        poppy.Wavefront class.


        Parameters
        --------------------
        beam_radius : astropy.Quantity of type length
            Radius of the illuminated beam at the initial optical plane.
            I.e. this would be the pupil aperture radius in an entrance pupil.
        units : astropy.units.Unit
            Astropy units of input parameters
        rayleigh_factor:
            Threshold for considering a wave spherical.
        oversample : float
            Padding factor to apply to the wavefront array, multiplying on top of the beam radius.


        References
        -------------------
        - Lawrence, G. N. (1992), Optical Modeling, in Applied Optics and Optical Engineering., vol. XI,
            edited by R. R. Shannon and J. C. Wyant., Academic Press, New York.

        - https://en.wikipedia.org/wiki/Gaussian_beam

        - IDEX Optics and Photonics(n.d.), Gaussian Beam Optics,
            [online] Available from:
            https://marketplace.idexop.com/store/SupportDocuments/All_About_Gaussian_Beam_OpticsWEB.pdf

        - Krist, J. E. (2007), PROPER: an optical propagation library for IDL,
            vol. 6675, p. 66750P-66750P-9.
            [online] Available from: http://dx.doi.org/10.1117/12.731179

        - Andersen, T., and A. Enmark (2011), Integrated Modeling of Telescopes, Springer Science & Business Media.

        """
        super(FresnelWavefront, self).__init__(
            diam=beam_radius.to(u.m).value * 2.0,
            oversample=oversample,
            **kwargs
        )
        try:
            units.to(u.m)
        except (AttributeError, u.UnitsError):
            raise ValueError("The 'units' parameter must be an astropy.units.Unit representing length.")
        self.units = units
        """`astropy.units.Unit` for measuring distance"""

        self.w_0 = beam_radius.to(self.units)  # convert to base units.
        """Beam waist radius at initial plane"""
        self.z = 0 * units
        """Current wavefront coordinate along the optical axis"""
        self.z_w0 = 0 * units
        """Coordinate along the optical axis of the latest beam waist"""
        self.waists_w0 = [self.w_0.to(u.m).value]
        """List of beam waist radii, in series as encountered during the course of an optical propagation."""
        self.waists_z = [self.z_w0.to(u.m).value]
        """List of beam waist distances along the optical axis, in series as encountered
        during the course of an optical propagation."""
        self.spherical = False
        """Is this wavefront spherical or planar?"""
        self.k = np.pi * 2.0 / self.wavelength
        """ Wavenumber"""
        self.rayleigh_factor = rayleigh_factor
        """Threshold for considering a wave spherical, in units of Rayleigh distance"""

        self.focal_length = np.inf * u.m
        """Focal length of the current beam, or infinity if not a focused beam"""

        if self.oversample > 1 and not self.ispadded:  # add padding for oversampling, if necessary
            self.wavefront = utils.pad_to_oversample(self.wavefront, self.oversample)
            self.ispadded = True
            logmsg = "Padded WF array for oversampling by {0:d}, to {1}.".format(
                self.oversample,
                self.wavefront.shape
            )
            _log.debug(logmsg)

            self.history.append(logmsg)
        else:
            _log.debug("Skipping oversampling, oversample < 1 or already padded ")

        if self.oversample < 2:
            _log.warning("Oversampling > 2x suggested for reliable results in Fresnel propagation.")

        self._y, self._x = np.indices(self.shape, dtype=float)
        self._y -= (self.wavefront.shape[0]) / 2.0
        self._x -= (self.wavefront.shape[1]) / 2.0
        """saves x and y indices for future use"""

        # FIXME MP: this self.n attribute appears unnecessary?
        if self.shape[0] == self.shape[1]:
            self.n = self.shape[0]
        else:
            self.n = self.shape

        if self.planetype == PlaneType.image:
            raise ValueError(
                "Input wavefront needs to be a pupil plane in units of m/pix. Specify a diameter not a pixelscale.")

    def display(self, *args, **kwargs):
        if 'use_angular_coordinates' not in kwargs:
            # Is this FresnelWavefront in angular units?
            return super(FresnelWavefront, self).display(
                *args,
                use_angular_coordinates=self.angular_coordinates,
                **kwargs
            )
        else:
            # ensure this FresnelWavefront's coordinates are
            # temporarily set to the requested type, so that
            # the self.coordinates() call will yield results
            # appropriate for displaying that type.
            tmp = self.angular_coordinates
            self.angular_coordinates = kwargs['use_angular_coordinates']
            retval = super(FresnelWavefront, self).display(
                *args, **kwargs
            )
            self.angular_coordinates = tmp
            return retval

    display.__doc__ = Wavefront.display.__doc__

    # properties and methods supporting fresnel propagation

    @property
    def z_r(self):
        """
        Rayleigh distance for the gaussian beam, based on
        current beam waist and wavelength.

        I.e. the distance along the propagation direction from the
        beam waist at which the area of the cross section has doubled.
        The depth of focus is conventionally twice this distance.
        """

        return np.pi * self.w_0 ** 2 / self.wavelength

    @property
    def divergence(self):
        """
        Half-angle divergence of the gaussian beam

        I.e.  the angle between the optical axis and the beam radius (at a large distance  from the waist) in radians.
        """
        return self.wavelength / (np.pi * self.w_0)

    @property
    def param_str(self):
        """
        Formatted string of gaussian beam parameters.
        """
        string = "w_0:{0:0.3e},".format(self.w_0) + " z_w0={0:0.3e}".format(self.z_w0) + "\n" + \
                 "z={0:0.3e},".format(self.z) + " z_r={0:0.3e}".format(self.z_r)
        return string

    @property
    def waists(self):
        """
        each [z_w_0,w_0] for each waist generated by an optic
        """
        return np.array([self.waists_z, self.waists_w0])

    def _fft(self):
        """
        Apply normalized forward 2D Fast Fourier Transform to wavefront
        """
        self.wavefront = accel_math.fft_2d(self.wavefront, forward=True, fftshift=False)

    def _inv_fft(self):
        """
        Apply normalized Inverse 2D Fast Fourier Transform to wavefront
        """
        self.wavefront = accel_math.fft_2d(self.wavefront, forward=False, fftshift=False)

    def r_c(self, z=None):
        """
        The gaussian beam radius of curvature as a function of distance z

        Parameters
        -------------
        z : float, optional
            Distance along the optical axis.
            If not specified, the wavefront's current z coordinate will
            be used, returning the beam radius of curvature at the current position.

        Returns
        -------
        Astropy.units.Quantity of dimension length

        """
        if z is None:
            z = self.z
        dz = (z - self.z_w0)  # z relative to waist
        if dz == 0:
            return np.inf * u.m
        return dz * (1 + (self.z_r / dz) ** 2)

    def spot_radius(self, z=None):
        """
        radius of a propagating gaussian wavefront, at a distance z

        Parameters
        -------------
        z : float, optional
            Distance along the optical axis.
            If not specified, the wavefront's current z coordinate will
            be used, returning the beam radius at the current position.

        Returns
        -------
        Astropy.units.Quantity of dimension length
        """
        if z is None:
            z = self.z
        return self.w_0 * np.sqrt(1.0 + ((z - self.z_w0) / self.z_r) ** 2)

    #  methods supporting coordinates, including switching between distance and angular units

    @staticmethod
    def pupil_coordinates(x, y, pixelscale):
        """Utility function to generate coordinates arrays for a pupil
        plane wavefront

        Parameters
        ------------
        x, y : array_like
            pixel indices
        pixelscale : float or 2-tuple of floats
            the pixel scale in meters/pixel, optionally different in
            X and Y

        Returns
        -------
        Y, X :  array_like
            Wavefront coordinates in either meters or arcseconds for pupil and image, respectively
        """
        # Override parent class method to provide one that's comparatible with
        # FFT indexing conventions. Centered one one pixel not on the middle
        # of the array.
        # This function is intentionally distinct from the regular Wavefront.coordinates(), and behaves
        # slightly differently. This is required for use in the angular spectrum propagation in the PTP and
        # Direct propagations.

        pixelscale_mpix = pixelscale.to(u.meter / u.pixel).value
        if not np.isscalar(pixelscale_mpix):
            pixel_scale_x, pixel_scale_y = pixelscale_mpix
        else:
            pixel_scale_x, pixel_scale_y = pixelscale_mpix, pixelscale_mpix

        if accel_math._USE_NUMEXPR:
            return ne.evaluate("pixel_scale_y * y"), ne.evaluate("pixel_scale_x * x")
        else:
            return pixel_scale_y * y, pixel_scale_x * x

    def coordinates(self):
        """ Return Y, X coordinates for this wavefront, in the manner of numpy.indices()

        This function knows about the offset resulting from FFTs. Use it whenever computing anything
        measured in wavefront coordinates.

        The behavior for Fresnel wavefronts is slightly different from
        Fraunhofer wavefronts, in that the optical axis is *not* the exact
        center of an array (the corner between pixels for an even number of pixels),
        but rather is a specific pixel (e.g. pixel 512,512 for a 1024x1024 array).
        This is for consistency with the array indexing convention used in FFTs since
        this class depends on FFTs rather than the more flexible matrix DFTs for its
        propagation.

        For Fresnel wavefronts, this depends on the focal length to get the image scale right.

        Returns
        -------
        Y, X :  array_like
            Wavefront coordinates in either meters or arcseconds for pupil and image, respectively
        """

        y, x = type(self).pupil_coordinates(self._x, self._y, self._pixelscale_m)

        # If the wavefront been explicitly set to use angular units,
        # for instance at an image plane,then
        # then convert to angular coordinates using the focal length
        if self.angular_coordinates:
            if not np.isfinite(self.focal_length.value):
                raise ValueError("Cannot convert to angular units for a beam with infinite focal length")
            platescale = (1 * u.radian / self.focal_length).to(u.arcsec / u.m)
            _log.debug("Converting to angular coords using plate scale = {}".format(platescale))
            y *= platescale.value
            x *= platescale.value

        return y, x

    @property
    def pixelscale(self):
        """ Pixelscale, in meters by default or in arcseconds if angular_coordinates is True """
        if self.angular_coordinates:
            return ((1 * u.radian / self.focal_length).to(u.arcsec / u.m)) * self._pixelscale_m
        else:
            return self._pixelscale_m

    @pixelscale.setter
    def pixelscale(self, value):
        if self.angular_coordinates:
            raise RuntimeError("Cannot set pixelscale of Fresnel wavefront while in angular units.")
        self._pixelscale_m = value

    @property
    def fov(self):
        """ FOV in arcseconds, if applicable"""
        if self.angular_coordinates:
            return np.asarray(self.wavefront.shape) * u.pixel * self.pixelscale
        else:
            return None

    # methods for optical propagation

    @utils.quantity_input(z=u.meter)
    def propagate_direct(self, z):
        """
        Implements the direct propagation algorithm as described in Andersen & Enmark (2011). Works best for
        far field propagation. Not part of the Gaussian beam propagation method.

        Parameters
        ----------
        z :  float or Astropy.Quantity length
            the distance from the current location to propagate the beam.
        """
        self.angular_coordinates = False  # coordinates must be in meters for propagation

        z_direct = z.to(u.m).value
        y, x = self.coordinates()
        k = np.pi * 2.0 / self.wavelength.to(u.meter).value
        s = self.n * u.pix * self.pixelscale  # S is "simulation size" and has length of meters
        _log.debug(
            "Propagation Parameters: k={0:0.2e},".format(k) + "S={0:0.2e},".format(s) + "z={0:0.2e},".format(z_direct))

        # TODO the following exponential code could be accelerated with numexpr
        quadphase_1st = np.exp(1.0j * k * (x ** 2 + y ** 2) / (2 * z_direct))  # eq. 6.68
        quadphase_2nd = np.exp(1.0j * k * z_direct) / (1.0j * self.wavelength.to(u.m).value * z_direct) * np.exp(
            1.0j * k * (x ** 2 + y ** 2) / (2 * z_direct))  # eq. 6.70

        stage1 = self.wavefront * quadphase_1st  # eq.6.67
        if z_direct > 0:
            result = accel_math._ifftshift(stage1)
            result = accel_math.fft_2d(result, forward=True, fftshift=False)
            result = accel_math._fftshift(result)
            result *= self.pixelscale.to(u.m / u.pix).value ** 2  # eq.6.69 and #6.80
        else:
            result = accel_math._fftshift(stage1)
            result = accel_math.fft_2d(result, forward=False, fftshift=False)
            result = accel_math._ifftshift(result)
            result *= self.pixelscale.to(u.m / u.pix).value ** 2 * self.n ** 2
        result *= quadphase_2nd

        self.pixelscale = self.wavelength * abs(z) / s / u.pix
        self.wavefront = result
        self.history.append("Direct propagation to z= {0:0.2e}".format(z))
        self.z += z

    @utils.quantity_input(distance=u.meter)
    def propagate_to(self, optic, distance):
        """Propagates a wavefront object to the next optic in the list, after
        some separation distance (which might be zero).
        Modifies this wavefront object itself.

        Transformations between most planes use Fresnel propagation.
        If the target plane is an image plane, the output wavefront will be set to provide its
        coordinates in arcseconds based on its focal length, but it retains its internal dimensions
        in meters for future Fresnel propagations.
        Transformations to a Detector plane are handled separately to allow adjusting the pixel scale
        to match the target scale.
        Transformations from any frame through a rotation plane simply rotate the wavefront accordingly.

        Parameters
        -----------
        optic : OpticalElement
            The optic to propagate to. Used for determining the appropriate optical plane.
        distance : astropy.Quantity of dimension length
            separation distance of this optic relative to the prior optic in the system.
        """
        msg = "  Propagating wavefront to {0} after distance {1} ".format(str(optic), distance)
        _log.debug(msg)
        self.history.append(msg)
        self.angular_coordinates = False  # coordinates must be in meters for propagation

        # Apply Fresnel propagation for the specified distance, regardless of
        # what type of plane is next
        if distance != 0 * u.m:
            self.propagate_fresnel(distance)

        self.current_plane_index += 1

        # Now we may do some further manipulations depending on the next plane
        if optic.planetype == PlaneType.rotation:  # rotate
            self.rotate(optic.angle)
            self.location = 'after ' + optic.name
        elif optic.planetype == PlaneType.image:
            self.location = 'before ' + optic.name
            self.angular_coordinates = True  # image planes want angular coordinates
            self.planetype = PlaneType.image  # needed for back compatibility when using image plane optics
        elif optic.planetype == PlaneType.detector:
            self._resample_wavefront_pixelscale(optic)
            self.location = 'at detector ' + optic.name
        else:
            self.location = 'before ' + optic.name

    @utils.quantity_input(dz=u.meter)
    def _propagate_ptp(self, dz):
        """ Plane-to-Plane Fresnel propagation.

        This function propagates a planar wavefront some distance
        while keeping it planar, yielding a planar output wavefront.
        This is used for propagation entirely within the Rayleigh
        distance of the beam waist.


        Parameters
        ----------
        dz :  float
            the distance from the current location to propagate the beam.

        References
        ----------
        Lawrence eq. 82, 86,87
        """

        if self.spherical:
            raise RuntimeError(
                '_propagate_ptp can only start from a planar wavefront, but was called with a spherical one.')

        if isinstance(dz, u.quantity.Quantity):
            z_direct = dz.to(u.m).value  # convert to meters.
        else:
            _log.warning("z= {0:0.2e}, has no units, assuming meters ".format(dz))
            z_direct = dz

        if np.abs(dz) < 1 * u.Angstrom:
            _log.debug("Skipping small dz = " + str(dz))
            # TODO: make this scale with physics and only skip un-interesting
            # distances instead of this arbitrary length -douglase
            return

        x, y = self.coordinates()  # meters
        meter_per_pix = self.pixelscale.to(u.m / u.pix).value
        rhosqr = accel_math._fftshift((x / (meter_per_pix ** 2 * self.n)) ** 2 + (
            y / (meter_per_pix ** 2 * self.n)) ** 2)
        # Transfer Function of diffraction propagation eq. 22, eq. 87
        wavelen_m = self.wavelength.to(u.m).value

        if accel_math._USE_NUMEXPR:
            exp_t = ne.evaluate("exp(-1.0j * pi * wavelen_m * (z_direct) * rhosqr)")
        else:
            exp_t = np.exp(-1.0j * np.pi * wavelen_m * z_direct * rhosqr)

        self._fft()

        self.wavefront *= exp_t  # eq. 6.68

        self._inv_fft()
        self.z += dz

        self.history.append("Propagated Plane-to-Plane, dz = " + str(z_direct))

    @utils.quantity_input(dz=u.meter)
    def _propagate_wts(self, dz):
        """ Waist-to-Spherical Fresnel propagation

        This function propagates a planar input wavefront to become a spherical wavefront.
        The starting position should be within the Rayleigh distance of the waist, and the
        ending position will be outside of that.

        Parameters
        -----------
        dz :  float
            the distance from the current location to propagate the beam.

        References
        ----------
         Lawrence eq. 83,88
        """
        _log.debug("Waist to Spherical propagation, dz=" + str(dz))

        if self.spherical:
            raise RuntimeError(
                '_propagate_wts can only start from a planar wavefront, but was called with a spherical one.')

        if dz == 0:
            _log.error("Waist to Spherical propagation stopped, no change in distance.")
            return

        self *= _QuadPhaseShifted(dz)

        if dz > 0:
            self._fft()
        else:
            self._inv_fft()

        self.pixelscale = self.wavelength * np.abs(dz) / (self.n * u.pixel * self.pixelscale) / u.pixel
        self.z += dz
        self.history.append("Propagated Waist to Spherical, dz = " + str(dz))
        self.spherical = True  # wavefront is now spherical

    @utils.quantity_input(dz=u.meter)
    def _propagate_stw(self, dz):
        """Spherical-to-Waist Fresnel propagation

        This function propagates a spherical wavefront to become a planar wavefront.
        The starting position should be outside the Rayleigh distance of the waist,
        and the ending position will be inside of it.


        Parameters
        ----------
        dz :  float
            the distance from the current location to propagate the beam, in meters

        References
        ----------
         Lawrence eq. 89
        """

        if not self.spherical:
            raise RuntimeError(
                '_propagate_stw can only start from a spherical wavefront, but was called with a planar one.')

        _log.debug("Spherical to Waist propagation, dz=" + str(dz))

        if dz == 0 * u.meter:
            _log.error("Spherical to Waist propagation stopped, no change in distance.")
            return

        if dz > 0 * u.meter:
            self._fft()
        else:
            self._inv_fft()

        # update to new pixel scale before applying curvature
        self.pixelscale = self.wavelength * np.abs(dz) / (self.n * u.pixel * self.pixelscale) / u.pixel
        self *= _QuadPhaseShifted(dz)
        self.z += dz
        self.history.append("Propagated Spherical to Waist, dz = " + str(dz))
        self.spherical = False  # wavefront is now planar

    def planar_range(self, z):
        """
        Returns True if the input range z is within the Rayleigh range of the waist.

        Parameters
        -----------
        z : float
            distance from the beam waist

        """
        return np.abs(self.z_w0 - z) < self.z_r

    @utils.quantity_input(delta_z=u.meter)
    def propagate_fresnel(self, delta_z, display_intermed=False):
        """Top-level routine for Fresnel diffraction propagation


        Each spherical wavefront is propagated to a waist and then to the next appropriate plane
         (spherical or planar).

        Parameters
        ----------
        delta_z :  float
            the distance from the current location to propagate the beam.
        display_intermed : boolean
             If True, display the complex start, intermediates waist and end surfaces.


        """
        self.angular_coordinates = False  # coordinates must be in meters for propagation
        z = self.z + delta_z
        if display_intermed:
            plt.figure()
            self.display('both', colorbar=True, title="Starting Surface")

        self.wavefront = accel_math._fftshift(self.wavefront)
        _log.debug("Beginning Fresnel Prop. Waist at z = " + str(self.z_w0))

        if not self.spherical:
            if self.planar_range(z):
                # Plane waves inside planar range:  use plane-to-plane
                _log.debug('  Plane to Plane Regime, dz=' + str(delta_z))
                _log.debug('  Constant Pixelscale: {}'.format(self.pixelscale))
                self._propagate_ptp(delta_z)
            else:
                # Plane wave to spherical. First use PTP to the waist, then WTS to Spherical
                _log.debug('  Plane to Spherical, inside Z_R to outside Z_R')
                _log.debug('  Starting Pixelscale: {}'.format(self.pixelscale))
                self._propagate_ptp(self.z_w0 - self.z)
                if display_intermed:
                    plt.figure()
                    self.display('both', colorbar=True)
                self._propagate_wts(z - self.z_w0)
        else:
            if self.planar_range(z):
                # Spherical to plane. First use STW to the waist, then PTP to the desired plane
                _log.debug('  Spherical to Plane Regime, outside Z_R to inside Z_R')
                self._propagate_stw(self.z_w0 - self.z)
                if display_intermed:
                    plt.figure()
                    self.display('both', colorbar=True, title='Intermediate Waist')
                self._propagate_ptp(z - self.z_w0)
            else:
                # Spherical to Spherical. First STW to the waist, then WTS to the desired spherical surface
                _log.debug('  Spherical to Spherical, Outside Z_R to waist (z_w0) to outside Z_R')
                _log.debug('  Starting Pixelscale: {}'.format(self.pixelscale))
                self._propagate_stw(self.z_w0 - self.z)
                _log.debug('  Intermediate Pixelscale: {}'.format(self.pixelscale))

                if display_intermed:
                    plt.figure()
                    self.display('both', colorbar=True, title='Intermediate Waist')
                self._propagate_wts(z - self.z_w0)
        if display_intermed:
            plt.figure()
            self.display('both', colorbar=True)

        self.wavefront = accel_math._fftshift(self.wavefront)
        self.planetype = PlaneType.intermediate
        _log.debug("------ Propagated to plane of type " + str(self.planetype) + " at z = {0:0.2e} ------".format(z))

    def __imul__(self, optic):
        """Multiply a Wavefront by an OpticalElement or scalar"""
        if isinstance(optic, QuadraticLens):
            # Special case: if we have a lens, call the routine for that,
            # which will modify the properties of this wavefront more fundamentally
            # than most other optics, adjusting beam parameters and so forth
            self.apply_lens_power(optic)
            return self
        else:
            # Otherwise fall back to the parent class
            return super(FresnelWavefront, self).__imul__(optic)

    def apply_lens_power(self, optic, ignore_wavefront=False):
        """
        Adds lens wavefront curvature to the wavefront
        corresponding to the lens' focal length f_l, and updates the
        Gaussian beam parameters of the wavefront.


        Parameters
        ----------
        optic : QuadraticLens
            An optic
        ignore_wavefront : boolean
            If True then only gaussian beam propagation parameters will be updated and the wavefront surface will not
            be calculated. Useful for quick calculations of gaussian laser beams

        """

        _log.debug("------ Applying Lens: " + str(optic.name) + " ------")
        _log.debug("  Pre-Lens Beam Parameters: " + self.param_str)

        # calculate beam radius at current surface
        spot_radius = self.spot_radius()
        _log.debug("  Beam radius at " + str(optic.name) + " ={0:0.2e}".format(spot_radius))

        # Is the incident beam planar or spherical?
        # We decided based on whether the last waist is outside the rayleigh distance.
        #  I.e. here we neglect small curvature just away from the waist
        # Based on that, determine the radius of curvature of the output beam
        if np.abs(self.z_w0 - self.z) > self.rayleigh_factor * self.z_r:
            _log.debug("spherical beam")
            _log.debug(self.param_str)
            r_input_beam = self.z - self.z_w0
            r_output_beam = 1.0 / (1.0 / self.r_c() - 1.0 / optic.fl)
            _log.debug(
                " input curved wavefront and " + str(optic.name) + " has output beam curvature of ={0:0.2e}".format(
                    r_output_beam))
        else:
            r_input_beam = np.inf * u.m
            # we are at a focus or pupil, so the new optic is the only curvature of the beam
            r_output_beam = -1 * optic.fl
            _log.debug(
                " input flat wavefront and " + str(optic.name) + " has output beam curvature of ={0:0.2e}".format(
                    r_output_beam))

        # update the wavefront parameters to the post-lens beam waist
        if self.r_c() == optic.fl:
            self.z_w0 = self.z
            self.w_0 = spot_radius
            _log.debug(str(optic.name) + " has a flat output wavefront")
        else:
            self.z_w0 = -r_output_beam / (
                1.0 + (self.wavelength * r_output_beam / (np.pi * spot_radius ** 2)) ** 2) + self.z
            self.w_0 = spot_radius / np.sqrt(1.0 + (np.pi * spot_radius ** 2 / (self.wavelength * r_output_beam)) ** 2)
            _log.debug(str(optic.name) + " has a curvature of ={0:0.2e}".format(r_output_beam))
            _log.debug(str(optic.name) + " has a curved output wavefront, with waist at {}".format(self.z_w0))

        _log.debug("Post Optic Parameters:" + self.param_str)

        # Update the focal length of the beam. This is closely related to but tracked separately from
        # the beam waist and radius of curvature; we keep track of it to use in optional conversion
        # of coordinates to angular units.
        if not np.isfinite(self.focal_length):
            self.focal_length = 1 * optic.fl
            _log.debug("Set output beam focal length to {}".format(self.focal_length))
        else:
            # determine magnification as the change in curvature of this optic
            mag = r_output_beam / r_input_beam
            self.focal_length *= mag
            _log.debug("Magnification: {}  from R_in = {}, R_out = {}".format(mag, r_input_beam, r_output_beam))
            _log.debug("Output beam focal length is now {}".format(self.focal_length))

        self.waists_z.append(self.z_w0.to(u.m).value)
        self.waists_w0.append(self.w_0.to(u.m).value)

        # update wavefront location:
        if optic.planetype != PlaneType.unspecified:
            self.planetype = optic.planetype

        if ignore_wavefront:
            # What we have done above is sufficient for Gaussian beam propagation,
            # and if that's all we're interested in we can skip updating the
            # wavefront array.
            _log.debug("------ Optic: " + str(optic.name) + " applied, for Gaussian beam parameters only ------")
            return

        # Now we need to figure out the phase term to apply to the wavefront
        # data array
        if not self.spherical:
            if np.abs(self.z_w0 - self.z) < self.z_r:
                _log.debug('Near-field, Plane-to-Plane Propagation.')
                z_eff = 1 * optic.fl

            else:
                # find the radius of curvature of the lens output beam
                # curvatures are multiplicative exponentials
                # e^(1/z) = e^(1/x)*e^(1/y) = e^(1/x+1/y) -> 1/z = 1/x + 1/y
                # z = 1/(1/x+1/y) = xy/x+y
                z_eff = 1.0 / (1.0 / optic.fl + 1.0 / (self.z - self.z_w0))
                _log.debug('Inside Rayleigh distance to Outside Rayleigh distance.')

                self.spherical = True

        else:  # spherical input wavefront
            if np.abs(self.z_w0 - self.z) > self.z_r:
                _log.debug('Spherical to Spherical wavefront propagation.')
                _log.debug("1/fl={0:0.4e}".format(1.0 / optic.fl))
                _log.debug("1.0/(R_input_beam)={0:0.4e}".format(1.0 / r_input_beam))
                _log.debug("1.0/(self.z-self.z_w0)={0:0.4e}".format(1.0 / (self.z - self.z_w0)))

                if (self.z - self.z_w0) == 0:
                    z_eff = 1.0 / (1.0 / optic.fl + 1.0 / (self.z - self.z_w0))
                else:
                    z_eff = 1.0 / (1.0 / optic.fl + 1.0 / (self.z - self.z_w0) - 1.0 / r_input_beam)

            else:
                _log.debug('Spherical to Planar.')
                z_eff = 1.0 / (1.0 / optic.fl - 1.0 / r_input_beam)
                self.spherical = False

        # Apply phase to the wavefront array
        effective_optic = QuadPhase(-z_eff, name=optic.name)
        self *= effective_optic

        _log.debug("------ Optic: " + str(optic.name) + " applied ------")

    def _resample_wavefront_pixelscale(self, detector):
        """ Resample a Fresnel wavefront to a desired detector sampling.

        The interpolation is done via the scipy.ndimage.zoom function, by default
        using cubic interpolation.  If you wish a different order of interpolation,
        set the `.interp_order` attribute of the detector instance.

        Parameters
        ----------
        detector : Detector class instance
            Detector that defines the desired pixel scale

        Returns
        -------
        The wavefront object is modified to have the appropriate pixel scale and spatial extent.

        """

        if self.angular_coordinates:
            raise NotImplementedError("Resampling to detector doesn't yet work in angular coordinates for Fresnel.")

        pixscale_ratio = (self.pixelscale / detector.pixelscale).decompose().value

        if np.abs(pixscale_ratio - 1.0) < 1e-3:
            _log.debug("Wavefront is already at desired pixel scale "
                       "{:.4g}.  No resampling needed.".format(self.pixelscale))
            self.wavefront = utils.pad_or_crop_to_shape(self.wavefront, detector.shape)
            return

        super(FresnelWavefront, self)._resample_wavefront_pixelscale(detector)

        self.n = detector.shape[0]

    @classmethod
    def from_wavefront(cls, wavefront):
        """Convert a Fraunhofer type wavefront object to a Fresnel one

        Note, for now this function only works if the input wavefront is at a
        pupil plane, so the Fraunhofer wavefront has pixelscale
        in meters/pix rather than arcsec/pix. Conversion from
        image planes may be added later.

        Parameters
        ----------
        wavefront : Wavefront
            The (Fraunhofer-type) wavefront to be converted

        """
        # Generate a Fresnel wavefront with the same sampling
        wf = wavefront

        if wf.planetype == PlaneType.image:
            raise NotImplementedError("Conversion from image planes to Fresnel is not yet implemented.")

        if wf.ispadded:
            beam_radius = wf.wavefront.shape[0] / wf.oversample / 2 * wf.pixelscale * u.pixel
        else:
            beam_radius = wf.wavefront.shape[0] / 2 * wf.pixelscale * u.pixel
        new_wf = FresnelWavefront(beam_radius=beam_radius,
                                  npix=wf.shape[0],
                                  oversample=wf.oversample,
                                  wavelength=wf.wavelength)
        # Deal with metadata
        new_wf.history = wf.history.copy()
        new_wf.history.append("Converted to Fresnel propagation")
        new_wf.history.append("  Fresnel array pixel scale = {:.4g}, oversample = {}".format(new_wf.pixelscale, new_wf.oversample))
        # Copy over the contents of the array
        new_wf.wavefront = utils.pad_to_size(wf.wavefront, new_wf.shape)
        # Copy over misc internal info
        if hasattr(wf, '_display_hint_expected_nplanes'):
            new_wf._display_hint_expected_nplanes = wf._display_hint_expected_nplanes
        new_wf.current_plane_index = wf.current_plane_index
        new_wf.location = wf.location

        return new_wf


class FresnelOpticalSystem(BaseOpticalSystem):
    """ Class representing a series of optical elements,
    through which light can be propagated using the Fresnel formalism.

    This is comparable to the "regular" (Fraunhofer-domain)
    OpticalSystem, but adds functionality for propagation to
    arbitrary optical planes rather than just pupil and image planes.

    Parameters
    -------------
    name : string
        descriptive name of optical system
    pupil_diameter : astropy.Quantity of dimension length
        Diameter of entrance pupil
    npix : int
        Number of pixels across the entrance pupil by default 1024
    beam_ratio : float
        Padding factor for the entrance pupil; what fraction of the array should
        correspond to the entrance pupil. Default is 0.5, which corresponds to
        Nyquist sampling (2 pixels per resolution element)
    verbose : bool
        whether to be more verbose with log output while computing
    """

    @u.quantity_input(pupil_diameter=u.m)
    def __init__(self, name="unnamed system", pupil_diameter=1 * u.m,
                 npix=1024, beam_ratio=0.5, verbose=True):
        super(FresnelOpticalSystem, self).__init__(name=name, verbose=verbose)
        self.pupil_diameter = pupil_diameter
        self.beam_ratio = beam_ratio
        del self.oversample  # use beam_ratio instead for fresnel systems
        self.npix = npix

        self.distances = []  # distance along the optical axis to each successive optic

    @u.quantity_input(distance=u.m)
    def add_optic(self, optic=None, distance=0.0 * u.m, index=None):
        """ Add an optic to the optical system

        Parameters
        ---------------
        optic : OpticalElement instance
            Some optic
        distance : astropy.Quantity of dimension length
            separation distance of this optic relative to the prior optic in the system.
        index : int
            Index at which to insert the new optical element

        """
        if index is None:
            # Optic is appended to the end of the system
            self.planes.append(optic)
            self.distances.append(distance.to(u.m))
        else:
            # Insert the optic into the middle of the beam train somewhere
            self.planes.insert(index, optic)
            self.distances.insert(index, distance.to(u.m))

        if self.verbose:
            _log.info("Added optic: {0} after separation: {1:.2e} ".format(self.planes[-1].name, distance))

        return optic

    @u.quantity_input(distance=u.m, pixelscale=u.micron / u.pixel)
    def add_detector(self, pixelscale=10 * u.micron / u.pixel, fov_pixels=10 * u.pixel,
                     distance=0.0 * u.m):
        """ Add a detector to the optical system

        Parameters
        ----------
        pixelscale : astropy.Quantity, with units micron/pixel or equivalent
            The pixel scale at the detector
        fov_pixels : astropy.Quantity with units pixel
            The number of pixels per axis of the detector. Assumes square detector.
        distance : astropy.Quantity of dimension length
            separation distance of this optic relative to the prior optic in the system.

        """
        super(FresnelOpticalSystem, self).add_detector(pixelscale=pixelscale, fov_pixels=fov_pixels)
        self.distances.append(distance)
        if self.verbose:
            _log.info("Added detector: {0} after separation: {1:.2e} ".format(self.planes[-1].name, distance))
    
    @utils.quantity_input(wavelength=u.meter)
    def input_wavefront(self, wavelength=1e-6 * u.meter, inwave=None):
        """Create a Wavefront object suitable for sending through a given optical system.

        Uses self.source_offset to assign an off-axis tilt, if requested.
        (FIXME does not work for Fresnel yet)

        Parameters
        ----------
        wavelength : float
            Wavelength in meters

        Returns
        -------
        wavefront : poppy.fresnel.FresnelWavefront instance
            A wavefront appropriate for passing through this optical system.

        """
        oversample = int(np.round(1 / self.beam_ratio))
        if isinstance(inwave, poppy.FresnelWavefront) :
            _log.info('Using user-defined wavefront for the input wavefront.')
            inwave = inwave
        elif inwave==None:
            _log.info('No input wavefront provided, generating input wavefront.' )
            inwave = FresnelWavefront(self.pupil_diameter / 2, wavelength=wavelength,
                                      npix=self.npix, oversample=oversample)
        else:
            raise ValueError("Input wavefront must be a FresnelWavefront() object when using FresnelOpticalSystem() or None.")
 
        _log.debug(
            "Input wavefront created with wavelength={0} microns,"
            "npix={1}, diam={3}, pixel scale={2}".format(
                wavelength.to(u.micron).value, self.npix, self.pupil_diameter / (self.npix * u.pixel), self.pupil_diameter
            ))
        inwave._display_hint_expected_nplanes = len(self)     # For displaying a multi-step calculation nicely
        return inwave
    
    def propagate(self,
                  wavefront,
                  normalize='none',
                  return_intermediates=False,
                  display_intermediates=False):
        """ Core low-level routine for propagating a wavefront through an optical system

        See docstring of OpticalSystem.propagate for details

        """
        intermediate_wfs = []

        for optic, distance in zip(self.planes, self.distances):
            # The actual propagation:
            wavefront.propagate_to(optic, distance)
            wavefront *= optic

            # Normalize if appropriate:
            if normalize.lower() == 'first' and wavefront.current_plane_index == 1:  # set entrance plane to 1.
                wavefront.normalize()
                _log.debug("normalizing at first plane (entrance pupil) to 1.0 total intensity")
            elif normalize.lower() == 'first=2' and wavefront.current_plane_index == 1:
                # this undocumented option is present only for testing/validation purposes
                wavefront.normalize()
                wavefront *= np.sqrt(2)
            elif normalize.lower() == 'exit_pupil':  # normalize the last pupil in the system to 1
                last_pupil_plane_index = np.where(
                    np.asarray([p.planetype is PlaneType.pupil for p in self.planes]))[0].max() + 1
                if wavefront.current_plane_index == last_pupil_plane_index:
                    wavefront.normalize()
                    _log.debug(
                        "normalizing at exit pupil (plane {0}) to 1.0 total intensity".format(wavefront.current_plane_index))
            elif normalize.lower() == 'last' and wavefront.current_plane_index == len(self.planes):
                wavefront.normalize()
                _log.debug("normalizing at last plane to 1.0 total intensity")

            # Optional outputs:
            if poppy.conf.enable_flux_tests:
                _log.debug("  Flux === " + str(wavefront.total_intensity))

            if return_intermediates:  # save intermediate wavefront, summed for polychromatic if needed
                intermediate_wfs.append(wavefront.copy())

            if display_intermediates:
                if poppy.conf.enable_speed_tests:
                    t0 = time.time()

                wavefront._display_after_optic(optic)

                if poppy.conf.enable_speed_tests:
                    t1 = time.time()
                    _log.debug("\tTIME %f s\t for displaying the wavefront." % (t1 - t0))

        if poppy.conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop - t_start))

        if return_intermediates:
            return wavefront, intermediate_wfs
        else:
            return wavefront

    def describe(self):
        """ Print out a string table describing all planes in an optical system"""
        res = (str(self) +
               "\n\tEntrance pupil diam:  {0}\tnpix: {1}\tBeam ratio:{2}".format(self.pupil_diameter, self.npix,
                                                                                 self.beam_ratio))
        for optic, distance in zip(self.planes, self.distances):
            if distance != 0:
                res += "\n\tPropagation distance:  {0}".format(distance)
            res += "\n\t" + str(optic)

        print(res)
        
    @utils.quantity_input(wavelength=u.meter)
    def calc_psf(self, wavelength=1e-6,
                 weight=None,
                 save_intermediates=False,
                 save_intermediates_what='all',
                 display=False,
                 return_intermediates=False,
                 return_final=False,
                 source=None,
                 normalize='first',
                 display_intermediates=False,
                 inwave=None):
        
        """Calculate a PSF, either multi-wavelength or monochromatic.

        The wavelength coverage computed will be:
        - multi-wavelength PSF over some weighted sum of wavelengths (if you provide a `source` argument)
        - monochromatic (if you provide just a `wavelength` argument)

        Parameters
        ----------
        wavelength : float or Astropy.Quantity, optional
            wavelength in meters, or some other length unit if specified as an astropy.Quantity. Either
            scalar for monochromatic calculation or list or ndarray for multiwavelength calculation.
        weight : float, optional
            weight by which to multiply each wavelength. Must have same length as
            wavelength parameter. Defaults to 1s if not specified.
        save_intermediates : bool, optional
            whether to output intermediate optical planes to disk. Default is False
        save_intermediate_what : string, optional
            What to save - phase, intensity, amplitude, complex, parts, all. Default is all.
        return_intermediates: bool, optional
            return intermediate wavefronts as well as PSF?
        return_final: bool, optional
            return the complex wavefront at the last surface propagation as well as the PSF.
            Useful for getting complex PSF without memory usage of `return_intermediates`
        source : dict
            a dict containing 'wavelengths' and 'weights' list.
        normalize : string, optional
            How to normalize the PSF. See the documentation for propagate_mono() for details.
        display : bool, optional
            whether to plot the results when finished or not.
        display_intermediates: bool, optional
            Display intermediate optical planes? Default is False. This option is incompatible with
            parallel calculations using `multiprocessing`. (If calculating in parallel, it will have no effect.)

        Returns
        -------
        outfits :
            a fits.HDUList
        intermediate_wfs : list of `poppy.Wavefront` objects (optional)
            Only returned if `return_intermediates` is specified.
            A list of `poppy.Wavefront` objects representing the wavefront at intermediate optical planes.
            The 0th item is "before first optical plane", 1st is "after first plane and before second plane", and so on.
        final_wfs : `poppy.Wavefront` object (optional)
            Only returned if `return_final` is specified.
           `poppy.Wavefront` objects representing the wavefront at the last of the optical planes.
        """

        tstart = time.time()
        if source is not None:
            wavelength = source['wavelengths']
            weight = source['weights']

            # Make sure the wavelength is unit-y
            if not isinstance(wavelength, u.Quantity):
                wavelength = np.asarray(wavelength) * u.meter

        # ensure wavelength is a quantity which is iterable:
        # (the check for a quantity of type length is applied in the decorator)
        if np.isscalar(wavelength.value):
            wavelength = np.asarray([wavelength.value], dtype=_float()) * wavelength.unit

        if weight is None:
            weight = [1.0] * len(wavelength)

        if len(tuple(wavelength)) != len(tuple(weight)):
            raise ValueError("Input source has different number of weights and wavelengths...")

        # loop over wavelengths
        if self.verbose:
            _log.info("Calculating PSF with %d wavelengths" % (len(wavelength)))
        outfits = None
        intermediate_wfs = None
        if save_intermediates or return_intermediates:
            _log.info("User requested saving intermediate wavefronts in call to poppy.calc_psf")
            retain_intermediates = True
        else:
            retain_intermediates = False

        normwts = np.asarray(weight, dtype=_float())
        normwts /= normwts.sum()

        _USE_FFTW = (conf.use_fftw and accel_math._FFTW_AVAILABLE)
        if _USE_FFTW:
            utils.fftw_load_wisdom()

        if conf.use_multiprocessing and len(wavelength) > 1:  # ######## Parallellized computation ############
            # Avoid a Mac OS incompatibility that can lead to hard-to-reproduce crashes.
            # see issues #23 and #176

            if _USE_FFTW:
                _log.warning('IMPORTANT WARNING: Python multiprocessing and fftw3 do not appear to play well together. '
                             'This may crash intermittently')
                _log.warning('   We suggest you set poppy.conf.use_fftw to False if you want to use multiprocessing().')
            if display:
                _log.warning('Display during calculations is not supported for multiprocessing mode. '
                             'Please set poppy.conf.use_multiprocessing = False if you want to use display=True.')
                _log.warning('(Plot the returned PSF with poppy.utils.display_psf.)')

            if return_intermediates:
                _log.warning('Memory usage warning: When preserving intermediate  planes in multiprocessing mode, '
                             'memory usage scales with the number of planes times number of wavelengths. Disable '
                             'use_multiprocessing if you are running out of memory.')
            if save_intermediates:
                _log.warning('Saving intermediate steps does not take advantage of multiprocess parallelism. '
                             'Set save_intermediates=False for improved speed.')

            # do *NOT* just blindly try to create as many processes as one has CPUs, or one per wavelength either
            # This is a memory-intensive task so that can end up swapping to disk and thrashing IO
            nproc = conf.n_processes if conf.n_processes > 1 \
                else utils.estimate_optimal_nprocesses(self, nwavelengths=len(wavelength))
            nproc = min(nproc, len(wavelength))  # never try more processes than wavelengths.
            # be sure to cast nproc to int below; will fail if given a float even if of integer value

            # Use forkserver method (requires Python >= 3.4) for more robustness, instead of just Pool
            # Resolves https://github.com/mperrin/poppy/issues/23
            ctx = multiprocessing.get_context('forkserver')
            pool = ctx.Pool(int(nproc))

            # build a single iterable containing the required function arguments
            _log.info("Beginning multiprocessor job using {0} processes".format(nproc))
            worker_arguments = [(self, wlen, retain_intermediates, return_final, normalize, _USE_FFTW)
                                for wlen in wavelength]
            results = pool.map(_wrap_propagate_for_multiprocessing, worker_arguments)
            _log.info("Finished multiprocessor job")
            pool.close()

            # Sum all the results up into one array, using the weights
            outfits, intermediate_wfs = results[0]
            outfits[0].data *= normwts[0]
            for idx, wavefront in enumerate(intermediate_wfs):
                intermediate_wfs[idx] *= normwts[0]
            _log.info("got results for wavelength channel {} / {} ({:g} meters)".format(
                0, len(tuple(wavelength)), wavelength[0]))
            for i in range(1, len(normwts)):
                mono_psf, mono_intermediate_wfs = results[i]
                wave_weight = normwts[i]
                _log.info("got results for wavelength channel {} / {} ({:g} meters)".format(
                    i, len(tuple(wavelength)), wavelength[i]))
                outfits[0].data += mono_psf[0].data * wave_weight
                for idx, wavefront in enumerate(mono_intermediate_wfs):
                    intermediate_wfs[idx] += wavefront * wave_weight
            outfits[0].header.add_history("Multiwavelength PSF calc using {} processes completed.".format(nproc))

        else:  # ######### single-threaded computations (may still use multi cores if FFTW enabled ######
            if display:
                plt.clf()
            for wlen, wave_weight in zip(wavelength, normwts):
                mono_psf, mono_intermediate_wfs = self.propagate_mono(
                    wlen,
                    retain_intermediates=retain_intermediates,
                    retain_final=return_final,
                    display_intermediates=display_intermediates,
                    normalize=normalize,
                    inwave=inwave
                )

                if outfits is None:
                    # for the first wavelength processed, set up the arrays where we accumulate the output
                    outfits = mono_psf
                    outfits[0].data *= wave_weight
                    intermediate_wfs = mono_intermediate_wfs
                    for wavefront in intermediate_wfs:
                        wavefront *= wave_weight  # modifies Wavefront in-place
                else:
                    # for subsequent wavelengths, scale and add the data to the existing arrays
                    outfits[0].data += mono_psf[0].data * wave_weight
                    for idx, wavefront in enumerate(mono_intermediate_wfs):
                        intermediate_wfs[idx] += wavefront * wave_weight

            # Display WF if requested.
            #  Note - don't need to display here if we are showing all steps already
            if display and not display_intermediates:
                cmap = getattr(matplotlib.cm, conf.cmap_sequential)
                cmap.set_bad('0.3')
                halffov_x = outfits[0].header['PIXELSCL'] * outfits[0].data.shape[1] / 2
                halffov_y = outfits[0].header['PIXELSCL'] * outfits[0].data.shape[0] / 2
                extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]
                unit = "arcsec"
                vmax = outfits[0].data.max()
                vmin = vmax / 1e4
                norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)  # vmin=1e-8,vmax=1e-1)
                plt.xlabel(unit)

                utils.imshow_with_mouseover(outfits[0].data, extent=extent, norm=norm, cmap=cmap,
                                            origin='lower')

        if save_intermediates:
            _log.info('Saving intermediate wavefronts:')
            for idx, wavefront in enumerate(intermediate_wfs):
                filename = 'wavefront_plane_{:03d}.fits'.format(idx)
                wavefront.writeto(filename, what=save_intermediates_what)
                _log.info('  saved {} to {} ({} / {})'.format(save_intermediates_what, filename,
                                                              idx, len(intermediate_wfs)))

        tstop = time.time()
        tdelta = tstop - tstart
        _log.info("  Calculation completed in {0:.3f} s".format(tdelta))
        outfits[0].header.add_history("Calculation completed in {0:.3f} seconds".format(tdelta))

        if _USE_FFTW and conf.autosave_fftw_wisdom:
            utils.fftw_save_wisdom()

        # TODO update FITS header for oversampling here if detector is different from regular?
        waves = np.asarray(wavelength)
        wts = np.asarray(weight)
        mnwave = (waves * wts).sum() / wts.sum()
        outfits[0].header['WAVELEN'] = (mnwave, 'Weighted mean wavelength in meters')
        outfits[0].header['NWAVES'] = (waves.size, 'Number of wavelengths used in calculation')
        for i in range(waves.size):
            outfits[0].header['WAVE' + str(i)] = (waves[i], "Wavelength " + str(i))
            outfits[0].header['WGHT' + str(i)] = (wts[i], "Wavelength weight " + str(i))
        ffttype = "pyFFTW" if _USE_FFTW else "numpy.fft"
        outfits[0].header['FFTTYPE'] = (ffttype, 'Algorithm for FFTs: numpy or fftw')
        outfits[0].header['NORMALIZ'] = (normalize, 'PSF normalization method')

        if self.verbose:
            _log.info("PSF Calculation completed.")

        if return_intermediates | return_final:
            return outfits, intermediate_wfs

        else:
            return outfits
        
    @utils.quantity_input(wavelength=u.meter)
    def propagate_mono(self,
                       wavelength=1e-6 * u.meter,
                       normalize='first',
                       retain_intermediates=False,
                       retain_final=False,
                       display_intermediates=False,
                       inwave=None):
            
        """Propagate a monochromatic wavefront through the optical system. Called from within `calc_psf`.
        Returns a tuple with a `fits.HDUList` object and a list of intermediate `Wavefront`s (empty if
        `retain_intermediates=False`).

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        normalize : string, {'first', 'last'}
            how to normalize the wavefront?
            * 'first' = set total flux = 1 after the first optic, presumably a pupil
            * 'last' = set total flux = 1 after the entire optical system.
            * 'exit_pupil' = set total flux = 1 at the last pupil of the optical system.
            * 'first=2' = set total flux = 2 after the first optic (used for debugging only)
        display_intermediates : bool
            Should intermediate steps in the calculation be displayed on screen? Default: False.
        retain_intermediates : bool
            Should intermediate steps in the calculation be retained? Default: False.
            If True, the second return value of the method will be a list of `poppy.Wavefront` objects
            representing intermediate optical planes from the calculation.
        retain_final : bool
            Should the final complex wavefront be retained? Default: False.
            If True, the second return value of the method will be a single element list
            (for consistency with retain intermediates) containing a `poppy.Wavefront` object
            representing the final optical plane from the calculation.
            Overridden by retain_intermediates.

        Returns
        -------
        final_wf : fits.HDUList
            The final result of the monochromatic propagation as a FITS HDUList
        intermediate_wfs : list
            A list of `poppy.Wavefront` objects representing the wavefront at intermediate optical planes.
            The 0th item is "before first optical plane", 1st is "after first plane and before second plane", and so on.
            (n.b. This will be empty if `retain_intermediates` is False and singular if retain_final is True.)
        """

        if conf.enable_speed_tests:
            t_start = time.time()
        if self.verbose:
            _log.info(" Propagating wavelength = {0:g}".format(wavelength))
        
        wavefront = self.input_wavefront(wavelength, inwave=inwave)  

        kwargs = {'normalize': normalize,
                  'display_intermediates': display_intermediates,
                  'return_intermediates': retain_intermediates}

        # Is there a more elegant way to handle optional return quantities?
        # without making them mandatory.
        if retain_intermediates:
            wavefront, intermediate_wfs = self.propagate(wavefront, **kwargs)
        else:
            wavefront = self.propagate(wavefront, **kwargs)
            intermediate_wfs = []

        if (not retain_intermediates) & retain_final:  # return the full complex wavefront of the last plane.
            intermediate_wfs = [wavefront]

        if conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop - t_start))

        return wavefront.as_fits(), intermediate_wfs
        
        
        
        