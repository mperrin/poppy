import numpy as np
import poppy
import matplotlib.pyplot as plt
import astropy.units as u

from .poppy_core import OpticalElement, Wavefront, PlaneType, _PUPIL, _IMAGE, _RADIANStoARCSEC

import logging
_log = logging.getLogger('poppy')

class subapertures(poppy.OpticalElement):
        """
        Example roadmap:
        
        #generate wavefront
        wf = poppy.wavefront() #or fresnel wavefront
        
        #...various surfaces
        
        subaperture(ARRAY OF OPTICS) 
        #initialize this new class, where the array of optics define the subapertures (e.g. lenslets)
        
        subapertures.sample_wf(wf) #this function takes the wavefront and subsamples it by the area of each optic
        subapertures.get_wavefront_array() #returns an array of input sub-wavefronts multipled by subaperture optics
        subapertures.get_psfs() #fraunhofer or fresnel propagation of each pupil  to the image/ waist
        image=subapertures.get_composite_wavefont() # returns wavefront of image plane of all the spots put back together
        subapertures.opd #single array made up of subapertures
        subapertures.amplitude  #single array made up of subapertures
        subapertures.getphasor #returns propagated array of spots from get_composite_wavefont
        
        
        Parameters
        ----------
        
        crosstalk: boolean
                  this variable sets whether light can leak from one lenslet image plane into the neighbors.
                  Default is False.
        x_y_offset: tuple
                  offset of central grid vertex from center of incident wavefront
        input_wavefront: None or poppy.Wavefront 
                  : the unsampled wavefront incident on the subapertures
        """
        
        def __init__(self,
                     dimensions = (2,2),
                     optic_array = np.array([[poppy.CircularAperture(radius=2.,planetype=PlaneType.pupil),
                                              poppy.CircularAperture(radius=2.,planetype=PlaneType.pupil)],
                                    [poppy.CircularAperture(radius=2.,planetype=PlaneType.pupil),
                                     poppy.CircularAperture(radius=2.,planetype=PlaneType.pupil)]]),
                     crosstalk=False,
                     x_y_offset=(0,0),
                     detector=None,
                     overwrite_inputwavefront=False,
                     display_intermediates=False,
                     optical_system=None,
                 **kwargs):
            self.n_apertures = dimensions[0]*dimensions[1]
            
            self.optic_array=optic_array
            self.crosstalk=crosstalk
            if crosstalk:
                raise ValueError("CROSS TALK NOT IMPLEMENTED <YET>")
            self.x_y_offset=x_y_offset
            self.amplitude = np.asarray([1.])
            self.opd = np.asarray([0.])
            self.pixelscale = None
            self.input_wavefront = None
            self.output_wavefront = None
            if detector == None:
                self.detector = poppy.Detector(0.01,fov_pixels=128)
            else:
                self.detector=detector
            self.optical_system = optical_system
            if  optical_system is not None:
                    raise ValueError("complete optical system after wavelets are not implemented yet")
            self.x_apertures=self.optic_array.shape[0]
            self.y_apertures=self.optic_array.shape[1]
            if self.x_apertures != self.y_apertures:
                raise ValueError("A square array of subapertures is currently required.")
            #initialize array of subsampled output wavefronts:
            self.wf_array = np.empty(self.optic_array.shape,dtype=np.object_)
            self.overwrite_inputwavefront = overwrite_inputwavefront
            self.display_intermediates = display_intermediates
            self._propagated_flag = False #can't have propagated when initializing
            poppy.OpticalElement.__init__(self, **kwargs)
                
        def sample_wf(self, wf):
            '''
            
            Parameters
            ----------
            
            '''
            #save the input wavefront
            self.input_wavefront = wf
            for i in range(self.x_apertures):
                 for j in  range(self.y_apertures):
                    opt = self.optic_array[i][j] #get an optic

                    #check for padding
                    if opt == None:
                        continue
                    
                    aper_per_dim = wf.diam /(opt.pupil_diam) #assuming squares
                    
                    self._w = opt.pupil_diam/wf.pixelscale #subaperture width in pixels 
                    #the generated number of subapertures might not match the input wavefront dimensions
                    #want to center the subapertures on the incoming wavefront
                    
                    self.c = wf.wavefront.shape[0]/2*u.pix #center of array
                    c=self.c
                    w=self._w
                    sub_wf=wf.copy() #new wavefront has all the previous wavefront properties
                    lower_x = int((c + w*(i)  - w*self.x_apertures/2).value)
                    lower_y = int((c + w*(j)  - w*self.y_apertures/2).value)
                    upper_x = int((c + w*(i+1)  - w*self.x_apertures/2).value)
                    upper_y = int((c + w*(j+1)  - w*self.y_apertures/2).value)
                    #print([i,j,c,w,lower_x,upper_x,lower_y,upper_y])

                    sub_wf.wavefront = wf.wavefront[lower_x:upper_x,lower_y:upper_y]
                    self.pixelscale
                    wf.pixelscale
                    self.wf_array[i][j] = sub_wf*opt
                    #print((i,j,c,w, sub_wf.shape,wf.shape))

                    if self.display_intermediates:
                        plt.figure()
                        self.wf_array[i][j].display()
                        
                    #print(sub_wave.shape)
            #slice wavefront
            #wf.Wavefront[,]
        
        #subsample input wavefront
        
        #generate subsampled grid of mini-pupils and return array of output wavefronts
        @property
        def subaperture_width(self):
            return self._w*self.input_wavefront.pixelscale
        
        #return a composite wavefront if an array of output wavefronts was generated
        def get_wavefront_array(self):
            """
            
            
            """
            if self.input_wavefront is None:
                raise ValueError("No input wavefront found.")
                

            if self._propagated_flag:
                #recalculate dimensions
                print("Tiling propagated wavefront arrays.")
                c = self.c_out
                w = self._w_out
                #create new output wavefront
                wf = poppy.Wavefront(wavelength = self.input_wavefront.wavelength, 
                                     npix = 2*self.c_out.value, 
                                     dtype = self.input_wavefront.wavefront.dtype, 
                                     pixelscale = self.detector.pixelscale,
                                     oversample = self.detector.oversample)
                
            else:
                c = self.c
                w = self._w
                wf = self.input_wavefront.copy()
            for i in range(self.x_apertures):
                 for j in  range(self.y_apertures):
                    sub_wf = self.wf_array[i][j] #get an subaperture wavefront
                    lower_x = int((c + w*(i)  - w*self.x_apertures/2).value)
                    lower_y = int((c + w*(j)  - w*self.y_apertures/2).value)
                    upper_x = int((c + w*(i+1)  - w*self.x_apertures/2).value)
                    upper_y = int((c + w*(j+1)  - w*self.y_apertures/2).value)
                    #check for padding

                    if sub_wf == None:
                        wf.wavefront[lower_x:upper_x,lower_y:upper_y] = np.nan
                    else:
                        wf.wavefront[lower_x:upper_x,lower_y:upper_y] = sub_wf.wavefront 
            if self.overwrite_inputwavefront:
                self.input_wavefront = wf
                
            return wf
        def get_psfs(self):
                if self.input_wavefront is None:
                        raise ValueError("No input wavefront found.")
            
            
                for i in range(self.x_apertures):
                    for j in  range(self.y_apertures):
                        sub_wf = self.wf_array[i][j]
                        sub_wf.propagateTo(self.detector)
                        if self.display_intermediates:
                            plt.figure()
                            sub_wf.display()
            
                self._w_out= self.wf_array[0][0].shape[0]*u.pix #subaperture width in pixels 
                self.c_out =  self._w_out*self.x_apertures/2 #center of array
            
                self._propagated_flag = True

        def multiply_all(self, optic):
                if self.input_wavefront is None:
                        raise ValueError("No input wavefront found.")
                for i in range(self.x_apertures):
                    for j in  range(self.y_apertures):
                        self.wf_array[i][j] *= optic
        

        def get_centroids(self,
                          cent_function=poppy.utils.measure_centroid,
                          asFITS=True,
                          **kwargs):
            """
                get centroid of intensity of each subwavefront

            """
            _log.debug("Centroid function:"+str(cent_function))
            if self.input_wavefront is None:
                raise ValueError("No input wavefront found.")
            if not self._propagated_flag:
                _log.warn("Getting centroid without having propagated.")
            self.centroid_list = np.zeros((2,self.x_apertures,self.y_apertures))
            for i in range(self.x_apertures):
                for j in  range(self.y_apertures):
                    sub_wf = self.wf_array[i][j]
                    if sub_wf.total_intensity == 0.0:
                        _log.warn("Setting centroid of aperture with no flux to NaN.")
                        self.centroid_list[:,i,j]=(np.nan,np.nan)
                        continue
                    if asFITS:
                        intensity_array=sub_wf.asFITS()
                    else:
                        intensity_array = sub_wf.intensity
                    self.centroid_list[:,i,j] = cent_function(intensity_array)
            return self.centroid_list
        
        def _replace_subwavefronts(self,replacement_array):
            for i in range(self.x_apertures):
                 for j in  range(self.y_apertures):
                    sub_wf = self.wf_array[i][j] #get an subaperture wavefront
                    lower_x = int((c + self._w*(i)  - self._w*self.x_apertures/2).value)
                    lower_y = int((c + self._w*(j)  - self._w*self.y_apertures/2).value)
                    upper_x = int((c + self._w*(i+1)  - self._w*self.x_apertures/2).value)
                    upper_y = int((c + self._w*(j+1)  - self._w*self.y_apertures/2).value)
                    #check for padding

                    if sub_wf == None:
                        wf.wavefront[lower_x:upper_x,lower_y:upper_y] = np.nan
                    else:
                        wf.wavefront[lower_x:upper_x,lower_y:upper_y] = sub_wf.wavefront 
class dispersion_plate(poppy.AnalyticOpticalElement):
    """
    Implements dispersion, defaulting to Sellemeier equation
    (https://en.wikipedia.org/wiki/Sellmeier_equation) for BK7.
        
    Parameters
    ----------
    d : astropy.unit of length
              the thickness of the dispersive element
    Sellmeier_coeffs : list or ndarray
       six coefficients in order: [B1,B2,B3,C1,C2,C3]
    material : str
       string describing a common dispersive material,
       including: 'BK7' and 'fused silica'
    custom_function :
        a custom dispersion function.
        For example the Sellmeier equation is implemented as:

            lambda wlengths : np.sqrt(1+B1*wlengths**2/(wlengths**2-C1)
               +B2*wlengths**2/(wlengths**2-C2)
               +B3*wlengths**2/(wlengths**2-C3))


    Notes:
    ----------

    Using Fused Silica Dispersion Constants from Malitson, I. H.(1965),
    Interspecimen comparison of the refractive index of fused silica,JOSA, 55(10), 120-1208.
     N-BK7 dispersion Constants from SCHOTT Datasheet
     
    """
    
    def __init__(self,
                 d=0*u.m,
                 Sellmeier_coeffs=None,
                 material='BK7',
                  planetype=_PUPIL,
                 custom_function=None,
                 name='dispersion plate',
                 **kwargs):
        
        poppy.AnalyticOpticalElement.__init__(self, name=name,  planetype= planetype,**kwargs)
        self.wavefront_display_hint = 'phase' # preferred display for wavefronts at this plane
        self.d = d
        
        if Sellmeier_coeffs is not None:
            if np.size(Sellmeier_coeffs) == 6:
                material = "coefficents"
                self.B1, self.B2, self.B3, self.C1, self.C2,self.C3 = Sellmeier_coeffs
                assert self.C1.is_equivalent(u.m**2)
                assert self.C2.is_equivalent(u.m**2)
                assert self.C3.is_equivalent(u.m**2)
            else:
                raise ValueError("Sellmeier equation takes 6 coefficients")
        elif material != "coefficients":
                print("Warning, no coefficients given")
                if material == 'fused silica':
                        self.B1=0.696166300
                        self.B2=0.407942600
                        self.B3=0.897479400
                        self.C1=4.67914826*10**(-3)*u.um**2
                        self.C2=1.35120631*10**(-2)*u.um**2
                        self.C3=97.9340025*u.um**2
                elif material == 'BK7':
                        self.B1=1.03961212
                        self.B2=0.231792344
                        self.B3=1.01046945
                        self.C1=6.00069867*10**(-3)*u.um**2
                        self.C2=2.00179144*10**(-2)*u.um**2
                        self.C3=103.5606535*u.um**2
                else:
                    raise ValueError("Requested material not implemented.")
        if custom_function is None:
            self.dispersion = self.n_sellmeier
        else:
            self.dispersion = custom_function
            
    def n_sellmeier(self,wlengths):
        """
        Sellmeier Equation for the initialized coefficients or material.
        
        Parameters
        ----------
        wlengths : astropy.unit of length, e.g. astropy.units.nm
        
        """
        
        n = np.sqrt(1 + self.B1*wlengths**2/(wlengths**2 - self.C1)
          + self.B2*wlengths**2/(wlengths**2 - self.C2)
          + self.B3*wlengths**2/(wlengths**2 - self.C3))
        return n

    
    def get_opd(self, wave):
        """
        returns optical path length through the dispersive element
        
        Parameters
        ----------
        wave :  a poppy.wavefront object
        
        """
        
        opl = self.d*n(wave.wavelength)
        return opl

