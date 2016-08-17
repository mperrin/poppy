import numpy as np
import poppy
import matplotlib.pyplot as plt
import astropy.units as u

from .poppy_core import OpticalElement, Wavefront, PlaneType, _PUPIL, _IMAGE, _RADIANStoARCSEC

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
            self.input_wavefront=wf
            for i in range(self.x_apertures):
                 for j in  range(self.y_apertures):
                    opt = self.optic_array[i][j] #get an optic

                    #check for padding
                    if opt == None:
                        continue
                    
                    aper_per_dim = wf.diam /(opt.radius*2) #assuming squares
                    
                    self.w= 2*opt.radius/wf.pixelscale #subaperture width in pixels 
                    #the generated number of subapertures might not match the input wavefront dimensions
                    #want to center the subapertures on the incoming wavefront
                    
                    self.c = wf.wavefront.shape[0]/2*u.pix #center of array
                    c=self.c
                    w=self.w
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
                w = self.w_out
                #create new output wavefront
                wf = poppy.Wavefront(wavelength = self.input_wavefront.wavelength, 
                                     npix = 2*self.c_out.value, 
                                     dtype = self.input_wavefront.wavefront.dtype, 
                                     pixelscale = self.detector.pixelscale,
                                     oversample = self.detector.oversample)
                
            else:
                c = self.c
                w = self.w
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
            
            self.w_out= self.wf_array[0][0].shape[0]*u.pix #subaperture width in pixels 
            self.c_out =  self.w_out*self.x_apertures/2 #center of array
            
            self._propagated_flag = True
            
