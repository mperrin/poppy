#Test functions for core poppy functionality

from .. import poppy_core
from .. import optics
from .. import geometry

import numpy as np
import astropy.io.fits as fits

# Test routines for antialiased fractional circle

def test_area_minimal_circle():
    """ Test that A = pi * r**2, where r==1 """
    res = geometry.filled_circle_aa((2,2), 0.5, 0.5, 1)
    assert np.abs(res.sum() -np.pi) < 1e-7


def test_clipping():

    res = geometry.filled_circle_aa((20,20), 10.5, 10.5, 1, clip=True,cliprange=(0,1))

    assert res.min() >= 0.0
    assert res.max() <= 1.0


# Come up with some representative plausible test cases for whcih we know the answers


# Test effect of shifting the center of the image by integer pixels

# Test effect of shifting the center of the image by fractional pixels
    # cross correlation of shifted & unshifted to demonstrate 1/2 pixel shifts? 

# Test using subpixel scaling of incput X and Y arrays

# Test the specific case at fault here. 






