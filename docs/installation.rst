Installation
==================

POPPY may be installed from PyPI in the usual manner for Python packages::

   % pip install poppy --upgrade

The source code is hosted in `this repository on GitHub
<https://github.com/spacetelescope/poppy>`_. It is possible to directly install the
latest development version from git::

   % git clone https://github.com/spacetelescope/poppy.git
   % cd poppy
   % pip install -e .

.. note::
   Users at STScI may also access POPPY through the standard `SSB software
   distributions <http://ssb.stsci.edu/ssb_software.shtml>`__.

Requirements
--------------

* Python 3.5, or more recent. Earlier versions of Python are no longer supported.
* The standard Python scientific stack: :py:mod:`numpy`, :py:mod:`scipy`,
  :py:mod:`matplotlib`
* POPPY relies upon the `astropy
  <http://www.astropy.org>`__ community-developed core library for astronomy.
  astropy, version 1.3 or more recent, is needed.

The following are *optional*.  The first, :py:mod:`pysynphot`, is recommended
for most users. The other optional installs are only worth adding for speed
improvements if you are spending substantial time running calculations.

* `pysynphot <http://pysynphot.readthedocs.org/en/latest/>`_ enables the simulation
  of PSFs with proper spectral response to realistic source spectra.  Without
  this, PSF fidelity is reduced. See below for :ref:`installation instructions
  for pysynphot <pysynphot_install>`.
* `psutil <https://pypi.python.org/pypi/psutil>`__ enables slightly better
  automatic selection of numbers of processes for multiprocess calculations.
* `pyFFTW <https://pypi.python.org/pypi/pyFFTW>`__. The FFTW library can speed
  up the FFTs used in multi-plane optical simulations such as coronagraphiy or
  slit spectroscopy. Since direct imaging simulations use a discrete matrix FFT
  instead, direct imaging simulation speed is unchanged.  pyFFTW is recommended
  if you expect to perform many coronagraphic calculations, particularly for
  MIRI.  (Note: POPPY previously made use of the PyFFTW3 package, which is
  *different* from pyFFTW.  The latter is more actively maintained and
  supported today, hence the switch.  Note also that some users have reported
  intermittent stability issues with pyFFTW for reasons that are not yet
  clear.) *At this time we recommend most users should skip installing pyFFTW
  while getting started with poppy*.
* Anaconda `accelerate <https://docs.anaconda.com/accelerate/>`_ and
  `numexpr <http://numexpr.readthedocs.io/en/latest/user_guide.html>`_.
  These optionally can provide improved performance particularly in the
  Fresnel code.

.. _pysynphot_install:

Installing or updating pysynphot
----------------------------------

`Pysynphot <http://pysynphot.readthedocs.org/en/latest/>`_ is an optional dependency, but is highly recommended.
See the `pysynphot installation docs here <http://pysynphot.readthedocs.org/en/latest/#installation-and-setup>`_
to install ``pysynphot`` and (at least some of) its CDBS data files.

*The minimum needed to have stellar spectral models available for use when
creating PSFs is pysynphot itself plus just one of the CDBS data files: the Castelli & Kurucz stellar atlas, file*
`synphot3.tar.gz <ftp://ftp.stsci.edu/cdbs/tarfiles/synphot3.tar.gz>`_ (18
MB). Feel free to ignore the rest of the synphot CDBS files unless you know you want a larger set of
input spectra or need the reference files for other purposes.


Testing your installation of poppy
----------------------------------

Poppy includes a suite of unit tests that exercise its functionality and verify
outputs match expectations. You can optionally run this test suite to verify
that your installation is working properly::

   >>> import poppy
   >>> poppy.test()
   ============================ test session starts =====================================
   Python 3.6.5, pytest-3.6.1, py-1.5.3, pluggy-0.6.0
   Running tests with Astropy version 3.0.3.
   ... [etc] ...
   ================= 126 passed, 1 skipped, 1 xfailed in 524.68 seconds ==================

Some tests may be automatically skipped depending on whether certain optional packaged are
installed, and other tests in development may be marked "expected to fail" (``xfail``), but
as long as no tests actually fail then your installation is working as expected.
