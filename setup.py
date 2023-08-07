#! /usr/bin/env python
"""The calf_milp package setup file.

Directions

1.  Update the code and documentation

2.  Test the template by running test_template.py

3.  Update the documentation, including the jupyter notebooks

3.1 Convert the notebook content to rst and integrate it into quick_start.rst and user_guide.rst

    jupyter nbconvert quick_start.ipynb --to markdown --output README.md
    jupyter nbconvert quick_start.ipynb --to rst --output quick_start.rst
    jupyter nbconvert user_guide.ipynb --to rst --output user_guide.rst

3.2 Check the documentation by reading it from _build/html

4.  Remove the old distribution: rm -r dist

5. cd to calf_milp and build the new dist folder:
python setup.py sdist bdist_wheel

Note that setup is deprecated and a replacement method is needed.

6. Upload to pip: twine upload dist/*


"""

import codecs

from setuptools import find_packages, setup

# noinspection PyProtectedMember
from calf_milp import _version

DISTNAME = 'calf_milp'
DESCRIPTION = 'A classifier that endeavors to solve the saddle point problem for AUC maximization.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Carlson Research, LLC'
MAINTAINER_EMAIL = 'hrolfrc@gmail.com'
URL = 'https://github.com/hrolfrc/calf-milp'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/hrolfrc/calf-milp'
VERSION = _version.__version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Development Status :: 3 - Alpha',
               'License :: OSI Approved',
               'Topic :: Scientific/Engineering',
               'Operating System :: OS Independent',
               'Programming Language :: Python :: 3']

EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
