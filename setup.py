#!/usr/bin/env python
from distutils.core import setup

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

setup(name="vmad", version=find_version("vmad/version.py"),
      author="Yu Feng",
      maintainter="Yu Feng",
      maintainter_email="rainwoodman@gmail.com",
      description="An extensible automated differentiation tool",
      zip_safe=True, # this should be pure python
      packages=[
                "vmad", "vmad.tests",
                "vmad.core", "vmad.core.tests",
                "vmad.lib", "vmad.lib.tests",
        ],
      license='GPLv3',
      install_requires=['numpy', 
                        'scipy', # currently needed for 1d linesearch
                       ]
      )

