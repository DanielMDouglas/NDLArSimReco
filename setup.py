#!/usr/bin/env python

import setuptools

VER = "0.0.1"

reqs = ["numpy",
        "plotly",
        "pyaml",
        "h5py",
        'LarpixParser @ git+https://github.com/DanielMDouglas/larpix_readout_parser',
        'NDeventDisplay @ git+https://github.com/DanielMDouglas/NDeventDisplay',
        "torch",
        "MinkowskiEngine",
        ]

setuptools.setup(
    name="NDLArSimReco",
    version=VER,
    author="Daniel D. and others",
    author_email="dougl215@slac.stanford.edu",
    description="CNN stuff NDLAr sim (testing!)",
    url="https://github.com/DanielMDouglas/NDLArSimReco",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.2',
)
