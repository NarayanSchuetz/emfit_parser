"""
 Created by Narayan Schuetz at 15.11.20 
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="emfit_qs_parser",
    version="0.0.1",
    author="Narayan Schütz",
    author_email="narayan.schuetz@artorg.unibe.ch",
    description="Used to parse raw EMFIT QS files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NarayanSchuetz/emfit_parser.git",
    packages=find_packages(),
    install_requires=[
        "markdown==3.0.1",
        "numpy>=1.16.2",
        "pandas>=0.24.2",
        "pyedflib>=0.1.19"
    ],  # if it doesn't work try replacing >= with ==
    zip_safe=False,
    python_requires='>=3.6'
)
