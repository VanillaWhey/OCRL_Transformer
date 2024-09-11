from setuptools import setup, find_packages
import os

__version__ = '0.0.1'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='ocrltransformer',
    version=__version__,
    author='Cedric Derstroff',
    author_email='cedric.derstroff@tu-darmstadt.de',
    packages=find_packages(),
    # package_data={'': extra_files},
    include_package_data=True,
    # package_dir={'':'src'},
    url='https://github.com/VanillaWhey/OCRL_Transformer',
    description='Object Centric Transformer',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
