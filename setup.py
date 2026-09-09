from setuptools import setup, find_packages
import os

__version__ = '1.0.0'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='orbit',
    version=__version__,
    author='Cedric Derstroff',
    author_email='cedric.derstroff@tu-darmstadt.de',
    packages=find_packages(),
    # package_data={'': extra_files},
    include_package_data=True,
    # package_dir={'':'src'},
    url='https://github.com/VanillaWhey/ORBiT',
    description='ORBiT',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
