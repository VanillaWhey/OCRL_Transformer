from setuptools import setup, find_packages
import os

__version__ = '0.0.1'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='OCRLTransformer',
    version=__version__,
    author='Cedric Derstroff',
    author_email='cedric.derstroff@tu-darmstadt.de',
    packages=find_packages(),
    # package_data={'': extra_files},
    include_package_data=True,
    # package_dir={'':'src'},
    # url='https://github.com/k4ntz/OC_Atari',
    description='Object Centric Transformer',
    long_description=long_description,
    long_description_content_type='text/markdown',
        install_requires=[
        "ocatari",
        "torch",
    ]
)
