"""Package setup script."""

from setuptools import find_packages, setup

setup(
    name="infscale",
    version="0.0.1",
    author="Infscale Maintainers",
    author_email="infscale-github-owners@cisco.com",
    include_package_data=True,
    packages=find_packages(),
    data_files=[],
    scripts=[],
    url="https://github.com/cisco-open/infscale/",
    license="LICENSE.txt",
    description="This package is a python library"
    " to serve ML model in the infscale system",
    long_description=open("README.md").read(),
    install_requires=[],
    extras_require={},
)
