import sys
import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="hd_var",
    author="Yvann Le Fay, Antoine Schoonaert",
    description="Efficient JAX implementation of high-dimensional VAR models using tensor factorization.",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax>=0.3.25",
        "jaxlib>=0.3.25",
        "pytest",
        "numpy>=1.24.3",
        "jaxopt>=0.8",
    ],
    long_description_content_type="text/markdown",
    keywords="linear model high-dimensional VAR vector autoregression tensor factorization key factors econometrics",
    license="MIT",
    license_files=("LICENSE",),
)