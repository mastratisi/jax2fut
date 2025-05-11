from setuptools import setup, find_packages

setup(
    name="jax2fut",
    version="0.1.0",
    description="A translator from JAX expressions to Futhark code",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "numpy>=1.22.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Compilers",
    ],
)
