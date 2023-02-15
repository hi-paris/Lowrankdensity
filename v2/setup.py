from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("simulations_paper_low_rank_cython_v2.pyx")
)
