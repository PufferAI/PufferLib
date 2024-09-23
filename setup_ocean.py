from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    # memory_cy
    Extension(
        name="pufferlib.environments.ocean.memory_cy.cy_memory_cy",
        sources=["pufferlib/environments/ocean/memory_cy/cy_memory_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    
    # spaces_cy
    Extension(
        name="pufferlib.environments.ocean.spaces_cy.cy_spaces_cy",
        sources=["pufferlib/environments/ocean/spaces_cy/cy_spaces_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    
    # bandit_cy
    Extension(
        name="pufferlib.environments.ocean.bandit_cy.cy_bandit_cy",
        sources=["pufferlib/environments/ocean/bandit_cy/cy_bandit_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
        
    # multiagent_cy
    Extension(
        name="pufferlib.environments.ocean.multiagent_cy.cy_multiagent_cy",
        sources=["pufferlib/environments/ocean/multiagent_cy/cy_multiagent_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
            
    # password_cy
    Extension(
        name="pufferlib.environments.ocean.password_cy.cy_password_cy",
        sources=["pufferlib/environments/ocean/password_cy/cy_password_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    
    # squared_cy
    Extension(
        name="pufferlib.environments.ocean.squared_cy.cy_squared_cy",
        sources=["pufferlib/environments/ocean/squared_cy/cy_squared_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
        
    # stochastic_cy
    Extension(
        name="pufferlib.environments.ocean.stochastic_cy.cy_stochastic_cy",
        sources=["pufferlib/environments/ocean/stochastic_cy/cy_stochastic_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
            
    # continuous_cy
    Extension(
        name="pufferlib.environments.ocean.continuous_cy.cy_continuous_cy",
        sources=["pufferlib/environments/ocean/continuous_cy/cy_continuous_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="cython_extensions",
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
