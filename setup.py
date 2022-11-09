"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    
    name='organoid_prediction_python',
    
    version='0.0.1',  
    
    description='A library for image ananalysis, data analysis and classification of organoid image data',  
    
    long_description=long_description,
    
    long_description_content_type='text/markdown',  
    
    url='https://github.com/Cryaaa/organoid_prediction_python',  

    
    author='Ryan Savill',  

    
    author_email='savill@mpi-cbg.de',

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        

        # Pick your license as you wish
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='image, analysis, development, organoids, classification',

    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    #package_dir={'': 'src'},  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(),  # Required

    python_requires='>=3.8, <3.10',

    install_requires=["seaborn","umap-learn","hdbscan","numpy", "pyopencl", "scikit-image", 
                      "scikit-learn", "pyclesperanto-prototype", "pandas",'pywebview',
                      "pythonnet==3.0.0a2", "vispy"],  

    project_urls={ 
        'Bug Reports': 'https://github.com/Cryaaa/organoid_prediction_python/issues',
        'Source': 'https://github.com/Cryaaa/organoid_prediction_python',
    },
)