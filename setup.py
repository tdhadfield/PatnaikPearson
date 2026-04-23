# setup.py  

from setuptools import setup, find_packages  

setup(  
    name="PatnaikPearson",  # Name used for pip installation (must be unique on PyPI)  
    version="0.1.0",             # Semantic versioning (MAJOR.MINOR.PATCH)  
    author="Tom Hadfield",          # Your name/handle  
    author_email="Thomas.Daniel.Hadfield@gmail.com",  # Contact email  
    description="Python package for using the Patnaik-Pearson intrinsic dimension estimator to analyse data manifolds",  # Short description  
    long_description=open("README.md").read(),  # Load long description from README  
    long_description_content_type="text/markdown",  # Format of long_description (Markdown)  
    url="https://github.com/tdhadfield/PatnaikPearson",  # Optional: Link to your repo  
    packages=find_packages(),   # Automatically find all packages (e.g., hello_world)  
    classifiers=[               # Optional: Categorize your package (for PyPI)  
        "Programming Language :: Python :: 3",  
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",  
    ],  
    python_requires=">=3.6",    # Minimum Python version required  
)  
