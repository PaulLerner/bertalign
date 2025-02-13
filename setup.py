from setuptools import setup, find_packages
import pkg_resources
import os

setup(
    name='Bertalign',
    version='0.1.0',
    url='https://github.com/bfsujason/bertalign',
    description='An automatic mulitlingual sentence aligner.',
    packages=find_packages(),    
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)