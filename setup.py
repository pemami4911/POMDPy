from setuptools import setup, find_packages

VERSION = '1.1.0'
DESCRIPTION = 'POMDPy: POMDPs in Python'

setup(name='POMDPy',
      version=VERSION,
      description=DESCRIPTION,
      author='Patrick Emami',
      author_email='pemami@ufl.edu',
      url='http://pemami4911.github.io/POMDPy/',
      packages=find_packages(),
      package_data={
          'pomdpy': ['config/*.json', 'config/*.txt'],
          'test': ['*.sh']
          },
      license='MIT License',
      install_requires=['numpy==1.11.2', 'matplotlib>=1.4.3', 'pytest==2.7.0', 'scipy>=0.15.1']
      )
