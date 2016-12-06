from setuptools import setup, find_packages

VERSION = '2.0.0'
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
      classifiers=[
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5'
      ]
)
