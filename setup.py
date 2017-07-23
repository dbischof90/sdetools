from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sdetools',
    version='0.1',
    packages=[''],
    url='https://github.com/dbischof90/sdetools',
    install_requires=requirements,
    license='MIT License',
    author='Daniel Bischof',
    author_email='daniel.bischof90@gmail.com',
    description='A library to estimate and simulate Stochastic Differential Equations.'
)
