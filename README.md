# SDEtools for Python 3 
###### A library to estimate and simulate Stochastic Differential Equations.

This library is a collection of statistical methods to simulate and estimate non-deterministic differential equations. Inference of SDEs is a topic I personally find very interesting but while the tools for their deterministic counterparts are well-developed (and especially well-implemented), stochastic differential equations lack a common tool set in Python.

![Alt text](misc/cirpaths.png?raw=true)

In current focus are one-dimensional Itô - diffusions and the focus will lie on such until the framework for the one-dimensional case is stable enough. The multidimensional case is in scope and the code is developed with the later extension to that case in mind.

## Usage

All numerical schemes and estimation algorithms rely on a stochastic differential equation, given by
```Python
SDE(drift, diffusion, timerange=[0,2])
```

with generic drift and diffusion functions. These can be specified in any way - it is useful though to ensure that the theoretical boundaries for existence of weak or strong solutions are set.

Schemes are implemented as iterators. To import the Euler-Mayurama scheme, use
```Python
from simulation.strong.taylor import Order_05 as Euler

[...]


for path_value in Euler(SDE, parameter, steps=50):
    do_stuff_with_it(path_value)
```
which gives you maximal flexibility, speed and effective memory management.

A short introduction on estimation methods will later be included.