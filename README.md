# SDEtools for Python 3 
###### A library to estimate and simulate Stochastic Differential Equations.

This library is a collection of statistical methods to simulate and estimate non-deterministic differential equations. Inference of SDEs is a topic I personally find very interesting but while the tools for their deterministic counterparts are well-developed (and especially well-implemented), stochastic differential equations lack a common tool set in Python.

![Alt text](misc/cirpaths.png?raw=true)

In current focus are one-dimensional Itô - diffusions and the focus will lie on such until the framework for the one-dimensional case is stable enough. The multidimensional case is in scope and the code is developed with the later extension to that case in mind.

## Usage

We will give a short overview about usage and tips for application here.

### Numerical discretization schemes
All numerical schemes and estimation algorithms rely on a stochastic differential equation, given by
```Python
SDE(drift, diffusion, timerange=[0,2])
```

with generic drift and diffusion functions. These can be specified in any way - it is useful though to ensure that the theoretical boundaries for existence of weak or strong solutions are set.

Schemes are implemented as iterators. To import the Euler-Mayurama scheme, use
```Python
from simulation.strong.explicit.taylor import Order_05 as Euler

[...]

for path_value in Euler(SDE, parameter, steps=50):
    do_stuff_with_it(path_value)
```
which gives you maximal flexibility, speed and effective memory management.

Weak schemes can also be used if one only needs to approximate moments of the underlying process.
To make the computation more performant, most schemes are brought down to as few driving stochastic sources as possible.
Do also note that it's far easier to compute different schemes of the same convergence order, the here presented implementations only represent one of them which seem to perform well on a general scale. There might be other schemes which fit your problem more specific, the convergence order will differ in `O(1)` only.

#### A short note on performance
A lot of inline optimization has been performed on the existing code base, down to the case that function calls in the discretization scheme are the by far most expensive operations.
When choosing a method, do consider how expensive your function call might be - it might be worth it to calculate derivatives analytically instead of relying on Runge-Kutta methods, which do perform just as nice, but the parameter functions need to be executed far more often.
This can make methods slower.
Another approach would be packages like `numba`, which applies JIT-compilation to the functions. In easy examples, this provided a speedup of around 20%.

Implicit methods are much slower given their nature - in every propagation a root-finding algorithm has to be solved. SDEtools does not assume linearity, therefore a more generic method has to be used. It is recommended to check if stability can not be reached otherwise first.

### Inference of stochastic processes
A short introduction on estimation methods will later be included.