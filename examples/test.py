from random import gauss, normalvariate
from time import time

from numpy import sqrt
from numpy.random import standard_normal, normal, randn

x = 0.0
t = time()
for i in range(1000000):
    x = standard_normal() * sqrt(6)

print(time() - t)

x = 0.0
t = time()
for i in range(1000000):
    x = normal(0, sqrt(6))

print(time() - t)

t = time()
for i in range(1000000):
    x = gauss(0, sqrt(6))

print(time() - t)

t = time()
for i in range(1000000):
    x = normalvariate(0, 1) * sqrt(6)

print(time() - t)

t = time()
for i in range(1000000):
    x = randn() * sqrt(6)

print(time() - t)
