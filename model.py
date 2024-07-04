import random
import numpy as np

print(np.random.rand(10))
print(np.random.randn(10))

np.random.seed(13)
print(np.random.randint(0, 100, 10))
