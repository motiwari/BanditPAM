"""Conversion examples python side."""
import numpy as np

import example_carma as carma

sample = np.asarray(
    np.random.random(size=(10, 2)),
    dtype=np.float64,
    order='F'
)

print(carma.manual_example(sample))
carma.update_example(sample)
print(carma.automatic_example(sample))


sample2 = np.asarray(
    np.random.random(size=(10, 2)),
    dtype=np.float64,
    order='F'
)

example_class = carma.ExampleClass(sample, sample2)
arr = example_class.member_func()
print(arr)
