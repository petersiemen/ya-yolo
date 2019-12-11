import numpy as np


def test_variance_assumptions():
    elems_1 = [1,2,3,4,5,6]
    print(np.var(elems_1))

    print(np.var(elems_1[:3]) + np.var(elems_1[3:]))