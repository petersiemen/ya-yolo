from collections import namedtuple
import numpy as np


class DofrAndPrice:
    def __init__(self, dofr, price):
        self.dofr = dofr
        self.price = price

    def __repr__(self):
        return f"DofrAndPrice({self.dofr},{self.price})"

def evaluate(clusters):
    for cluster in clusters:
        np.var(cluster)

def cluster(items):
    """
    minimize sum of clusters variances
    minimize number of clusters


    200, 201, 202, 400, 402

    optimal partitioning: (200, 201, 202), (400, 402)



    loss = sum(variance(cluster)) +


    Parameters
    ----------
    items

    Returns
    -------

    """
    items.sort(key=lambda i: i.dofr)

    elems = np.array([[i.dofr, i.price] for i in items])

    np.mean(elems[:, 1])
    np.var(elems[:,1])
    np.mean(elems)

    print(items)
    pass
