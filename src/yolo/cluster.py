from collections import namedtuple
import numpy as np
from itertools import combinations_with_replacement

class DofrAndPrice:
    def __init__(self, dofr, price):
        self.dofr = dofr
        self.price = price

    def __repr__(self):
        return f"DofrAndPrice({self.dofr},{self.price})"

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, DofrAndPrice):
            return self.dofr == other.dofr and self.price == other.price
        return NotImplemented

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))


def evaluate(clusters):
    return np.sum([np.var(cluster) for cluster in clusters])


def generate_partitions(dofr_and_prices):
    num_buckets = len(dofr_and_prices)
    combinations_with_replacement([i for i in range(num_buckets)])


def step(clusters, tail):
    if len(clusters) == 0:
        if len(tail) == 0:
            return clusters
        else:
            step([[tail[0]]], tail[1:])
    else:
        if len(tail) == 0:
            return clusters



        cur_total_var = evaluate(clusters)
        cur_last_cluster_var = np.var(clusters[-1])
        var_with_added_element = np.var(clusters[-1] + [tail[0]])
        std_dev_of_last_cluster = np.sqrt(cur_last_cluster_var)
        mean_of_last_cluster = np.mean(clusters[-1])
        std_dev_with_added_element =np.sqrt(var_with_added_element)
        if abs(tail[0] - mean_of_last_cluster) < std_dev_of_last_cluster:
            step(clusters[:-1] + [clusters[-1] + tail[0]], tail[1:])


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



def blocksNo(A, maxBlock):
    # Initially set the A[0] being an individual block
    blocksNumber = 1    # The number of blocks, that A could
                        # be divided to with the restriction
                        # that, the sum of each block is less
                        # than or equal to maxBlock
    preBlockSum = A[0]
    for element in A[1:]:
        # Try to extend the previous block
        if preBlockSum + element > maxBlock:
            # Fail to extend the previous block, because
            # of the sum limitation maxBlock
            preBlockSum = element
            blocksNumber += 1
        else:
            preBlockSum += element
    return blocksNumber

def solution(K, A):
    blocksNeeded = 0    # Given the restriction on the sum of
                        # each block, how many blocks could
                        # the original A be divided to?
    resultLowerBound = max(A)
    resultUpperBound = sum(A)
    result = 0          # Minimal large sum
    # Handle two special cases
    if K == 1:      return resultUpperBound
    if K >= len(A): return resultLowerBound
    # Binary search the result
    while resultLowerBound <= resultUpperBound:
        resultMaxMid = (resultLowerBound + resultUpperBound) / 2
        blocksNeeded = blocksNo(A, resultMaxMid)
        if blocksNeeded <= K:
            # With large sum being resultMaxMid or resultMaxMid-,
            # we need blocksNeeded/blocksNeeded- blocks. While we
            # have some unused blocks (K - blocksNeeded), We could
            # try to use them to decrease the large sum.
            resultUpperBound = resultMaxMid - 1
            result = resultMaxMid
        else:
            # With large sum being resultMaxMid or resultMaxMid-,
            # we need to use more than K blocks. So resultMaxMid
            # is impossible to be our answer.
            resultLowerBound = resultMaxMid + 1
    return result