import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from datasets.detected_car import DetectedCar
from logging_config import *

logger = logging.getLogger(__name__)


def sum_of_variance(clusters):
    return np.sum([np.var(cluster) for cluster in clusters])


def cluster(detected_cars, n_clusters):
    X = np.array([[detected_car.price, detected_car.date_of_first_registration]
                  for detected_car in detected_cars])

    X = MinMaxScaler().fit_transform(X)
    X[:, 1] = X[:, 1] * 3 # the date_of_first_registration should overrule the price

    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    return kmeans.labels_


def find_best_cluster(detected_cars, min_n_custers, max_n_clusters):
    all_labels = []
    all_variances = []
    for n_clusters in range(min_n_custers, max_n_clusters+1):
        labels = cluster(detected_cars, n_clusters)
        all_labels.append(labels)
        clusters = DetectedCar.to_price_clusters(detected_cars, labels)
        variance = sum_of_variance(clusters)
        all_variances.append(variance)

    best = np.argmin(all_variances)
    return all_labels[best]
