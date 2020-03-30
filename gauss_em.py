'''
compute and plot the Euclidean (l2) distance and the l1 distance for the clusters as a function of EM iterations;
2) modify the program to find 3 clusters and 4 clusters instead of 2.
Report your plots and results for both parts, and explain your observations and conclusions.

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from itertools import cycle, islice
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances



#import Data
data = np.loadtxt('faithful.txt', float, delimiter=' ')
plt.figure(1)
plt.scatter(data[:, 0], data[:, 1], s=5)
# plt.show()

# 1st iteration
gmm = GaussianMixture(n_components=2, covariance_type='full', init_params='random', max_iter=1)
gmm.fit(data)
gmm_result = gmm.predict(data)
mu1, mu2 = gmm.means_
# l1 =
# l2 =
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(gmm_result) + 1))))
plt.figure(2)
plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[gmm_result])
plt.show()

for k in range(4):
    gmm = GaussianMixture(n_components=2, covariance_type='full', means_init=(mu1, mu2), max_iter=1)
    gmm.fit(data)
    gmm_result = gmm.predict(data)
    mu1, mu2 = gmm.means_
    # l1 =
    # l2 =
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(gmm_result) + 1))))
    plt.figure(k+3)
    plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[gmm_result])
    plt.show()

