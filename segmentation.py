import numpy as np
import matplotlib.pyplot as plt
from luv import *
np.random.seed(42)
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

def get_diff(img, currentPoint, tmpPoint):
    x=int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y])
    return abs(x)


def select_p(p):
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1),
                    Point(1, 0), Point(1, 1), Point(0, 1),
                    Point(-1, 1), Point(-1, 0)]
    if p == 1:
        return connects
    

def euclidean_distance(x1, x2):
    dist=np.sqrt(np.sum((x1 - x2) ** 2))
    return dist


def clusters_distance(cluster1, cluster2):
    # cluster1 and cluster2 are lists of lists of points
    dist=[euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2]
    return max(dist)

def clusters_distance_centers(cluster1, cluster2):
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize k centroids randomly
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimize clusters
        for i in range(self.max_iters):
            # create clusters by assigning sample points to the nearest centroid
            self.clusters = self.create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)

            # if clusters have converged, break the loop
            if self.is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify sample points as the index of their clusters
        return self.get_cluster_labels(self.clusters)

    def get_cluster_labels(self, clusters):
        # each sample point will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_idx
        return labels

    def create_clusters(self, centroids):
        # assign the sample points to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.get_closest_index(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    @staticmethod
    def get_closest_index(sample, centroids):
        # distance of the current sample to each centroid
        distances = np.linalg.norm(centroids - sample, axis=1)
        closest_index = np.argmin(distances)
        return closest_index

    def get_centroids(self, clusters):
        # find new centroid of each cluster
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_converged(self, centroids_old, centroids):
        # distances between each old and new centroid, for all centroids
        distances = np.linalg.norm(centroids_old - centroids, axis=1)
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index]
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()

    def cent(self):
        return self.centroids
# class KMeans:
#     def __init__(self, K=5, max_iters=100, plot_steps=False):
#         self.K = K 
#         self.max_iters = max_iters
#         self.plot_steps = plot_steps
#         # initialize a list of indices for each cluster
#         self.clusters = [[] for _ in range(self.K)]
#         # initialize a list of the centers (mean feature vector) for each cluster
#         self.centroids = []

#     def predict(self, X):
#         self.X = X
#         self.n_samples, self.n_features = X.shape
#         # initialize k centroids randomly
#         random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
#         self.centroids = [self.X[idx] for idx in random_sample_idxs]
#         # optimize clusters
#         for _ in range(self.max_iters):
#             # create clusters by assigning sample points to the nearest centroid
#             self.clusters = self.create_clusters(self.centroids)
#             if self.plot_steps:
#                 self.plot()
#             # calculate new centroids from the clusters
#             centroids_old = self.centroids
#             self.centroids = self.get_centroids(self.clusters)
#             # if clusters have converged break the loop
#             if self.is_converged(centroids_old, self.centroids):
#                 break
#             if self.plot_steps:
#                 self.plot()
#         # classify sample points as the index of their clusters
#         return self.get_cluster_labels(self.clusters)

#     def get_cluster_labels(self, clusters):
#         # each sample point will get the label of the cluster it was assigned to
#         labels = np.empty(self.n_samples)
#         for cluster_idx, cluster in enumerate(clusters):
#             for index in cluster:
#                 labels[index] = cluster_idx
#         return labels

#     def create_clusters(self, centroids):
#         # assign the sample points to the closest centroids to create clusters
#         clusters = [[] for _ in range(self.K)]
#         for idx, sample in enumerate(self.X):
#             centroid_idx = self.get_closest_index(sample, centroids)
#             clusters[centroid_idx].append(idx)
#         return clusters

#     @staticmethod
#     def get_closest_index(sample, centroids):
#         # distance of the current sample to each centroid
#         distances = [euclidean_distance(sample, point) for point in centroids]
#         closest_index = np.argmin(distances)
#         return closest_index

#     def get_centroids(self, clusters):
#         # find new centroid of each cluster
#         centroids = np.zeros((self.K, self.n_features))
#         for cluster_idx, cluster in enumerate(clusters):
#             cluster_mean = np.mean(self.X[cluster], axis=0)
#             centroids[cluster_idx] = cluster_mean
#         return centroids

#     def is_converged(self, centroids_old, centroids):
#         # distances between each old and new centroids, fol all centroids
#         distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
#         return sum(distances) == 0

#     def plot(self):
#         fig, ax = plt.subplots(figsize=(12, 8))
#         for i, index in enumerate(self.clusters):
#             point = self.X[index].T
#             ax.scatter(*point)

#         for point in self.centroids:
#             ax.scatter(*point, marker="x", color='black', linewidth=2)
#         plt.show()

#     def cent(self):
#         return self.centroids


class MeanShift:
    def __init__(self, source: np.ndarray, threshold: int):
        im = np.copy(source)
        self.threshold = threshold
        self.current_mean_random = True
        self.current_mean_arr = []
        size = im.shape[0], im.shape[1], 3
        self.output_array = np.zeros(size, dtype=np.uint8)
        self.feature_space = self.create_feature_space(source=im)

    def run_mean_shift(self):
        while len(self.feature_space) > 0:
            below_threshold_arr, self.current_mean_arr = self.calculate_euclidean_distance(current_mean_random=self.current_mean_random, threshold=self.threshold)
            self.get_new_mean(below_threshold_arr=below_threshold_arr)


    def get_output(self):
        return self.output_array

    @staticmethod
    def create_feature_space(source: np.ndarray):
        im = np.copy(source)
        rows = im.shape[0]
        columns = im.shape[1]
        feature_space = np.zeros((rows*columns, 5))
        counter = 0
        for i in range(rows):
            for j in range(columns):
                array = im[i][j]
                feature_space[counter][0:3] = array[0:3]
                feature_space[counter][3] = i
                feature_space[counter][4] = j
                counter += 1
        return feature_space

    def calculate_euclidean_distance(self, current_mean_random: bool, threshold: int):
        if current_mean_random:
            current_mean = np.random.randint(0, len(self.feature_space))
            self.current_mean_arr = self.feature_space[current_mean]

        diff = self.feature_space[:, 0:4] - self.current_mean_arr[0:4]
        distances = np.linalg.norm(diff, axis=1)

        below_threshold_arr = np.where(distances < threshold)[0]

        return below_threshold_arr, self.current_mean_arr

    def get_new_mean(self, below_threshold_arr: list):
        iteration = 0.01
        mean = np.mean(self.feature_space[below_threshold_arr], axis=0)
        mean_e_distance = np.linalg.norm(mean[0:4] - self.current_mean_arr[0:4])

        if mean_e_distance < iteration:
            new_arr = np.zeros((1, 3))
            new_arr[0][0:3] = mean[0:3]
            for i in range(len(below_threshold_arr)):
                m = int(self.feature_space[below_threshold_arr[i]][3])
                n = int(self.feature_space[below_threshold_arr[i]][4])
                self.output_array[m][n] = new_arr
                self.feature_space[below_threshold_arr[i]][0] = -1

            self.current_mean_random = True
            new_d = self.feature_space[self.feature_space[:, 0] != -1]
            self.feature_space = new_d

        else:
            self.current_mean_random = False
            self.current_mean_arr[0:4] = mean[0:4]
# class MeanShift:
#     def __init__(self, source: np.ndarray, threshold: int):
#         im=np.copy(source)
#         self.threshold = threshold
#         self.current_mean_random = True
#         self.current_mean_arr = []
#         size = im.shape[0], im.shape[1], 3
#         self.output_array = np.zeros(size, dtype=np.uint8)
#         self.feature_space = self.create_feature_space(source=im)
#     def run_mean_shift(self):
#         while len(self.feature_space) > 0:
#             below_threshold_arr, self.current_mean_arr = self.calculate_euclidean_distance(
#                 current_mean_random=self.current_mean_random,
#                 threshold=self.threshold)
#             self.get_new_mean(below_threshold_arr=below_threshold_arr)

#     def get_output(self):
#         return self.output_array

#     @staticmethod
#     def create_feature_space(source: np.ndarray):
#         im=np.copy(source)
#         rows = im.shape[0]
#         columns = im.shape[1]
#         feature_space = np.zeros((rows * columns, 5))
#         counter = 0
#         for i in range(rows):
#             for j in range(columns):
#                 array = im[i][j]
#                 for k in range(5):
#                     if (k >= 0) & (k <= 2):
#                         feature_space[counter][k] = array[k]
#                     else:
#                         if k == 3:
#                             feature_space[counter][k] = i
#                         else:
#                             feature_space[counter][k] = j
#                 counter += 1
#         return feature_space

#     def calculate_euclidean_distance(self, current_mean_random: bool, threshold: int):
#         below_threshold_arr = []
#         if current_mean_random:
#             current_mean = np.random.randint(0, len(self.feature_space))
#             self.current_mean_arr = self.feature_space[current_mean]
#         for f_indx, feature in enumerate(self.feature_space):
#             ecl_dist = euclidean_distance(self.current_mean_arr, feature)
#             if ecl_dist < threshold:
#                 below_threshold_arr.append(f_indx)
#         return below_threshold_arr, self.current_mean_arr

#     def get_new_mean(self, below_threshold_arr: list):
    
#         iteration = 0.01
#         mean_1 = np.mean(self.feature_space[below_threshold_arr][:, 0])
#         mean_2 = np.mean(self.feature_space[below_threshold_arr][:, 1])
#         mean_3 = np.mean(self.feature_space[below_threshold_arr][:, 2])
#         mean_i = np.mean(self.feature_space[below_threshold_arr][:, 3])
#         mean_j = np.mean(self.feature_space[below_threshold_arr][:, 4])
#         mean_e_distance = (euclidean_distance(mean_1, self.current_mean_arr[0]) +
#                            euclidean_distance(mean_2, self.current_mean_arr[1]) +
#                            euclidean_distance(mean_3, self.current_mean_arr[2]) +
#                            euclidean_distance(mean_i, self.current_mean_arr[3]) +
#                            euclidean_distance(mean_j, self.current_mean_arr[4]))
#         if mean_e_distance < iteration:
#             new_arr = np.zeros((1, 3))
#             new_arr[0][0] = mean_1
#             new_arr[0][1] = mean_2
#             new_arr[0][2] = mean_3
#             for i in range(len(below_threshold_arr)):
#                 m = int(self.feature_space[below_threshold_arr[i]][3])
#                 n = int(self.feature_space[below_threshold_arr[i]][4])
#                 self.output_array[m][n] = new_arr
#                 self.feature_space[below_threshold_arr[i]][0] = -1
#             self.current_mean_random = True
#             new_d = np.zeros((len(self.feature_space), 5))
#             counter_i = 0
#             for i in range(len(self.feature_space)):
#                 if self.feature_space[i][0] != -1:
#                     new_d[counter_i][0] = self.feature_space[i][0]
#                     new_d[counter_i][1] = self.feature_space[i][1]
#                     new_d[counter_i][2] = self.feature_space[i][2]
#                     new_d[counter_i][3] = self.feature_space[i][3]
#                     new_d[counter_i][4] = self.feature_space[i][4]
#                     counter_i += 1
#             self.feature_space = np.zeros((counter_i, 5))
#             counter_i -= 1
#             for i in range(counter_i):
#                 self.feature_space[i][0] = new_d[i][0]
#                 self.feature_space[i][1] = new_d[i][1]
#                 self.feature_space[i][2] = new_d[i][2]
#                 self.feature_space[i][3] = new_d[i][3]
#                 self.feature_space[i][4] = new_d[i][4]
#         else:
#             self.current_mean_random = False
#             self.current_mean_arr[0] = mean_1
#             self.current_mean_arr[1] = mean_2
#             self.current_mean_arr[2] = mean_3
#             self.current_mean_arr[3] = mean_i
#             self.current_mean_arr[4] = mean_j

class AgglomerativeClustering:
    def __init__(self, image: np.ndarray, clusters_numbers: int = 2, initial_k: int = 25):
        self.clusters_num = clusters_numbers
        self.initial_k = initial_k
        src = np.copy(image.reshape((-1, 3)))
        self.fit(src)
        self.output_image = [[self.predict_center(list(src)) for src in row] for row in image]
        self.output_image = np.array(self.output_image, np.uint8)

    def initial_clusters(self, points):
        # consider each data point a single-point cluster
        groups = {}
        d = int(256 / self.initial_k)
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            # take the two closest distance clusters by single linkage method and unify them
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]

    def fit(self, points):
        # initialize each point to a distinct cluster
        self.clusters_list = self.initial_clusters(points)
        while len(self.clusters_list) > self.clusters_num:
            # find the closest pair of clusters
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: clusters_distance_centers(c[0], c[1]))

            # remove the two clusters from the clusters list and merge them together
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]
            merged_cluster = cluster1 + cluster2

            # add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def predict_cluster(self, point):
        return self.cluster[tuple(point)]

    def predict_center(self, point):
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center