import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

source = "cn"
max_n_cluster = 10
perplexity = 8
dir_path = "output_data/" + source + "/"

data_encode = np.genfromtxt(dir_path + "data.csv", delimiter = ",")
print(data_encode.shape)

tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = 3407)
data_2d = tsne.fit_transform(data_encode)
plt.scatter(data_2d[:, 0], data_2d[:, 1], marker = ".", alpha = 0.5)
plt.savefig(f"{dir_path}cluster/tsne.png")
plt.show()

avg_inter_dist = []
avg_cross_dist = []
pop_array = np.zeros((max_n_cluster - 2, max_n_cluster - 1))
for k in range(2, max_n_cluster):
    kmeans = KMeans(n_clusters = k, random_state = 3407)
    kmeans.fit(data_encode)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c = kmeans.labels_, marker = ".", alpha = 0.5)
    plt.savefig(f"{dir_path}cluster/tsne_{k}_clusters.png")

    inter_dist_mat = kmeans.transform(data_encode)
    avg_inter_dist.append(np.mean(np.min(inter_dist_mat, axis = 1)))

    cross_dist_mat = np.zeros((k, k))
    for i in range(k - 1):
        for j in range(i + 1, k):
            cross_dist_mat[i, j] = np.linalg.norm(kmeans.cluster_centers_[i, :] - kmeans.cluster_centers_[j, :], ord = 2)

    avg_cross_dist.append(np.sum(cross_dist_mat) / k / (k - 1))

    clust_pop = np.unique(kmeans.labels_, return_counts = True)
    for i, population in enumerate(np.sort(-clust_pop[1])):
        pop_array[k - 2, i] = -population

plt.plot(range(2, max_n_cluster), avg_inter_dist, ".-")
plt.xlabel("Number of Clusters")
plt.ylabel("Inter-Cluster Distance")
plt.savefig(f"{dir_path}cluster/inter_dist.png")
plt.show()

plt.plot(range(2, max_n_cluster), avg_cross_dist, ".-")
plt.xlabel("Number of Clusters")
plt.ylabel("Cross-Cluster Distance")
plt.savefig(f"{dir_path}cluster/cross_dist.png")
plt.show()

print(pop_array)
cum_pop = np.cumsum(pop_array, axis = 1)
plt.bar(range(2, max_n_cluster), pop_array[:, 0])
for i in range(max_n_cluster - 2):
    plt.bar(range(2, max_n_cluster), pop_array[:, i + 1], bottom = cum_pop[:, i])

plt.xlabel("Number of clusters")
plt.ylabel("Population in each cluster")
plt.savefig(f"{dir_path}cluster/clust_pop.png")
plt.show()
