from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import pickle

def K_Means_elbow_best_model(df,k_min,k_max, step, folder_name):

    X = df

    n_clusters_range = range(k_min,k_max+1)


    for n_clusters in n_clusters_range:
        
        if 2+step*n_clusters >= k_max:
            break
        else:
            time_start = time.time()
            kmeans = KMeans(n_clusters=(2+step*n_clusters), init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            with open(f'/Users/edoardorossi/Documents/GitHub/Tesi/MDP_data/kmeans/{folder_name}/kmeans_{2+step*n_clusters}.pkl', 'wb') as f:
                pickle.dump(kmeans, f)
            time_elapsed = time.time() - time_start
            print("KMeans with", 2+step*n_clusters, "clusters computed in", time_elapsed, "seconds")
        


def K_Medoids_elbow_best_model(df,k_min, k_max):

    X = df

    n_clusters_range = range(k_in,k_max+1)

    wcss = []

    for n_clusters in n_clusters_range:
        
        time_start = time.time()
        kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean')
        kmedoids.fit(X)
        wcss.append(kmedoids.inertia_)
        time_elapsed = time.time() - time_start
        print("KMedoids with", n_clusters, "clusters computed in", time_elapsed, "seconds")

    plt.plot(n_clusters_range, wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def agglomerative_elbow(df,k_max):

    X = df
    sse = []

    for i in range(2,k_max+1):
        time_start = time.time()
        clustering = AgglomerativeClustering(n_clusters=i)
        clustering.fit(X)
        sse.append(clustering.inertia_)
        time_elapsed = time.time() - time_start
        print("Agglomerative with", i, "clusters computed in", time_elapsed, "seconds")
    plt.plot(range(2,k_max+1), sse)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

