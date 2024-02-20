import sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score, silhouette_score

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.
def load_data(filepath):
    data = pd.read_csv(filepath, low_memory=False)
    # Drop non-numeric columns
    data = data.select_dtypes(include=[np.number])
    return data

# kmeans, https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# DBI: https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
# https://www.geeksforgeeks.org/k-means-clustering-introduction/
# SI, https://www.geeksforgeeks.org/silhouette-index-cluster-validity-index-set-2/
def kmeans_dm(data, num_clusters):
    start_time = time.time()
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(data)
    dbi = davies_bouldin_score(data, labels)
    si = silhouette_score(data, labels)
    runtime = time.time() - start_time
    return dbi, si, runtime

# kmedoids, https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
def kmedoids_dm(data, num_clusters):
    start_time = time.time()
    kmedoids = KMedoids(n_clusters=num_clusters)
    labels = kmedoids.fit_predict(data)
    dbi = davies_bouldin_score(data, labels)
    si = silhouette_score(data, labels)
    runtime = time.time() - start_time
    return dbi, si, runtime

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
def hierarchicalclustering_dm(data, num_clusters):
    start_time = time.time()
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    labels = hierarchical.fit_predict(data)
    dbi = davies_bouldin_score(data, labels)
    si = silhouette_score(data, labels)
    runtime = time.time() - start_time
    return dbi, si, runtime

def plot_results(results, cluster_sizes):
    fig, axes = plt.subplots(3, 1)
    metrics = ['DBI', 'SI', 'Runtime']
    for i, metric in enumerate(metrics):
        for alg, values in results.items():
            axes[i].plot(cluster_sizes, values[metric], label=f'{alg} ({metric})')
        axes[i].set_xlabel('Number of Clusters')
        axes[i].set_ylabel(metric)
        axes[i].legend()
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Ooooooooooooooooof, something wrong")
        print("Please follow: python3 cluster_analysis.py <input_data> <number of clusters>")
        sys.exit(1)

    input_data = sys.argv[1]
    num_clusters = int(sys.argv[2])
    data = load_data(input_data)

    # add def to algorithms dictionary
    algorithms = {
        'K-Means': kmeans_dm,
        'K-Medoids': kmedoids_dm,
        'Hierarchical': hierarchicalclustering_dm
    }


    results = {}
    for alg in algorithms.keys():
        results[alg] = {'DBI': [], 'SI': [], 'Runtime': []}

    total_runtimes = {}
    for alg_name in algorithms.keys():
        total_runtimes[alg_name] = 0

    cluster_sizes = range(2, num_clusters + 1) # Evaluating for different cluster sizes

    for size in cluster_sizes:
        # Iterate over each algorithm and function
        for alg_name, alg_func in algorithms.items():
            start_time = time.time()
            # Execute the clustering function for the current algorithm with cluster size
            clustering_result = alg_func(data, size)
            # Extract the results: DBI, SI, and runtime
            dbi = clustering_result[0]
            si = clustering_result[1]
            runtime = clustering_result[2]
            # Append DBI result to results dictionary
            dbi_list = results[alg_name]['DBI']
            dbi_list.append(dbi)
            results[alg_name]['DBI'] = dbi_list
            # Append SI result to results dictionary
            si_list = results[alg_name]['SI']
            si_list.append(si)
            results[alg_name]['SI'] = si_list
            # Append Runtime result to results dictionary
            runtime_list = results[alg_name]['Runtime']
            runtime_list.append(runtime)
            results[alg_name]['Runtime'] = runtime_list
            end_time = time.time()
            runtime = end_time - start_time
            total_runtimes[alg_name] += runtime
            print(f"{alg_name} runtime for {size} clusters: {runtime:.2f} seconds")

    for alg_name, total_runtime in total_runtimes.items():
        print(f"Total runtime for {alg_name}: {total_runtime:.2f} seconds")

    plot_results(results, cluster_sizes)


if __name__ == "__main__":
    main()
