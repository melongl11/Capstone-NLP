from ensemble_k_means import *
import time

model_path = 'weight/namuwiki-2.model'
n_clusters = 8
n_ensembles = 40
start = time.time()
cluster = get_cluster(key_word="안전", model_path=model_path, n_clusters=n_clusters, n_ensembles=n_ensembles)
print(time.time() - start)
for i in range(n_clusters):
    print("Cluster ", i, " ", cluster[i])