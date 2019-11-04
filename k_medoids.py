from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.elbow import elbow
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils.metric import distance_metric, type_metric
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd


model = Word2Vec.load('weight/namuwiki-2.model')
result = model.most_similar('안전', topn=100)

word_vectors = []
num_clusters = 8
word_names = []
for r in result:
    word_vectors.append(model.wv[r[0]])
    word_names.append(r[0])

tsne = PCA(n_components=2)

X_tsne = tsne.fit_transform(word_vectors)

kmin, kmax = 1, 10
elbow_instance = elbow(X_tsne, kmin, kmax)

elbow_instance.process()
amount_clusters = elbow_instance.get_amount()
wce = elbow_instance.get_wce()

centers = kmeans_plusplus_initializer(X_tsne, amount_clusters, amount_candidates=kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
k_means_instance = kmeans(X_tsne, centers)
k_means_instance.process()

clusters = k_means_instance.get_clusters()
centers = k_means_instance.get_centers()


init = 0
initial_medoids = []
for i in range(amount_clusters):
    initial_medoids.append(init)
    init += int(100 / amount_clusters)
kmedoids_instance = kmedoids(X_tsne, initial_medoids)

kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
print(clusters)


index_to_word = [[] for i in range(num_clusters)]
idx = 0
for c in clusters:
    for i in c:
        index_to_word[idx].append(word_names[i])
    idx += 1

for i in range(num_clusters):
    print("Cluster ", i, " ", index_to_word[i])
print(index_to_word)


visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, X_tsne)
visualizer.show()

#print(k_means_instance.get_total_wce())
#kmeans_visualizer.show_clusters(X_tsne, clusters, centers)