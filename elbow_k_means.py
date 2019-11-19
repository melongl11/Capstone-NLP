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


def elbow_k_means(key_word, model_path):
    model = Word2Vec.load(model_path)
    result = model.most_similar(key_word, topn=100)

    word_vectors = []
    num_clusters = 8
    word_names = []
    word_correlation = []
    for r in result:
        word_vectors.append(model.wv[r[0]])
        word_names.append(r[0])
        word_correlation.append(r[1])

    tsne = PCA(n_components=2)

    X_tsne = tsne.fit_transform(word_vectors)

    kmin, kmax = 1, 10
    elbow_instance = elbow(X_tsne, kmin, kmax)

    elbow_instance.process()
    amount_clusters = elbow_instance.get_amount()
    wce = elbow_instance.get_wce()

    centers = kmeans_plusplus_initializer(X_tsne,
                                          amount_clusters,
                                          amount_candidates=kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
    k_means_instance = kmeans(X_tsne, centers)
    k_means_instance.process()

    clusters = k_means_instance.get_clusters()
    centers = k_means_instance.get_centers()

    index_to_word = [[] for i in range(len(clusters))]
    index_to_correlation = []
    idx = 0
    for c in clusters:
        for i in c:
            index_to_word[idx].append(word_names[i])
            index_to_correlation.append(word_correlation[i])
        idx += 1


    return len(clusters), index_to_word, index_to_correlation
