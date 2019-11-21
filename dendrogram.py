import numpy as np
import time
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster import cluster_visualizer
import Word2VecSingleton

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)

logger = Word2VecSingleton.Logger('weight/namuwiki-2-window10.model')

model = logger.model

start = time.time()

result = model.most_similar('고양이', topn=100)

word_vectors = []
num_clusters = 8
word_names = []
for r in result:
    word_vectors.append(model.wv[r[0]])
    word_names.append(r[0])

tsne = PCA(n_components=2)

X_tsne = tsne.fit_transform(word_vectors)

agglomerative_instance = agglomerative(X_tsne, 8, type_link.SINGLE_LINK, True)

agglomerative_instance.process()

clusters = agglomerative_instance.get_clusters()
print(clusters)
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, X_tsne)
visualizer.show()