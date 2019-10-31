from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)
model = Word2Vec.load('weight/1570397982.644223word2vec.model')
result = model.most_similar('안전', topn=70)
print(len(model.wv['강아지'])) # 이런 방식으로 각 단어 vector에 접근 가능함.
print(len(model.wv.syn0[0])) # 해당 모델에서 각 단어의 vector를 가지고 있음. 차원은 n * 100 n=단어 수

word_vectors = []
num_clusters = 20
word_names = []
for r in result:
    word_vectors.append(model.wv[r[0]])
    word_names.append(r[0])

print(word_vectors[0][:10])
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(word_vectors)
df = pd.DataFrame(X_tsne, index=word_names, columns=['x', 'y'])  # 10차원 데이터를 2차원으로.

inertia = []
for k in range(1, 20):
    kmeans_clustering = KMeans(n_clusters=k)
    idx = kmeans_clustering.fit_predict(df)
    inertia.append(int(kmeans_clustering.inertia_))

print(inertia)
index = [i for i in range(1, 20)]
fig = go.Figure(data=[go.Table(header=dict(values=['k', 'inertia']),
                               cells=dict(values=[index, inertia]))
                      ])
fig.show()
