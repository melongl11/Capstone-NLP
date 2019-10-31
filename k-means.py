from gensim.models import Word2Vec
from sklearn.cluster import KMeans

model = Word2Vec.load('weight/namuwiki-2.model')
result = model.most_similar('안전', topn=90)
print(len(model.wv['강아지'])) # 이런 방식으로 각 단어 vector에 접근 가능함.
print(len(model.wv.syn0[0])) # 해당 모델에서 각 단어의 vector를 가지고 있음. 차원은 n * 100 n=단어 수

word_vectors = []
num_clusters = 8
word_names = []
for r in result:
    print(model.wv[r[0]])
    word_vectors.append(model.wv[r[0]])
    word_names.append(r[0])

inertia = []
for k in range(1, 20):
    kmeans_clustering = KMeans(n_clusters=k)
    kmeans_clustering.fit(word_vectors)
    inertia.append(kmeans_clustering.inertia_)

print(inertia)
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

idx = list(idx)
print(idx)

cluster = [list() for _ in range(num_clusters)]

for i in range(len(idx)):
    cluster[idx[i]].append(word_names[i])

for i in range(num_clusters):
    print("Cluster ", i, " ", cluster[i])

print(result)