from gensim.models import Word2Vec
from sklearn.cluster import KMeans

model = Word2Vec.load('weight/1570397982.644223word2vec.model')
result = model.most_similar('강아지', topn=100)
print(len(model.wv['강아지'])) # 이런 방식으로 각 단어 vector에 접근 가능함.
print(len(model.wv.syn0[0])) # 해당 모델에서 각 단어의 vector를 가지고 있음. 차원은 n * 100 n=단어 수

word_vectors = []
num_clusters = 8
word_names = []
for r in result:
    print(model.wv[r[0]])
    word_vectors.append(model.wv[r[0]])
    word_names.append(r[0])


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