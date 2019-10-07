from gensim.models import Word2Vec

model = Word2Vec.load('weight/1570397982.644223word2vec.model')
result = model.most_similar('강아지')
print(result)