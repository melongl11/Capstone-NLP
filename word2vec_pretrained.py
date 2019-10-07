from gensim.models import Word2Vec

model = Word2Vec.load('ko.bin')

result = model.most_similar("최민섭", topn=100)

print(result)

print(len(result))