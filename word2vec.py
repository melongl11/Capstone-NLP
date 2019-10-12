from konlpy.tag import Okt, Kkma, Mecab
from gensim.models import Word2Vec
import time

def generate_data(filename):
    kkma = Kkma()
    fread = open(filename, encoding='utf8')

    n = 0
    result = []

    while True:
        line = fread.readline()
        if not line: break
        n = n + 1

        if n % 5000 == 0:
            print("%d 번째 While 문"%n)

        try:
            tokenlist = kkma.pos(line)
        except Exception:
            continue
        temp = []
        for word in tokenlist:
            if word[1] in ['NNG', 'NNP']:
                temp.append((word[0]))
        if temp:
            result.append(temp)
    fread.close()

    return result

textset = ['dataset/wikiAB.txt', 'dataset/wikiAC.txt', 'dataset/wikiAD.txt', 'dataset/wikiAE.txt', 'dataset/wikiAF.txt']

weight = "1570397982.644223word2vec.model"
for t in textset:
    result = generate_data(t)

    model = Word2Vec(size=100, window=5, min_count=5, workers=4, sg=1)
    model.load(weight)
    model.build_vocab(result)
    model.train(result, total_examples=model.corpus_count, epochs=model.iter)

    weight = str(time.time()) + "word2vec.model"
    model.save(str(time.time()) + "word2vec.model")

