from konlpy.tag import Okt, Kkma
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


result = generate_data('dataset/wikiAA.txt')

model = Word2Vec(result, size=100, window=5, min_count=5, workers=4, sg=0)

model.save(str(time.time()) + "word2vec.model")

