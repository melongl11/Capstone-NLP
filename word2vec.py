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


def read_data(filename):
    fread = open(filename, encoding='utf8')

    result = []
    n = 0
    while True:
        line = fread.readline()
        if not line: break
        n = n + 1
        if n % 5000 == 0:
            print("%d 번째 While 문"%n)
        splitted = line.split(' ')

        result.append(splitted[:len(splitted) - 1])

    return result

result = read_data('dataset/namuwiki_data_khaiii/2/namuwiki_data_khaiii_2.txt')

# initialize train model
# model = Word2Vec(size=100, window=5, min_count=5, workers=4, sg=1)
# update model
model = Word2Vec.load('weight/namuwiki-1.model')

model.build_vocab(result, update=True)
model.train(result, total_examples=model.corpus_count, epochs=model.iter)

model.save(str(time.time()) + "word2vec.model")

