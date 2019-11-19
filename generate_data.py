
from koalanlp.Util import initialize, finalize
from koalanlp.proc import *
from koalanlp import API

from konlpy.tag import Okt, Kkma, Komoran, Hannanum

'''
n = 0
result = []


initialize(java_options="-Xmx4g -Dfile.encoding=utf-8", KKMA="2.0.2", EUNJEON="2.0.2", ETRI="2.0.2")

parser = Parser(API.KKMA)
parsed = parser('안녕하세요. 저는 홍길동입니다. 무엇을 도와 드릴까요? 4차산업혁명에는 블록체인이 대세다')


for dep in parsed[0].getEntities():
    print(dep)

'''

sentence = u"""실업계 학과 개편(관상원예과, 산업유통정보과)"""

okt = Okt()
print(okt.nouns(sentence))

kkma = Kkma()
print(kkma.nouns(sentence))

komoran = Komoran()
print(komoran.nouns(sentence))

hannanum = Hannanum()
print(hannanum.nouns(sentence))