from konlpy.tag import Okt, Kkma, Mecab
import MeCab

kkma = Kkma()
mecab = MeCab.Tagger()


n = 0
result = []

print(mecab.parse('안녕하세요. 저는 홍길동입니다. 무엇을 도와 드릴까요? 4차산업혁명에는 블록체인이 대세다'))
print(kkma.pos('안녕하세요. 저는 홍길동입니다. 무엇을 도와 드릴까요? 4차산업혁명에는 블록체인이 대세다'))
