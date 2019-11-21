from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gensim
import gensim.models as g
import pandas as pd
import Word2VecSingleton

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)

logger = Word2VecSingleton.Logger('weight/namuwiki-2-window10.model')
model = logger.model

word_n = 500

result = model.most_similar('강아지', topn=100)
word_vectors = []
num_clusters = 8
word_names = []
for r in result:
    word_vectors.append(model.wv[r[0]])
    word_names.append(r[0])

tsne = PCA(n_components=2)

X_tsne = tsne.fit_transform(word_vectors)

df = pd.DataFrame(X_tsne, index=word_names, columns=['x', 'y'])

fig = plt.figure()
fig.set_size_inches(20, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=10)
plt.show()

