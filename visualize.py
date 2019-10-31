from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gensim
import gensim.models as g
import pandas as pd

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)

model_name = 'weight/1572369367.7740755word2vec.model'
model = g.Doc2Vec.load(model_name)

word_n = 500

vocab = list(model.wv.vocab)
X = model[vocab]

tsne = TSNE(n_components=2)

X_tsne = tsne.fit_transform(X[:word_n, :])

df = pd.DataFrame(X_tsne, index=vocab[:word_n], columns=['x', 'y'])

fig = plt.figure()
fig.set_size_inches(20, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=10)
plt.show()