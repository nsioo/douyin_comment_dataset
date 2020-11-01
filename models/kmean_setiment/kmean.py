import sys
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

def kmean(path):
    word_vectors = Word2Vec.load(path).wv
    model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50)\
                .fit(X=word_vectors.vectors)
    word_vectors.similar_by_vector(model.cluster_centers_[0], topn=10, restrict_vocab=None)
    positive_cluster_center = model.cluster_centers_[0]
    negative_cluster_center = model.cluster_centers_[1]

    words = pd.DataFrame(word_vectors.vocab.keys())
    words.columns = ['words']
    words['vectors'] = words['words'].apply(lambda x: word_vectors.wv[f'{x}'])
    words['cluster'] = words['vectors'].apply(lambda x: model.predict([np.array(x)]))
    words['cluster'] = words['cluster'].apply(lambda x: x[0])
    words['cluster_value'] = [1 if i==0 else -1 for i in words['cluster']]
    words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x['vectors']]).min()), axis=1)
    words['sentiment_coeff'] = words['closeness_score'] * words['cluster_value']
    print(words.head(10))

    words[['words', 'sentiment_coeff']].to_csv('sentiment_score_cluster.csv', index=False)
if __name__ == "__main__":
    w2v_path = sys.argv[1]
    kmean(w2v_path)