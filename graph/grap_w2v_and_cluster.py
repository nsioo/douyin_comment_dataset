import pandas as pd
import numpy as np
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

def load_w2v_result(path):
    kvp = {}
    f = open(path, 'r', encoding='utf-8')
    for line in tqdm(f.readlines()[1:]):
        line_list = line.split(' ')
        word = line_list[0]
        value = np.mean(list(map(float, line_list[1:])))
        kvp[word] = value
    # sort
    kvp = dict(sorted(kvp.items(), key=lambda x : x[-1], reverse=True))
    return kvp

def load_keam_result(path):
    df = pd.read_csv(path, encoding='utf-8')
    keys = list(df['words'])
    values = list(df['sentiment_coeff'].apply(lambda x : float(x)))
    kvp = dict(zip(keys, values))
    kvp = dict(sorted(kvp.items(), key=lambda x : x[-1], reverse=True))
    return kvp

def visualization(kvp):

    # graph
    x = [i for i in range(len(kvp))]
    y = list(kvp.values())
    print(y)
    n = list(kvp.keys())
    # plt.figure(figsize=(10, 15))
    plt.plot(x, y, 'bo-', ms=0.001)
    if 'cluster' in kvp:
        plt.title('kmean result visualization')
    else:
        plt.title('word2vec result visualization')
    plt.xlabel('value of sentiment coefficient')
       # for i, k in enumerate(n):
    #     # ran = randint(-15, 15)
    #     plt.text(x=x[i], y=y[i], s=k, fontsize=15, color='mediumvioletred')
    plt.show()
    print(list(kvp.keys())[:100])
    print(list(kvp.keys())[-100:])

     





if __name__ == "__main__":
    path_w2v = 'model_music_word_ch.vec'
    path_kmean = 'sentiment_score_cluster.csv'
    kvp_w2v = load_w2v_result(path_w2v)
    kvp_cluster = load_keam_result(path_kmean)
    visualization(kvp_w2v)
    visualization(kvp_cluster)