# coding = utf-8

from __future__ import division


from six.moves import xrange
from sys import exit
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import scipy
import math



'''
word embedding , embedding lookup
'''

# word is pos/neg words
def emblookup(words, word2vec):
    ret = []
    for w in words:
        w = w.lower()
        # if word2vec do not find the target word , jist skip
        if w not in word2vec:
            continue
        # saving the target word vector into ret
        ret.append(word2vec[w])
    print('the size of sentiment words found from embedding table have ', len(ret))
    return ret




'''
regularization = (Σx^2)^1/2
every vector value / regularization -- > normalizer
'''
def normalizer(myvector):
    my_sum = 0.
    for myvalue in myvector:
        my_sum += myvalue * myvalue # power2 and sum
    if my_sum <= 0.:  # 
        return myvector
    my_sum = math.sqrt(my_sum) # regularization
    newvector = []
    for myvalue in myvector:
        newvector.append(myvalue / my_sum) # divided by regularization
    # return normailized vector
    return newvector



def line_process(l):
    try:
        l = l.decode('utf-8').strip().split(' ')
    except:
        print(l[0]) # exception 
        return (None, None)
    # 拿到当前word
    word = l[0].lower()
    # 循环 该word的vector并正则话
    vals = normalizer([float(v) for v in l[1:]])
    return (word, vals)


def word2vec(emb_path):
    word2vect = {} # 返回字典
    p = Pool(cpu_count())
    # 多线程处理
    # [token,idx=1], [vector],[vector],[vector]....
    with open(emb_path, 'rb') as f:
        # 因为第一行是 vec file 的shape 即 999995 300
        # 返回的 pairs 为 [(word, vals), (word2, vals), ...]
        pairs = p.map(line_process, f.readlines()[1:])
    p.close()
    p.join()
    _pairs = []
    # 去除None word
    for i in pairs:
        if i[0] is not None:
            _pairs.append(i)
    print('success')
    return  dict(_pairs)



def read(emb_path):
    with open(emb_path, 'rb') as f:

        # print(f.readlines()[1].strip().split(' '))
        for i in f.readlines():
            try:
                print(i.decode('utf-8').strip().split()[:5])
                # print(type(i.decode('utf-8')))
            except:
                break


if __name__ == "__main__":
    pass
    # read('../../../wiki-news-300d-1M.vec')
    # x = word2vec('../TikTok-300d-170h.vec')
    # print(len(x))