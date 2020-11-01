import os, sys
import pandas as pd
import numpy as np
import codecs
import re
from re import sub
import multiprocessing
import jieba
from tqdm import tqdm
jieba.load_userdict('../../pos_neg/emoji.txt')
from unidecode import unidecode

from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile   
from gensim.models import KeyedVectors


from time import time 
from collections import defaultdict

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)



def stop_word(path):
    buff = []
    with codecs.open(path) as fp:
        for line in fp:
            buff.append(line[:-2].strip())
    return set(buff)



def process_txt(path, word=False):
    cut_result = []
    result = []
    cut_filter = set()
    sw = stop_word('../../pos_neg/stopword.txt')
    print(sw)
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = map(lambda x : ''.join(re.findall("[\u4e00-\u9fff,!?.\/'+]", x)), line.strip())
            line = ''.join(line)
            cut_line = jieba.cut(line, cut_all=False, HMM=True)
            cut_list = list(cut_line)
            cut_list = [i for i in cut_list if i not in sw]
            cut_str = ' '.join(cut_list)
            if len(cut_list) == 0:
                continue
            if cut_str in cut_filter:
                continue
            else:
                cut_filter.add(cut_str)
            cut_result.append(cut_str+'\n')
    print("total data : %d" % len(cut_result))
    return cut_result, 



def word2vec_word(data, save_path):
    w2v_model = Word2Vec(min_count=3,
                window=4,
                size=300,
                sample=1e-5, 
                alpha=0.03, 
                min_alpha=0.0007, 
                negative=20,
                workers=multiprocessing.cpu_count()-1)
    # init
    start = time()
    w2v_model.build_vocab(LineSentence(data), progress_per=50000)
    print('Time to build vocab: {} mins'.format(round((time() - start) / 60, 2)))
    # train
    start = time()
    w2v_model.train(LineSentence(data), total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - start) / 60, 2)))
    w2v_model.init_sims(replace=True)
    w2v_model.save("word2vec_word.model")
    w2v_model.wv.save_word2vec_format(save_path, binary=False)
    
       

def word2vec_sentence(data,  save_path):
    phrases = Phrases(data, min_count=1, progress_per=50000)
    bigrame = Phraser(phrases)
    sentences = bigrame[data]
    print(sentences)
    w2v_model = Word2Vec(min_count=3,
                        window=4,
                        size=300,
                        sample=1e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=multiprocessing.cpu_count()-1)
    # init
    start = time()
    w2v_model.build_vocab(sentences, progress_per=50000)
    print('Time to build vocab: {} mins'.format(round((time() - start) / 60, 2)))
    # train
    start = time()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - start) / 60, 2)))
    w2v_model.init_sims(replace=True)
    w2v_model.save("word2vec.model")
    w2v_model.wv.save_word2vec_format(save_path, binary=False)
    

if __name__ == "__main__":
    clean_data = process_txt(sys.argv[1])  
    f = open('clean_data.txt', 'w', newline='',  encoding='utf-8')
    f.writelines(clean_data)
    word2vec_word('clean_data.txt', save_path='model_music_word_ch.vec')
    # word2vec_sentence(clean_data, save_path='model_music.vec')

