import re
import os
import copy
import numpy as np
import jieba 
import jieba.analyse as anls
import jieba.posseg as pseg
import matplotlib.pyplot as plt

# jieba.load_userdict('../pos_neg/emoji.txt')


def process_v2_txt(path):
    cut_result = set()
    key_words = set()
    phrase = {}
    with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = map(lambda x : ''.join(re.findall('[\u4e00-\u9fff]', x)), line.strip())
                line = ''.join(line)
                cut_line = jieba.cut(line, cut_all=False, HMM=True)
                # cut_line_2 = copy.deepcopy(cut_line)
                # split words
                # print(list(cut_line))
                for c in list(cut_line):
                    cut_result.add(c)

                # phrase
                words = pseg.cut(line)
                for _, flag in words:
                    if flag not in phrase:
                        phrase[flag] = 0
                    phrase[flag] += 1

    # # key words
    # with open(path, 'r', encoding='utf-8') as f2:
    #     for line in f2.readlines():
    #         line = map(lambda x : ''.join(re.findall('[\u4e00-\u9fff]', x)), line.strip())
    #         line = ''.join(line)
    #         cut_line = jieba.cut(line, cut_all=False, HMM=True)
    #         space_line = ' '.join(list(line))
    #         kw = anls.extract_tags(sentence=space_line, topK=20, allowPOS=('ns', 'n'))
    #         for i in kw:
    #             key_words.add(i)


    
    print(len(cut_result))
    # print(key_words, len(key_words))
    print(phrase)
    # return cut_result


path = '../data_s2/single_vedio/dance/dance_result.txt'
process_v2_txt(path)

