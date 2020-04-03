# coding = utf-8



# 导入 jieba
import jieba
import jieba.posseg as pseg #词性标注
import jieba.analyse as anls #关键词提取

# 使用方法
# https://blog.csdn.net/qq_27586341/article/details/90025288
# https://zhuanlan.zhihu.com/p/40016964
from gensim.models import Word2Vec



'''
this script can transfer word to vector , the following step will use:
1. comment preparing
2. cut comment to single word
3. useing gensim module transfer word to vector and build origin word embedding table for comments

'''





