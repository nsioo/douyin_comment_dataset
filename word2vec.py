# coding = utf-8

# 导入 jieba
import re
import os, sys
import logging
import jieba
import jieba.posseg as pseg #词性标注
import jieba.analyse as anls #关键词提取
import pandas as pd
# 使用方法
# https://blog.csdn.net/qq_27586341/article/details/90025288
# https://zhuanlan.zhihu.com/p/40016964
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from multiprocessing import Pool
from multiprocessing import cpu_count
'''
this script can transfer word to vector , the following step will use:
1. comment preparing
2. cut comment to single word
3. useing gensim module transfer word to vector and build origin word embedding table for comments

'''

def build_emoji_corpus(data):

    emoji = './output/emoji.txt'
    if os.path.exists(emoji):
        os.remove(emoji) 

    with open('./single_vedio/'+data+'.txt', 'r', encoding='utf_8_sig') as f:
        for c in f:
            try:
                if '[' in c and ']' in c:
                    sidx = c.index('[')
                    eidx = c.index(']')
                    c = c[sidx+1:eidx]

                    with open(emoji, 'a', newline='\n', encoding='utf_8_sig') as f:
                        f.write(c+'\r\n')
            except:
                continue


def data_process_xlsx(data):
    jieba.load_userdict('./output/emoji.txt')
    df = pd.read_excel(data+'_update.xlsx', sheet_name='Sheet1', encoding = 'utf-8')
    df = df['comment'].map(lambda x : ''.join(re.findall('[\u4e00-\u9fff]', str(x))))

    sl = []
    for c in df:
        s = jieba.lcut(c, cut_all=False, HMM=True)
        s = ' '.join(s)
        sl.append(s)
    print('good'+data)
    return sl


def data_process_txt(data):    
    jieba.load_userdict('./output/emoji.txt')
    with open('./single_vedio/'+data+'.txt', encoding = 'utf_8_sig') as f:
        s1 = []
        for l in f.readlines():
            l = ''.join(re.findall('[\u4e00-\u9fff]', str(l)))
            s = jieba.lcut(l, cut_all=False, HMM=True)
            s1.append(s)
        print('good'+data)
        return (s1, data)


def add_sentence_corpus(sl, data, save_path='./output/'):

    if os.path.exists(save_path+data+'_s.txt'):
        os.remove(save_path+data+'_s.txt')
        open(save_path+data+'_s.txt', 'w', encoding='utf_8_sig')

    for line in sl:
        with open(save_path+data+'_s.txt', 'a', newline='\n', encoding='utf_8_sig') as f:
            f.write(' '.join(line)+'\r\n')    
    print('good2'+data)




def build_w2v(st_path, save_path):
    # sys.argv[0]获取的是脚本文件的文件名称
    program = os.path.basename(sys.argv[0])
    print(program)

    # 建立logger 实例
    logger = logging.getLogger(program)

    #1.format: 指定输出的格式和内容，format可以输出很多有用信息，
    #%(asctime)s: 打印日志的时间
    #%(levelname)s: 打印日志级别名称
    #%(message)s: 打印日志信息
    logging.basicConfig(filename='./log/log.txt' ,
                        format='%(asctime)s: %(levelname)s: %(message)s', 
                        level= logging.INFO)

    # 打印一个通知日志
    logging.info('running %s ' % ' '.join(sys.argv))

    if len(sys.argv):
        globals()['__doc__'] = locals()
        print(globals()['__doc__'])
    '''
    LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
    size：是每个词的向量维度;
    window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词； 
    min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃； 
    workers：是训练的进程数（需要更精准的解释，请指正），默认是当前运行机器的处理器核数。这些参数先记住就可以了。
    sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
    alpha (float, optional) – 初始学习率
    iter (int, optional) – 迭代次数，默认为5
    '''
    model = Word2Vec(LineSentence(st_path), size=300, window=2, min_count=5, workers=cpu_count())
    
    # model 保存
    # model.save(save_path)

    #不以C语言可以解析的形式存储词向量
    model.wv.save_word2vec_format(save_path, binary=False)




def evaluate(testwords, model='./output/TikTok_word2vec.model'):
    TikTok_w2v_model = Word2Vec.load(model)
    for i in testwords:
        res = TikTok_w2v_model.most_similar(i)
        print(i, 'the probability of similar:', res)



if __name__ == "__main__":
    # build_emoji_corpus('雷军评论')

    # 制作 setence corpus
    dsl = ['减肥健康', '情感恋爱', '抗疫', '搞笑2', '搞笑3', '明星李易峰', '罗永浩', '董明珠', '雷军评论']
    # p = Pool(4)
    # sls = p.map(data_process_txt, dsl)
    # p.close()
    # p.join()
    # for ls, data in sls:
    #     # print('data:', data, 'ls', ls)
    #     # print('--'*20)
    #     add_sentence_corpus(ls, data)

    for p, _, fs in [i for i in os.walk('./output')]:
        for f in fs[1:]:
            f_p = p+'/'+f
            #制作 word2vector
            build_w2v(f_p, p+'/'+f[:3]+'.vec')    

    # test word2vector
    # testwords = ['哈哈', '主播', '可爱', '厉害', '心疼']
    # evaluate(testwords)