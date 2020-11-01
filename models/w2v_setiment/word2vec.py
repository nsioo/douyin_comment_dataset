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
from functools import partial
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

def stop_word(sw_path):
    with open(sw_path, 'r', encoding='utf_8_sig') as f:
        swl = []
        for i in f.readlines():
            swl.append(i.strip('\n'))
        return swl

def data_process_excel(data, sw):
    jieba.load_userdict('./output/emoji.txt')
    df = pd.read_excel('./radom_vedio/'+data+'_update.xlsx', sheet_name=0, encoding = 'utf-8')
    df = df['comment'].map(lambda x : ''.join(re.findall('[\u4e00-\u9fff]', str(x))))

    sl = []
    for c in df:
        s = jieba.lcut(c, cut_all=False, HMM=True)
        # print('打印 s:', s)
        s = [i for i in s if not i in sw]
        # s = ' '.join(s)
        sl.append(s)
    print('good'+data)
    return sl, data


def data_process_txt(data, sw):    
    jieba.load_userdict('./output/emoji.txt')
    with open('./douyin_crawl/result/'+data+'.txt', encoding = 'utf_8_sig') as f:
        s1 = []
        for l in f.readlines():
            l = ''.join(re.findall('[\u4e00-\u9fff]', str(l)))
            s = jieba.lcut(l, cut_all=False, HMM=True)
            # 去停用词
            s = [i for i in s if not i in sw]
            s1.append(s)
        print('good'+data)
        return (s1, data)


def add_sentence_corpus(sl, data='result_v2_process', save_path='./output/'):

    if clean:
        os.remove(save_path+data+'.txt')
        open(save_path+data+'.txt', 'w', encoding='utf_8_sig')

    for line in sl:
        # 空行和单个字行去除
        if line == []:
            continue
        elif len(line) == 1:
            continue
        with open(save_path+data+'.txt', 'a', newline='\n', encoding='utf_8_sig') as f:
            f.write(' '.join(line)+'\r\n')    
    print('good2'+data)


def add_sentence_corpus_tocsv(sl, data='result_v2_process', save_path='./output/'):

    df = pd.DataFrame(columns=['word'])    
    bag = []
    for line in sl:
        # 空行和单个字行去除
        if line == []:
            continue
        elif len(line) == 1:
            continue
        else:
            for l2 in line:
                bag.append(l2)

    df['word'] = bag
    df.to_csv(save_path+data+'.csv', index=False,  encoding='utf-8-sig')
    print('good2'+data)








def build_w2v(st_path, save_path):
    program = os.path.basename(sys.argv[0])
    print(program)

    logger = logging.getLogger(program)

    logging.basicConfig(filename='./log/log.txt' ,
                        format='%(asctime)s: %(levelname)s: %(message)s', 
                        level= logging.INFO)

    logging.info('running %s ' % ' '.join(sys.argv))

    if len(sys.argv):
        globals()['__doc__'] = locals()
        print(globals()['__doc__'])
   
   
    model = Word2Vec(LineSentence(st_path), size=300, window=4, min_count=5, workers=cpu_count())
    
    # model 保存
    # model.save(save_path)

    model.wv.save_word2vec_format(save_path, binary=False)


def evaluate(testwords, model='./output/TikTok_word2vec.model'):
    TikTok_w2v_model = Word2Vec.load(model)
    for i in testwords:
        res = TikTok_w2v_model.most_similar(i)
        print(i, 'the probability of similar:', res)



if __name__ == "__main__":

    swl = stop_word('./output/stopwords.txt')

    # build_emoji_corpus('雷军评论')

    # 制作 setence corpus

    # dsl = ['dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 
    # 'dataset9', 'dataset10', 'dataset11', 'dataset12', 'dataset13', 'dataset14', 'dataset15',
    # 'dataset16', 'dataset17','dataset18']
    # dsl = ['减肥健康', '情感恋爱', '抗疫', '搞笑2', '搞笑3', '明星李易峰', '罗永浩', '董明珠', '雷军评论']
    dsl = ['result_v2']
    data = 'result_v2_process'
    p = Pool(cpu_count())
    data_process_txt_v2 = partial(data_process_txt, sw=swl)
    sls = p.map(data_process_txt_v2, dsl)
    p.close()
    p.join()

    for ls, data in sls:
        # print('data:', data, 'ls', ls)
        print('--'*20)
        add_sentence_corpus_tocsv(ls, data=data)
    
    # ================================================== #
    # for p, _, fs in [i for i in os.walk('./output')]:
    #     for f in fs[2:]:
    #         f_p = p+'/'+f
    #         print(f_p)
    #         #制作 word2vector
    #         build_w2v(f_p, p+'/'+'vec/'+f[:3]+'.vec')    

    # ================================================== #

    # build_w2v('./output/total.txt', './output/vec/total.vec')


    # test word2vector
    # testwords = ['哈哈', '主播', '可爱', '厉害', '心疼']
    # evaluate(testwords)