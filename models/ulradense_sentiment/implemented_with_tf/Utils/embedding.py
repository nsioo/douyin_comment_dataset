# coding = utf-8
import numpy as np
import pandas as pd

class Embedding:
    def __init__(self, matrix, vocabulary, word2index, normalize):
        '''
        Args:
            matrix          A numpy array, words associated with rows
            vocabulary      List of strings
            word2index      Dictionary mapping word to its index in 
                            "vocabulary".
            normalized      Boolean
        '''

        self.m = matrix
        self.normalized = normalize
        if normalize:
            self.normalize()
        self.dim = self.m.shape[1]
        self.wi = word2index
        self.iw = vocabulary
    

    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        self.m = self.m / norm[:, np.newaxis]
        self.normalized = True
    
    
    def represent(self, w):
        # 根据word 取到对应向量 (word 来源于正向/负向词对)
        # 如果 seed word 在 原始word embedding table 中找得到
        # 返回对应向量, 找不到返回同等大小的0向量
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            return np.zeros(self.dim)



    @classmethod
    def from_fasttext_vec(cls,
                          path,
                          vocab_limit=None,
                          normalize=False):
        
        with open(path, 'r', encoding='utf-8') as f:
            
            vectors = [] # 2d-matrix, 一行就是一个word 的 embedding
            wi = {} # word 的 index
            iw = [] # 存放word

            first_line = f.readline().split() # vec 文件首行为 [vocab_size, dims]
            vocab_size = int(first_line[0])
            dim = int(first_line[1])

            if vocab_limit is None:
                vocab_limit = vocab_size
            
            for count in range(vocab_limit):
                line = f.readline().strip()

                parts = line.split() # 一行分割
                word = ' '.join(parts[:-dim]) # 取字
                vec = [float(x) for x in parts[-dim:]] # 取向量
                # 存储
                iw += [word]
                wi[word] = count
                vectors.append(vec)

        return cls(matrix=np.array(vectors),
                   vocabulary=iw,
                   word2index=wi,
                   normalize=normalize)            

def load_anew99(path='./source/anew99.csv'):
    anew = pd.read_csv(path, encoding='utf-8')
    anew.columns = ['Word', 'Valence', 'Arousal', 'Dominance']

    anew.set_index('Word', inplace=True)
    return anew

def load_cnseed(path='./source/cn_seed.csv'):
    cn_seed = pd.read_csv(path, encoding='utf-8')
    cn_seed.set_index('word', inplace=True)

    return cn_seed




if __name__ == "__main__":
    #emb = Embedding.from_fasttext_vec(path='./source/TikTok-300d-170h.vec')
    # print(emb.m)
    # print(emb.wi)

    # anew test
    #print(load_anew99())
    #print(load_cnseed(path='./source/cn_seed_v2_5.csv').shape)
    df = load_cnseed(path='./source/cn_seed_v2.csv')

    print(type(df))
    print(df.columns)

