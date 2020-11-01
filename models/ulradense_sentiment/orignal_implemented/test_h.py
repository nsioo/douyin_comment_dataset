

import scipy
import numpy as np
from scipy.stats import special_ortho_group
from random import randint 
import itertools
import sys



# print(Q[0, :].shape)

# batch_sz = 50
# X = np.random.rand(50, 300) * Q[0, :].reshape(300, 1)
# x =np.absolute(X)


# print(Q[0, :].reshape(300, 1).shape)
# print(x.shape)
# print(x[1][0,0])


 Q = np.matrix(scipy.stats.ortho_group.rvs(10, random_state=5))


def gradient(loss, vec_diff):

    # U --> Q[0, 1]                 shape [1, 10]
    # S --> vec_diff                shape [10, 1]
    # V --> np.transpose vec_diff   shape [1, 10]
    return Q[0, :] * vec_diff * np.transpose(vec_diff) / loss


def train (pos_vecs, neg_vecs):
    diff_ps = [i for i in itertools.product(pos_vecs, neg_vecs)]

    same_ps = [ i for i in itertools.combinations(pos_vecs, 2)] + \
              [i for i in itertools.combinations(neg_vecs, 2)]


    # 差异元组 25 = 5 x 5   正向5和5负向两两组合
    print(len(diff_ps))
    # 相似元组 20 = 5x2 + 5x2   正向5两两组合4+3+2+1+0，  负向5两两组合4+3+2+1+0
    print(len(same_ps))

    # 差异， 批次为25
    # ew 正向， ev 负向
    EW, EV = [], []
    for ew, ev in diff_ps:
        EW.append(np.asarray(ew))
        EV.append(np.asarray(ev))

    EW = np.asarray(EW)
    EV = np.asarray(EV)
    print('正向table', EW.shape)
    print('负向table', EV.shape)


    VEC_DIFF = EW - EV

    print('差异table:', VEC_DIFF[0].shape)

    DIFF_LOSS = np.absolute(VEC_DIFF * Q[0, :].reshape(10,1))
    print('差异损失:', DIFF_LOSS.shape) # S

    print('25个差异对中的其中一个差异对见的欧式距离', DIFF_LOSS[0][0, 0])    

    diff_grad = []
    for idx in range(len(EW)):
        # VEC_DIFF[idx] 对当前词的差异的向量表达 增加维度保持一致 [10] -> [10, 1]       
        diff_grad_step = gradient(DIFF_LOSS[idx][0, 0], VEC_DIFF[idx].reshape(10, 1))
        diff_grad.append(diff_grad_step)
    

    print('其中一个词差异梯度表达',diff_grad[0].shape)


    diff_grad = np.mean(diff_grad, axis=0)
    print('所有词的差异梯度均值', diff_grad.shape)







if __name__ == "__main__":
    pos_v = [[randint(1, 100) for i in range(10)] for i in range(5)]
    neg_v = [[randint(1, 100) for i in range(10)] for i in range(5)]


    train(pos_v, neg_v)


    Q = np.matrix(scipy.stats.ortho_group.rvs(10, random_state=5))

    print(Q[0,:])












# from sympy.matrices import Matrix,GramSchmidt
 
# A = np.array([[1,0,0],[0,2,1],[0,1,3]])
# # 将numpy 矩阵转为sympy的列向量集合
# MA = [Matrix(col) for col in A.T]
# # 施密特正交化
# gram = GramSchmidt(MA)

# print('orthogonal', gram)
