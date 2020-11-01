# coding=utf-8
import scipy.stats as st

def rmse(a,b):
	return np.sqrt(np.mean(np.power(a-b,2)))

def mae(a,b):
	return np.mean(np.absolute(a-b))

# use to final evaluation
def evall(true, prediction, metric='r'):
	'''
	Expects pandas data frames.
	'''
	metrics={'r':lambda x,y:st.pearsonr(x,y)[0],
			'rmse':rmse,
			'mae':mae
		}
	metric=metrics[metric]
	row=[]
	for var in list(prediction):
		# p-value pearsonr value
		# 1）输入：x为特征，y为目标变量.
		# 2）输出：r： 相关系数 [-1，1]之间，p-value: p值。
		#p值越小，表示相关系数越显著，一般p值在500个样本以上时有较高的可靠性。
		value=metric(prediction[var], true[var])
		row+=[value]
	return row




## function that rescale data
def scaleInRange(x, oldmin, oldmax, newmin, newmax):
	# linear scaling (koeper 2016) (softmax makes no sense)
	return ((newmax - newmin)*(x - oldmin)) / (oldmax - oldmin) + newmin

def scale_prediction_to_seed(preds, seed_lexicon):
    seed_mins = seed_lexicon.min(axis=0)
    seed_maxes = seed_lexicon.max(axis=0)
    pred_mins = preds.min(axis=0)
    pred_maxes = preds.max(axis=0)
    
    # V A D rescaling size
    for var in list(preds):
        preds[var] = scaleInRange(preds[var],
                                  oldmin=pred_mins[var],
                                  oldmax=pred_maxes[var],
                                  newmin=seed_mins[var],
                                  newmax=seed_maxes[var])

    return preds


def average_results_df(results_df):
    avg=results_df.mean(axis=0)
    sd=results_df.std(axis=0)
    results_df.loc['Average']=avg
    results_df.loc['SD']=sd
    results_df['Average']=results_df.mean(axis=1)
    return results_df

