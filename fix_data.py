# coding = utf-8

import pandas as pd




def fix(name, sign):
    df = pd.read_csv(name+'.csv', encoding='utf-8')

    # print(df.head(50))
    print(len(df))

    # drop NaN line
    df.dropna(axis=0, how='any', inplace=True)

    # select mean columns
    df = df[['\'序号', '\'评论人', '\'UID', '\'抖音号', '\'获赞数↓', '\'性别↓', '\'子评数↓','\'年龄↓', '\'评论时间↓', '\'评论详情']]
    # fix header
    df = df.rename(columns={'\'序号':'id', '\'评论人':'user', '\'UID':'UID', '\'抖音号':'account','\'获赞数↓':'num_like', '\'性别↓':'gender', '\'子评数↓':'sub_comment_num','\'年龄↓':'age', '\'评论时间↓':'comment_time', '\'评论详情':'comment'})
    # drop the '\''
    if sign in name:    
        df = df.applymap(lambda x: x[1:] if not type(x) is float else x )
    else:
        df = df.applymap(lambda x: x[1:])
    # reset index
    df = df.reset_index()

    print(df)
    df.to_excel(name+'_update.xlsx', sheet_name='Sheet1', index=False)



if __name__ == "__main__":
    # for name in ['dataset'+str(i) for i in range(2, 7)]:
        # fix(name) 

    n = '18'
    fix('dataset'+n, n)
