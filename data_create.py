import pandas as pd
import numpy as np
import time
import datetime
from sklearn.feature_extraction.text import CountVectorizer
#生成特征脚本

#加载数据
now=datetime.datetime.now()
today=datetime.datetime.strftime( now, '%Y%m%d%H%M')
train_df=pd.read_csv('./data/train.csv',na_values='\\N')
test_df=pd.read_csv('./data/test.csv',na_values='\\N')
#repay_df=pd.read_csv('./data/user_repay_logs.csv',na_values='\\N')
listing_df=pd.read_csv('./data/listing_info.csv',na_values='\\N')
user_df=pd.read_csv('./data/user_info.csv',na_values='\\N')
user_taglist_df=pd.read_csv('./data/user_taglist.csv',na_values='\\N')
#1.5G 接下来单独处理
#user_behavior_df=pd.read_csv('./data/user_behavior_logs.csv',na_values='\\N'

#repay_df.due_date=pd.to_datetime(repay_df.due_date)
#listing_df.auditing_date=pd.to_datetime(listing_df.auditing_date)
#listing_df.auditing_date=pd.to_datetime(listing_df.auditing_date)
#user_taglist_df.insertdate=pd.to_datetime(user_taglist_df.insertdate)

#train test
train_df['label']=train_df.repay_date.isnull()
train_df=train_df.drop(['repay_amt','repay_date'],axis=1)

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
df.due_date=pd.to_datetime(df.due_date)
df=df.sort_values(by='due_date',ascending=True)
df=df.reindex()

#listing
listing_df.auditing_date=pd.to_datetime(listing_df.auditing_date)
listing_df=listing_df.sort_values(by='auditing_date',ascending=True)

def describe(df,col_name):
    ret=pd.DataFrame(df.groupby('user_id')[col_name].nunique())
    ret.columns=['uid_'+col_name+'_nunique']
    ret['uid_'+col_name+'_sum']=df.groupby('user_id')[col_name].sum()
    ret['uid_'+col_name+'_mean']=df.groupby('user_id')[col_name].mean()
    ret['uid_'+col_name+'_max']=df.groupby('user_id')[col_name].max()
    ret['uid_'+col_name+'_min']=df.groupby('user_id')[col_name].min()
    ret['uid_'+col_name+'_range']=ret['uid_'+col_name+'_max']-ret['uid_'+col_name+'_min']
    #ret['uid_'+col_name+'_mode']=df.groupby('user_id')[col_name].agg(lambda x: np.mean(pd.Series.mode(x)))
    return ret

#user历史listing信息
group=listing_df.groupby('user_id')
#listing_id 个数
listing_ret=pd.DataFrame(group.user_id.count())
listing_ret.columns=['listing_id_cnt']
#auditing_date 最大值 最小值
listing_ret['auditing_date_max']=group.auditing_date.last()
listing_ret['auditing_date_min']=group.auditing_date.first()
#term 类别数 求和数 平均数 最大值 最小值 众数 range
listing_ret=listing_ret.join(describe(listing_df,'term'))
#rate 类别数 求和数 平均数 最大值 最小值 众数 range
listing_ret=listing_ret.join(describe(listing_df,'rate'))
#principal 类别数 求和数 平均数 最大值 最小值 众数 range
listing_ret=listing_ret.join(describe(listing_df,'principal'))
listing_ret=listing_ret.reset_index()
df=df.merge(listing_ret,on='user_id',how='left')

#当前listing信息
listing_df.rename(columns={'auditing_date': 'listing_date'}, inplace=True)
del listing_df['user_id']
df=df.merge(listing_df,on='listing_id',how='left')


#user_df 只保留最新的记录
user_df.rename(columns={'insertdate': 'user_date'}, inplace=True)
user_df=user_df.sort_values(by='user_date',ascending=False)
user_df=user_df.drop_duplicates('user_id')
df=df.merge(user_df,on='user_id',how='left')

##user_taglist_df
#user_taglist_df.rename(columns={'insertdate': 'user_tag_date'}, inplace=True)
#user_taglist_df=user_taglist_df.sort_values(by='user_tag_date',ascending=False)
#user_taglist_df=user_taglist_df.drop_duplicates('user_id')
#user_taglist_df['taglist'] = user_taglist_df['taglist'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())
##防止把字符长度为1的字符串当作停用词去除
#counter = CountVectorizer(min_df=0, token_pattern='\w+')
#tag_list_df=pd.DataFrame(counter.fit_transform(user_taglist_df['taglist']).toarray(),columns=counter.get_feature_names())
#tag_list_df['tag_cnt']=tag_list_df.sum(axis=1)
#user_taglist_df=pd.concat([user_taglist_df,tag_list_df],axis=1)
#df=df.merge(user_taglist_df,on='user_id',how='left')

#repay_df
repay_df=df[['user_id']].merge(repay_df,on='user_id',how ='inner')
ret_repay=repay_df.groupby('user_id').count()
repay_df.due_date=pd.to_datetime(repay_df.due_date)
repay_df.repay_date=pd.to_datetime(repay_df.repay_date)



df.to_csv('data.csv',index=False)
print(1)
