import pandas as pd
import numpy as np
import time
import datetime

now=datetime.datetime.now()
today=datetime.datetime.strftime( now, '%Y%m%d%H%M')
train_df=pd.read_csv('./data/train.csv',na_values='\\N')
test_df=pd.read_csv('./data/test.csv',na_values='\\N')
repay_df=pd.read_csv('./data/user_repay_logs.csv',na_values='\\N')
##过滤只要第一期的
#repay_df=repay_df[repay_df.order_id==1]
test_df.due_date=pd.to_datetime(test_df.due_date)
repay_df=test_df[['user_id']].merge(repay_df,on='user_id')
train_df.repay_date=train_df.repay_date.fillna('2200-01-01')
train_df['order_id']=1
#df=train_df.append(repay_df,ignore_index=True)
df=repay_df
df=df.drop_duplicates(subset=['listing_id','order_id'])
df=df.sort_values(by='due_date',ascending=True)
df=df.reindex()
df.due_date=pd.to_datetime(df.due_date)
df.repay_date=pd.to_datetime(df.repay_date)
df['is_overdue']=df.repay_date=='2200-01-01'
df['before_days']=df.due_date-df.repay_date
df['before_days']=df['before_days'].dt.days
df.loc[df.is_overdue==True,'before_days']=-8
before_mean=df.groupby('user_id').before_days.mean()
before_mean=before_mean.reset_index()
test_df=test_df.merge(before_mean,on='user_id',how='left')
test_df.to_csv('test2.csv',index=False)
#test_df=pd.read_csv('test2.csv')
test_df.due_date=pd.to_datetime(test_df.due_date)
test_df.auditing_date=pd.to_datetime(test_df.auditing_date)
def excute_ret(ser):
    #user_id	listing_id	auditing_date	due_date	due_amt	before_days	is_overdue
    #1.是否新用户
    if np.isnan(ser.before_days):
        return ser.due_date
    #是否高逾期
    elif ser.before_days<=-1:
        return np.nan
    #大量提前还款用户
    elif ser.due_date-pd.Timedelta(days=int(ser.before_days))<ser.auditing_date:
        return ser.due_date
    #正常用户
    else:
        return ser.due_date-pd.Timedelta(days=int(ser.before_days))
    

test_df['repay_date']=test_df.apply(excute_ret,axis=1)
ret_df=test_df[['listing_id','repay_date','due_amt']]
ret_df.columns=['listing_id','repay_date','repay_amt']
ret_df.loc[ret_df.repay_date.isnull(),'repay_amt']=None
ret_df.to_csv('ret'+str(today)[:10]+'.csv',index=False)
print(1)
