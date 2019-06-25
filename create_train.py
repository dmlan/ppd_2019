import pandas as pd
import numpy as np
from itertools import combinations
import os
import geohash


def excute_feature(data_frame):
    #数值型特征
    data_ret = pd.DataFrame(data_frame.groupby('UID')['day'].min())
    data_ret.columns=['uid_early_day']
    data_ret['uid_max_day'] = data_frame.groupby('UID')['day'].max()
    data_ret['uid_range_day'] = data_ret['uid_max_day'] - data_ret['uid_early_day']

    data_ret['uid_min_long'] = data_frame.groupby('UID')['longitude'].min()
    data_ret['uid_max_long'] = data_frame.groupby('UID')['longitude'].max()
    data_ret['uid_range_long'] = data_ret['uid_max_long'] - data_ret['uid_min_long']

    data_ret['uid_min_lat'] = data_frame.groupby('UID')['latitude'].min()
    data_ret['uid_max_lat'] = data_frame.groupby('UID')['latitude'].max()
    data_ret['uid_range_lat'] = data_ret['uid_max_lat'] - data_ret['uid_min_lat']

    data_ret['uid_min_amt'] = data_frame.groupby('UID')['trans_amt'].min()
    data_ret['uid_max_amt'] = data_frame.groupby('UID')['trans_amt'].max()
    data_ret['uid_range_amt'] = data_ret['uid_max_amt'] - data_ret['uid_min_amt']

    data_ret['uid_min_bal'] = data_frame.groupby('UID')['bal'].min()
    data_ret['uid_max_bal'] = data_frame.groupby('UID')['bal'].max()
    data_ret['uid_range_bal'] = data_ret['uid_max_bal'] - data_ret['uid_min_bal']

    for wd in range(7):
        data_ret['uid_week_'+str(wd)+'_cnt'] = data_frame[data_frame['nweek_day'] == wd].groupby('UID')['day'].count()
    data_ret['uid_all_cnt'] = data_frame.groupby('UID')['day'].count()
    #小时数
    time_period = [0, 7, 12, 16, 25]
    for tp in range(4):
        data_ret['uid_time_period_'+str(tp)+'_cnt'] = data_frame[((data_frame['nhour'] >= time_period[tp]) &
                        (data_frame['nhour'] < time_period[tp+1]))].groupby('UID')['day'].count()

    #名义型特征
    nunique_var = ['acc_id1', 'acc_id2', 'acc_id3', 'amt_src1', 'amt_src2', 'bal', 'channel', 'code1', 'code2', 'day',
                   'device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub',
                   'ip2', 'ip2_sub', 'mac1', 'mac2', 'market_code', 'market_type', 'merchant', 'mode', 'nweek_day',
                   'os', 'success', 'trans_amt', 'trans_type1', 'trans_type2', 'version', 'wifi', 'nhour', 'longitude'
                   , 'latitude']

    for nv in nunique_var:
        data_ret['uid_'+nv+'_nunique'] = data_frame.groupby('UID')[nv].nunique()
        data_ret['uid_'+nv+'_notnull'] = data_frame[data_frame[nv].notnull()].groupby('UID')['UID'].count()

    data_ret.drop(['uid_nhour_notnull', 'uid_day_notnull', 'uid_nweek_day_notnull'], axis=1, inplace=True)
    data_ret['uid_tran_cnt'] = data_frame[data_frame['mode'] == 'transaction'].groupby('UID')['UID'].count()

    
    #名义变量
    var = ['acc_id1', 'acc_id2', 'acc_id3', 'amt_src1', 'amt_src2', 'version', 'code1', 'code2',
                  'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub', 'ip2',
                  'ip2_sub', 'mac1', 'mac2', 'market_code', 'merchant', 'wifi', 'mode']
    #名义变量组合
    pair = ['code1', 'code2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 
                   'ip1_sub', 'ip2', 'ip2_sub', 'mac1', 'mac2', 'merchant', 'wifi', 'mode']
    #单个特征
    for rv in var:
        sample_data = data_frame[['UID', rv]].drop_duplicates()
        group_data = data_group(data_frame, rv)
        sample_data = sample_data.merge(group_data, on=rv, how='left')
        data_ret['cnt_uid_' + rv + '_max'] = sample_data.groupby('UID')['cnt_uid_' + rv].max()
        data_ret['cnt_uid_' + rv + '_min'] = sample_data.groupby('UID')['cnt_uid_' + rv].min()
        data_ret['cnt_uid_' + rv + '_mean'] = sample_data.groupby('UID')['cnt_uid_' + rv].mean()
            
        data_ret['early_day_' + rv + '_max'] = sample_data.groupby('UID')['early_day_' + rv].max()
        data_ret['early_day_' + rv + '_min'] = sample_data.groupby('UID')['early_day_' + rv].min()
        data_ret['early_day_' + rv + '_mean'] = sample_data.groupby('UID')['early_day_' + rv].mean()
        
        data_ret['later_day_' + rv + '_max'] = sample_data.groupby('UID')['later_day_' + rv].max()
        data_ret['later_day_' + rv + '_min'] = sample_data.groupby('UID')['later_day_' + rv].min()
        data_ret['later_day_' + rv + '_mean'] = sample_data.groupby('UID')['later_day_' + rv].mean()
        
        data_ret['range_day_' + rv + '_max'] = sample_data.groupby('UID')['range_day_' + rv].max()
        data_ret['range_day_' + rv + '_min'] = sample_data.groupby('UID')['range_day_' + rv].min()
        data_ret['range_day_' + rv + '_mean'] = sample_data.groupby('UID')['range_day_' + rv].mean()
        
        if rv != 'geo_code':
            data_ret['close_long_' + rv + '_max'] = sample_data.groupby('UID')['close_long_' + rv].max()
            data_ret['close_long_' + rv + '_min'] = sample_data.groupby('UID')['close_long_' + rv].min()
            data_ret['close_long_' + rv + '_mean'] = sample_data.groupby('UID')['close_long_' + rv].mean()
            
            data_ret['far_long_' + rv + '_max'] = sample_data.groupby('UID')['far_long_' + rv].max()
            data_ret['far_long_' + rv + '_min'] = sample_data.groupby('UID')['far_long_' + rv].min()
            data_ret['far_long_' + rv + '_mean'] = sample_data.groupby('UID')['far_long_' + rv].mean()
            
            data_ret['range_long_' + rv + '_max'] = sample_data.groupby('UID')['range_long_' + rv].max()
            data_ret['range_long_' + rv + '_min'] = sample_data.groupby('UID')['range_long_' + rv].min()
            data_ret['range_long_' + rv + '_mean'] = sample_data.groupby('UID')['range_long_' + rv].mean()
            
            data_ret['close_lat_' + rv + '_max'] = sample_data.groupby('UID')['close_lat_' + rv].max()
            data_ret['close_lat_' + rv + '_min'] = sample_data.groupby('UID')['close_lat_' + rv].min()
            data_ret['close_lat_' + rv + '_mean'] = sample_data.groupby('UID')['close_lat_' + rv].mean()
            
            data_ret['far_lat_' + rv + '_max'] = sample_data.groupby('UID')['far_lat_' + rv].max()
            data_ret['far_lat_' + rv + '_min'] = sample_data.groupby('UID')['far_lat_' + rv].min()
            data_ret['far_lat_' + rv + '_mean'] = sample_data.groupby('UID')['far_lat_' + rv].mean()
            
            data_ret['range_lat_' + rv + '_max'] = sample_data.groupby('UID')['range_lat_' + rv].max()
            data_ret['range_lat_' + rv + '_min'] = sample_data.groupby('UID')['range_lat_' + rv].min()
            data_ret['range_lat_' + rv + '_mean'] = sample_data.groupby('UID')['range_lat_' + rv].mean()
            
        data_ret['least_ratio_' + rv + '_max'] = sample_data.groupby('UID')['least_ratio_' + rv].max()
        data_ret['least_ratio_' + rv + '_min'] = sample_data.groupby('UID')['least_ratio_' + rv].min()
        data_ret['least_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['least_ratio_' + rv].mean()
        
        data_ret['most_ratio_' + rv + '_max'] = sample_data.groupby('UID')['most_ratio_' + rv].max()
        data_ret['most_ratio_' + rv + '_min'] = sample_data.groupby('UID')['most_ratio_' + rv].min()
        data_ret['most_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['most_ratio_' + rv].mean()
        
        data_ret['range_ratio_' + rv + '_max'] = sample_data.groupby('UID')['range_ratio_' + rv].max()
        data_ret['range_ratio_' + rv + '_min'] = sample_data.groupby('UID')['range_ratio_' + rv].min()
        data_ret['range_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['range_ratio_' + rv].mean()
    #俩俩组合遍历    
    for rv in combinations(pair, 2):

        rv2 = '_'.join(rv)
        sample_data = data_frame[['UID'] + list(rv)].drop_duplicates()
        group_data = data_group_pair(data_frame, rv)
        sample_data = sample_data.merge(group_data, on=rv, how='left')

        data_ret['cnt_uid_' + rv2 + '_max'] = sample_data.groupby('UID')['cnt_uid_' + rv2].max()
        data_ret['cnt_uid_' + rv2 + '_min'] = sample_data.groupby('UID')['cnt_uid_' + rv2].min()
        data_ret['cnt_uid_' + rv2 + '_mean'] = sample_data.groupby('UID')['cnt_uid_' + rv2].mean()
        
        data_ret['early_day_' + rv2 + '_max'] = sample_data.groupby('UID')['early_day_' + rv2].max()
        data_ret['early_day_' + rv2 + '_min'] = sample_data.groupby('UID')['early_day_' + rv2].min()
        data_ret['early_day_' + rv2 + '_mean'] = sample_data.groupby('UID')['early_day_' + rv2].mean()
        
        data_ret['later_day_' + rv2 + '_max'] = sample_data.groupby('UID')['later_day_' + rv2].max()
        data_ret['later_day_' + rv2 + '_min'] = sample_data.groupby('UID')['later_day_' + rv2].min()
        data_ret['later_day_' + rv2 + '_mean'] = sample_data.groupby('UID')['later_day_' + rv2].mean()
        
        data_ret['range_day_' + rv2 + '_max'] = sample_data.groupby('UID')['range_day_' + rv2].max()
        data_ret['range_day_' + rv2 + '_min'] = sample_data.groupby('UID')['range_day_' + rv2].min()
        data_ret['range_day_' + rv2 + '_mean'] = sample_data.groupby('UID')['range_day_' + rv2].mean()
    #TODO 距离变化特征    
    return data_ret


def data_group_pair(data, col):
    col2 = '_'.join(col)
    groupby_col=data.groupby(col)
    df = pd.DataFrame(groupby_col['UID'].nunique())
    df.columns = ['cnt_uid_' + col2]
    df['early_day_' + col2] = groupby_col['day2'].min()
    df['later_day_' + col2] = groupby_col['day2'].max()
    df['range_day_' + col2] = df['later_day_' + col2] - df['early_day_' + col2]
    df = df.reset_index()
    return df



def data_group(data, col):
    # 有多少UID写了这个属性
    df = pd.DataFrame(data.groupby(col)['UID'].nunique())
    df.columns = ['cnt_uid_' + col]
    df['early_day_' + col] = data.groupby(col)['day2'].min()
    df['later_day_' + col] = data.groupby(col)['day2'].max()
    df['range_day_' + col] = df['later_day_' + col] - df['early_day_' + col]
    if col != 'geo_code':
        # longitude
        df['close_long_' + col] = data.groupby(col)['longitude'].min()
        df['far_long_' + col] = data.groupby(col)['longitude'].max()
        df['range_long_' + col] = df['far_long_' + col] - df['close_long_' + col]
        # latitude
        df['close_lat_' + col] = data.groupby(col)['latitude'].min()
        df['far_lat_' + col] = data.groupby(col)['latitude'].max()
        df['range_lat_' + col] = df['far_lat_' + col] - df['close_lat_' + col]

    df['least_ratio_' + col] = data.groupby(col)['ratio'].min()
    df['most_ratio_' + col] = data.groupby(col)['ratio'].max()
    df['range_ratio_' + col] = df['most_ratio_' + col] - df['least_ratio_' + col]
    df = df.reset_index()
    return df


if __name__ == '__main__':
    operation_train = pd.read_csv('operation_train_new.csv')
    transaction_train = pd.read_csv('transaction_train_new.csv')
    tag_train = pd.read_csv('tag_train_new.csv')
    operation_round1 = pd.read_csv('operation_round1_new.csv')
    transaction_round1 = pd.read_csv('transaction_round1_new.csv')
    tag_test = pd.read_csv('tag_test_new.csv')[['UID']]
    transaction_train['mode'] = 'transaction'
    action_train = operation_train.append(transaction_train).reset_index(drop=True)
    action_train = action_train.sort_values(by=['UID', 'day', 'time'], ascending=[True, True, True])
    action_train = action_train.merge(tag_train, on='UID')
    action_train['day2'] = action_train['day']
    transaction_round1['mode'] = 'transaction'
    action_round1 = operation_round1.append(transaction_round1).reset_index(drop=True)
    action_round1 = action_round1.sort_values(by=['UID', 'day', 'time'], ascending=[True, True, True])
    action_round1['day2'] = action_round1['day'] + 30
    all_data = action_train.append(action_round1).reset_index(drop=True)
    #TODO测试集星期和训练集可能不同
    all_data['nweek_day'] = all_data['day'].apply(lambda x: x % 7)
    all_data['version'] = all_data.version.fillna('0.0.0')
    all_data['nhour'] = all_data['time'].apply(lambda x: int(x[:2]))
    all_data['longitude_latitude']=all_data['geo_code'].apply(lambda x: geohash.decode(x) if isinstance(x, str) else np.nan)
    all_data['longitude'] = all_data['longitude_latitude'].apply(lambda x:x[0] if isinstance(x, tuple) else np.nan)
    all_data['latitude'] = all_data['longitude_latitude'].apply(lambda x:x[1] if isinstance(x, tuple) else np.nan)
    all_data['ratio'] = all_data['trans_amt'] / all_data['bal']
    del all_data['longitude_latitude']
    data_var = excute_feature(all_data)
    data_var = data_var.reset_index()
    train = tag_train.merge(data_var, on='UID')
    valid = tag_test.merge(data_var, on='UID')  
    #删除nan过多的列
    train_na_num=train.apply(lambda x:x.isna().sum())
    valid_na_num=valid.apply(lambda x:x.isna().sum())
    train_na_num=train_na_num[train_na_num<30000]
    valid_na_num=valid_na_num[valid_na_num<30000]
    #drop_train = train.T.drop_duplicates().T
    #drop_valid = valid.T.drop_duplicates().T
    features = [i for i in train_na_num.index.values if i in valid_na_num.index.values]
    print(len(features))
    train[features + ['Tag']].to_csv('create_train_data2.csv', index=False)
    valid[features].to_csv('create_test_data2.csv', index=False)
    train[features].to_csv('create_train_data_no_tag2.csv', index=False)




