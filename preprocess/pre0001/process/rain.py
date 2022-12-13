import pandas as pd
import numpy as np
import gc
import math

def rain_process1(rain, rain_st):
    ###################
    # data.csv側の処理 #
    ###################

    ##### 全体でdrop_duplicates
    rain.drop_duplicates(inplace=True)
    
    ##### (date, statition, city)が重複している行 -> 
    #####     その行の中でfloatが最も含まれている行を採用
    # (date, station, city)でグループを作った時、2行以上ある場合は重複なのでそのindexだけ取り出す
    nunique_date_st_city = rain.groupby(['date', 'station', 'city']).nunique()
    nunique_date_st_city['max_count'] = nunique_date_st_city.max(axis=1)

    dup_date_st_city = nunique_date_st_city.query('max_count >= 2')[['max_count']]
    dup_date_st_city_idx =  dup_date_st_city.index
    # 重複した行それぞれに含まれる値でfloatである値をカウント
    dup_date_st_city_df = rain.set_index(['date', 'station', 'city']).loc[dup_date_st_city_idx]
    dup_date_st_city_df['num_count'] = dup_date_st_city_df.apply(lambda x: \
        24 - pd.to_numeric(x, errors='coerce').isnull().sum(),axis=1)
    # floatが最も多い1行だけをとり出してconcat
    concat_df = None
    for _, df in dup_date_st_city_df.groupby(['date', 'station', 'city']):
        df = df.sort_values('num_count', na_position='first', ascending=False).iloc[0:1, :]
        if concat_df is None:
            concat_df = df.copy()
        else:
            concat_df = pd.concat([concat_df, df], axis=0)
    concat_df.drop('num_count', inplace=True, axis=1)

    # 重複していない行たちとconcat
    unique_date_st_city = nunique_date_st_city.query('max_count == 1')
    unique_date_st_city =  unique_date_st_city.index
    unique_date_st_city_df = rain.set_index(['date', 'station', 'city']).loc[unique_date_st_city]
    rain = pd.concat([unique_date_st_city_df, concat_df])

    del nunique_date_st_city, dup_date_st_city, dup_date_st_city_df, \
        unique_date_st_city, unique_date_st_city_df, concat_df, df
    gc.collect()

    rain.reset_index(inplace=True)
    rain.sort_values(['date', 'station', 'city'], inplace=True)

    ##### (おそらく)同じstationである行の値をマージ
    # 観測日数が31日のstationは、そのstation名に(電)のついたものと同じstationと考えられるのでマージ
    for st in rain['station'].value_counts()[rain['station'].value_counts() == 31].index:
        st_ = st + '(電)'
        # 変更前後の２つのstationに含まれるデータに日付の重なりがなければマージ
        if len(rain.query('station in (@st, @st_)')) == rain.query('station in (@st, @st_)')['date'].nunique():
            rain.loc[rain['station'] == st, 'station'] = st_

    # station.csvには存在しないstationで、(国)をつけたものなら存在するものはおなじstationとしてマージ
    for st in set(rain['station'].unique()) - set(rain_st['観測所名称'].unique()):
        bool_ = rain_st['観測所名称'].str.contains(st)
        if (bool_).any():
            st_ = rain_st[bool_]['観測所名称'].iloc[-1]
            if f'{st}(国)' == st_:
                # 変更前後の２つのstationに含まれるデータに日付の重なりがなければマージ
                if len(rain.query('station in (@st, @st_)')) == rain.query('station in (@st, @st_)')['date'].nunique():
                    rain.loc[rain['station'] == st, 'station'] = st_
    
    ######################
    # station.csv側の処理 #
    ######################

    ##### station名に(砂防)が含まれているものは入力時使用も0であり、ないものとマージできる
    rain_st.loc[:, '観測所名称'] = rain_st['観測所名称'].str.replace(r'\(砂防\)', '')

    ###################
    # データベースを作成 #
    ###################
    
    # idに(station, city)を対応させたテーブルを作る
    keys = rain.groupby(['station', 'city']).count().index
    rain_db = pd.DataFrame(index=keys).reset_index()
    rain_db['id'] = range(len(rain_db))
    rain_db = rain_db.reindex(columns=['id', 'station', 'city'])

    # column名に変更を加える
    rain_st = rain_st.rename(columns={'観測所名称': 'station', '市町': 'city'})

    # station.csvのcityがnanのもののうち、data.csvから埋められるものは埋める
    for data in rain_st.iterrows(): # stationを一列ずつ取り出す
        city = data[1]['city'] # cityを取り出す
        if isinstance(city, float) and math.isnan(city): # そのcityがnanの時のみ
            st = data[1]['station']
            city = rain.query('station==@st')['city'].unique()[0] # data.csvからそのstationを検索してなんのcityかをみる
            rain_st.loc[(rain_st['station'] == st), 'city'] = city
    
    # data.csvの(station, city)をidに置き換える
    rain = rain_db.merge(rain, on=['station', 'city'], how='left')
    rain.drop(['station', 'city'], axis=1, inplace=True)

    rain_st = rain_db.merge(rain_st, on=['station', 'city'], how='left')
    rain_st['入力時使用'] = rain_st['入力時使用'].fillna(0.0)

    return rain, rain_st