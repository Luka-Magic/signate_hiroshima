import pandas as pd

def water_process1(water, water_st):
    # data.csvで異常値を取るものは、異常値をとらなくなってからの値を使う。
    # 今回は「異常値」 = 「前後で100近くの変化があった箇所」としている。
    # (以下三つ以外ではめちゃくちゃ大きくても10前後)
    water = water[~((water['date'] <= 1304) & (water['station'] == '大谷池'))]
    water = water[~((water['date'] <= 699) & (water['station'] == "白川"))]
    water = water[~((water['date'] <= 94) & (water['station'] == "山手左岸(国)"))]

    # data.csvのstation名についている(電)は全て消して良い
    water['station'] = water['station'].str.replace(r'\(電\)', '')

    # data.csvだけに存在し、かつstation.csv側では(国)がついているstation名は全て(国)をつける
    for st in set(water['station'].unique()) - set(water_st['観測所名称'].unique()):
        bool_ = water_st['観測所名称'].str.contains(st)
        if (bool_).any():
            st_ = water_st[bool_]['観測所名称'].iloc[-1]
            if f'{st}(国)' == st_:
                if len(water.query('station in (@st, @st_)')) == water.query('station in (@st, @st_)')['date'].nunique():
                    water.loc[water['station'] == st, 'station'] = st_
    
    # 山手は(国)をつける
    water.loc[water['station'] == '山手', 'station'] = '山手(国)'

    # 河川名の改行を全て消す (太田川\n放水路だけは\n込みで入力時使用が1となっているので後で\nをつけ直す)
    water_st.loc[:, '河川名'] = water_st['河川名'].str.replace('\\n', '')
    
    # database化
    keys = water.groupby(['station', 'river']).count().index
    water_db = pd.DataFrame(index=keys).reset_index()
    water_db['id'] = range(len(water_db))
    water_db = water_db.reindex(columns=['id', 'station', 'river'])

    water = water_db.merge(water, on=['station', 'river'], how='left')
    water.drop(['station', 'river'], axis=1, inplace=True)
    
    water_st = water_st.rename(columns={'観測所名称': 'station', '河川名': 'river'})
    water_st = water_db.merge(water_st, on=['station', 'river'], how='left')

    # 改行があるとうまく処理されなかったので最後に改行を付け直す
    water_st.loc[water_st['river']=='太田川放水路', 'river'] = '太田川\\n放水路'
    # 一連の処理で入力時使用・評価対象がnanになるものがいくつか出てくるので0で埋める。
    water_st['入力時使用'] = water_st['入力時使用'].fillna(0.0)
    water_st['評価対象'] = water_st['評価対象'].fillna(0.0)

    return water, water_st
