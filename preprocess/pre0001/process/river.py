import pandas as pd
import numpy as np

def river_process1(rain_st, tide_st,  water_st, dam):
    # 水系をまとめる
    # その他たちは沿岸部に集中しているのでまとめて「沿岸部」とする
    rain_st.loc[rain_st['水系名']=="その他", '水系名'] = "沿岸部"
    tide_st.loc[tide_st['水系名']=="中国その他", '水系名'] = "沿岸部"

    # カラム名の操作
    def column_rename(df):
        renames = {'水系名': 'system', '河川名': 'river'}
        return df.rename(columns=renames)
    
    water_st = column_rename(water_st)
    rain_st = column_rename(rain_st)
    tide_st = column_rename(tide_st)
    dam = column_rename(dam)

    riv_sys_db = water_st[['system', 'river']]
    riv_sys_db = pd.concat([riv_sys_db, rain_st[['system', 'river']]])
    riv_sys_db = pd.concat([riv_sys_db, tide_st[['system', 'river']]])
    riv_sys_db = pd.concat([riv_sys_db, dam[['system', 'river']]])

    riv2sys = {}
    for i, row in riv_sys_db.iterrows():
        if not isinstance(row['system'], str) and np.isnan(row['system']): # systemがnanのものを辞書に入れない
            continue
        riv2sys[row['river']] = row['system']

    water_st.loc[water_st['system'].isnull(), 'system'] = water_st.loc[water_st['system'].isnull(), 'river'].map(riv2sys)
    
    rivers = riv_sys_db['river'].sort_values().unique().tolist()
    systems = riv_sys_db['system'].sort_values().unique().tolist()

    river_db = pd.DataFrame([[i, v] for i, v in enumerate(rivers)], columns=['id', 'river'])
    system_db = pd.DataFrame([[i, v] for i, v in enumerate(systems)], columns=['id', 'system'])

    def riv_sys2id(df):
        riv2id = river_db.set_index('river').to_dict()['id']
        assert not set(df['river'].unique()) - set(riv2id.keys())
        df['river'] = df['river'].apply(lambda x:riv2id[x])

        sys2id = system_db.set_index('system').to_dict()['id']
        assert not set(df['system'].unique()) - set(sys2id.keys())
        df['system'] = df['system'].apply(lambda x:sys2id[x])
        return df
    
    water_st = riv_sys2id(water_st)
    rain_st = riv_sys2id(rain_st)
    tide_st = riv_sys2id(tide_st)
    dam = riv_sys2id(dam)

    return (rain_st, tide_st, water_st, dam, river_db, system_db)