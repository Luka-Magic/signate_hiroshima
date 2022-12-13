import pandas as pd

def water_process1(water, water_st):

    water['station'] = water['station'].str.replace(r'\(電\)', '')

    for st in set(water['station'].unique()) - set(water_st['観測所名称'].unique()):
        bool_ = water_st['観測所名称'].str.contains(st)
        if (bool_).any():
            st_ = water_st[bool_]['観測所名称'].iloc[-1]
            if f'{st}(国)' == st_:
                if len(water.query('station in (@st, @st_)')) == water.query('station in (@st, @st_)')['date'].nunique():
                    water.loc[water['station'] == st, 'station'] = st_
    
    water.loc[water['station'] == '山手', 'station'] = '山手(国)'

    keys = water.groupby(['station', 'river']).count().index
    water_db = pd.DataFrame(index=keys).reset_index()
    water_db['id'] = range(len(water_db))
    water_db = water_db.reindex(columns=['id', 'station', 'river'])

    water = water_db.merge(water, on=['station', 'river'], how='left')
    water.drop(['station', 'river'], axis=1, inplace=True)

    water_st = water_st.rename(columns={'観測所名称': 'station', '河川名': 'river'})
    water_st = water_db.merge(water_st, on=['station', 'river'], how='left')

    return water, water_st
