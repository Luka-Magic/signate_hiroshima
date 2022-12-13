import pandas as pd

def tide_process1(tide, tide_st):
    tide.loc[tide['station'] == '柿浦漁港', 'station'] = '柿浦港'
    tide.loc[tide['station'] == '呉阿賀港', 'station'] = '呉(阿賀)港'
    tide.loc[tide['station'] == '倉橋漁港', 'station'] = '倉橋港'

    keys = tide.groupby(['station', 'city']).count().index
    tide_db = pd.DataFrame(index=keys).reset_index()
    tide_db['id'] = range(len(tide_db))
    tide_db = tide_db.reindex(columns=['id', 'station', 'city'])

    tide = tide_db.merge(tide, on=['station', 'city'], how='left')
    tide.drop(['station', 'city'], axis=1, inplace=True)

    tide_st = tide_st.rename(columns={'観測所名': 'station'})
    tide_st = tide_db.merge(tide_st, on=['station'], how='left')
    
    return tide, tide_st