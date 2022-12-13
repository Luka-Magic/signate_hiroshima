import pandas as pd

def convert_timeseries(df):
    all_station = df['station'].unique()
    all_date = df['date'].unique()
    null_value = 'x'

    data = []
    for st in all_station:
        for date in all_date:
            data.append([date, st])
    
    old_df = df.copy()
    old_df = old_df[~old_df[['date', 'station']].duplicated()]
    old_df.fillna(null_value, inplace=True)

    new_df = pd.DataFrame(data, columns=['date', 'station'])
    new_df = pd.merge(new_df, old_df, on=['date', 'station'], how='left')
    del old_df
    return new_df