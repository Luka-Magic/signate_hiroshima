import pandas as pd

def convert_timeseries(df):
    all_id = df['id'].unique()
    all_date = df['date'].unique().astype(int)
    all_id.sort()
    all_date.sort()
    null_value = 'x'

    data = []
    for st in all_id:
        for date in all_date:
            data.append([date, st])
    
    old_df = df.copy()
    old_df = old_df[~old_df[['date', 'id']].duplicated()]
    old_df.fillna(null_value, inplace=True)

    new_df = pd.DataFrame(data, columns=['date', 'id'])
    new_df = pd.merge(new_df, old_df, on=['date', 'id'], how='left')
    new_df = new_df.sort_values(['id', 'date']).reset_index(drop=True)
    del old_df
    
    series = new_df.iloc[:, 2:].values.reshape(472, -1).T
    timeseries_df = pd.DataFrame(series, columns=all_id)

    dates = []
    hours = []
    for date in all_date:
        for hour in range(24):
            dates.append(date)
            hours.append(hour)
    timeseries_df['date'] = dates
    timeseries_df['hour'] = hours
    timeseries_df = timeseries_df.reindex(columns=['date', 'hour'] + list(all_id))
    return timeseries_df