from pathlib import Path
import pandas as pd
import sys
sys.path.append('process')

from water import water_process1
from rain import rain_process1
from tide import tide_process1
from timeseries import convert_timeseries
from river import river_process1
import warnings
warnings.simplefilter('ignore')

def main(data_dir):
    '''

        data_dir: str or Path
            signateからダウンロードできるデータ(train.zip)を解凍したディレクトリ
            "train"がdata_dirの最後に来るようにする
        
        --------------------------------

        前処理の大きな流れ

        1. data_dirから計7つのデータを読み込む。

        2.  3つのデータ(water, rain, tide)についてそれぞれ,
                - duplicate
                - 同一station
                - 途中でstataion名が変更されたもの
                - 明らかな外れ値をとるもの
                    ...など
            に対し前処理を行う。
            (station, city)または(station, river)の組み合わせを
            主キー(id)としたdatabaseを作る。
        
        3.  測定値の入ったデータ(data.csv)を、columnsをstationに
            して測定値を時系列で一列に変換する。

        4.  河川名と水系名について前処理を施し、河川名、水系名それぞれ
            主キー(id)としたdatabaseを作る。全データの河川名と水系名
            をidで表す。
        
        5. save_dirに9つのcsvファイルを保存する。

        --------------------------------


    '''
    
    data_dir = Path(data_dir)
    # save_dir = Path(save_dir)
    assert data_dir.name == 'train', f'data_dirのディレクトリ名の最後が"train"である必要があります。input: {str(data_dir)}'

    # 1. データを読み込む
    print('1. データ読み込み')
    rain = pd.read_csv(data_dir / 'rainfall' / 'data.csv')
    rain_st = pd.read_csv(data_dir / 'rainfall' / 'stations.csv')
    tide = pd.read_csv(data_dir / 'tidelevel' / 'data.csv')
    tide_st = pd.read_csv(data_dir / 'tidelevel' / 'stations.csv')
    water = pd.read_csv(data_dir / 'waterlevel' / 'data.csv')
    water_st = pd.read_csv(data_dir / 'waterlevel' / 'stations.csv')
    dam = pd.read_csv(data_dir / 'dam.csv')
    print(' => complete')

    # 2. 3つのデータに対し前処理 & database化
    print('2. 3つのデータに前処理 & database化')
    rain, rain_st = rain_process1(rain, rain_st)
    tide, tide_st = tide_process1(tide, tide_st)
    water, water_st = water_process1(water, water_st)
    print(' => complete')

    # 3. 1つのstationに対し値を一列に時系列で並べる
    print('3. 1つのstationに対し値を一列に時系列で並べる')
    rain = convert_timeseries(rain)
    tide = convert_timeseries(tide)
    water = convert_timeseries(water)
    print(' => complete')

    # 4. 河川名、水系名について処理
    print('4. 河川名、水系名について処理')
    (rain_st, tide_st, water_st, dam, river_table, river_system_table) = \
        river_process1(rain_st, tide_st, water_st, dam)
    print(' => complete')

    # 5. 保存
    # rain.to_csv(save_dir / 'rainfall' / 'data.csv', index=False)
    # rain_st.to_csv(save_dir / 'rainfall' / 'stations.csv', index=False)
    # tide.to_csv(save_dir / 'tidelevel' / 'data.csv', index=False)
    # tide_st.to_csv(save_dir / 'tidelevel' / 'stations.csv', index=False)
    # water.to_csv(save_dir / 'waterlevel' / 'data.csv', index=False)
    # water_st.to_csv(save_dir / 'waterlevel' / 'stations.csv', index=False)
    # dam.to_csv(save_dir / 'dam.csv', index=False)
    # river_table.to_csv(save_dir / 'river_table.csv', index=False)
    # river_system_table.to_csv(save_dir / 'river_system_table.csv', index=False)

    return (rain, rain_st, tide, tide_st, water, water_st, dam, river_table, river_system_table)