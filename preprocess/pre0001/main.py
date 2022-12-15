from pathlib import Path
import pandas as pd
import sys
sys.path.append('process')

from water import water_process1
from rain import rain_process1
from tide import tide_process1
from timeseries import convert_timeseries
from river import river_process1

def main(data_dir: Path):
    '''

        前処理の大きな流れ

        1. データを読み込む

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

    '''

    # 1. データを読み込む
    rain = pd.read_csv(data_dir / 'rainfall' / 'data.csv')
    rain_st = pd.read_csv(data_dir / 'rainfall' / 'stations.csv')
    tide = pd.read_csv(data_dir / 'tidelevel' / 'data.csv')
    tide_st = pd.read_csv(data_dir / 'tidelevel' / 'stations.csv')
    water = pd.read_csv(data_dir / 'waterlevel' / 'data.csv')
    water_st = pd.read_csv(data_dir / 'waterlevel' / 'stations.csv')
    dam = pd.read_csv(data_dir / 'dam.csv')

    # 2. 3つのデータに対し前処理 & database化
    rain = rain_process1(rain, rain_st)
    tide = tide_process1(tide, tide_st)
    water = water_process1(water, water_st)

    # 3. 1つのstationに対し値を一列に時系列で並べる
    rain = convert_timeseries(rain)
    tide = convert_timeseries(tide)
    water = convert_timeseries(water)

    # 4. 河川名、水系名について処理
    (rain_st, tide_st, water_st, dam, river, system) = \
        river_process1(rain_st, tide_st, water_st, dam)

    return (rain, rain_st, tide, tide_st, water, water_st, dam, river, system)

