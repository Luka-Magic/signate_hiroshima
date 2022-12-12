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
        idとしたdatabaseを作る。
    
    3.  測定値の入ったデータ(data.csv)を、columnsをstationに
        して測定値を時系列で一列に変換する。

    4.  河川名と水系名について前処理を施し、河川名、水系名それぞれ
        idとしたdatabaseを作る。全データの河川名と水系名をidで
        表す。

'''

from pathlib import Path
import pandas as pd

from .process.water_process1 import water_preprocess1
from .process.rain_process1 import rain_preprocess1
from .process.tide_process1 import tide_preprocess1
from .process.to_timeseries import to_timeseries
from .process.river_process import river_preprocess1

def main(data_dir: Path):
    # 1. データを読み込む
    rain = pd.read_csv(data_dir / 'rainfall' / 'data.csv'),
    rain_st = pd.read_csv(data_dir / 'rainfall' / 'stations.csv'),
    tide = pd.read_csv(data_dir / 'tidelevel' / 'data.csv'),
    tide_st = pd.read_csv(data_dir / 'tidelevel' / 'stations.csv'),
    water = pd.read_csv(data_dir / 'waterlevel' / 'data.csv'),
    water_st = pd.read_csv(data_dir / 'waterlevel' / 'stations.csv'),
    dam = pd.read_csv(data_dir / 'dam.csv')

    # 2. 3つのデータに対し前処理 & database化
    rain = rain_preprocess1(rain, rain_st)
    tide = tide_preprocess1(tide, tide_st)
    water = water_preprocess1(water, water_st)

    # 3. 1つのstationに対し値を一列に時系列で並べる
    rain = to_timeseries(rain, 'rain')
    tide = to_timeseries(tide, 'tide')
    water = to_timeseries(water, 'water')

    # 4. 河川名、水系名について処理
    (rain, rain_st, tide, tide_st, water, water_st, dam, river, system) = \
        river_preprocess1(rain, rain_st, tide, tide_st, water, water_st, dam)
    
    return (rain, rain_st, tide, tide_st, water, water_st, dam, river, system)

