'''

    前処理の大きな流れ

    2. dataを読み込む

    1.  3つのデータ(water, rain, tide)についてそれぞれ,
            - duplicate
            - 同一station
            - 途中でstataion名が変更されたもの
            - 明らかな外れ値をとるもの
                ...など
        に対し前処理を行う。
    
    2.  (station, city)または(station, river)の組み合わせを
        idとしたdatabaseを作る。
    
    3.  測定値の入ったデータ(data.csv)を、columnsをstationに
        して測定値を時系列で一列に変換する。

    4.  河川名と水系名について前処理を施し、河川名、水系名それぞれ
        idとしたdatabaseを作る。全データの河川名と水系名をidで
        表す。

'''
from pathlib import Path


def main():
    pass