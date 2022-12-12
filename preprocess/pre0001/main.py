'''
    前処理の大きな流れ
    
    1.  3つのデータ(water, rain, tide)についてそれぞれ,
            - duplicate
            - 同一station
            - 途中でstataion名が変更されたもの
            - 明らかな外れ値をとるもの
                ...など
        に対し前処理を行う。
    
    2.  (station, city)または(station, river)の組み合わせを
        keyとしたdatabaseを作る。
    
    3.  

'''