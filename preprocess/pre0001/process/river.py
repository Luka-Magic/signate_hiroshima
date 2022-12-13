def river_process1(rain, rain_st, tide, tide_st, water, water_st, dam):
    # 水系をまとめる
    water_sys = set(water_st['水系名'].unique())
    tide_sys = set(tide_st['水系名'].unique())
    rain_sys = set(rain_st['水系名'].unique())
    dam_sys = set(dam['水系名'].unique())

    # 
