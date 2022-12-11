from pathlib import Path
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from hydra.experimental import compose, initialize_config_dir


def preprocess(cfg, df):
    # string -> floatに (欠損値を全てnanとする)
    df = df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)

    # 数値部分(date, hour以外)をnan埋め
    df_meta = df[['date', 'hour']]
    df_data = df.drop(columns=['date', 'hour'])

    # 標準化
    df_zscore_data = (df_data - df_data.mean(skipna=True)) / df_data.std(skipna=True)
    df = pd.concat([df_meta, df_zscore_data], axis=1)
    
    # 標準化に使った値は後で戻す時のために変数に入れておく
    st2mean = df_data.mean(skipna=True).to_dict()
    st2std = df_data.std(skipna=True).to_dict()
    st2info = {st: {'mean': st2mean[st], 'std': st2std[st]} for st in st2mean.keys()}

    return df, st2info


class HiroshimaDataset(Dataset):
    def __init__(self, cfg, df, st2info):
        super().__init__()
        self.st2info = st2info
        inputs_ = df.iloc[-1*cfg.input_sequence_size:].values.T[:, :, np.newaxis]
        # 入力の長さがRNNの入力に足りないとき => 前にpadding
        if cfg.input_sequence_size > input_.shape[1]:
            pad_length = cfg.input_sequence_size - input_.shape[1]
            pad = np.tile(np.array(input_[:, 0, :][:, np.newaxis, :]), (1, pad_length, 1))
            input_ = np.concatenate([pad, input_], axis=1)
            input_ = input_.fillna(method='ffill') # まず新しいデータで前のnanを埋める
            input_ = input_.fillna(method='bfill') # 新しいデータがnanだった場合は古いデータで埋める
            input_ = input_.fillna(0.) # 全てがnanなら０埋め
        self.inputs = input_.tolist()
        self.stations = df.drop(columns=['date', 'hour']).columns

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ = self.inputs[idx]

        meta = self.st2info[self.stations[idx]]
        meta['station'] = self.stations[idx]
        input_ = torch.tensor(input_) # input_: (len_of_series, input_size)
        return input_, meta


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    def forward(self, x, h0=None):
        '''
            x: (bs, len_of_series ,input_size)
            h0 = Tuple(h, c)
        '''
        _, h = self.lstm(x, h0) # -> x: (bs, len_of_series, hidden_size)
        return h


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h):
        '''
            x: (bs, len_of_series ,hidden_size)
            h = Tuple(h, c)
        '''
        x, h = self.lstm(x, h) # -> x: (bs, len_of_series, hidden_size)
        x = self.fc(x) # -> x: (bs, len_of_series, output_size)
        return x


def load_models(cfg, model_dir, device):
    models = []
    for model_path in model_dir.glob('*.pth'):
        encoder = Encoder(cfg.input_size, cfg.hidden_size).to(device)
        decoder = Decoder(cfg.hidden_size, cfg.output_size).to(device)
        path_dict = torch.load(model_path)
        encoder.load_state_dict(path_dict['encoder'])
        decoder.load_state_dict(path_dict['decoder'])
        model = {
            'encoder': encoder,
            'decoder': decoder
        }
        models.append(model)
    return models


def input2seriesdf(input):
    date = input['date']
    stations = input['stations']
    waterlevel = input['waterlevel']

    # 土台としてsize=(24, len(stations))のnanで埋まったdataframeを作成
    waterlevel_series = pd.DataFrame(data=np.full([24, len(stations)], None), columns=stations)
    waterlevel_series['hour'] = np.arange(24)

    # 入力をsize=(24, len(stations))の形に変換
    waterlevel_input = pd.DataFrame(waterlevel)
    waterlevel_input = waterlevel_input.pivot(index='hour', columns='station', values='value').reset_index()
    waterlevel_input.columns.name = None
    waterlevel_input = waterlevel_input.set_index('hour')

    # 土台に合成
    waterlevel_series = waterlevel_series.combine_first(waterlevel_input)
    # 日付のカラムを追加
    waterlevel_series['date'] = date

    return waterlevel_series


def ensemble(preds_all):
    '''
        pres_all: List of preds
            (preds: np.array, size=(len(stations), 24))
    '''
    # とりあえず平均取るだけ
    preds_all = np.mean(preds_all, axis=0)
    return preds_all


def postprocess(preds_all, stations):
    df = pd.DataFrame(preds_all.T, columns=stations)
    df.index.name = 'hour'
    df.columns.name = 'station'
    output = df.stack().reset_index().rename(columns={0: 'value'}).to_dict('records')
    return output


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """
        Get model method
        Args:
            model_path (str): Path to the trained model directory.
        Returns:
            bool: The return value. True for success, False otherwise.
        """
        cls.water_df = None
        cls.rain_df = None
        cls.tide_df = None
        cls.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

        root_dir = Path(model_path).parent
        
        with initialize_config_dir(config_dir=root_dir / 'src' / 'config'):
            cfg = compose(config_name='config.yaml')
            cls.cfg = cfg
        
        cls.models = load_models(cfg, model_path, device)

        return True

    @classmethod
    def predict(cls, input):
        """Predict method
        Args:
            input (str): path to the image you want to make inference from
        Returns:
            dict: Inference for the given input.
        """

        cfg = cls.cfg

        # input -> stationがcolumnの、長さ24(時間分)のdataframe
        waterlevel_series = input2seriesdf(input)
        
        # 過去のdfと結合
        if cls.water_df is None:
            cls.water_df = waterlevel_series.copy()
        else:
            cls.water_df = pd.concat([cls.water_df, waterlevel_series]).reset_index()
        
        # 前処理
        input_df = cls.water_df.loc[:, input['stations']]
        input_df, st2info = preprocess(cfg, input_df) # df: date, hour, station_1, ..., station_n 

        # dataloaderの用意
        test_ds = HiroshimaDataset(cfg, input_df, st2info)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.valid_bs,
            shuffle=False,
            num_workers=cfg.n_workers,
            pin_memory=True
        )

        # 予測結果を入れる
        preds_all = []
        stations = []

        # 予測
        for i, (encoder, decoder) in enumerate(cls.models):
            preds_one_model = []
            
            for (data, meta) in test_loader:
                data = data.to(cls.device).float()

                with torch.no_grad():
                    h, c = encoder(data)
                    repeat_input = h.transpose(1, 0).repeat(1, cfg.output_sequence_size, 1)
                    pred = decoder(repeat_input, (h, c)).squeeze()

                    preds = (pred.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy())
                    preds_one_model.append(preds)
                    
                if i == 0:
                    stations += meta['station']
            
            preds_one_model = np.concatenate(preds_one_model)
            preds_all.append(preds_one_model)
        
        preds_all = ensemble(preds_all)
        output = postprocess(preds_all, stations)
        return output