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


def preprocess(cfg, train_fold_df, valid_fold_df):
    # dataframeの前処理
    train_fold_df = train_fold_df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)
    train_fold_df = train_fold_df.drop(columns=['75', '152'])
    valid_fold_df = valid_fold_df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)
    valid_fold_df = valid_fold_df.drop(columns=['75', '152'])

    # train_fold_dfの数値部分を標準化
    train_meta = train_fold_df[['date', 'hour']]
    train_data = train_fold_df.drop(columns=['date', 'hour', 'fold'])
    train_zscore_data = (train_data - train_data.mean(skipna=True)) / train_data.std(skipna=True)
    train_fold_df = pd.concat([train_meta, train_zscore_data], axis=1)

    # valid_fold_dfはtrainの平均と標準偏差で標準化
    valid_meta = valid_fold_df[['date', 'hour']]
    valid_data = valid_fold_df.drop(columns=['date', 'hour', 'fold'])
    valid_zscore_df = (valid_data - train_data.mean(skipna=True)) / train_data.std(skipna=True)
    valid_fold_df = pd.concat([valid_meta, valid_zscore_df], axis=1)

    st2mean = train_data.mean(skipna=True).to_dict()
    st2std = train_data.std(skipna=True).to_dict()
    st2info = {st: {'mean': st2mean[st], 'std': st2std[st]} for st in st2mean.keys()}

    return train_fold_df, valid_fold_df, st2info


class HiroshimaDataset(Dataset):
    def __init__(self, cfg, df, st2info, phase):
        super().__init__()
        self.st2info = st2info
        # 1. 入力に使える開始日((input_size // 24) + 1)と終了日(最後の24h以外)以内を切り取る
        # 2. input: (input_size分の時間 ~23hが最後, all station)、target: (24h, all station) でconcat
        #    stationに対しnullが存在しないデータだけtrain, testのlistに追加
        self.inputs = []
        self.targets = []
        self.stations = []
        self.borders = []

        first_index = df.index[0]
        start_row = (((cfg.input_sequence_size-1) // 24) + 1) * 24 # 例えばsequenceの長さが30(時間)の場合、入力に使うのは"2日目の23時までの30時間"にする
        last_row = len(df) # 最後の行まで使う

         # borderはある日の0時を示し、inputはそれより前のsequenceの長さ時間(例えば30時間分)、targetはborder以降の24時間分を使う
        for border in range(start_row, last_row, 24):
            assert df.iloc[border]['hour'] == 0, '行が0時スタートになってない。'
            input_ = df.iloc[border-cfg.input_sequence_size:border, :].drop(columns=['date', 'hour'])
            target = df.iloc[border:border+24, :].drop(columns=['date', 'hour'])

            cat = pd.concat([input_, target]) # 一度concatしてnull値チェック
            cat = cat.loc[:, ~cat.isnull().any(axis=0)] # input, target共にnullがない列だけ抜き出す
            self.stations += cat.columns.tolist()
            self.borders += [first_index + border]*len(cat.columns)
            input_, target = cat.iloc[:-24], cat.iloc[-24:] # concatを戻す
            self.inputs += input_.values.T[:, :, np.newaxis].tolist()
            self.targets += target.values.T.tolist()
        print(f'{phase} datas: {len(self.inputs)}')

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        target = self.targets[idx]
        meta = self.st2info[self.stations[idx]]
        meta['station'] = self.stations[idx]
        meta['border'] = self.borders[idx]

        input_ = torch.tensor(input_) # input_: (len_of_series, input_size)
        # TODO padding操作
        target = torch.tensor(target) # target: (len_of_series)

        return input_, target, meta


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


def load_models(cfg, model_dir):
    models = []
    for model_path in model_dir.glob('*.pth'):
        encoder = Encoder(cfg.input_size, cfg.hidden_size).to(cfg.device)
        decoder = Decoder(cfg.hidden_size, cfg.output_size).to(cfg.device)
        path_dict = torch.load(model_path)
        encoder.load_state_dict(path_dict['encoder'])
        decoder.load_state_dict(path_dict['decoder'])
        model = {
            'encoder': encoder,
            'decoder': decoder
        }
        models.append(model)
    return models


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method
        Args:
            model_path (str): Path to the trained model directory.
        Returns:
            bool: The return value. True for success, False otherwise.
        """
        # try:
        root_dir = Path(model_path).parent

        with initialize_config_dir(config_dir=root_dir / 'src' / 'config'):
            cfg = compose(config_name='config.yaml')
            cls.cfg = cfg
        
        cls.model = load_models(cfg, model_path)
        # with open(reference_meta_path) as f:
        #     reference_meta = json.load(f)
        # embeddings, ids = make_reference(
        #     cfg, reference_path, reference_meta, cls.model)
        # cls.embeddings = embeddings
        # cls.ids = ids
        return True
        # except:
        #     return False

    @classmethod
    def predict(cls, input):
        """Predict method
        Args:
            input (str): path to the image you want to make inference from
        Returns:
            dict: Inference for the given input.
        """
        stations = input['stations']
        waterlevel = input['waterlevel']
        merged = pd.merge(pd.DataFrame(stations, columns=['station']), pd.DataFrame(waterlevel))
        merged['value'] = merged['value'].replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        merged['value'] = merged['value'].fillna(0.0)
        merged['value'] = merged['value'].astype(float)
        prediction = merged[['hour', 'station', 'value']].to_dict('records')

        return prediction