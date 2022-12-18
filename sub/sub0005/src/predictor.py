from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hydra.experimental import compose, initialize_config_dir

from preprocess.preprocess import get_data


def preprocess(cfg, df):
    # string -> floatに (strの欠損値を全てnanとする)
    df = df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)

    # # 標準化
    # df_meta = df[['date', 'hour']]
    # df_data = df.drop(columns=['date', 'hour'])
    # df_zscore_data = (df_data - df_data.mean(skipna=True)) / df_data.std(skipna=True)
    # df = pd.concat([df_meta, df_zscore_data], axis=1)
    
    # # 標準化に使った値は後で戻す時のために変数に入れておく
    # st2mean = df_data.mean(skipna=True).to_dict()
    # st2std = df_data.std(skipna=True).to_dict()
    # st2info = {st: {'mean': st2mean[st], 'std': st2std[st]} for st in st2mean.keys()}

    return df
    

class HiroshimaDataset(Dataset):
    def __init__(self, cfg, df, st2info):
        super().__init__()
        self.st2info = st2info

        self.d = cfg.tde_d
        self.tau = cfg.tde_tau
        self.input_sequence_size = cfg.input_sequence_size
        
        input_ = df.iloc[-1*cfg.input_sequence_size:].drop(columns=['date', 'hour'])

        input_length = self.input_sequence_size + (self.d - 1) * self.tau

        # nan埋め
        input_ = input_.fillna(method='ffill') # まず新しいデータで前のnanを埋める
        input_ = input_.fillna(method='bfill') # 新しいデータがnanだった場合は古いデータで埋める
        input_ = input_.fillna(0.) # 全てがnanなら０埋め

        # dataframe -> np.array
        input_ = input_.values.T # size=(len(station), len(時間))

        # 入力の長さがRNNの入力に足りないとき => 前にpadding
        if input_length > input_.shape[1]:
            pad_length = input_length - input_.shape[1]
            pad = np.tile(np.array(input_[:, 0][:, np.newaxis]), (1, pad_length))
            input_ = np.concatenate([pad, input_], axis=1)
        
        self.inputs = input_.tolist()
        self.stations = df.drop(columns=['date', 'hour']).columns

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ = np.array(self.inputs[idx]).astype(np.float32)

        meta = self.st2info[self.stations[idx]]
        meta['station'] = self.stations[idx]
        if self.d > 1:
            input_tde = np.zeros((self.input_sequence_size, self.d))
            for i in range(self.d):
                input_tde[:, i] = np.roll(input_, i*self.tau)[(self.d - 1)*self.tau:]
            input_ = torch.tensor(input_tde) # input_: (len_of_series, input_size)
        else:
            input_ = torch.tensor(input_).unsqueeze(-1)
        return input_, meta


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    def forward(self, x, h0=None):
        '''
            x: (bs, len_of_series ,input_size)
            h0 = Tuple(h, c)
        '''
        hs, h = self.lstm(x, h0) # -> x: (bs, len_of_series, hidden_size)
        return hs, h


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention=True):
        super().__init__()
        self.attention = attention

        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
        if attention:
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, hs):
        '''
            x: (bs, output_seq_size=1, output_size=1)
            h = Tuple(h, c)
            hs = (bs, input_seq_size, hidden_size)
        '''
        x, h = self.lstm(x, h) # x: (bs, output_seq_size=1, hidden_size)
        if self.attention:
            x_t = torch.transpose(x, 1, 2)
            attention_weight = self.softmax(torch.bmm(hs, x_t)) # attention_weight: (bs, input_seq_size, output_seq_size=1)
            weighted_hs = hs * attention_weight # weighted_hs: (bs, input_seq_size, hidden_size)
            context = torch.sum(weighted_hs, dim=1, keepdim=True) # context: (bs, 1, hidden_size)
            x = torch.cat([x, context], dim=2) # (bs, 1, hidden_size*2)
        x = self.fc(x) # x: (bs, 1, output_size)
        return x, h


class Model(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.output_size = cfg.output_size
        self.output_sequence_size = cfg.output_sequence_size
        self.device = device

        self.encoder = Encoder(self.input_size, self.hidden_size, self.output_size)
        self.decoder = Decoder(self.hidden_size, self.output_size, attention=cfg.attention)
    
    def forward(self, x):
        hs, encoder_state = self.encoder(x, h0=None)
        decoder_input = x[:, -1:, :1]
        decoder_state = encoder_state

        pred = torch.empty(len(x), self.output_sequence_size, self.output_size).to(self.device).float()

        for i in range(self.output_sequence_size):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state, hs) # decoder_output: (bs, 1, output_size=1)
            pred[:, i:i+1, :] = decoder_output
            decoder_input = decoder_output
        return pred


def load_models(cfg, model_dir, device):
    models = []
    for model_path in model_dir.glob('*.pth'):
        model = Model(cfg, device)
        path_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(path_dict['model'])
        model = {
            'model': model
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


def postprocess(preds_all, df):
    value = preds_all.reshape(-1)
    df['value'] = value
    
    # formatを整える & nan埋め
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.fillna(df.mean(skipna=True))

    df['hour'] = df['hour'].astype(int)
    df['station'] = df['station'].astype(str)    
    df['value'] = df['value'].astype(float)
    output = df[['hour', 'station', 'value']].to_dict('records')
    return output


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        
        cls.water_df = None
        cls.rain_df = None
        cls.tide_df = None
        cls.device = 'cpu'

        model_path = Path(model_path)
        root_dir = model_path.parent.resolve()
        data_dir = root_dir / 'model' / 'train'
        
        with initialize_config_dir(config_dir=str(root_dir / 'src' / 'config')):
            cfg = compose(config_name='config.yaml')
            cls.cfg = cfg

        cls.train_rain, cls.rain_st, cls.train_tide, cls.tide_st, cls.train_water, \
            cls.water_st, cls.dam, cls.river, cls.river_system = get_data(data_dir)


        # 学習データを標準化
        df_data = cls.train_water.drop(columns=['date', 'hour'])
        st2mean = df_data.mean(skipna=True).to_dict()
        st2std = df_data.std(skipna=True).to_dict()
        st_id2info = {st: {'mean': st2mean[st], 'std': st2std[st]} for st in st2mean.keys()}
        print(st_id2info)

        # st2id
        st2id = cls.water_st[['id', 'station']].set_index('station')['id'].to_dict()
        cls.st2info = {st: st_id2info[st2id[st]] for st in st2id.keys()}

        cls.models = load_models(cfg, model_path, cls.device)

        return True

    @classmethod
    def predict(cls, input):

        cfg = cls.cfg

        # input -> stationがcolumnの、長さ24(時間分)のdataframe
        waterlevel_series = input2seriesdf(input)
        stations = list(input['stations'])
        waterlevel = input['waterlevel']
        output_df = pd.merge(pd.DataFrame(stations, columns=['station']), pd.DataFrame(waterlevel))
        
        # 過去のdfと結合
        if cls.water_df is None:
            cls.water_df = waterlevel_series.copy()
        else:
            cls.water_df = pd.concat([cls.water_df, waterlevel_series]).reset_index(drop=True)
        
        # 前処理
        input_columns = stations + ['date', 'hour']
        input_df = cls.water_df.loc[:, input_columns]
        input_df = preprocess(cfg, input_df)

        # dataloaderの用意
        test_ds = HiroshimaDataset(cfg, input_df, cls.st2info)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.test_bs,
            shuffle=False,
            num_workers=cfg.n_workers,
            pin_memory=True
        )

        # 予測結果を入れる
        preds_all = []

        # 予測
        for models in cls.models:
            preds_one_model = []
            model = models['model'].to(cls.device)
            
            for (data, meta) in test_loader:
                data = data.to(cls.device).float()

                with torch.no_grad():
                    pred = model(data).squeeze()

                    preds = (pred.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy())
                    preds_one_model.append(preds)
            
            preds_one_model = np.concatenate(preds_one_model)
            preds_all.append(preds_one_model)
        
        preds_all = ensemble(preds_all)
        output = postprocess(preds_all, output_df)
        return output