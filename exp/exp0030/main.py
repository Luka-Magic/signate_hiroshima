from pathlib import Path
import gc
import random
import numpy as np
import pandas as pd
import os, psutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

from tqdm import tqdm

from utils import rmse, AverageMeter, seed_everything

import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize_config_dir
import wandb

def load_data(cfg, root_path):  
    df = pd.read_csv(str(root_path / cfg.water_csv_path))
    water_st_df = pd.read_csv(str(root_path / cfg.water_st_csv_path))
    rain_df = pd.read_csv(str(root_path / cfg.rain_csv_path))
    rain_st_df = pd.read_csv(str(root_path / cfg.rain_st_csv_path))

    dates = df['date'].astype(int).unique()
    dates.sort()
    folds = TimeSeriesSplit(n_splits=cfg.n_folds).split(dates)
    df['fold'] = -1
    for fold, (_, valid_dates) in enumerate(folds):
        df.loc[df['date'].isin(valid_dates), 'fold'] = fold
    return df, water_st_df, rain_df, rain_st_df


def preprocess(cfg, train_fold_df, valid_fold_df, water_st_df, rain_df, rain_st_df):
    # dataframeの前処理
    train_fold_df = train_fold_df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)
    valid_fold_df = valid_fold_df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)

    # train_fold_dfの数値部分(date, hour以外)の処理
    train_meta = train_fold_df[['date', 'hour']]
    train_data = train_fold_df.drop(columns=['date', 'hour', 'fold'])

    ## train_fold_dfの標準化
    train_zscore_data = (train_data - train_data.mean(skipna=True)) / train_data.std(skipna=True)
    train_fold_df = pd.concat([train_meta, train_zscore_data], axis=1)

    # valid_fold_dfはtrainの平均と標準偏差で標準化
    valid_meta = valid_fold_df[['date', 'hour']]
    valid_data = valid_fold_df.drop(columns=['date', 'hour', 'fold'])
    valid_zscore_df = (valid_data - train_data.mean(skipna=True)) / train_data.std(skipna=True)
    valid_fold_df = pd.concat([valid_meta, valid_zscore_df], axis=1)

    # 標準化に使った値は後で戻す時のために変数に入れておく
    st2mean = train_data.mean(skipna=True).to_dict()
    st2std = train_data.std(skipna=True).to_dict()
    st2info = {st: {'mean': st2mean[st], 'std': st2std[st]} for st in st2mean.keys()}

    # rain_dfの前処理
    rain_df = rain_df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)
    rain_meta = rain_df[['date', 'hour']]
    rain_data = rain_df.drop(columns=['date', 'hour'])
    # 差分をとる
    rain_data = rain_data - rain_data.shift()
    # nan埋め
    rain_data = rain_data.fillna(0.)
    # minmaxscale
    rain_data = ((rain_data - np.nanmin(rain_data.values)) / (np.nanmax(rain_data.values) - np.nanmin(rain_data.values))) * 255.
    # int化
    rain_data = rain_data.astype(np.uint8)

    rain_df = pd.concat([rain_meta, rain_data], axis=1)
    return train_fold_df, valid_fold_df, st2info, rain_df


class HiroshimaDataset(Dataset):
    def __init__(self, cfg, df, st2info, phase, water_st_df, rain_df, rain_st_df):
        super().__init__()
        self.df = df
        self.st2info = st2info
        self.rain_map_size = cfg.rain_map_size
        self.clip_map_size = cfg.clip_map_size
        self.input_sequence_size = cfg.input_sequence_size
        self.rain_df = rain_df
        self.rain_st_df = rain_st_df
        self.water_st_df = water_st_df

        self.inputs = []
        self.targets = []
        self.stations = []
        self.borders = []

        self.first_index = df.index[0]
        start_row = 24 # 最低でもはじめの24hは使う
        last_row = len(df) # 最後の行まで使う (最後の24hは推論用)

         # borderはある日の0時を示し、inputはそれより前のsequenceの長さ時間(例えば30時間分)、targetはborder以降の24時間分を使う
        for border in tqdm(range(start_row, last_row, 24)):
            assert df.iloc[border]['hour'] == 0, '行が0時スタートになってない。'
            
            input_ = df.iloc[max(border-cfg.input_sequence_size, 0):border, :].drop(columns=['date', 'hour'])
            input_ = input_.fillna(method='ffill') # まず新しいデータで前のnanを埋める
            input_ = input_.fillna(method='bfill') # 新しいデータがnanだった場合は古いデータで埋める
            input_ = input_.fillna(0.) # 全てがnanなら０埋め
            
            target = df.iloc[border:border+24, :].drop(columns=['date', 'hour'])

            target = target.loc[:, ~target.isnull().any(axis=0)] # input, target共にnullがない列だけ抜き出す
            self.stations += target.columns.tolist()
            self.borders += [self.first_index + border]*len(target.columns)
            input_ = input_.loc[:, target.columns] # targetに使われるinputだけ取り出す
            input_ = input_.values.T[:, :, np.newaxis] # size=(len(station), len(時間), 1)

            # 入力の長さがRNNの入力に足りないとき => 前にpadding
            if cfg.input_sequence_size > input_.shape[1]:
                pad_length = cfg.input_sequence_size - input_.shape[1]
                pad = np.tile(np.array(input_[:, 0, :][:, np.newaxis, :]), (1, pad_length, 1))
                input_ = np.concatenate([pad, input_], axis=1)
            
            self.inputs += input_.tolist()
            self.targets += target.values.T.tolist()
        print(f'{phase} datas: {len(self.inputs)}')

        self._create_rain_map()
    
    def _create_rain_map(self):
        """
            rain_mapを作成する。

            rain_map:
                rainデータと緯度経度の情報から2次元のgridを作成し、セル内にrainfallのデータを入れる(ない場所はnanにして後に埋める)。
                cfg.rain_map_sizeが1辺のサイズ。左下を(0, 0)として各セルの座標を(x, y)とする。
                時間方向を合わせて3次元になる(time, x, y)。
        """
        # rain_id2xy: rain_idがそれぞれどの座標(x. y)にあたるかを示すdataframe
        rain_id2xy = self.rain_st_df[['id', '緯度', '経度']].dropna()
        x = rain_id2xy['経度'].values
        y = rain_id2xy['緯度'].values
        rain_id2xy['x'] = ((self.rain_map_size-1) * ((x - x.min()) / (x.max() - x.min()))).astype(np.uint8)
        rain_id2xy['y'] = ((self.rain_map_size-1) * ((y - y.min()) / (y.max() - y.min()))).astype(np.uint8)
        rain_id2xy = rain_id2xy[['id', 'x', 'y']]
        self.rain_id2xy = rain_id2xy[~rain_id2xy[['x', 'y']].duplicated()]
        del rain_id2xy

        # xyhour: 最終的なデータをこのdataframeにleftjoinして, valuesをreshapeして(time, x, y)の形にする。
        data_list = []
        for hour in range(len(self.df)):
            for y_ in range(self.rain_map_size-1, -1, -1):
                for x_ in range(self.rain_map_size):
                    data_list.append([x_, y_, hour])
        self.xyhour = pd.DataFrame(data=data_list, columns=['x', 'y', 'hour'])
        self.xyhour = self.xyhour.astype(np.uint8)
        del data_list
        gc.collect()

        # water_id2xy: water_idがそれぞれどの座標(x, y)にあたるかを示すdict
        water_id2xy = self.water_st_df[['id', '緯度', '経度']].dropna()
        water_x = water_id2xy['経度'].values
        water_y = water_id2xy['緯度'].values
        water_id2xy['x'] = ((self.rain_map_size-1) * ((water_x - x.min()) / (x.max() - x.min()))).astype(np.uint8)
        water_id2xy['y'] = ((self.rain_map_size-1) * ((water_y - y.min()) / (y.max() - y.min()))).astype(np.uint8)
        self.water_id2xy = water_id2xy[['id', 'x', 'y']].set_index('id').to_dict()

        # rain_id2value: idとvalueの組み合わせをrainfallデータから作成
        rain_border = self.rain_df.drop(columns=['date', 'hour'])
        rain_id2value = rain_border.stack(dropna=False).reset_index() \
            .rename(columns={'level_0': 'index', 'level_1': 'id', 0: 'value'}).astype({'id': int, 'value': np.uint8}) # id, value
        rain_id2value = rain_id2value.rename(columns={'index': 'hour'})
        
        # xy2value: mergeによりそれぞれのrainfallのデータがどの座標かを対応させる。
        xy2value = pd.merge(rain_id2value, self.rain_id2xy, on='id', how='left')
        del rain_id2value, rain_border
        # mapping_df: mergeにより全ての(time, x, y)においてどのデータがあるかを対応させる
        print(f'RAM memory used (before creating mapping_df): {psutil.virtual_memory()[2]}%')
        mapping_df = pd.merge(self.xyhour, xy2value, on=['x', 'y', 'hour'], how='left')
        del self.xyhour, xy2value
        gc.collect()
        print(f'RAM memory used (created mapping_df): {psutil.virtual_memory()[2]}%')
        mapping_df = mapping_df.fillna(0).astype({'value': np.uint8})
        print(f'RAM memory used (mapping_df preprocess): {psutil.virtual_memory()[2]}%')
        gc.collect()
        print(f'RAM memory used (before create rain_map): {psutil.virtual_memory()[2]}%')

        # arrayに変換してnan埋め
        self.rain_map = mapping_df['value'].values.reshape(-1, self.rain_map_size, self.rain_map_size)
        print(f'RAM memory used (created rain_map): {psutil.virtual_memory()[2]}%')
        del mapping_df
        gc.collect()
        self.rain_map = np.nan_to_num(self.rain_map, nan=0)
        gc.collect()
        print(f'RAM memory used (rain_map all completed!): {psutil.virtual_memory()[2]}%')
    
    def __len__(self):
        return len(self.inputs)
    
    def _get_clip_map(self, border, station):
        """
            clip_mapを作成
            clip_map:
                入力に使う水位の時間と観測所の位置からrain_mapの一部を切り取り入力に使う。
                cfg.clip_map_sizeを一辺、時間の長さはcfg.input_sequence_size
                また端っこでも切り取れるように先にpaddingをしてから切り取り

        """
        # もしｗaterデータの座標がない場合は全て0
        if int(station) not in self.water_id2xy['x']:
            return np.zeros((self.input_sequence_size, self.clip_map_size, self.clip_map_size))
        
        # 時間方向に切り取り
        index = border - self.first_index
        rain_map = self.rain_map[max(index - self.input_sequence_size, 0):index, :, :]
        # paddingしてから2次元で切り取る
        rain_map_pad = np.pad(rain_map, ((0, 0), (self.clip_map_size, self.clip_map_size), (self.clip_map_size, self.clip_map_size)))
        water_x, water_y = self.water_id2xy['x'][int(station)], self.water_id2xy['y'][int(station)]
        center_x, center_y = self.rain_map_size - water_y, water_x
        center_x_pad, center_y_pad = center_x + self.clip_map_size, center_y + self.clip_map_size
        half_clip_size = (self.clip_map_size - 1) // 2
        clip_map = rain_map_pad[:, (center_x_pad-half_clip_size):(center_x_pad+half_clip_size+1), (center_y_pad-half_clip_size):(center_y_pad+half_clip_size+1)]

        # 時間方向で足りない場合は前を0埋め
        if clip_map.shape[0] < self.input_sequence_size:
            pad_length = self.input_sequence_size - clip_map.shape[0]
            clip_map = np.pad(clip_map, ((pad_length, 0), (0, 0), (0, 0)))
        
        return clip_map
    
    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        target = self.targets[idx]
        meta = self.st2info[self.stations[idx]]
        meta['station'] = self.stations[idx]
        meta['border'] = self.borders[idx]

        # 水位の入力の時間と位置からrain_mapを作成
        rain_map = self._get_clip_map(meta['border'], meta['station'])
        rain_map = torch.tensor(rain_map).float()

        input_ = torch.tensor(input_) # input_: (len_of_series, input_size)
        target = torch.tensor(target) # target: (len_of_series)

        return input_, rain_map, target, meta


def prepare_dataloader(cfg, train_fold_df, valid_fold_df, st2info, water_st_df, rain_df, rain_st_df):
    train_ds = HiroshimaDataset(cfg, train_fold_df, st2info, 'train', water_st_df, rain_df, rain_st_df)
    valid_ds = HiroshimaDataset(cfg, valid_fold_df, st2info, 'valid', water_st_df, rain_df, rain_st_df)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.n_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.n_workers,
        pin_memory=True
    )
    return train_loader, valid_loader


class RainFeatureExtractor(nn.Module):
    def __init__(self, clip_map_size, rain_feature_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )
        image_feature_size = 6
        self.fc = nn.Linear(4*image_feature_size**2, rain_feature_size)
    
    def forward(self, x):
        b, seq, h, w = x.size()
        x = x.view(b*seq, 1, h, w)
        x = self.conv(x) # (b*seq, 1, h_c, w_c)
        x = x.view(b, seq, -1)
        x = self.fc(x)
        return x


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
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        '''
            x: (bs, len_of_series ,output_size)
            h = Tuple(h, c)
        '''
        x, h = self.lstm(x, h0) # -> x: (bs, len_of_series, hidden_size)
        x = self.fc(x) # -> x: (bs, len_of_series, output_size)
        return x, h


class Model(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.input_size = cfg.input_size
        self.clip_map_size = cfg.clip_map_size
        self.hidden_size = cfg.hidden_size
        self.output_size = cfg.output_size
        self.rain_feature_size = cfg.rain_feature_size
        self.output_sequence_size = cfg.output_sequence_size
        self.device = device

        self.rain_feature_extractor = RainFeatureExtractor(self.clip_map_size, self.rain_feature_size)
        self.encoder = Encoder(self.input_size + self.rain_feature_size, self.hidden_size)
        self.decoder = Decoder(self.hidden_size, self.output_size)
    
    def forward(self, x, rain_map, target, teacher_forcing_ratio):
        rain_map_feature = self.rain_feature_extractor(rain_map)
        x = torch.cat([x, rain_map_feature], dim=-1)
        encoder_state = self.encoder(x, h0=None)
        decoder_input = x[:, -1:, :1]
        decoder_state = encoder_state

        pred = torch.empty(len(x), self.output_sequence_size, self.output_size).to(self.device).float()

        for i in range(self.output_sequence_size):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state) # decoder_output: (bs, 1, output_size=1)
            pred[:, i:i+1, :] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target[:, i:i+1].unsqueeze(-1) if teacher_force else decoder_output
        return pred


def train_one_epoch(cfg, epoch, dataloader, model, loss_fn, device, optimizer, scheduler, scheduler_step_time):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    model.train()

    preds_all = []
    targets_all = []

    losses = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    teacher_forcing_ratio = cfg.teacher_forcing_ratio_rate ** (epoch - 1)

    for step, (data, rain_map, target, meta) in pbar:
        if torch.isnan(rain_map).any():
            print(rain_map)
            raise ValueError
        data = data.to(device).float() # (bs, len_of_series, input_size)
        rain_map = rain_map.to(device).float() # (bs, len_of_series, rain_map_size, rain_map_size)
        target = target.to(device).float() # (bs, len_of_series)

        pred = model(data, rain_map, target, teacher_forcing_ratio).squeeze()

        # 評価用のlossの算出
        loss = 0
        for i in range(pred.size()[1]):
            loss += loss_fn(pred[:, i], target[:, i])
        losses.update(loss.item(), len(data))
        
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if scheduler is not None and scheduler_step_time=='step':
            scheduler.step()
        
        losses.update(loss.item(), len(data))
        pred = (pred.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy())
        target = (target.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy())
        preds_all += [pred]
        targets_all += [target]
        pbar.set_postfix(OrderedDict(loss=losses.avg))
        if np.isnan(losses.avg):
            raise ValueError
    
    if scheduler is not None and scheduler_step_time=='epoch':
        scheduler.step()

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    score = rmse(targets_all, preds_all)

    lr = get_lr(optimizer)
    
    print(f'Train - Epoch {epoch}: losses {losses.avg:.4f}, scores: {score}')
    return score, losses.avg, lr, teacher_forcing_ratio


def valid_one_epoch(cfg, epoch, dataloader, model, loss_fn, device):
    model.eval()

    preds_all = []
    targets_all = []

    losses = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, (data, rain_map, target, meta) in pbar:
        data = data.to(device).float() # (bs, len_of_series, input_size)
        rain_map = rain_map.to(device).float() # (bs, len_of_series, rain_map_size, rain_map_size)
        target = target.to(device).float() # (bs, len_of_series)

        with torch.no_grad():
            pred = model(data, rain_map, target, 0.).squeeze()

            # 評価用のlossの算出
            loss = 0
            for i in range(pred.size()[1]):
                loss += loss_fn(pred[:, i], target[:, i]) # output_sizeは1なのでpredの3次元目をsqueeze
            losses.update(loss.item(), len(data))

            # 評価用にRMSEを算出
            pred = (pred.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy())
            target = (target.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy())
            preds_all += [pred]
            targets_all += [target]

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    score = rmse(targets_all, preds_all)

    print(f'Valid - Epoch {epoch}: losses {losses.avg:.4f}, scores {score:.4f}')
    return score, losses.avg


def main():
    exp_path = Path.cwd()
    root_path = Path.cwd().parents[2]
    save_path = root_path / 'outputs' / exp_path.name
    save_path.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(exp_path / 'config')):
        cfg = compose(config_name='config.yaml')
    
    seed_everything(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.use_wandb:
        wandb.login()
    
    df, water_st_df, rain_df, rain_st_df = load_data(cfg, root_path)

    for fold in range(cfg.n_folds):
        if fold not in cfg.use_folds:
            continue
        
        if cfg.use_wandb:
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True)
            wandb.init(project=cfg.wandb_project, entity='luka-magic',
                        name=f'{exp_path.name}', config=wandb.config)
            wandb.config.fold = fold
        train_fold_df = df[df['fold'] < fold]
        valid_fold_df = df[df['fold'] == fold]
        train_fold_df = train_fold_df.sort_values(['date', 'hour'])
        valid_fold_df = valid_fold_df.sort_values(['date', 'hour'])

        train_fold_df, valid_fold_df, st2info, rain_df_p = preprocess(cfg, train_fold_df, valid_fold_df, water_st_df, rain_df, rain_st_df)
        train_loader, valid_loader = prepare_dataloader(cfg, train_fold_df, valid_fold_df, st2info, water_st_df, rain_df_p, rain_st_df)
        del train_fold_df, valid_fold_df, st2info, rain_df_p
        gc.collect()

        model = Model(cfg, device).to(device)

        if cfg.loss_fn == 'MSELoss':
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.MSELoss()
        
        if cfg.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
        if cfg.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, total_steps=cfg.n_epochs * len(train_loader), max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
        else:
            scheduler  = None
        scheduler_step_time = cfg.scheduler_step_time

        best_dict = dict(
            score=float('inf'),
            loss=float('inf'),
        )
        
        for epoch in range(1, cfg.n_epochs+1):
            # 学習
            train_score, train_loss, lr, teacher_forcing_ratio = train_one_epoch(cfg, epoch, train_loader, model, loss_fn, device, optimizer, scheduler, scheduler_step_time)
            # 推論
            valid_score, valid_loss = valid_one_epoch(cfg, epoch, valid_loader, model, loss_fn, device)

            wandb_dict = dict(
                epoch=epoch,
                train_score=train_score,
                train_loss=train_loss,
                valid_score=valid_score,
                valid_loss=valid_loss,
                lr = lr,
                teacher_forcing_ratio = teacher_forcing_ratio
            )
            if valid_score < best_dict['score']:
                best_dict['score'] = valid_score
                wandb.run.summary['best_score'] = best_dict['score']
                save_dict = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                }
                torch.save(save_dict, str(save_path / f'best_score_fold{fold}.pth'))
                print(f'score update!: {best_dict["score"]:.4f}')
            if valid_loss < best_dict['loss']:
                wandb.run.summary['best_loss'] = best_dict['loss']
                best_dict['loss'] = valid_loss
            if cfg.use_wandb:
                wandb.log(wandb_dict)
        
        wandb.finish()
        del model, train_loader, valid_loader, loss_fn, optimizer, best_dict
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()