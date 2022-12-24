from pathlib import Path
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

from utils import rmse, AverageMeter, seed_everything

import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize_config_dir
import wandb

def load_data(cfg, root_path):
    df = pd.read_csv(str(root_path / cfg.water_csv_path))
    st_df = pd.read_csv(str(root_path / cfg.water_st_csv_path))
    dates = df['date'].astype(int).unique()
    dates.sort()
    folds = TimeSeriesSplit(n_splits=cfg.n_folds).split(dates)
    df['fold'] = -1
    for fold, (_, valid_dates) in enumerate(folds):
        df.loc[df['date'].isin(valid_dates), 'fold'] = fold
    return df, st_df


def preprocess(cfg, train_fold_df, valid_fold_df):
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

    return train_fold_df, valid_fold_df, st2info


def prepare_dataloader(cfg, train_fold_df, valid_fold_df, st2info):
    train_ds = HiroshimaDataset(cfg, train_fold_df, st2info, 'train')
    valid_ds = HiroshimaDataset(cfg, valid_fold_df, st2info, 'valid')

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


class HiroshimaDataset(Dataset):
    def __init__(self, cfg, df, st2info, phase):
        super().__init__()
        self.st2info = st2info

        self.inputs = []
        self.targets = []
        self.stations = []
        self.borders = []

        first_index = df.index[0]
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
            self.borders += [first_index + border]*len(target.columns)
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

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        target = self.targets[idx]
        meta = self.st2info[self.stations[idx]]
        meta['station'] = self.stations[idx]
        meta['border'] = self.borders[idx]

        input_ = torch.tensor(input_) # input_: (len_of_series, input_size)
        target = torch.tensor(target) # target: (len_of_series)

        return input_, target, meta


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, stations_embed):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size+stations_embed[1], hidden_size=hidden_size, batch_first=True)
        self.st_embeddings = nn.Embedding(stations_embed[0], stations_embed[1])
        print(stations_embed)
    
    def forward(self, x, st, h0=None):
        '''
            x: (bs, input_seq_size ,input_size)
            h0 = Tuple(h, c)
            st: (bs, 1)
        '''
        st = st.repeat(1, x.size()[1]) # (bs, len_of_series)
        st_embed = self.st_embeddings(st) # (bs, input_seq_size, station_embed_size)
        x = torch.cat([x, st_embed], axis=-1)
        _, h = self.lstm(x, h0) # -> x: (bs, input_seq_size, hidden_size)
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
    def __init__(self, cfg, n_stations, device):
        super().__init__()
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.output_size = cfg.output_size
        self.output_sequence_size = cfg.output_sequence_size
        self.device = device
        self.stations_embed = [n_stations, cfg.station_embed_size]
        self.encoder = Encoder(self.input_size, self.hidden_size, self.stations_embed)
        self.decoder = Decoder(self.hidden_size, self.output_size, cfg.station_embed_size)
    
    def forward(self, x, target, stations, teacher_forcing_ratio):
        encoder_state = self.encoder(x, stations, h0=None)
        decoder_input = x[:, -1:, :]
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

    for step, (data, target, meta) in pbar:
        data = data.to(device).float() # (bs, len_of_series, input_size)
        target = target.to(device).float() # (bs, len_of_series)
        stations = torch.tensor(list(map(int, meta['station']))).unsqueeze(-1).to(device).long()
        pred = model(data, target, stations, teacher_forcing_ratio).squeeze()
        break

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
        pred = pred.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy()
        target = target.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy()
        preds_all += [pred]
        targets_all += [target]
    
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

    for step, (data, target, meta) in pbar:
        data = data.to(device).float() # (bs, len_of_series, input_size)
        target = target.to(device).float() # (bs, len_of_series)

        with torch.no_grad():
            pred = model(data, target, 0.).squeeze()

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
    
    df, st_df = load_data(cfg, root_path)
    n_stations = st_df['id'].nunique()

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

        train_fold_df, valid_fold_df, st2info = preprocess(cfg, train_fold_df, valid_fold_df)
        train_loader, valid_loader = prepare_dataloader(cfg, train_fold_df, valid_fold_df, st2info)

        model = Model(cfg, n_stations, device).to(device)

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
        del model, train_fold_df, valid_fold_df, train_loader, valid_loader, loss_fn, optimizer, best_dict
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()