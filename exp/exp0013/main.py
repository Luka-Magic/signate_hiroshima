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
    dates = df['date'].astype(int).unique()
    dates.sort()
    folds = TimeSeriesSplit(n_splits=cfg.n_folds).split(dates)
    df['fold'] = -1
    for fold, (_, valid_dates) in enumerate(folds):
        df.loc[df['date'].isin(valid_dates), 'fold'] = fold
    return df


def preprocess(cfg, train_fold_df, valid_fold_df):
    # dataframeの前処理
    train_fold_df = train_fold_df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)
    train_fold_df = train_fold_df.drop(columns=['75', '152'])
    valid_fold_df = valid_fold_df.apply(lambda x:pd.to_numeric(x, errors='coerce')).astype(float)
    valid_fold_df = valid_fold_df.drop(columns=['75', '152'])

    # train_fold_dfの数値部分(date, hour以外)をnan埋め
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
        if phase == 'train':
            border_interval = 1
        else:
            border_interval = 24
        for border in tqdm(range(start_row, last_row, border_interval)):
            if phase == 'train' and random.random() < cfg.train_data_ratio:
                continue
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
            if len(target) != cfg.output_sequence_size:
                continue
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
        # print(input_.shape, target.shape)

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


def valid_function(cfg, fold, epoch, dataloader, encoder, decoder, loss_fn, device, best_dict, save_path):
    encoder.eval()
    decoder.eval()

    preds_all = []
    targets_all = []

    losses = AverageMeter()

    # pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar = enumerate(dataloader)

    for step, (data, target, meta) in pbar:
        data = data.to(device).float() # (bs, len_of_series, input_size)
        target = target.to(device).float() # (bs, len_of_series)

        with torch.no_grad():
            h, c = encoder(data) # h: (layers=1, bs, hidden_size), c: (layers=1, bs, hidden_size) 
            repeat_input = h.transpose(1, 0).repeat(1, cfg.output_sequence_size, 1) # repeat_input: (bs, len_of_series, hidden_size)
            pred = decoder(repeat_input, (h, c)).squeeze() # pred: (bs, len_of_series, output_size)

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
    valid_score = rmse(targets_all, preds_all)
    valid_loss = losses.avg

    print(f'Valid - Epoch {epoch}: losses {losses.avg:.4f}, scores {valid_score:.4f}')

    # log
    if valid_score < best_dict['score']:
        best_dict['score'] = valid_score
        wandb.run.summary['best_score'] = best_dict['score']
        save_dict = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
        }
        torch.save(save_dict, str(save_path / f'best_score_fold{fold}.pth'))
        print(f'score update!: {best_dict["score"]:.4f}')
    if valid_loss < best_dict['loss']:
        wandb.run.summary['best_loss'] = best_dict['loss']
        best_dict['loss'] = valid_loss
    
    return valid_score, valid_loss


def train_valid_one_epoch(cfg, fold, epoch, train_loader, valid_loader, encoder, decoder, loss_fn, device, \
                        encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, scheduler_step_time, best_dict, save_path):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    encoder.train()
    decoder.train()

    preds_all = []
    targets_all = []

    losses = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    step_valid = len(pbar) // cfg.num_valids_per_epoch # validを行うstep数

    for step, (data, target, meta) in pbar:
        data = data.to(device).float() # (bs, len_of_series, input_size)
        target = target.to(device).float() # (bs, len_of_series)

        h, c = encoder(data) # h: (layers=1, bs, hidden_size), c: (layers=1, bs, hidden_size) 
        repeat_input = h.transpose(1, 0).repeat(1, cfg.output_sequence_size, 1) # repeat_input: (bs, len_of_series, hidden_size)
        pred = decoder(repeat_input, (h, c)).squeeze() # decoder_output: (bs, len_of_series,)

        loss = 0
        for i in range(pred.size()[1]):
            loss += loss_fn(pred[:, i], target[:, i]) # output_sizeは1なのでpredの3次元目を0としている
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        if encoder_scheduler is not None and scheduler_step_time=='step':
            encoder_scheduler.step()
            decoder_scheduler.step()

        losses.update(loss.item(), len(data))
        pred = (pred.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy())
        target = (target.detach().cpu().numpy() * meta['std'].unsqueeze(-1).numpy() + meta['mean'].unsqueeze(-1).numpy())
        preds_all += [pred]
        targets_all += [target]
        
        # validとlog、モデルの保存などを行う
        if step % step_valid == 0:
            preds_concat = np.concatenate(preds_all)
            targets_concat = np.concatenate(targets_all)
            train_score = rmse(targets_concat, preds_concat)
            train_loss = losses.avg

            valid_score, valid_loss = valid_function(cfg, fold, epoch, valid_loader, encoder, decoder, loss_fn, device, best_dict, save_path)
            
            wandb_dict = dict(
                epoch = (cfg.num_valids_per_epoch * (epoch - 1)) + (step // step_valid), # wandbでは1回のvalidを1epochとする
                train_score = train_score,
                train_loss = train_loss,
                valid_score = valid_score,
                valid_loss = valid_loss
            )

            if cfg.use_wandb:
                wandb.log(wandb_dict)
            encoder.train()
            decoder.train()
    
    if encoder_scheduler is not None and scheduler_step_time=='epoch':
        encoder_scheduler.step()
        decoder_scheduler.step()


def main():
    exp_path = Path.cwd()
    root_path = Path.cwd().parents[2]
    save_path = root_path / 'outputs' / exp_path.name
    save_path.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(exp_path / 'config')):
        cfg = compose(config_name='config.yaml')
    
    seed_everything(cfg.seed)

    if cfg.use_wandb:
        wandb.login()
    
    df = load_data(cfg, root_path)

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

        device = torch.device(cfg.device)

        encoder = Encoder(cfg.input_size, cfg.hidden_size).to(device)
        decoder = Decoder(cfg.hidden_size, cfg.output_size).to(device)

        if cfg.loss_fn == 'MSELoss':
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.MSELoss()
        
        if cfg.optimizer == 'AdamW':
            encoder_optimizer = torch.optim.AdamW(
                encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            decoder_optimizer = torch.optim.AdamW(
                decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
        if cfg.scheduler == 'OneCycleLR':
            encoder_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                encoder_optimizer, total_steps=cfg.n_epochs * len(train_loader), max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
            decoder_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                decoder_optimizer, total_steps=cfg.n_epochs * len(train_loader), max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
        else:
            encoder_scheduler, decoder_scheduler  = None, None
        scheduler_step_time = cfg.scheduler_step_time

        best_dict = dict(
            score=float('inf'),
            loss=float('inf'),
        )
        
        for epoch in range(1, cfg.n_epochs+1):
            # 学習 & 推論
            train_valid_one_epoch(cfg, fold, epoch, train_loader, valid_loader, encoder, decoder, loss_fn, device, \
                        encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, scheduler_step_time, best_dict, save_path)

        wandb.finish()
        del encoder, decoder, train_fold_df, valid_fold_df, train_loader, valid_loader, loss_fn, encoder_optimizer, decoder_optimizer, best_dict
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()