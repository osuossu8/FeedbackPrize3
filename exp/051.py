import os
import gc
import re
import sys
sys.path.append("/root/workspace/FeedbackPrize3")
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

from sklearn import metrics
from sklearn.metrics import mean_squared_error


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle, asMinutes, timeSince


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFG:
    EXP_ID = '051'
    debug = False
    apex = True
    train = True
    sync_bn = True
    n_gpu = 2
    seed = 55
    model = 'microsoft/deberta-v3-large'
    max_len = 512 # 640
    targets = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    target_size = len(targets)
    n_accumulate=1
    print_freq = 100
    eval_freq = 230 # 390 # 780
    scheduler = 'cosine'
    batch_size = 4 #1
    num_workers = 0
    lr = 7e-6
    weigth_decay = 0.01
    min_lr= 1e-6
    num_warmup_steps = 0
    num_cycles=0.5
    epochs = 4
    n_fold = 20
    trn_fold = [i for i in range(n_fold)]
    freezing = True
    gradient_checkpoint = True
    reinit_layers = 1
    tokenizer = AutoTokenizer.from_pretrained(model)


LOGGER = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")
OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def setup_tokenizer(CFG):
    CFG.tokenizer.add_tokens([f"\n"], special_tokens=True)
    CFG.tokenizer.add_tokens([f"\r"], special_tokens=True)


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = CFG.max_len
        self.text = df['full_text'].values
        self.tokenizer = CFG.tokenizer
        self.targets = df[CFG.targets].values

    def __len__(self):
        return len(self.df)

    # staticmethod に書き換えたい
    def cut_head_and_tail(self, text):
        input_ids = self.tokenizer.encode(text)
        n_token = len(input_ids)

        if n_token == self.max_len:
            input_ids = input_ids
            attention_mask = [1 for _ in range(self.max_len)]
            token_type_ids = [1 for _ in range(self.max_len)]
        elif n_token < self.max_len:
            pad = [1 for _ in range(self.max_len-n_token)]
            input_ids = input_ids + pad
            attention_mask = [1 if n_token > i else 0 for i in range(self.max_len)]
            token_type_ids = [1 if n_token > i else 0 for i in range(self.max_len)]
        else:
            harf_len = (self.max_len-2)//2
            _input_ids = input_ids[1:-1]
            input_ids = [0]+ _input_ids[:harf_len] + _input_ids[-harf_len:] + [2]
            attention_mask = [1 for _ in range(self.max_len)]
            token_type_ids = [1 for _ in range(self.max_len)]

            if len(input_ids) < self.max_len:
                diff = self.max_len - len(input_ids)
                input_ids = [0]+ _input_ids[:harf_len] + _input_ids[-(harf_len+diff):] + [2]

        d = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }
        return d

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.cut_head_and_tail(text)
        labels = torch.tensor(self.targets[index], dtype=torch.float)
        return inputs, labels


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


def freeze(module):
    """
    Freezes module's parameters.
    """

    for parameter in module.parameters():
        parameter.requires_grad = False


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()

        self.cfg = CFG
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0

        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        self.output = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.cfg.target_size)
        )

        # Freeze
        if self.cfg.freezing:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:2])

        # Gradient Checkpointing
        # if self.cfg.gradient_checkpoint:
        #     self.model.gradient_checkpointing_enable() 

        if self.cfg.reinit_layers > 0:
            layers = self.model.encoder.layer[-self.cfg.reinit_layers:]
            for layer in layers:
                for module in layer.modules():
                    self._init_weights(module)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.model(ids, mask, token_type_ids)
        else:
            transformer_out = self.model(ids, mask)

        # simple CLS
        sequence_output = transformer_out[0][:, 0, :]

        logits = self.output(sequence_output)

        return logits



# CPMP utility to gather values from all workers
def my_gather(output, local_rank, world_size):
    # output must be a tensor on the cuda device
    # output must have the same size in all workers
    result = None
    if local_rank == 0:
        result = [torch.empty_like(output) for _ in range(world_size)]
    torch.distributed.gather(output, gather_list=result, dst=0)
    return result


def criterion(outputs, targets):
    loss_fct = nn.MSELoss() # nn.SmoothL1Loss(reduction='mean') # nn.MSELoss()
    loss = loss_fct(outputs, targets)
    return loss


def get_score(outputs, targets):
    mcrmse = []
    for i in range(CFG.target_size):
        mcrmse.append(
            metrics.mean_squared_error(
                targets[:, i],
                outputs[:, i],
                squared=False,
            ),
        )
    mcrmse = np.mean(mcrmse)
    return mcrmse


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler


def train_one_epoch(rank, model, optimizer, scheduler, dataloader, valid_dataloader, epoch, fold, best_score, valid_dataset, valid_labels):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()

    start = end = time.time()

    for step, (data, targets) in enumerate(dataloader):
        data = collate(data)
        ids = data['input_ids'].to(rank, dtype=torch.long)
        mask = data['attention_mask'].to(rank, dtype=torch.long)
        targets = targets.to(rank, dtype=torch.float)

        batch_size = ids.size(0)
 
        with autocast(enabled=CFG.apex):
            outputs = model(ids, mask)
            loss = criterion(outputs, targets)

        loss = loss / CFG.n_accumulate
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        if (step +1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        end = time.time()

        epoch_loss = torch.tensor([losses.avg], device=rank)
        epoch_loss = my_gather(epoch_loss, rank, CFG.n_gpu)

        if rank == 0:
            epoch_loss = torch.cat(epoch_loss).mean().item()

            if step % CFG.print_freq == 0 or step == (len(dataloader)-1):
                LOGGER.info('Epoch: [{0}][{1}/{2}] '
                            'Elapsed {remain:s} '
                            'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                            .format(epoch+1, step, len(dataloader),
                                    remain=timeSince(start, float(step+1)/len(dataloader)),
                                    loss=losses,
                                    ))

        else:
            epoch_loss = None

        if (step > 0) & (step % CFG.eval_freq == 0):

            valid_epoch_loss, pred = valid_one_epoch(rank, model, valid_dataloader, valid_dataset, epoch)

            if rank == 0:
                score = get_score(pred, valid_labels)
                LOGGER.info(f'Epoch {epoch+1} Step {step} - avg_train_loss: {epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}')
                LOGGER.info(f'Epoch {epoch+1} Step {step} - Score: {score:.4f}')

                if score < best_score:
                    best_score = score
                    LOGGER.info(f'Epoch {epoch+1} Step {step} - Save Best Score: {best_score:.4f} Model')
                    torch.save({'model': model.state_dict(),
                                'predictions': pred},
                                OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

                # model.train()

    gc.collect()

    return epoch_loss, valid_epoch_loss, pred, best_score


@torch.no_grad()
def valid_one_epoch(rank, model, dataloader, valid_dataset, epoch):
    model.eval()
    losses = AverageMeter()

    start = end = time.time()
    preds = []

    for step, (data, targets) in enumerate(dataloader):
        data = collate(data)
        ids = data['input_ids'].to(rank, dtype=torch.long)
        mask = data['attention_mask'].to(rank, dtype=torch.long)
        targets = targets.to(rank, dtype=torch.float)

        batch_size = ids.size(0)
        outputs = model(ids, mask)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), batch_size)
        preds.append(outputs.detach()) # keep preds on GPU

        end = time.time()

        if rank == 0:
            if step % CFG.print_freq == 0 or step == (len(dataloader)-1):
                LOGGER.info('EVAL: [{0}/{1}] '
                            'Elapsed {remain:s} '
                            'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                            .format(step, len(dataloader),
                                    remain=timeSince(start, float(step+1)/len(dataloader)),
                                    loss=losses,
                                    ))

    predictions = torch.cat(preds)

    loss_avg = torch.tensor([losses.avg], device=rank)
    loss_avg = my_gather(loss_avg, rank, CFG.n_gpu)
    predictions = my_gather(predictions, rank, CFG.n_gpu)
 
    if rank == 0:
        loss_avg = torch.cat(loss_avg).mean().item()
        predictions = torch.stack(predictions)
        _, _, t = predictions.shape
        predictions = predictions.transpose(0, 1).reshape(-1, t) 
        # DistributedSampler pads the dataset to get a multiple of world size
        predictions = predictions[:len(valid_dataset)]
        predictions = predictions.cpu().numpy()
        return loss_avg, predictions
    else:
        return None, None


def train_loop(rank, CFG, fold, return_dict):

    LOGGER.info(f"Running basic DDP example on rank {rank}.")
    setup_ddp(rank, CFG.n_gpu)

    set_seed(CFG.seed)

    if rank == 0:
        LOGGER.info(f'-------------fold:{fold} training-------------')

    train = pd.read_csv('input/train_folds.csv')

    skf = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for i,(train_index, val_index) in enumerate(skf.split(train,train[CFG.targets])):
        train.loc[val_index,'kfold'] = i

    if CFG.debug:
        train = train.sample(n=128)
        CFG.print_freq = 8
        CFG.eval_freq = 32
        CFG.epochs = 1

    train_data = train[train.kfold != fold].reset_index(drop=True)
    valid_data = train[train.kfold == fold].reset_index(drop=True)
    valid_labels = valid_data[CFG.targets].values

    train_dataset = FeedBackDataset(train_data, CFG.tokenizer, CFG.max_len)
    valid_dataset = FeedBackDataset(valid_data, CFG.tokenizer, CFG.max_len)

    train_sampler = DistributedSampler(train_dataset, num_replicas=CFG.n_gpu, rank=rank, shuffle=True, seed=CFG.seed, drop_last=True,)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, sampler=train_sampler, pin_memory=True, drop_last=True)

    valid_sampler = DistributedSampler(valid_dataset, num_replicas=CFG.n_gpu, rank=rank, shuffle=False, seed=CFG.seed, drop_last=False,)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size * 2, sampler=valid_sampler, pin_memory=True, drop_last=False)

    model = FeedBackModel(CFG.model)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # wrap for DDP
    if CFG.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weigth_decay)
    num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # loop
    best_score = 100

    for epoch in range(CFG.epochs):

        start_time = time.time()

        train_epoch_loss, valid_epoch_loss, pred, best_score = train_one_epoch(rank, model, optimizer, scheduler, train_dataloader, valid_dataloader, epoch, fold, best_score, valid_dataset, valid_labels)

        if rank == 0:
            score = get_score(pred, valid_labels)

            elapsed = time.time() - start_time

            LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}  time: {elapsed:.0f}s')
            LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

            if score < best_score:
                best_score = score
                LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'predictions': pred},
                            OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    if rank == 0:
        predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                                 map_location=torch.device('cpu'))['predictions']
        valid_data['pred_0'] = predictions[:, 0]
        valid_data['pred_1'] = predictions[:, 1]
        valid_data['pred_2'] = predictions[:, 2]
        valid_data['pred_3'] = predictions[:, 3]
        valid_data['pred_4'] = predictions[:, 4]
        valid_data['pred_5'] = predictions[:, 5]

        torch.cuda.empty_cache()
        gc.collect()

        return_dict[fold] = valid_data

        cleanup_ddp()


def get_result(oof_df):
    labels = oof_df[CFG.targets].values
    preds = oof_df[['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values
    score = get_score(preds, labels)
    LOGGER.info(f'Score: {score:<.4f}')


def main():
    manager = mp.Manager()
    return_dict = manager.dict()

    setup_tokenizer(CFG)

    if torch.cuda.device_count() > 1:
        LOGGER.info(f"We have available {torch.cuda.device_count()}, GPUs! but using {CFG.n_gpu} GPUs")

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            mp.spawn(train_loop, args=(CFG, fold, return_dict), nprocs=CFG.n_gpu, join=True)
            _oof_df = return_dict[fold]
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)

        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_csv(OUTPUT_DIR+f'oof_df.csv', index=False)


if __name__=="__main__":
    main()

