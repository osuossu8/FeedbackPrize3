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
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle, asMinutes, timeSince


class CFG:
    EXP_ID = '028'
    apex = True
    train = True
    debug = False
    sync_bn = True
    n_gpu = 2
    seed = 2022
    model = 'microsoft/deberta-v3-large'
    n_splits = 5
    max_len = 1536
    targets = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    target_size = len(targets)
    n_accumulate=1
    print_freq = 100
    eval_freq = 780 # * 2
    scheduler = 'cosine'
    batch_size = 1
    num_workers = 0
    lr = 5e-6
    weigth_decay = 0.01
    min_lr=1e-6
    num_warmup_steps = 0
    num_cycles=0.5
    epochs = 4
    n_fold = 5
    trn_fold = [i for i in range(n_fold)]
    freezing = True
    gradient_checkpoint = True
    reinit_layers = 4 # 3
    tokenizer = AutoTokenizer.from_pretrained(model)


LOGGER = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


def setup_tokenizer(CFG):
    CFG.tokenizer.add_tokens([f"\n"], special_tokens=True)
    CFG.tokenizer.add_tokens([f"[START]"], special_tokens=True)
    CFG.tokenizer.add_tokens([f"[END]"], special_tokens=True)


def add_token_preprocess(row):
    res = ["[START]"]
    for sentence in row.split('\n'):
        if sentence == '':
            res.append("[END] \n [START]")
        else:
            res.append(sentence)

    res.append("[END]")

    return ' '.join(res)


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
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable() 

        #if self.cfg.reinit_layers > 0:
        #    layers = self.model.encoder.layer[-self.cfg.reinit_layers:]
        #    for layer in layers:
        #        for module in layer.modules():
        #            self._init_weights(module)

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
    loss_fct = nn.MSELoss()
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


def train_one_epoch(rank, model, optimizer, train_dataset, train_return_list, epoch):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=CFG.n_gpu)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    sampler = DistributedSampler(train_dataset, num_replicas=CFG.n_gpu, rank=rank, shuffle=True, seed=CFG.seed, drop_last=True,)
    dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, sampler=sampler, pin_memory=True, drop_last=True)

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

        if rank == 0:
            if step % CFG.print_freq == 0 or step == (len(dataloader)-1):
                LOGGER.info('Epoch: [{0}][{1}/{2}] '
                            'Elapsed {remain:s} '
                            'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                            .format(epoch+1, step, len(dataloader),
                                    remain=timeSince(start, float(step+1)/len(dataloader)),
                                    loss=losses,
                                    ))

    loss_avg = torch.tensor([losses.avg], device=rank)
    loss_avg = my_gather(loss_avg, rank, CFG.n_gpu)

    if rank == 0:
        loss_avg = torch.cat(loss_avg).mean().item()
    else:
        loss_avg = None
 
    train_return_list.append(loss_avg)


@torch.no_grad()
def valid_one_epoch(rank, model, valid_dataset, valid_return_list, epoch):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=CFG.n_gpu)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(valid_dataset, num_replicas=CFG.n_gpu, rank=rank, shuffle=False, seed=CFG.seed, drop_last=True,)
    dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size * 2, sampler=sampler, pin_memory=True, drop_last=False)

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
        valid_return_list.append([loss_avg, predictions])
    else:
        valid_return_list.append([None, None])


def main(fold):
    manager = mp.Manager()
    train_return_list = manager.list()
    valid_return_list = manager.list()

    LOGGER.info(f'-------------fold:{fold} training-------------')

    train = pd.read_csv('input/train_folds.csv')
    train['full_text'] = train['full_text'].map(add_token_preprocess)

    train_data = train[train.kfold != fold].reset_index(drop=True)
    valid_data = train[train.kfold == fold].reset_index(drop=True)
    valid_labels = valid_data[CFG.targets].values

    trainDataset = FeedBackDataset(train_data, CFG.tokenizer, CFG.max_len)
    validDataset = FeedBackDataset(valid_data, CFG.tokenizer, CFG.max_len)

    model = FeedBackModel(CFG.model)
    torch.save(model.config, OUTPUT_DIR+'config.pth')

    # wrap for DDP
    if CFG.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weigth_decay)

    epoch = 0

    mp.spawn(train_one_epoch,
        args=(model, optimizer, trainDataset, train_return_list, epoch),
        nprocs=CFG.n_gpu,
        join=True)

    mp.spawn(valid_one_epoch,
        args=(model, validDataset, valid_return_list, epoch),
        nprocs=CFG.n_gpu,
        join=True)

    LOGGER.info(train_return_list)
    LOGGER.info(valid_return_list)


if __name__=="__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(CFG.seed)

    setup_tokenizer(CFG)

    OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fold = 0
    main(fold)

