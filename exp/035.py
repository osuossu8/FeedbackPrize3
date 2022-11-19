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
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle, asMinutes, timeSince, trace
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from sklearn.metrics import make_scorer
from joblib import dump, load
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


device = set_device()
if str(device) == 'cpu':
    from sklearn.svm import SVR
else:
    from cuml.svm import SVR
    import cuml


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFG:
    EXP_ID = '035'
    debug = False
    apex = True
    train = True
    sync_bn = True
    n_gpu = 2
    seed = 2022
    #model = 'microsoft/deberta-v3-base' # 'microsoft/deberta-v3-large'
    n_splits = 5
    max_len = 1536
    targets = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    target_size = len(targets)
    n_accumulate=1
    print_freq = 100
    eval_freq = 390 # 780
    scheduler = 'cosine'
    batch_size = 2 #1
    num_workers = 0
    lr = 7e-6 # 5e-7 # 7e-6
    weigth_decay = 0.01
    min_lr= 1e-6 # 1e-7 # 1e-6
    num_warmup_steps = 0
    num_cycles=0.5
    epochs = 5 # 4
    n_fold = 5
    trn_fold = [i for i in range(n_fold)]
    freezing = True
    gradient_checkpoint = True
    reinit_layers = 1
    #tokenizer = AutoTokenizer.from_pretrained(model)


set_seed(CFG.seed)
LOGGER = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")
OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def encode_text(cfg, text):
    if cfg.pretrained:
        inputs = cfg.tokenizer(
            text,
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=cfg.max_len,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
    else:
        inputs = cfg.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            #max_length=CFG.max_len,
            #pad_to_max_length=True,
            #truncation=True
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.max_len

    def __len__(self):
        return len(self.texts)

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
        text = self.texts[index]
        # inputs = self.cut_head_and_tail(text)
        inputs = encode_text(self.cfg, text)
        return inputs


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def init(self, kwargs):
        super().init(kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@torch.no_grad()
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


class Inferencer:
    def __init__(self, input_path=None, cfg=None, inference_weight=1):
        if cfg == None:
            self.cfg = load_config(input_path, inference_weight)
        else:
            self.cfg = cfg
    
    def predict(self, test_loader, device, stat_fn=np.mean):
        preds = []
        start = time.time()
        print('#'*10, cfg.path, '#'*10)
        for fold in self.cfg.trn_fold:
            LOGGER.info(f'Predicting fold {fold}...')
            model = load_model(self.cfg, fold)
            pred = inference_fn(test_loader, model, device)
            preds.append(pred)
            del model, pred; gc.collect()
            torch.cuda.empty_cache()
        end = time.time() - start
        print('#'*10, f'ETA: {end:.2f}s', '#'*10, '\n')
        
        self.preds = stat_fn(preds, axis=0) 
        self.preds = np.clip(self.preds, 1, 5)
        return self.preds
    
    def get_oof_result(self):
        return get_result(pd.read_pickle(os.path.join(cfg.path, 'oof_df.pkl')))

    @torch.no_grad()
    def get_text_embedding(self, data_loader, device, fold=None): 
        # pretrained=True: not fine-tuned models.
        if not self.cfg.pretrained:
            model = load_model(self.cfg, fold, pool=self.cfg.pool_head)            
        else:
            model = AutoModel.from_pretrained(self.cfg.model)
        model.to(device)
        model.eval()
            
        fold_emb = []
        for inputs in tqdm(data_loader, total=len(data_loader)):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            if not self.cfg.pretrained:
                emb = model.feature(**inputs)
            else:
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)
                
                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                emb = mean_pooling(output, attention_mask.detach().cpu())
                emb = F.normalize(emb, p=2, dim=1)
                emb = emb.squeeze(0)
            fold_emb.extend(emb.detach().cpu().numpy())
            del emb; gc.collect(); torch.cuda.empty_cache();
          
        fold_emb = np.array(fold_emb)
        return fold_emb


# model_set=['roberta_base', 'roberta_large', 'deberta_base', 'deberta_large', 'deberta_v2_xlarge', 'deberta_v2_xxlarge', 'deberta_v3_base', 'deberta_v3_large'];   score=0.4499109371935949
# model_set=['deberta_large_mnli', 'roberta_base', 'roberta_large', 'xlnet_base_cased', 'xlnet_large_cased', 'deberta_base', 'deberta_large', 'deberta_xlarge', 'deberta_v2_xlarge', 'deberta_v2_xxlarge', 'deberta_v3_base', 'deberta_v3_large'];   score=0.4499581917973121

MAX_LEN1 = 1024 # 640
MAX_LEN2 = 1024 # 512

##################################################
deberta_base = Config(
    model='microsoft/deberta-base',
    file_name='microsoft_deberta_base_768',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)
deberta_large = Config(
    model='microsoft/deberta-large',
    file_name='microsoft_deberta_large_1024',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)
deberta_xlarge = Config(
    model='microsoft/deberta-xlarge',
    file_name='microsoft_deberta_xlarge_1024',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)
deberta_v2_xlarge = Config(
    model='microsoft/deberta-v2-xlarge',
    file_name='microsoft_deberta_v2_xlarge_1536',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)
deberta_v2_xxlarge = Config(
    model='microsoft/deberta-v2-xxlarge',
    file_name='microsoft_deberta_v2_xxlarge_1536',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)

deberta_v3_base = Config(
    model='microsoft/deberta-v3-base',
    file_name='microsoft_deberta_v3_base_768',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)
deberta_v3_large = Config(
    model='microsoft/deberta-v3-large',
    file_name='microsoft_deberta_v3_large_1024',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)
deberta_large_mnli = Config(
    model='microsoft/deberta-large-mnli',
    file_name='microsoft_deberta_large_mnli_1024',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)

roberta_base = Config(
    model='roberta-base',
    file_name='roberta_base_768',
    pretrained=True, inference_weight=1, max_len=MAX_LEN2)
roberta_large = Config(
    model='roberta-large',
    file_name='roberta_large_1024',
    pretrained=True, inference_weight=1, max_len=MAX_LEN2)

xlnet_base = Config(
    model='xlnet-base-cased',
    file_name='xlnet_base_cased_768',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)
xlnet_large = Config(
    model='xlnet-large-cased',
    file_name='xlnet_large_cased_1024',
    pretrained=True, inference_weight=1, max_len=MAX_LEN1)

##################################################

target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']


BASE_PATH = 'input'
SUBMISSION_PATH = os.path.join(BASE_PATH, 'sample_submission.csv')
TRAIN_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_PATH = os.path.join(BASE_PATH, 'test.csv')

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


svr_folds = 15

skf = MultilabelStratifiedKFold(n_splits=svr_folds, shuffle=True, random_state=42)
for i,(train_index, val_index) in enumerate(skf.split(train,train[target_cols])):
    train.loc[val_index,'fold'] = i

# train = train.head(50)


def get_text_embedding(cfg, dfs):
    cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    infer_ = Inferencer(cfg=cfg, inference_weight=cfg.inference_weight)
    if cfg.model == 'gpt2':
        cfg.tokenizer.pad_token = cfg.tokenizer.eos_token
    text_embs = []
    for df in dfs:
        dataset = TestDataset(cfg, df)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False)

        # Text embedding for SVM
        test_text_emb = []
        if not cfg.pretrained:
            for fold in infer_.cfg.trn_fold:
                test_text_emb.append(infer_.get_text_embedding(loader, device, fold))
            text_emb = np.mean(text_emb, axis=0)
        else:
            text_emb = infer_.get_text_embedding(loader, device)
        text_embs.append(text_emb)
        del dataset, loader; gc.collect(); torch.cuda.empty_cache();
    del infer_; gc.collect(); torch.cuda.empty_cache();
    return text_embs


def learner_cv(features, learner, folds=15, save=False, verbose=False):
    scores = []
    oof_list = []
    id_list = []
    fold_list = []
    for fold in range(folds):
        dftr_ = train[train['fold']!=fold]
        dfev_ = train[train['fold']==fold]

        tr_text_feats = features[list(dftr_.index),:]
        ev_text_feats = features[list(dfev_.index),:]

        clf = MultiOutputRegressor(learner)
        clf.fit(tr_text_feats, dftr_[target_cols].values)
        ev_preds = clf.predict(ev_text_feats)
        
        score, _ = mc_rmse(dfev_[target_cols].values, ev_preds)
        scores.append(score)

        id_list.append(dfev_['text_id'].values)
        oof_list.append(ev_preds)
        fold_list.append(dfev_['fold'].values)
        if verbose:
            LOGGER.info('#'*25)
            LOGGER.info(f'### Fold {fold+1}')
            LOGGER.info("Score: {}".format(score))
        if save:
            dump(clf, f'{OUTPUT_DIR}/svr_{fold}.model')

    oof_df = pd.DataFrame()
    oof_df['text_id'] = np.concatenate(id_list, 0)
    oof_df[['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = np.concatenate(oof_list, 0)
    LOGGER.info(oof_df.shape)
    LOGGER.info(oof_df.columns)
    oof_df.to_csv(f'{OUTPUT_DIR}/svr_oof_df.csv', index=False)
    return np.mean(scores)


def get_learner_score(models_cfg, learner, folds=5, save=False, verbose=False):
    for i, model_cfg in enumerate(models_cfg):
        model_name = model_cfg.model.split('/')[-1].replace('-', '_')
        models_cfg[i].model_name = model_name
        model_file = f'{OUTPUT_DIR}/train_text_emb_{model_cfg.file_name}.npy'
        if 'embedding' in model_cfg:
            continue
        with open(model_file, 'rb') as f:
            models_cfg[i].embedding = np.load(f)   
    embeddings = np.concatenate([model_cfg.embedding for model_cfg in models_cfg], axis=1)
    svr_score = learner_cv(embeddings, learner, folds=folds, save=save, verbose=verbose)
    LOGGER.info('\n')
    LOGGER.info(f'model_set={[m.model_name for m in models_cfg]};   score={svr_score}')
    return svr_score, models_cfg


def mc_rmse(y_true, y_pred):
    scores = []
    ncols = y_true.shape[1]
    
    for n in range(ncols):
        yn_true = y_true[:, n]
        yn_pred = y_pred[:, n]
        rmse_ = mean_squared_error(yn_true, yn_pred, squared=False)
        scores.append(rmse_)
    score = np.mean(scores) 
    return score, scores


pretrained_models_cfg = [
    deberta_large_mnli,
    #gpt2,
    roberta_base,
    roberta_large,
    xlnet_base, 
    xlnet_large,
    deberta_base, 
    deberta_large, 
    deberta_xlarge,
    deberta_v2_xlarge, 
    deberta_v2_xxlarge,
    deberta_v3_base, 
    deberta_v3_large,
]

for cfg in tqdm(pretrained_models_cfg):
    with trace(f'{cfg.model} start.'):
        test_text_emb = get_text_embedding(cfg, [train])[0]
        model_file = f'{OUTPUT_DIR}/train_text_emb_{cfg.file_name}.npy'
        np.save(model_file, test_text_emb)
        del test_text_emb; gc.collect(); torch.cuda.empty_cache();
    LOGGER.info(f'{cfg.model} saved.')

gc.collect(); torch.cuda.empty_cache();


learner = SVR(C=2.0)
svr_score, models_cfg = get_learner_score(pretrained_models_cfg, learner, folds=svr_folds, save=True, verbose=True)

print('done')

