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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle, asMinutes, timeSince


# https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x

class CFG:
    EXP_ID = '043'
    debug = False # True
    apex = True
    train = True
    sync_bn = True
    n_gpu = 2
    seed = 42 # 77 # 38 # 2022
    model = 'microsoft/deberta-v3-large'
    n_splits = 5
    max_len = 1536
    targets = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    target_size = len(targets)
    n_accumulate=1
    print_freq = 100
    eval_freq = 732 # 780 # * 2
    scheduler = 'linear' # 'cosine'
    batch_size = 1
    num_workers = 0
    lr = 5e-6
    weigth_decay = 0.01
    min_lr = 1e-6
    num_warmup_steps = 0
    num_cycles=0.5
    epochs = 4
    n_fold = 4 # 5
    trn_fold = [i for i in range(n_fold)]
    freezing = True
    gradient_checkpoint = False
    reinit_layers = 0
    tokenizer = AutoTokenizer.from_pretrained(model)


LOGGER = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")
OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


dftr = pd.read_csv("input/train.csv")
dftr["src"]="train"
dfte = pd.read_csv("input/test.csv")
dfte["src"]="test"
print('Train shape:',dftr.shape,'Test shape:',dfte.shape,'Test columns:',dfte.columns)
df = pd.concat([dftr,dfte],ignore_index=True)


target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]


import sys
#sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
FOLDS = 25
skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
for i,(train_index, val_index) in enumerate(skf.split(dftr,dftr[target_cols])):
    dftr.loc[val_index,'FOLD'] = i
LOGGER.info('Train samples per fold:')
LOGGER.info(dftr.FOLD.value_counts())

from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


BATCH_SIZE = 2 # 4

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        text = self.df.loc[idx,"full_text"]
        tokens = tokenizer(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN,return_tensors="pt")
        tokens = {k:v.squeeze(0) for k,v in tokens.items()}
        return tokens

ds_tr = EmbedDataset(dftr)
embed_dataloader_tr = torch.utils.data.DataLoader(ds_tr,\
                        batch_size=BATCH_SIZE,\
                        shuffle=False)
ds_te = EmbedDataset(dfte)
embed_dataloader_te = torch.utils.data.DataLoader(ds_te,\
                        batch_size=BATCH_SIZE,\
                        shuffle=False)


tokenizer = None
MAX_LEN = 1024 # 640

def get_embeddings(MODEL_NM='', MAX=640, BATCH_SIZE=4, verbose=True):
    global tokenizer, MAX_LEN
    DEVICE="cuda"
    model = AutoModel.from_pretrained( MODEL_NM )
    tokenizer = AutoTokenizer.from_pretrained( MODEL_NM )
    MAX_LEN = MAX

    model = model.to(DEVICE)
    model.eval()
    all_train_text_feats = []
    for batch in tqdm(embed_dataloader_tr,total=len(embed_dataloader_tr)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        all_train_text_feats.extend(sentence_embeddings)
    all_train_text_feats = np.array(all_train_text_feats)
    if verbose:
        print('Train embeddings shape',all_train_text_feats.shape)

    te_text_feats = []
    for batch in tqdm(embed_dataloader_te,total=len(embed_dataloader_te)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        te_text_feats.extend(sentence_embeddings)
    te_text_feats = np.array(te_text_feats)
    if verbose:
        print('Test embeddings shape',te_text_feats.shape)

    return all_train_text_feats, te_text_feats


MODEL_NM = 'microsoft/deberta-base'
#MODEL_NM = '../input/huggingface-deberta-variants/deberta-base/deberta-base'
all_train_text_feats, te_text_feats = get_embeddings(MODEL_NM)
to_pickle(OUTPUT_DIR+f'{MODEL_NM.replace("/", "-")}_train_text_feats.pkl', all_train_text_feats)

MODEL_NM = 'microsoft/deberta-v3-large'
#MODEL_NM = '../input/deberta-v3-large/deberta-v3-large'
all_train_text_feats2, te_text_feats2 = get_embeddings(MODEL_NM)
to_pickle(OUTPUT_DIR+f'{MODEL_NM.replace("/", "-")}_train_text_feats.pkl', all_train_text_feats2)

MODEL_NM = 'microsoft/deberta-large'
#MODEL_NM = '../input/huggingface-deberta-variants/deberta-large/deberta-large'
all_train_text_feats3, te_text_feats3 = get_embeddings(MODEL_NM)
to_pickle(OUTPUT_DIR+f'{MODEL_NM.replace("/", "-")}_train_text_feats.pkl', all_train_text_feats3)

MODEL_NM = 'microsoft/deberta-large-mnli'
#MODEL_NM = '../input/huggingface-deberta-variants/deberta-large-mnli/deberta-large-mnli'
all_train_text_feats4, te_text_feats4 = get_embeddings(MODEL_NM, MAX=512)
to_pickle(OUTPUT_DIR+f'{MODEL_NM.replace("/", "-")}_train_text_feats.pkl', all_train_text_feats4)

MODEL_NM = 'microsoft/deberta-xlarge'
#MODEL_NM = '../input/huggingface-deberta-variants/deberta-xlarge/deberta-xlarge'
all_train_text_feats5, te_text_feats5 = get_embeddings(MODEL_NM, MAX=512)
to_pickle(OUTPUT_DIR+f'{MODEL_NM.replace("/", "-")}_train_text_feats.pkl', all_train_text_feats5)

all_train_text_feats = np.concatenate([all_train_text_feats,all_train_text_feats2,
                                       all_train_text_feats3,all_train_text_feats4,
                                       all_train_text_feats5],axis=1)

te_text_feats = np.concatenate([te_text_feats,te_text_feats2,
                                te_text_feats3,te_text_feats4,
                                te_text_feats5],axis=1)

del all_train_text_feats2, te_text_feats2
del all_train_text_feats3, te_text_feats3
del all_train_text_feats4, te_text_feats4
del all_train_text_feats5, te_text_feats5
gc.collect()

print('Our concatenated embeddings have shape', all_train_text_feats.shape )

from cuml.svm import SVR
import cuml
print('RAPIDS version',cuml.__version__)


from sklearn.metrics import mean_squared_error

preds = []
scores = []
def comp_score(y_true,y_pred):
    rmse_scores = []
    for i in range(len(target_cols)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    return np.mean(rmse_scores)


oof_list = []
id_list = []
fold_list = []
#for fold in tqdm(range(FOLDS),total=FOLDS):
for fold in range(FOLDS):
    print('#'*25)
    print('### Fold',fold+1)
    print('#'*25)

    dftr_ = dftr[dftr["FOLD"]!=fold]
    dfev_ = dftr[dftr["FOLD"]==fold]

    tr_text_feats = all_train_text_feats[list(dftr_.index),:]
    ev_text_feats = all_train_text_feats[list(dfev_.index),:]

    ev_preds = np.zeros((len(ev_text_feats),6))
    test_preds = np.zeros((len(te_text_feats),6))
    for i,t in enumerate(target_cols):
        print(t,', ',end='')
        clf = SVR(C=1)
        clf.fit(tr_text_feats, dftr_[t].values)
        ev_preds[:,i] = clf.predict(ev_text_feats)
        test_preds[:,i] = clf.predict(te_text_feats)
    print()
    score = comp_score(dfev_[target_cols].values,ev_preds)
    scores.append(score)

    id_list.append(dfev_['text_id'].values)
    oof_list.append(ev_preds)
    fold_list.append(dfev_['FOLD'].values)

    print("Fold : {} RSME score: {}".format(fold,score))
    preds.append(test_preds)

print('#'*25)
print('Overall CV RSME =',np.mean(scores))

oof_df = pd.DataFrame()
oof_df['text_id'] = np.concatenate(id_list, 0)
oof_df[['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = np.concatenate(oof_list, 0)
print(oof_df.shape)
print(oof_df.columns)
oof_df.to_csv(OUTPUT_DIR+f'svr_chris_oof_df.csv', index=False)



