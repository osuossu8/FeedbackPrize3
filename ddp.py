import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle, asMinutes, timeSince


class CFG:
    apex = True
    train = True
    n_gpu = 2
    n_accumulate=1
    scheduler = 'cosine'
    batch_size = 30 # 1
    num_workers = 0
    lr = 5e-6
    min_lr=1e-6
    weigth_decay = 0.01
    num_warmup_steps = 0
    num_cycles=0.5
    epochs = 4
    n_fold = 5
    trn_fold = [i for i in range(n_fold)]


class RandomDataset(Dataset):

    def __init__(self, input_size, output_size, length):
        self.len = length
        self.data = torch.randn(length, input_size)
        self.label = torch.randn(length, output_size)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


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


def train_one_epoch(rank, model, optimizer, train_dataset, return_list):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=CFG.n_gpu)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    sampler = DistributedSampler(train_dataset, num_replicas=CFG.n_gpu, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, sampler=sampler)

    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()

    start = end = time.time()

    for step, (data, labels) in enumerate(train_loader):
        inputs = data.to(rank)
        labels = labels.to(rank)

        batch_size = inputs.size(0)
 
        with autocast(enabled=CFG.apex):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

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

    loss_avg = torch.tensor([losses.avg], device=rank)
    loss_avg = my_gather(loss_avg, rank, CFG.n_gpu)

    if rank == 0:
        loss_avg = torch.cat(loss_avg).mean().item()
    else:
        loss_avg = None
 
    return_list.append(loss_avg)


def main():
    manager = mp.Manager()
    return_list = manager.list()

    input_size = 5
    output_size = 2
    data_size = 100
    train_data = RandomDataset(input_size, output_size, data_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    model = Model(input_size, output_size)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weigth_decay)

    mp.spawn(train_one_epoch,
        args=(model, optimizer, train_data, return_list),
        nprocs=CFG.n_gpu,
        join=True)

    print(return_list)


if __name__=="__main__":
    main()
