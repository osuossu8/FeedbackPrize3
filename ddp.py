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


from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle, asMinutes, timeSince


class CFG:
    apex = True
    n_accumulate=1
    lr = 5e-6
    weigth_decay = 0.01


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


def train_one_epoch(rank, n_gpu, input_size, output_size, batch_size, train_dataset):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=n_gpu)
    # create local model
    model = Model(input_size, output_size).to(rank)
    # construct DDP model
    model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weigth_decay)
    sampler = DistributedSampler(train_dataset, num_replicas=n_gpu, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()

    start = end = time.time()

    for step, (data, labels) in enumerate(train_loader):
        inputs = data.to(rank)
        labels = labels.to(rank)

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

        end = time.time()

    loss_avg = torch.tensor([losses.avg], device=rank)
    print(loss_avg)
    loss_avg = my_gather(loss_avg, rank, n_gpu)

    if rank == 0:
        loss_avg = torch.cat(loss_avg).mean().item()
    else:
        loss_avg = None
 
    print(loss_avg)
    return loss_avg


def main():
    n_gpu = 2
    input_size = 5
    output_size = 2
    batch_size = 30
    data_size = 100
    dataset = RandomDataset(input_size, output_size, data_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train_one_epoch,
        args=(n_gpu, input_size, output_size, batch_size, dataset),
        nprocs=n_gpu,
        join=True)



if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "29500"
    main()
