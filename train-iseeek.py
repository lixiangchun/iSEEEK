import datetime
import os
import time
import math
from tqdm import tqdm

import torch
import torch.utils.data
from torch import nn

from transformers import (
    BertForMaskedLM,
    AdamW,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset

import utils


class GeneRankingDataset(Dataset):
    def __init__(self, text_file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = self.load_lines(text_file)

    def load_lines(self, text_file):
        f = open(text_file)
        for line in f:
            line = line.strip()
            if line.isspace():
                continue
            a = line.split()
            if len(a) <= max_len:
                lines.append(line)
            else:
                lines.append(" ".join(a[0:self.max_len]))
        f.close()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.tokenizer(self.lines[idx], add_special_tokens=True, truncation=True, padding=True, max_length=self.max_len)

def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        
        for k, v in batch.items():batch[k] = v.to(device, non_blocking=True)
        output = model(**batch)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = batch['input_ids'].shape[0]
        
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['perplexity'].update(loss.exp().item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def evaluate(model, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            
            for k, v in batch.items():batch[k] = v.to(device, non_blocking=True)
            output = model(**batch)
            loss = output.loss
            
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = batch['input_ids'].shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['perplexity'].update(loss.exp().item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Perplexity {top1.global_avg:.3f}'.format(top1=metric_logger.perplexity))
    
    return metric_logger.perplexity.global_avg

def load_data(train_file, valid_file, tokenizer, max_len, distributed):
    # Data loading code
    print("Loading data")
    
    dataset = GeneRankingDataset(train_file, tokenizer, max_len)
    dataset_test = GeneRankingDataset(valid_file, tokenizer, max_len)
    
    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler, tokenizer


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    tokenizer = PreTrainedTokenizerFast.from_pretrained("TJMUCH/transcriptome-iseeek")
    model = BertForMaskedLM.from_pretrained("TJMUCH/transcriptome-iseeek")

    dataset, dataset_test, train_sampler, test_sampler, tokenizer = load_data(
        args.train_file, args.val_file, tokenizer, args.max_len, args.distributed)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=data_collator, 
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, collate_fn=data_collator,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)
        
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)


    num_update_steps_per_epoch = math.ceil(len(data_loader))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_train_steps)


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.dont_load_optim_sche is False:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq)
        evaluate(model, data_loader_test, device=device)
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
            
            model_without_ddp.save_pretrained(args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train-file', help='training set')
    parser.add_argument('--val-file', help='validation set')
    parser.add_argument('--max-len', help='max_len [128]', type=int, default=128)
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--lr', default=1.0e-5, type=float, help='initial learning rate')
    parser.add_argument('--lr-step-size', default=80, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--dont_load_optim_sche', help='do not load the parameters of optimizer and scheduler for resume', action='store_true')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    parser.add_argument("--lr_scheduler_type", type=str,
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        default="linear", help="The scheduler type to use.")


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
