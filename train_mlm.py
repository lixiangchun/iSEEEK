import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms

import math
import os
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    AlbertConfig, AlbertForMaskedLM,
    AdamW,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


import utils

try:
    from apex import amp
except ImportError:
    amp = None

class MaskedLMDataset(Dataset):
    def __init__(self, text_file, tokenizer, max_len, shuffle=True):
        self.tokenizer = tokenizer
        self.lines = self.load_lines(text_file, max_len)
        self.max_len = max_len

        if shuffle:np.random.shuffle(self.lines)

    def load_lines(self, text_file, max_len):
        with open(text_file) as f:
            lines = [
                #line.strip()
                self.truncate_line(line, max_len)
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines

    def truncate_line(self, line, max_len):
        a = line.strip().split()
        if len(a) <= max_len:
            return line
        return " ".join(a[0:max_len]) 

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.tokenizer(self.lines[idx], add_special_tokens=True, truncation=True, max_length=self.max_len)



def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch, print_freq, apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    #for i, batch in metric_logger.log_every(enumerate(data_loader), print_freq, header):
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        
        for k, v in batch.items():batch[k] = v.to(device, non_blocking=True)
        output = model(**batch)
        loss = output.loss
        
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
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


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def build_tokenizer(files, tokenizer_file=None):
    assert type(files) == list and len(files) > 0
    if tokenizer_file and os.path.exists(tokenizer_file):
        print('Tokenizer file `{}` exists, exit!'.format(tokenizer_file))
        return None
    
    if tokenizer_file is None:
        print('WARNING: Tokenizer will not be save given that tokenizer_file is not set')
        
    # Build word-level tokenizer, i.e. tokenize sentences by whitespace.
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)
    
    if tokenizer_file:
        tokenizer.save(tokenizer_file)
    
    return tokenizer
    
def load_tokenizer(tokenizer_file, max_len, tokenizer=None):
    # Data loading code
    print("Loading tokenizer")
    
    if tokenizer is None:
        tokenizer = Tokenizer.from_file(tokenizer_file)
        
    # We might want our tokenizer to automatically add special tokens, like "[CLS]" or "[SEP]".
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[PAD]", tokenizer.token_to_id("[PAD]")),
            ("[MASK]", tokenizer.token_to_id("[MASK]"))
            ],
        )
    
    # Instantiate with a tokenizer object
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, model_max_length=max_len,
        mask_token='[MASK]', cls_token='[CLS]', sep_token='[SEP]', 
        pad_token='[PAD]', unk_token='[UNK]')
    
    return tokenizer

def load_data(traindir, valdir, tokenizer_file, max_len, distributed):
    # Data loading code
    print("Loading data")
    
    tokenizer = load_tokenizer(tokenizer_file, max_len)
    
    dataset = MaskedLMDataset(traindir, tokenizer, max_len)
    dataset_test = MaskedLMDataset(valdir, tokenizer, max_len)
    
    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler, tokenizer


def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    #train_dir = os.path.join(args.data_path, 'train')
    #val_dir = os.path.join(args.data_path, 'val')
    train_dir = args.train_file
    val_dir = args.val_file
    dataset, dataset_test, train_sampler, test_sampler, tokenizer = load_data(
        train_dir, val_dir, args.dict, args.max_len, args.distributed)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=data_collator, 
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, collate_fn=data_collator,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    
    config = BertConfig(
        vocab_size=tokenizer.vocab_size, intermediate_size=1536, num_attention_heads=8, hidden_size=576,
        num_hidden_layers=8, max_position_embeddings=384, pad_token_id=tokenizer.pad_token_id)

    if args.bert_config:
        config = BertConfig.from_json_file(args.bert_config)
    if args.single_layer_bert:
        config = BertConfig(vocab_size=tokenizer.vocab_size, num_hidden_layers=1, pad_token_id=tokenizer.pad_token_id)
    
    model = BertForMaskedLM(config)
        
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

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    ##lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
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
        train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq, args.apex)
        ##lr_scheduler.step()
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
    parser.add_argument('--num-classes', help='number of classes for the objective task', type=int)
    parser.add_argument('--max-len', help='max_len [128]', type=int, default=128)
    parser.add_argument('--dict', help='dictionary file', type=str)
    
    #parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=1.0e-5, type=float, help='initial learning rate')
    #parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
     #                   help='momentum')
    #parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
     #                   metavar='W', help='weight decay (default: 1e-4)',
      #                  dest='weight_decay')
    parser.add_argument('--lr-step-size', default=80, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--dont_load_optim_sche', help='do not load the parameters of optimizer and scheduler for resume', action='store_true')
    parser.add_argument('--bert_config', default=None, help='bert config json file')
    parser.add_argument('--single_layer_bert', help='built a single-layer bert', action='store_true')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
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
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    parser.add_argument("--lr_scheduler_type", type=str,
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        default="linear", help="The scheduler type to use.")


    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

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
