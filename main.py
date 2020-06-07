import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import WikiText2
from torchtext.data import Field, BPTTIterator
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import yaml
import math

import model
from utils import AttrDict, init_logger



parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:2")
parser.add_argument('--resume_training', action='store_true', 
                    help='if action, resume training')
parser.add_argument('--seed', type=int, default=1111)
args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

def train(net, trainLoader, criterion, optimizer, config, logger):
    net.train()
    running_loss = 0.0
    for batchIdx, batch in enumerate(trainLoader):
        inputs = batch.text
        targets = batch.target
        
        optimizer.zero_grad()
        outs, raw_output, dropped_output = net(inputs)
        loss = criterion(outs.view(-1, outs.size(-1)), targets.view(-1))
        
        # Activiation Regularization
        if config.model.rnn.alpha:
            loss = loss + config.model.rnn.alpha*dropped_output.pow(2).mean()
        # Temporal Activation Regularization (slowness)
        if config.model.rnn.beta:
            loss = loss + config.model.rnn.beta*(raw_output[1:]-raw_output[:-1]).pow(2).mean()
        
        loss.backward()
#         if config.training.model_type == 'rnn':
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), config.training.grad_clip)
        optimizer.step()
        running_loss += loss.item()
        
        N = len(trainLoader)//10
        if batchIdx % N == N-1:
            running_loss /= N
            logger.info(f'| epoch {epoch:3d} | {batchIdx:4d}/{len(trainLoader):4d} batches | '
                          f'loss {running_loss:5.4f} | ppl {math.exp(running_loss):5.2f}')
            running_loss = 0.0
        
        
def evaluate(net, devLoader, criterion):
    net.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batchIdx, batch in enumerate(devLoader):
            inputs = batch.text
            targets = batch.target

            outs, _, _ = net(inputs)
            loss = criterion(outs.view(-1, outs.size(-1)), targets.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(devLoader)
        
        
        
###############################################################################
# Load data
###############################################################################
configfile = open('./config.yaml')
config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
device = torch.device(args.device)

# ? include lenghts 
TEXT = Field(lower=True, include_lengths=False, batch_first=False)
# TEXT： split string into tokens
trainSet, devSet, testSet = WikiText2.splits(
    text_field=TEXT, root=config.data.data_root) 
TEXT.build_vocab(trainSet)
# TEXT: numericalize, pad, add init_token and eos_token
trainLoader, devLoader, testLoader = BPTTIterator.splits(
    (trainSet, devSet, testSet), batch_size=config.data.BSZ, 
    bptt_len = config.data.bptt_len, device=device)
assert len(TEXT.vocab)==config.data.vocabSize


###############################################################################
# Define model
###############################################################################
if config.training.model_type == 'rnn':
    net = model.RNNLM(config).to(device)
elif config.training.model_type == 'transformer':
    net = model.TransformerLM(config).to(device)
if config.training.optim_type == 'sgd':
    optimizer = optim.SGD(
        net.parameters(), lr=config.training.lr, 
        weight_decay=config.training.weight_decay, 
        momentum=config.training.momentum)
elif config.training.optim_type == 'adam':
    optimizer = optim.Adam(
        net.parameters(), lr=config.training.lr, 
        weight_decay=config.training.weight_decay)
elif config.training.optim_type == 'asgd':
    optimizer = optim.ASGD(
        net.parameters(), lr=config.training.lr, t0=0, lambd=0., 
        weight_decay=config.training.weight_decay)
criterion = nn.CrossEntropyLoss()

save_root = config.data.save_root
save_root = os.path.join(save_root, 'model_1')
if not os.path.exists(save_root):
    os.mkdir(save_root)
# writer_path = os.path.join(save_root, 'writer')
logger_path = os.path.join(save_root, 'lm.log')
ckpt_path = os.path.join(save_root, 'lm.pth')
# writer = SummaryWriter(writer_path)
logger = init_logger(logger_path)


###############################################################################
# Training code
###############################################################################
if args.resume_training:
    ckpt = torch.load(ckpt_path)
    start_epoch = ckpt['epoch']+1
    best_dev_loss = ckpt['best_dev_loss']
    net.load_state_dict(ckpt['net_state_dict'])
    logger.info(
        f'resume training from epoch {start_epoch} with best_dev_ppl {best_dev_ppl:5.2f}')
else:
    start_epoch = 0
    best_dev_loss = float('inf')
    logger.info(config)
    logger.info('start training ...')

for epoch in range(40):
    train(net, trainLoader, criterion, optimizer,  config, logger)
    train_loss = evaluate(net, trainLoader, criterion)
    dev_loss = evaluate(net, devLoader, criterion)
    logger.info('-' * 70)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info(f'| end of epoch {epoch:3d} | lr {cur_lr:2.3e} | '
                  f'train ppl {math.exp(train_loss):5.2f} | dev ppl {math.exp(dev_loss):5.2f}')
    logger.info('-' * 70)
    
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        torch.save({'epoch':epoch, 'best_dev_loss':best_dev_loss, 
                    'net_state_dict':net.state_dict()}, ckpt_path)
        logger.info(f'model saved with best dev loss {best_dev_loss:5.4f}')
    else:
        optimizer.param_groups[0]['lr'] /= config.training.lr_decay_factor
    







