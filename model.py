import torch
import torch.nn as nn

import math


class RNNLM(nn.Module):
    def __init__(self, config):
        super(RNNLM, self).__init__()
        vocabSize = config.data.vocabSize
        nemd = config.model.rnn.nemd
        drop_ratio = config.model.rnn.drop_ratio
        nhid = config.model.rnn.nhid
        nlayer = config.model.rnn.nlayer
        activation = config.model.rnn.activation
        tie_weight = config.model.rnn.tie_weight
        self.drop_emd = config.model.rnn.drop_emd
        
        self.embedding = nn.Embedding(vocabSize, nemd)
        self.dropout = nn.Dropout(drop_ratio)
        self.rnn = nn.LSTM(nemd, nhid, nlayer, dropout=drop_ratio, batch_first=False)
        self.out = nn.Linear(nhid, vocabSize)
        
        # if not pretrained embedding
        self.init_weights()
        
        if tie_weight:
            if nemd!=nhid:
                raise ValueError('When using the tied flag, nemd must be equal to nhid')
            self.out.weight = self.embedding.weight
            
       
    # init hidden?
    def init_weights(self):
        # init the embedding and softmax layers
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.out.weight, -initrange, initrange)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, inputs, hidden=None):
        embeded = self.embedding(inputs)
        if self.drop_emd:
            embeded = self.dropout(embeded)
        outputs, hidden = self.rnn(embeded, hidden)
        # ?
        outputs = self.dropout(outputs)
        outputs = self.out(outputs)
        return outputs
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        max_seq_len = 5000
        nemd = config.model.transformer.nemd
         
        pe = torch.zeros(max_seq_len, nemd)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000)*torch.arange(0, nemd, 2).float()/nemd)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        # ? 自动识别device？
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[0:x.size(0),:]

    
class TransformerLM(nn.Module):
    def __init__(self, config):
        super(TransformerLM, self).__init__()
        vocabSize = config.data.vocabSize
        self.nemd = config.model.transformer.nemd
        drop_ratio = config.model.transformer.drop_ratio
        self.drop_emd = config.model.transformer.drop_emd
        nhead = config.model.transformer.nhead
        nhid = config.model.transformer.nhid
        nlayer = config.model.transformer.nlayer
        self.src_mask=None
        tie_weight = config.model.transformer.tie_weight
        
        self.embedding = nn.Embedding(vocabSize, nemd)
        self.pos_encoder = PositionalEncoding(config)
        self.dropout = nn.Dropout(drop_ratio)
        encoder_layers = nn.TransformerEncoderLayer(self.nemd, nhead, nhid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayer)
        self.out = nn.Linear(nemd, vocabSize)
        
        # if not pretrained embedding
        self.init_weights()
        
        if tie_weight:
            self.out.weight = self.embedding.weight
        
    def init_weights(self):
        # init the embedding and softmax layers
        initrange = 0.1
        nn.init_uniform_(self.embedding.weight, -initrange, initrange)
        nn.init_uniform_(self.out.weight, -initrange, initrange)
        nn.init_zeros_(self.out.bias)
        
    def _generate_square_subsequent_mask(self, src):
        seq_len = src.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 0, float(0.0)).masked_fill(mask==1, float('-inf'))
        return mask
    
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            self.src_mask = self._generate_square_subsequent_mask(src).to(device)
        embeded = self.embedding(src)*math.sqrt(self.nemd)
        embeded = pos_encoder(embeded)
        if self.drop_emd:
            embeded = self.dropout(embeded)
        # src_key_padding_mask ?
        output = self.transformer_encoder(embeded, mask=self.src_mask)
        output = self.out(output)
        return output
        
            
    
    
    
    
    
    