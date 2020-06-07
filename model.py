import torch
import torch.nn as nn
import math

from locked_dropout import LockedDropout
from embed_regularize import embedded_dropout


class RNNLM(nn.Module):
    def __init__(self, config):
        super(RNNLM, self).__init__()
        vocabSize = config.data.vocabSize
        nemd = config.model.rnn.nemd
        nhid = config.model.rnn.nhid
        self.nlayer = config.model.rnn.nlayer
        tie_weight = config.model.rnn.tie_weight
        rnn_type = config.model.rnn.rnn_type
        self.embed_drop_ratio = config.model.rnn.embed_drop_ratio
        self.locked_drope = config.model.rnn.locked_drope
        self.locked_droph = config.model.rnn.locked_droph
        self.locked_dropo = config.model.rnn.locked_dropo
        
        self.embedding = nn.Embedding(vocabSize, nemd)
        self.lockdrop = LockedDropout()
        rnns = [getattr(nn, rnn_type)(
            nemd if l==0 else nhid, nhid if l!=self.nlayer-1 else nemd, 
            dropout=0, batch_first=False) for l in range(self.nlayer)]
        self.rnns = nn.ModuleList(rnns)
        self.out = nn.Linear(nemd, vocabSize)
        
        # if not pretrained embedding
        self.init_weights()
        
        if tie_weight:
            self.out.weight = self.embedding.weight
            
    def init_weights(self):
        # init the embedding and softmax layers
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.out.weight, -initrange, initrange)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, inputs):
        embedded = embedded_dropout(
            self.embedding, inputs, dropout=self.embed_drop_ratio if self.training else 0)
        embedded = self.lockdrop(embedded, self.locked_drope)
        raw_output = embedded
        for l, rnn in enumerate(self.rnns):
            raw_output, _ = rnn(raw_output)
            if l != self.nlayer-1:
                raw_output = self.lockdrop(raw_output, self.locked_droph)
        outputs = self.lockdrop(raw_output, self.locked_dropo)
        dropped_output = outputs
        outputs = self.out(outputs)
        return outputs, raw_output, dropped_output
    
   
    
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
        return x

    
    
class TransformerLM(nn.Module):
    def __init__(self, config):
        super(TransformerLM, self).__init__()
        vocabSize = config.data.vocabSize
        self.nemd = config.model.transformer.nemd
        emd_drop_ratio = config.model.transformer.emd_drop_ratio
        hidden_drop_ratio = config.model.transformer.hidden_drop_ratio
        nhead = config.model.transformer.nhead
        nhid = config.model.transformer.nhid
        nlayer = config.model.transformer.nlayer
        self.src_mask=None
        tie_weight = config.model.transformer.tie_weight
        
        self.embedding = nn.Embedding(vocabSize, self.nemd)
        self.pos_encoder = PositionalEncoding(config)
        self.dropout = nn.Dropout(emd_drop_ratio)
        encoder_layers = nn.TransformerEncoderLayer(self.nemd, nhead, nhid, hidden_drop_ratio)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayer)
        self.out = nn.Linear(self.nemd, vocabSize)
        
        # if not pretrained embedding
        self.init_weights()
        
        if tie_weight:
            self.out.weight = self.embedding.weight
        
    def init_weights(self):
        # init the embedding and softmax layers
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.out.weight, -initrange, initrange)
        nn.init.zeros_(self.out.bias)
        
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
        embeded = self.dropout(self.pos_encoder(embeded))
        # src_key_padding_mask ?
        output = self.transformer_encoder(embeded, mask=self.src_mask)
        output = self.out(output)
        return output
        
            
    
    
    
    
    
    