import torch

def embedded_dropout(embedding, inputs, dropout=0.1, scale=None):
    if dropout:
        mask = embedding.weight.data.new().resize_(embedding.weight.size(0), 1).bernoulli_(1-dropout)
        mask = mask.expand_as(embedding.weight)/(1-dropout)
        masked_embed_weight = mask * embedding.weight
    else:
        masked_embed_weight = embedding.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
    padding_idx = embedding.padding_idx
    if padding_idx is None:
        padding_idx = -1
    embedded = torch.nn.functional.embedding(
        inputs, masked_embed_weight, padding_idx, embedding.max_norm, 
        embedding.norm_type, embedding.scale_grad_by_freq, embedding.sparse)
    return embedded