# Word-level language modeling
Regularizations for word-level language model with LSTM and transformer. The project is based on the paper "Regularizing and Optimizing LSTM Language Models". [Code here](https://github.com/salesforce/awd-lstm-lm)


## Dependencies
- PyTorch 1.5.0
- torchtext 0.6.0


## Regularizations

|  LSTM                              | train/dev ppl            |  epoch  |
| :------------------------------    | :----------------------: | :-----: |
|   Baseline [1]                     |  102.78/168.42           |    6    | 
|   + tie embedding [2-3]            |  80.06/158.41            |    8    |
|   + variational/locked dropout [4] |  38.19/109.17            |    40   |
|   + embedding dropout [4]          |  46.69/101.54            |    40   |
|   + AR [5]                         |  52.07/97.22             |    40   |
|   + TAR [5]                        |  53.87/96.91             |    40   |
|   + fasttext embedding [6]         |  48.80/88.01             |    40   |
|   + adaptive softmax [7]           |                          |         |
|   + layer normalize [8]            |                          |         |
|   + skip connection                |                          |         |


**Setup**
- dataset: WikiText-2
- nemd:400, nhid:1150, nlayer:3, batch size: 80, bptt:70 
- optimizer: SGD without momentum, lr:30(constant), wdecay: 1.2e-6 
- locked dropout: default from paper, embedding dropout: 0.2 (paper default: 0.1)

**Note**
- SGD without momentum has been proved empirically better than adaptive optimizer(like Adam) in LM task.
- Larger regularized model performs better than smaller model without regularizations.
- Shallow transformer performes worse than regularized LSTM model.
- Pretrained embedding accelerates training preocess.
- Hyper-parameters tunning (eg. learning schedule) can further improve the results.



## Best results on three open datasets

| train/dev ppl  |   LSTM  |   Transformer  |
|:-------------: | :-----: | :-------------:|
| PennTreeBank   |         |                |
| WikiText-2     |         |                |
| WikiText103    |         |                |




## Ref

1. https://github.com/pytorch/examples/tree/master/word_language_model
2. Using the Output Embedding to Improve Language Models
3. Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling
4. A theoretically grounded application of dropout in recurrent neural networks
5. Revisiting activation regularization for language rnns
6. Enriching Word Vectors with Subword Information
7. Efficient softmax approximation for GPUs
8. Layer Normalization
