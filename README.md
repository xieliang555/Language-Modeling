# Word-level language modeling
Regularizations for word-level language model with LSTM and transformer.


## Dependencies
- PyTorch 1.5.0
- torchtext 0.6.0


## Regularizations

|  LSTM                              | train/dev ppl            |  epoch  |
| :------------------------------    | :----------------------: | :-----: |
|   Default                          |  102.78/168.42           |    6    | 
|   + tie embedding [1-2]            |  80.06/158.41            |    8    |
|   + variational/locked dropout [3] |  38.19/109.17            |    40   |
|   + embedding dropout [3]          |  46.69/101.54            |    40   |
|   + AR [4]                         |  52.07/97.22             |    40   |
|   + TAR [4]                        |  53.87/96.91             |    40   |
|   + adaptive softmax [5]           |                          |         |
|   + pretrained embedding           |                          |         |
|   + layer normalize                |                          |         |
|   + skip connection                |                          |         |


**Default Setup**
- nemd:400, nhid:1150, nlayer:3, batch size: 80, bptt:70 
- optimizer: SGD without momentum, lr:30(constant), wdecay: 1.2e-6 
- locked dropout: default, embedding dropout: 0.2

**Note**
- SGD without momentum has been proved empirically better than adaptive optimizer(like Adam) in LM task.
- Larger regularized model performs better than smaller model without regularizations.
- Shallow transformer performes worse than regularized LSTM model.



## Best results on three open datasets

| train/dev ppl  | LSTM    |   Transformer  |
|:-------------: | :-----: | :-------------:|
| PennTreeBank   |         |                |
| WikiText-2     |         |                |
| WikiText103    |         |                |




## Ref

1. Using the Output Embedding to Improve Language Models
2. Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling
3. A theoretically grounded application of dropout in recurrent neural networks
4. Revisiting activation regularization for language rnns
5. Efficient softmax approximation for GPUs
6. Regularizing and Optimizing LSTM Language Models
7. https://github.com/pytorch/examples/tree/master/word_language_model
8. https://github.com/salesforce/awd-lstm-lm
