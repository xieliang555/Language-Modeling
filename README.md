# Word-level language modeling
This repository is a showcast of how regualarizations can help improve word-level language model. 

The code is partly based on the implementation of [Regularizing and Optimizing LSTM Language Models](https://github.com/salesforce/awd-lstm-lm) paper and the [PyTorch example](https://github.com/pytorch/examples/tree/master/word_language_model).


## Dependencies
- install all the dependencies by the command ```pip install -r requirements.txt```


## Regularizations

|  LSTM                              | train/dev ppl            |  epoch  |
| :------------------------------    | :----------------------: | :-----: |
|   Baseline                         |  102.78/168.42           |    6    | 
|   + tie embedding [1-2]            |  80.06/158.41            |    8    |
|   + variational/locked dropout [3] |  38.19/109.17            |    40   |
|   + embedding dropout [3]          |  46.69/101.54            |    40   |
|   + AR [4]                         |  52.07/97.22             |    40   |
|   + TAR [4]                        |  53.87/96.91             |    40   |
|   + fasttext embedding [5]         |  48.80/88.01             |    40   |
|   + adaptive softmax [6]           |                          |         |
|   + layer normalize [7]            |                          |         |
|   + skip connection                |                          |         |


**Setup**
- dataset: WikiText-2
- nemd:400, nhid:1150, nlayer:3, batch size: 80, bptt:70 
- optimizer: SGD without momentum, lr:30(constant), wdecay: 1.2e-6 
- locked dropout: default from paper, embedding dropout: 0.2 (paper default: 0.1)

**Note**
- SGD without momentum has been proved empirically better than the adaptive optimizer(like Adam) in the LM task.
- The larger regularized model performs better than the smaller model without regularizations.
- The shallow transformer performes worse than the regularized LSTM model.
- The pretrained embedding helps accelerate the training preocess.
- Current PyTorch version does not support the **WeightDrop** implementation refered in this [paper](https://arxiv.org/abs/1708.02182).
- The example config provided were not fine-tunned. Hyper-parameters tunning (eg. learning schedule) may further improve the results.



## Best results on three open datasets

| train/dev ppl  |   LSTM  |   Transformer  |
|:-------------: | :-----: | :-------------:|
| PennTreeBank   |         |                |
| WikiText-2     |         |                |
| WikiText103    |         |                |




## Reference

1. [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
2. [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/abs/1611.01462)
3. [A theoretically grounded application of dropout in recurrent neural networks](https://arxiv.org/abs/1512.05287)
4. [Revisiting activation regularization for language rnns](https://arxiv.org/abs/1708.01009)
5. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
6. [Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309)
7. [Layer Normalization](https://arxiv.org/abs/1607.06450)
