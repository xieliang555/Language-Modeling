# Word-level language modeling
Some training tricks that matter for word-level language modeling using LSTM and transformer


## Requirements
- PyTorch 1.5.0
- torchtext 0.6.0


## Tricks

>         
- Default dataset: WikiText-2  
- Total training epoch: 40   
- Model scale:  
   - LSTM
      - small: 50dim for embedding + 128dim for hidden units + 1hidden layer 
      - medium: 100dim for embedding + 256dim for hidden units + 2hidden layer
      - large: 150dim for embedding + 512dim for hidden units + 3 hidden layer
   - Transformer
      - small: 50dim for embedding+ 128dim for hidden units + 1hidden layer + 1 head
      - medium: 100dim for embedding + 251dim for hidden units + 2 hidden layer + 2head
      - large: 150dim for embedding + 512dim for hidden units + 3 hidden layer + 3head

1. **model scale**

|  train/dev ppl   | LSTM          | Transformer |    
| :--------------: | :-----------: | :---------: | 
|  small           | 79.17/158.06  |             | 
| medium           | 55.65/149.35  |             |
| large            |               |             |


2. **batch size**

|train/dev ppl    | LSTM    | Transformer | 
| :-------------: | :-----: |:----------: | 
|   32            |         |             | 
|   64            |         |             |
|   128           |         |             |


3. **pretrained embedding**

|  train/dev ppl  | LSTM    | Transformer | 
| :-------------: |:------: | :---------: |
| None            |         |             |
| GloVe           |         |             | 
| FastText        |         |             |
| CharNGram       |         |             |
   
   
4. **tie embedding**

|train/dev ppl    | LSTM    |  Transformer | 
| :-------------: | :-----: |:-----------: | 
| False           |         |              | 
| True            |         |              |


5. **embedding dropout**

| drop ratio   | LSTM    | Transformer | 
| :----------: | :-----: | :---------: |
| 0            |         |             | 
| 0.2          |         |             |
| 0.5          |         |             |


6. **optimizer**

|  train/dev  ppl          | LSTM    | Transformer | 
| :----------------------: | :-----: | :---------: | 
| sgd                      |         |             |
| adam                     |         |             | 
| sgd + momentum(0.9)      |         |             |
| sgd + weight decay(1e-4) |         |             |
| adam + weght decay(1e-4) |         |             |

SGD is proved better than adaptive optimizer like Adam in LM task empirically


7. **only for LSTM**

|                  | skip connection    |  
| :--------------: | :----------------: | 
|  train/dev ppl   |                    |            



## Results on three open datasets
usingthe combined tricks of ...

|  train/dev ppl | LSTM    | Transformer |
|:-------------: | :-----: | :---------: | 
| WikiText-2     |         |             |
| WikiText103    |         |             |
| PennTreeBank   |         |             |

```{.python .input}

```
