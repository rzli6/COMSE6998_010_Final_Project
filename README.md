# Prediction of short-term stock price by A-CNN+
## COMSE6998_010_Final_Project

Author: [Ruizhe Li (rl3070)](https://github.com/rzli6) \& [Ruochen Nan (rn2498)](https://github.com/marina32).

## Introduction
This project applies order-encoding methods and deep learning methods to do the short-term stock price prediction. Using [NYSE TAQ data](https://www.nyse.com/market-data/historical) on Oct 7th, 2019 in the A-CNN, A-CNN+ and A-LSTM model solves the classification problem of two classes (Up, Down) and the mid-price is the target of prediction.

## Implementation Details
**Dataset:** TAQ NYSE Equities on Oct 7th, 2019

**Hardware:** 2vCPUs | 13 GB memory | 1 x NVIDIA Tesla P100 / Tesla T4

**Platform:** Google Cloud Platform / Colab

**Framework:** Tensorflow 2

## Models
This project includes below models:

**CNN:**
- Embedding of order-type
- (with/without) Average pooling
- Convolutional layers (with/without inception module)
- Softmax classifier

And tested models 
- with/without average pooling
- with/without inception structures

**LSTM:**
- Embedding
- Several LSTM layers
- Batch-norm and dropout layers used in between
- Softmax classifier on the top

And tested the LSTM model 
- with different layers
- with/without average pooling layers

## Results
**CNN:**
For CNN models, we can achieve a 60% F1 score. However, average pooling and larger kernel sizes do not work, because our dataset only contains high frequency market orders, where low resolution information is not quite useful. Our parallel-kernel model can give similar performance as small-kernel models. It is expected to give better results on dataset containing both high frequency and low frequency data.

**LSTM:**
On the other hand, LSTM models can achieve 70-80% F1 score, which is much better than CNN models. For the same reason, average pooling does not work. We compared 1-layer model versus 2-layer model, and found the 2-layer model is easier to be overfitted. 1-layer model is better because it gives similar results and could be trained much faster. But both 1-layer and 2-layer models are much slower than CNN models, thus it could be a bottleneck in a real-world setting.

## References

1. [Daigo Tashiro, Hiroyasu Matsushima, Kiyoshi Izumi \& Hiroki Sakaji (2019) Encoding of high-frequency order information and prediction of short-term stock price by deep learning, Quantitative Finance, 19:9, 1499-1506, DOI: 10.1080/14697688.2019.1622314](https://www.tandfonline.com/doi/abs/10.1080/14697688.2019.1622314)

2. [Bao W, Yue J, Rao Y. A deep learning framework for financial time series using stacked autoencoders and long-short term memory. PLoS One. 2017 Jul 14;12(7):e0180944. doi: 10.1371/journal.pone.0180944. PMID: 28708865; PMCID: PMC5510866.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944)

3. [Dixon, Matthew. "Sequence classification of the limit order book using recurrent neural networks." Journal of computational science 24 (2018): 277-286.](https://www.sciencedirect.com/science/article/abs/pii/S1877750317309675)

4. [Cont, R., A. Kukanov, and S. Stoikov. "The price impact of order book events. arXiv. org Quantitative Finance Papers." (2011).](https://arxiv.org/abs/1011.6402) 

5. [Kercheval, A. and Zhang, Y., Modeling high-frequency limit order book dynamics with support vector machines. Quant. Finance, 2015, 15, 1315–1329.](https://www.math.fsu.edu/~aluffi/archive/paper462.pdf)
