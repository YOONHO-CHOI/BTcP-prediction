# Break through cancer pain (BTcP) prediction

Breakthrough cancer pain (BTcP) is a challenging clinical problem in managing cancer pain. 
We investigated the clinical relevance of deep neural network models that predicts the onset time of BTcP for patients.

BTcP was defined as the pain with a numerical rating scale (NRS) score of 4 or above. 
![그림2](https://user-images.githubusercontent.com/54790722/183016167-3a49ae7a-3cd8-456d-bce2-fa19ce72ce3f.png)

The model consisted of three basic LSTM blocks stacked followed by dense layers.
![그림1](https://user-images.githubusercontent.com/54790722/183016602-8a90d8c2-d7fc-4e80-bad9-1ef5c8156419.png)

The model was trained for 300 epochs with a batch size of 100 with balanced cross entropy loss and was optimized by stochastic weight averaging (SWA,(23)) with an initial learning rate of 1e-4, start averaging of 5, and the average period of 1. Our model was programmed in Python 3.7, Tensorflow 2.4.1 version, and experimented with NVIDIA Geforce RTX 2080. 
![120_24_LSTM_12 Learning curves](https://user-images.githubusercontent.com/54790722/183015026-9f541567-a9eb-45fd-b438-02d87aba8a47.jpg)

We evaluated the performance of the model based on the Matthews correlation coefficient (MCC).
## MCC=  (TP⋅TN-FP⋅FN)/√((TP+FP)⋅(TP+FN)⋅(TN+FP)⋅(TN+FN))

