# Break through cancer pain (BTcP) prediction


Breakthrough cancer pain (BTcP) is a challenging clinical problem in managing cancer pain.

We investigated the clinical relevance of deep neural network models that predicts the onset time of BTcP.



BTcP was defined as the pain with a numerical rating scale (NRS) score of 4 or above. 

<p align="center"><img src = "https://user-images.githubusercontent.com/54790722/183018583-7b36e2f9-ff1f-4c81-9f70-b056e70fa5bf.png"/></p>
<p align="center"><img src = "https://user-images.githubusercontent.com/54790722/183026447-1c816468-92ab-40a5-b5d1-d6350c419954.png"/></p>


## Model 
The model consisted of three basic LSTM blocks stacked followed by dense layers.
<p align="center"><img src = "https://user-images.githubusercontent.com/54790722/183020377-9a6af0b4-952a-4a7c-ba42-fb49d3c11900.jpg" width="500" height="600"/></p>



The model was trained for 300 epochs with a batch size of 100 with balanced cross entropy loss and was optimized by stochastic weight averaging (SWA,(23)) with an initial learning rate of 1e-4, start averaging of 5, and the average period of 1. Our model was programmed in Python 3.7, Tensorflow 2.4.1 version, and experimented with NVIDIA Geforce RTX 2080.

## Evaluation metric
Matthews correlation coefficient (MCC)
<p align="center"><img src = "https://user-images.githubusercontent.com/54790722/183021941-e4f26737-0507-4364-9717-487ce2baabc3.jpg"/></p>


## Results
<p align="center"><img src = "https://user-images.githubusercontent.com/54790722/183023920-447dd7e7-697e-4cbd-bb06-f5ac53b5f41e.jpg"/></p>
<p align="center"><img src = "https://user-images.githubusercontent.com/54790722/183023853-3f4b7e2b-f987-489a-937e-515bc7027b0a.png"/></p>



## Cases of prediction for onset timing of BTcP
<p align="center"><img src = "https://user-images.githubusercontent.com/54790722/183026501-5dfacc8b-7199-4948-ab41-96f1a7014ea8.png"/></p>

