# CIL-Project

## 1. Traditional Matrix Factorization

### Alternating Least Squares

* [Matrix Factorization techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
* [Improving regularized singular value decomposition for collaborative filtering](https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf)

## 2. Autoencoders

### I-AutoRec and U-Autorec

I-AutoRec seems to perform better than U-AutoRec.

### Enhance Dataset with Autoencoders

* [AutoRec: Autoencoders Meet Collaborative Filtering](http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf)
* [Collaborative Filtering with Stacked Denoising AutoEncoders and Sparse Inputs](https://hal.inria.fr/hal-01256422v1/document)
* [Training Deep AutoEncoders for Collaborative Filtering](https://arxiv.org/pdf/1708.01715.pdf)

## 3. Neural Networks

### Deep and Wide Neural Networks

### Include Reliabilities in Neural Networks

### Graph Neural Networks

* [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)
* [Deep Learning Architecture for Collaborative Filtering Recommender Systems](https://www.researchgate.net/publication/340416554_Deep_Learning_Architecture_for_Collaborative_Filtering_Recommender_Systems)
* [Deep Learning based Recommender System: A Survey and New Perspectives](https://arxiv.org/pdf/1707.07435.pdf)
* [Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf) --> [code](https://github.com/metahexane/ngcf_pytorch_g61/blob/master/ngcf.py) and [medium](https://medium.com/@yusufnoor_88274/implementing-neural-graph-collaborative-filtering-in-pytorch-4d021dff25f3)

# CIL-Roadmap

1. (28.04) Test out some other baseline approaches
 * Integrate GCF in ```pl.LightningModule``` [DONE]
 * Prediction-backward NNs [DONE]
 * Implementation of NCF [DONE]

2. (05.05) Finish baselines
 * Try out AEs [DONE]
 * End-to-end implementation of baselines (uniformity in code) [NOT DONE]

3. (12.05) Think about final model
 * Integrate GCF in general code framework [DONE]
 * Describe ideas of (concrete) final models
   *  Input --> (V)AE --*Denoised version of the data*--> GNN: embedding + feed-forward network (NCF) --> Prediction
   *  Bagging
   *  Include error in feed-forward network
 * Try out VAEs (and if time remains try out denoising through AEs) [DONE]
 * Try out bagging of various models [DONE]
 * End-to-end implementation of baselines (uniformity in code) [CLOSE]

4. (19.05) Implement various combinations of baselines 
 * (V)AE + GNN
 * GNN + NCF 
 * NCF full-implementation
 * End-to-end implementation of baselines (uniformity in code)

5. (26.05) Implement final model
 * Improve NN approaches: Xavier, learning rate schedulers,  
 * Implement easy baselines: Bayesian SVD++ and ensembling those
 * Data preprocessing: reduce data size and stack in to GNN + NCF
 * NCF + Error fed back into

6. (02.06) Add pretraining
7.

### To-do list:
1. Easy baselines [Lasse]
2. Dimensionality reduction [Andras]
3. NCF with feeback loop [Silas]
4. NN improvements (and AE if time) [Matteo] 
5. Code cleanup: Flake8 (Low Priority)


# Datasets
* [MovieLens](https://grouplens.org/datasets/movielens/)
* [Amazon Review Data](https://nijianmo.github.io/amazon/index.html)

