# Cross-Lingual Transfer Learning

## Language Adverserial Network for Cross-Lingual Text Classification

#### Source code: [Language-Adversarial Training](https://github.com/htanwar922/Language-Adversarial-Network/tree/master/Work "Source code on Github")

## Introduction

__Language-Adversarial Training__ technique for the cross-lingual model transfer problem learns a language-invariant hidden feature space to achieve better cross-lingual generalization.

This project is a tensorflow implementation of language-adversarial training approach in the context of __Cross-lingual text classification (CLTC)__

We classify a product review into five categories corresponding to its star rating. 

<a href="https://user-images.githubusercontent.com/33962145/86307612-cfde6000-bc34-11ea-9472-c43f4565df1a.png"><img src="https://user-images.githubusercontent.com/33962145/86307612-cfde6000-bc34-11ea-9472-c43f4565df1a.png" width="250" height="400"></a>


LAN has two branches. There are four main components in the network: 
- Embedding averager __EA__ that maps input sequence x to embeddings and gives an averaging of the embeddigs of the sequence.
- Joint Feature extractor __F__ that maps the averaged embeddings to a fixed-length feature vector in the shared feature space.
- Sentiment classifier __P__ that predicts the label for x given the feature representation F (x).
- Language discriminator __Q__ that also takes F (x) but predicts a scalar score indicating whether x is from SOURCE (1) or TARGET (-1).

We adopt the __Deep Averaging Network (DAN)__ for the (EA + F). DAN takes the arithmetic mean of the word vectors as input, and passes it through several fully-connected layers until a softmax for classification.

EA takes the arithmetic mean of the word vectors as input, F passes it through several fully-connected layers until a softmax for classification in Semantic Classifier (P). In LAN, EA first calculates the average of the word vectors in the input sequence, then passes the average through a feed-forward network with ReLU nonlinearities. The activations of the last layer in F are considered the extracted features for the input and are then passed on to P and Q. The sentiment classifier P and the language discriminator Q are standard feed-forward networks. P has a softmax layer on top for text classification and Q ends with a tanh layer of output width 1 to assign a language identification score (-1 for target and 1 for source).

### Setup
To run the program on Colab: 

Copy the Work folder to the Google Drive.
The directory should be: 'My Drive/Colab Notebooks/Work/'

In that two directories get created after running extract_data.ipynb:

> - 'My Drive/Colab Notebooks/Work/Amazon reviews/'
>> - 'My Drive/Colab Notebooks/Work/Amazon reviews/train'
>> - 'My Drive/Colab Notebooks/Work/Amazon reviews/dev'
>> - 'My Drive/Colab Notebooks/Work/Amazon reviews/test'

> - 'My Drive/Colab Notebooks/Work/bwe/'
>> - 'My Drive/Colab Notebooks/Work/bwe/vectors'

### Note
___More detailed descriptions of running instructions can be found in folders in the repository.___
<br></br>
#### References :
[Language-Adversarial Training for Cross-Lingual Text Classification (TACL)](https://github.com/ccsasuke/adan "Source Code on github")<br>
[**Adversarial Deep Averaging Networks for Cross-Lingual Sentiment Classification**](https://arxiv.org/abs/1606.01614)<br>
[Xilun Chen](http://www.cs.cornell.edu/~xlchen/),
Yu Sun,
[Ben Athiwaratkun](http://www.benathiwaratkun.com/),
[Claire Cardie](http://www.cs.cornell.edu/home/cardie/),
[Kilian Weinberger](http://kilian.cs.cornell.edu/)
<br>
Transactions of the Association for Computational Linguistics (TACL)
<br>
[paper (arXiv)](https://arxiv.org/abs/1606.01614),
[bibtex (arXiv)](http://www.cs.cornell.edu/~xlchen/resources/bibtex/adan.bib),
[paper (TACL)](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00039),
[bibtex](http://www.cs.cornell.edu/~xlchen/resources/bibtex/adan_tacl.bib),
[talk@EMNLP2018](https://vimeo.com/306129914)

