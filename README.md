# Evolving Artificial Neural Networks

This repository is dedicated for the 1st course project of CS421 Advanced Artificial Intelligence, during the fall semester of 2018. This project is to use evolutionary algorithms to evolve neural networks to solve the 5-Parity Problem.

A more detailed description is available at this [report](https://github.com/Neutrino42/Advanced_AI_evolving_ANN/blob/master/README.pdf)


## 1. Introduction

The main task of this assignment is to evolve artificial neural networks (ANNs) for the *5-Parity* problem. The ANNs are famous for their properties of fitting any non-linear function. The work presented in this report is a modificiation of the paper [1] publishied by Yao and Liu. They adopted a evolutionary strategy to change the structure (e.g. number of nodes, connections) of the ANNs in order to produce some better results. This report will be divided into four parts. The second part will involve the detailed description of the algorithm. Several sub-parts will follows, introducing the procedure of each sub-components of the whole algorithm. Then I will present the experimental results and some parameter settings. Finally, the conclusion and future works.



###1.1 The *N-Parity Problem*

The *N-Parity problem* takes a *N*-dimensional binary vector as input and returns 1 or 0 depending on weather
the vector has an even number of 1s or not. A more formal definition is illustrated as follows:

![5-parity problem](res/5-parity.png)

In this assignment, we only consider *N = 5*, which is the *5-Parity Problem*.



## 2. Algorithm Design

### 2.1 Overview

An EP algorithm, which does not use crossover, is adopted in this network evolving process. This algorithm is similar with EPNet [1] by Yao and Liu, but the implementation details and design are slightly different.
![ANN](res/ANN.png)

​						Fig. 1 A fully connected feedforward ANN

A more detailed description is available at this [report](https://github.com/Neutrino42/Advanced_AI_evolving_ANN/blob/master/README.pdf)




### References

[1]X. Yao and Y. Liu, "A new evolutionary system for evolving artificial neural networks", *IEEE Transactions on Neural Networks*, vol. 8, no. 3, pp. 694-713, 1997.

[2]P. Werbos, "Backpropagation through time: what it does and how to do it", *Proceedings of the IEEE*, vol. 78, no. 10, pp. 1550-1560, 1990.

[3]W. Finnoff, F. Hergert and H. Zimmermann, "Improving model selection by nonconvergent methods", *Neural Networks*, vol. 6, no. 6, pp. 771-783, 1993.

