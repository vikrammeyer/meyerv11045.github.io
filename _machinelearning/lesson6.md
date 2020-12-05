---
title: "What is Deep Learning?"
lesson: 6
---

Deep learning (DL) is a subfield of machine learning that is focused on creating large neural networks in order to solve problems in supervised, unsupervised, and reinforcement learning. These large neural networks are defined by their depth, or number of hidden layers. A “deep” neural network has, by definition, more than one hidden layer. Deep neural networks (DNNs) commonly have between 2-8 hidden layers, with 10+ layers being considered very deep and computationally expensive \cite{DL_Blog}.

![Shallow vs. Deep Neural Network](/assets/images/ML_images/AI/ShallowVsDeepNN.png)

Each additional hidden layer learns a distinct set of features based on the output of the previous layer. By aggregating and combining features from previous layers, deep neural networks are able to recognize more complex features as a result of more hidden layers, an ability referred to as \textit{feature hierarchy} (More on feature hierarchies in the following section) \cite{DL_Pathmind}.

![Feature Heirarchy](/assets/images/ML_images/AI/FeatureHeirarchy.png)

In order to learn these feature hierarchies, deep neural networks require large amounts of data. In general, the more high-quality data a deep neural network can train on, the more accurate it will become as seen below in figure \ref{fig:DL_Data}. In fact, a not-sophisticated algorithm trained on lots of high-quality data can outperform a very sophisticated algorithm trained on very little high-quality data or trained on low-quality data \cite{DL_Pathmind}.

![Deep Learning Data](/assets/images/ML_images/AI/DL_Data.png)

