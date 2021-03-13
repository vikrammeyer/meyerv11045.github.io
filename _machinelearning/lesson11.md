---
title: "Logistic Regression"
lesson: 11
---

While linear regression makes continuous value predictions (e.g. any number between 0 and 100), logistic regression makes discrete predictions (e.g. $y$ is either 1 or 0). These discrete predictions can be used to classify inputs such as whether an email is spam or not spam or whether a tumor is malign or benign. In order to achieve this binary output, the hypothesis for linear regression must be slightly modified so that it maps all values of $\theta^TX$ to between 0 and 1 so that they can be treated as probabilities. This mapping is achieved using the sigmoid or logistic function ($\sigma(z)$) whose equation and graph is shown below.

$$\begin{equation}
    \sigma(z) = \frac{1}{1 + e^{-z}}
\end{equation}$$

![Sigmoid Function](/assets/images/ML_images/SupervisedLearning/sigmoidFn.png)

The sigmoid function is incorporated into the hypothesis like so:
$$\begin{equation}
    h_\theta(X) = \sigma(\theta^TX) = \frac{1}{1 + e^{-\theta^TX}}
\end{equation}$$

Now that the output of the hypothesis is a probability between 0 and 1, a decision boundary to determine the class of input data:
    if $h_\theta(x) >= 0.5$ then predict $y=1$
    if $h_\theta(x) < 0.5$ then predict $y=0$.

What is happening with this decision boundary is that the hypothesis predicts an output the same way as linear regression and then maps this output to a probability between 0 and 1 unlike linear regression. Then a threshold, 0.5 in most cases, is applied to determine the class (1 or 0) for the input data (classification).

Now that we understand the hypothesis used in logistic regression and how the hypothesis is interpreted to create a decision boundary, we can begin to look at the cost function that will be used by the gradient descent algorithm to find the best parameters in order to minimize the cost function. Logistic regression cannot use the same cost function as linear regression with this binary classification because it would produce a non-convex cost surface with many local minimum, making it much harder to find the global minimum compared to a convex surface \cite{LogisticRegression}. To create a cost function that will produce a convex surface for this binary classification problem, we use two separate cost functions for the two classes ($y=1$ and $y=0$) as seen in equation \ref{piecewiseCost}. Using the different cost functions for the different classes allows incorrect classifications to be highly penalized since the cost function approaches infinity the farther the prediction is from the actual class as seen in the graphs in figure \ref{fig:LogRegCost}. This cost function is referred to as the cross-entropy loss or log loss function.

$$\begin{equation}
\operatorname{cost}\left(h_{\theta}(x), y\right)=\left\{\begin{array}{ll}
-\log \left(h_{\theta}(x)\right) & \text { if\ } y=1 \\
-\log \left(1-h_{\theta}(x)\right) & \text { if\ } y=0
\end{array}\right.
\end{equation}$$

![Logistic Regression Cost](/assets/images/ML_images/SupervisedLearning/LogisticRegressionCost.png)

In practice, it is much more efficient to remove if-else statements when calculating the cost, so below is the one-line version of the cross-entropy loss function for logistic regression. It utilizes the fact that the label is either 0 or 1 in order to be be equivalent to the cost function expressed below when the values are plugged in. For example, when $y=1$, $J(\theta) = -log(h_\theta(x)$ because the other term goes to 0 when 1 is substituted in. Note that both equation \ref{LogRegCost1} and \ref{LogRegCost2} are equivalent, they just have the negative signs in different places and can be seen both ways in blog posts and documentation. 

$$\begin{equation} \label{LogRegCost1}
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}[y^{(i)} \log (h_{\theta}(x^{(i)}))+(1-y^{(i)}) \log (1-h_{\theta}(x^{(i)}))]
\end{equation}$$

$$\begin{equation} \label{LogRegCost2}
J(\theta)=\frac{1}{m} \sum_{i=1}^{m}[-y^{(i)} \log (h_{\theta}(x^{(i)}))-(1-y^{(i)}) \log (1-h_{\theta}(x^{(i)}))]
\end{equation}$$

With the cost function defined for logistic regression, we can use the gradient descent algorithm discussed in the next section to find the optimal values for the parameter matrix $\theta$ in order to produce the most accurate predictions and the lowest cost.