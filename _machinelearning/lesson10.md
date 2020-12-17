---
title: "Linear Regression"
lesson: 10
---

The goal of linear regression is to make a continuous value prediction for a given input after having trained the model on a labeled data set. Take for instance this simple problem of predicting the salary for an employee based on the number of years of experience. 

![Linear Regression Problem](/assets/images/ML_images/SupervisedLearning/LinRegProblem.png)

In order to predict the salary for an employee $y$ for an input of years of experience $x$, we will construct a hypothesis that outputs a prediction for an input $x$: $h_\theta(x) = \theta_0 + \theta_1 x_1$. In this simple case of linear regression, the hypothesis is in the form of the equation of a line ($y = mx +b$) where the slope is $\theta_1$ and the y-intercept is $\theta_0$. Once the parameters $\theta_0$ and $\theta_1$ are optimized to the best value using the gradient descent algorithm that will be discussed later, the result is a line of best fit (pictured above) that can be used to make predictions for new inputs.

When implementing this linear regression, it is best to use matrices so we slightly modify the hypothesis by adding a new feature $x_0$ that is always equal to 1 and passing a matrix of features to the hypothesis. This additional feature helps the shapes of the parameter matrix and feature matrix align better for matrix multiplication so that the hypothesis can be represented by a concise statement of matrix multiplication as seen in equation \ref{linRegEqn} (Note the parameter matrix $\theta$ is transposed so that the matrix multiplication is (1 x 2) x (2 x 1) = (1 x 1). 

$$\begin{equation}
    h_\theta(X) = \theta_0x_0 + \theta_1x_1
\end{equation}$$

$$\begin{equation}
X=\left[\begin{array}{l}
x_{0} \\
x_{1}
\end{array}\right] \quad \theta=\left[\begin{array}{l}
\theta_{0} \\
\theta_{1}
\end{array}\right]
\end{equation}$$

$$\begin{equation} \label{linRegEqn}
    h_\theta(X) = \theta^TX
\end{equation}$$

Next, in order to find the optimal values for $\theta_0$ and $\theta_1$ through gradient descent, we need to define a cost function that determines the error between the predicted output and actual output. A popular cost function for simple supervised learning tasks on continuous values is the \textbf{Mean Squared Error (MSE)} cost function. This cost function simply finds the difference between the predicted ($h(x)$) and actual ($y$) and then squares that difference (The $i$ superscript denotes the ith training example). The cost for one iteration is the average of the cost for all the training examples (as evident by the summation symbol). Note that $m$ is the number of training examples.

$$\begin{equation}
    J(\theta_0,\theta_1) = \frac{1}{2m} \sum\limits_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
\end{equation}$$

With the cost function defined, we can use the gradient descent algorithm discussed later on to find the optimal values for $\theta_0$ and $\theta_1$ that produce the most accurate predictions and the lowest cost.

Below is the update rule for gradient descent for linear regression. It is important to note that the parameters should be updated simultaneously. This means that the derivative of the cost function with respect to $\theta_0$ and $\theta_1$ should be calculated first before updating both $\theta_0$ and $\theta_1$. This is important so that the update of one parameter does not effect the update of the other simply because of the order in which they were calculated.

![Linear Regression Update Rule](/assets/images/ML_images/SupervisedLearning/LinRegUpdateRule.png)

For those with a calculus background who are interested in what the actual derivatives are for linear regression's MSE cost function, the equations are displayed below. While being able to derive the derivatives of the cost function is helpful in building an intuition for how gradient descent works, most all machine learning projects now have functions that automatically compute the numerous derivatives required for complex machine learning problems.

$$\begin{array}{l}
\frac{\partial}{\partial \theta_{0}} J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \\
\frac{\partial}{\partial \theta_{1}} J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x^{(i)}
\end{array}$$

### Multivariate Linear Regression
The previous example involved a problem with one feature. However, oftentimes relationships are more complex and involve multiple features to predict an output, this is where multivariate linear regression can be used.
The only difference between the previously discussed univariate linear regression and multivariate linear regression is that the latter makes predictions based on more features. For example, instead of predicting the salary of an employee based on years of experience, the salary might also depend on education level, college degree, area of residence. 

Take the table below as an example, where the goal is to predict the numbers of units of product sold based on the advertising spent on TV, radio, and newspaper ads. In this example, there are 3 features (TV, radio, and newspaper spending) and there are 4 observations/training examples represented in the table. In reality, it would take many times more training examples to get an accurate model.

![Multivariate Linear Regression](/assets/images/ML_images/SupervisedLearning/multivariate.png)

We can use the table above to establish some notation that will be useful later. $x^{(i)}$ refers to the feature vector of the ith training example. In order to specify specific features, we use another index as a subscript $x^{(i)}_j$ to indicate the jth feature in the ith training example. For example, in the above table, $x^3_2$ refers to the 2nd feature of the training example at index 3 which is 41.9 (remember arrays/lists are 0 indexed). It is important to note that features and training examples are 0-indexed and the feature at $x_0$ always has a value of 1. 

In the case of one feature the hypothesis can be visualized as a line in 2D space with two parameters. In the case of two features the hypothesis can be visualized as a plane in 3D space with three parameters. Any number of features above two becomes nearly impossible to visualize that is why it is important to build the intuition with basic examples and use your understanding of the mathematical notation to support your intuition in more complex problems.

Luckily, the use of matrix notation to setup the linear regression algorithm allows us to easily extend the problem from one feature to $n$ features such that the hypothesis may look like this: $h_\theta(x)=\theta_{0} x_{0}+\theta_{1} x_{1}+\ldots+\theta_{n} x_{n}$ but still is expressed as $h_\theta(x) = \theta^T x$ where $\theta$ and $x$ are vectors from $\theta_0$ to $\theta_n$ and $x_0$ to $x_n$ respectively. Thus, when implemented in code univariate and multivariate linear regression are the same and it is merely the data passed in as features that distinguishes between the two types.