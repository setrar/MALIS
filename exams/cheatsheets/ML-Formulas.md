For an efficient cheat sheet based on the provided lecture topics, focusing on key formulas is crucial. Below is a list of essential formulas and mathematical concepts that are fundamental to understanding the discussed machine learning topics:

### Linear Models for Regression
- **Ordinary Least Squares (OLS)**: $\hat{\beta} = (X^TX)^{-1}X^TY$

### Linear Classifiers: LDA & Logistic Regression
- **Logistic Regression Model**: $P(Y=1|X) = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$
- **LDA Decision Rule**: $X > c$ if $\frac{P(Y=1)P(X|Y=1)}{P(Y=0)P(X|Y=0)} > c$, for some threshold $c$.

### Gradient Descent
- **Gradient Descent Update Rule**: $\theta := \theta - \alpha \frac{\partial J(\theta)}{\partial \theta}$
  
### The Perceptron & Bias-Variance Decomposition
- **Perceptron Update Rule**: If $y_i(\mathbf{w} \cdot \mathbf{x}_i) \leq 0$, then $\mathbf{w} := \mathbf{w} + \eta y_i \mathbf{x}_i$
- **Bias-Variance Tradeoff**: $\text{Error}(x) = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$

### Support Vector Machines & Kernels
- **SVM Optimization Problem**: Minimize $\frac{1}{2}\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$ for all $i$.
- **Kernel Function**: $K(x, x') = \phi(x) \cdot \phi(x')$, for some mapping $\phi$.

### Kernels, Bias-Variance Decomposition, and Regularization
- **Regularization Terms**: 
  - L1: $||\beta||_1 = \sum |\beta_i|$
  - L2: $||\beta||_2^2 = \sum \beta_i^2$

### Regularization, Validation & Trees
- **Cross-Validation Error**: $\frac{1}{k} \sum_{i=1}^{k} \text{Error}_i$
- **Decision Tree Splitting Criterion**: Information Gain, Gini Impurity.

### Ensembles
- **Random Forest**: No explicit formula, but conceptually combines multiple tree predictions.
- **Boosting Weight Update**: Weights updated based on error of previous model's performance.

### Unsupervised Learning
- **k-Means Clustering**: Objective to minimize $\sum_{i=1}^k \sum_{ x \in S_i } || x - \mu_i ||^2$, where $\mu_i$ is the mean of points in $S_i$.
- **PCA**: Find the $k$ eigenvectors of the covariance matrix $X^TX$ that correspond to the $k$ largest eigenvalues.

### Neural Networks
- **Backpropagation**: Given a loss function $L$, $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$, where $z = wx + b$ and $a$ is the activation.

Including these formulas on your cheat sheet provides a solid mathematical foundation for understanding the algorithms and techniques discussed in the lectures. This selection prioritizes breadth and relevance, ensuring coverage of key topics while maintaining the cheat sheet's utility and conciseness.
