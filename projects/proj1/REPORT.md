# REPORT

## Part I – Implementing a kNN from scratch:

- [ ] Task 1.

By updating the `knn.py` source file representing the kNN Classifier, we wre able to add/fill the nescessary methods that allows the `experiments` notebook to run.

- The `knn.py` source code mimics the `sklearn.neighbors.KNeighborsClassifier` from the Python package `scikit-learn `.
- The distance metrics used are `Euclidian distance` and `Minkowski distance` using the same function but changing the parameters 

```math
D(X,Y) = ( \sum_{i=1}^n | x_i - y_i |^p )^{\dfrac{1}{p}}
```

- [ ] Task 2.

Studying the `curse of dimensionality` issue, I figured there was no need to exceed the 300 neighbor count so I tested through the notebook by ,first, adding manual code and then looping through a function.

The loop returned with an accuracy score of 80.54% at 246 neighbor point. 
