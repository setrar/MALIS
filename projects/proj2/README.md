# Project 2: The Perceptron

The perceptron is a machine learning algorithm for learning a binary classifier of the form:

```math
\begin{gather}
\hat{𝑦} = ℎ(𝒙) = 𝑠𝑖𝑔𝑛(𝒘^𝑇𝒙 + 𝑏)
\\
\text{ where } w \text{ is a vector of real-valued weights and } b text{ is denoted the bias }
\end{gather}
```

&#x274C; NOT FINISHED

- [ ] Task 4. Solve exercises 2.3 and 2.4 from The Elements of Statistical Learning.

## Part III – Getting familiar with the gradient:

- [ ] Task 6. Solve exercises 5.1-5.3, 5.5-5.7 and 5.9 from Chapter 5 in [The Mathematics of Machine Learning](https://mml-book.github.io/book/mml-book.pdf) [(1)](#1-httpsmml-bookgithubiobookmml-bookpdf)
.

---

## Report:

You need to prepare a 1-page report explaining how your model was trained and in which ways you used
the validation and test sets. Report the accuracy of your model. the results obtained in the validation set
and the strategy you used to choose a model. Use as many pages as you require for Task 6. For task 6, if
it is easier, you can deliver scanned copies of derivations done by hand


## Deliverables:

Upload a zip file named <group-name>.zip containing the report, the perceptron.py, the experiment.ipynb
file and any instructions required to run it. If you scanned the derivations, an additional pdf with them.

# Evaluation:

| Criteria | Score |
|-|-|
| Your code runs and works as expected on any given dataset | 6 |
| Your model achieves a good accuracy on a test set (not seen at training), i.e. above 80% | 2 |
| Among all submissions, your model achieves the highest accuracy | 1 | 
| Among all submissions, your model is the fastest during testing | 2 |
| Your answers to part II are correct (task 3 or task 4) | 4 |
| Your report | 5 |


Important:
- Failing to submit a report leads to a mark of zero (0).
- If ChatGPT is used, failing to report it and explaining its use leads to a mark of zero (0).
- A group will be chosen at random to present their solution during the lecture. Failing to justify the submitted solution leads to a mark of zero (0).

---
### [1] https://mml-book.github.io/book/mml-book.pdf

# References
 - [ ] [sci-kit learn best for machine learning with Julia?](https://www.reddit.com/r/Julia/comments/u83fzz/scikit_learn_best_for_machine_learning_with_julia/)
 - [ ] [Handwritten Digit Recognition using MNIST dataset and Julia Flux](https://github.com/crhota/Handwritten-Digit-Recognition-using-MNIST-dataset-and-Julia-Flux/blob/master/src/Handwriting%20Recognition.ipynb)
 - [ ] [An introduction to Deep Learning using Flux Part I: A simple linear regression example](https://medium.com/p/5c44be0c5661)
 - [ ] [An introduction to Deep Learning using Flux Part II: Multi-layer Perceptron](https://medium.com/@sophb/an-introduction-to-deep-learning-using-flux-part-ii-multi-layer-perceptron-32526b323474)
