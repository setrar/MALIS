# Project 1: Undertanding k-NNs

kNN is considered a non-parametric method given that it makes few assumptions about the form of the data distribution. This approach is _memory-based_ as it requires no model to be fit.

Nearest-neighbor methods use the observations from the training set closest in input space to 𝑥 to form 𝑦̂ . It assumes that if a sample's features are similar to the ones of points of one particular class then it belongs to that class. These points are known as nearest neighbors.


```math
\text { The specific case where } \mathit{𝑘}=1 \text{ is denoted the nearest neighbor algorithm. }
\text{ Here } \hat{\mathit{y}} \text { is assigned the value } \mathit{y}_l  \text { of the closest point } \mathit{x}_𝑙 \text { to } \mathit{x} \text{ in the training data.}
```

 This corresponds to a [_Voronoi tessellation_](https://en.wikipedia.org/wiki/Voronoi_diagram) of the training data.

## Part I – Implementing a kNN from scratch:

- [ ] Task 1.

- [ ] Task 2.

## Part II – The curse of dimensionality:

To get familiar with it, first read section 2.5 from the book [The Elements of Statistical Learning](https://hastie.su.domains/Papers/ESLII.pdf). 

After reading, **choose one** of the two tasks below.

- [ ] Task 3.

- [ ] Task 4. Solve exercises 2.3 and 2.4 from The Elements of Statistical Learning.

---

## Report:

## Deliverables:


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


