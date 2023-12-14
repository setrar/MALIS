# REPORT

Project 2: The Perceptron

Team: Ekemini Ekong, Brice Robert 

## Part I – Implementing a perceptron algorithm from scratch:

- [ ] Task 1. Several experiments has been implemented to create the  `perceptron.jl` source code.

First, as [simple perceptron](experiments/perceptron.jl) file has been layed out with a `mutable` structure `Perceptron` since Julia like C uses data structure rather than object oriented encapsulation `Class`, then all necessary functions have been added to allow the perceptron to do some basic training.

During the training, the Perceptron data will be mutated this is why `mutable` has been added. By default, Julia tends to take the functional programming side and favors immutability and function first approach.

```julia
# Perceptron class definition
mutable struct Perceptron
    weights::Vector{Float64}
    bias::Float64
end
# Constructor for Perceptron
function Perceptron(input_size::Int) ...
# Activation function (step function)
function activate(x) ...
# Prediction method for Perceptron
function predict(perceptron::Perceptron, input::Vector{Float64}) ...
# Train the perceptron model
function train_perceptron!(model, train_data, epochs = 10, lr = 0.1) ...
```

1. The [simple Perceptron](experiments/perceptron.jl) uses the `sign` step function as its activation function, during training the prediction is called and from the first [experiment](experiments/experiment.ipynb) a mere `.098` acuracy score was achieved. The training was already using the MINST Dataset.
2. We then tried to use the sigmoid function as its activation function and the result was literrally the same.
3. We started thinking about decomposing the predition/training work. We extended the simple Perceptron by allowing multiple instances to call the process. We created the [MultiClass Perceptron](perceptron.jl)
