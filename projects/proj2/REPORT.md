# REPORT

Project 2: The Perceptron

Team: Ekemini Ekong, Brice Robert 

## Part I – Implementing a perceptron algorithm from scratch:

- [ ] Task 1. Several experiments has been implemented to create the  `perceptron.jl` source code.

First, as simple perceptron file has been layed out. with a `mutable` structure `Perceptron` since Julia like C uses data structure rather than object oriented encapsulation `Class`, the all necessary functions has been added 


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
