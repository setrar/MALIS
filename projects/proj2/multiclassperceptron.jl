# Define a struct for a single binary perceptron classifier
mutable struct BinaryPerceptron
    weights::Vector{Float64}
    bias::Float64
end

# Constructor for the binary perceptron
function BinaryPerceptron(input_size::Int)
    weights = randn(input_size)
    bias = randn()
    return BinaryPerceptron(weights, bias)
end

# Sigmoid activation function
function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end


# Prediction function for the binary perceptron
function predict(model::BinaryPerceptron, input::Vector{Float64})
    return sigmoid(dot(model.weights, input) + model.bias) > 0.5 ? 1 : 0
end


# Training function for the binary perceptron
function train!(model::BinaryPerceptron, inputs::Matrix{Float64}, targets::Vector{Int}, epochs::Int, learning_rate::Float64)
    for epoch in 1:epochs
        for i in 1:size(inputs, 2)
            x = inputs[:, i]
            y = targets[i]
            ŷ = predict(model, x)
            error = y - ŷ
            model.weights .+= learning_rate * error * x
            model.bias += learning_rate * error
        end
    end
end

# Define the multi-class perceptron model
struct MultiClassPerceptron
    binary_classifiers::Vector{BinaryPerceptron}
end

# Constructor for the multi-class perceptron
function MultiClassPerceptron(input_size::Int, num_classes::Int)
    classifiers = [BinaryPerceptron(input_size) for _ in 1:num_classes]
    return MultiClassPerceptron(classifiers)
end

# Training function for the multi-class perceptron
function train!(model::MultiClassPerceptron, inputs::Matrix{Float64}, targets::Vector{Int}, epochs::Int, learning_rate::Float64)
    num_classes = length(model.binary_classifiers)
    for i in 1:num_classes
        # Create binary targets for the current classifier
        binary_targets = [target == i ? 1 : 0 for target in targets]
        train!(model.binary_classifiers[i], inputs, binary_targets, epochs, learning_rate)
    end
end

# Prediction function for the multi-class perceptron
function predict(model::MultiClassPerceptron, input::Vector{Float64})
    # Get the predictions from each binary classifier
    predictions = [predict(classifier, input) for classifier in model.binary_classifiers]
    # Return the index of the classifier with the highest prediction score
    return argmax(predictions)
end
