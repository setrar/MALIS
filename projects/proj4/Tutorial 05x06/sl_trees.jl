using DecisionTree

using Random, Statistics

# load Data

X, y = load_data("iris")

# cast data types

X = float.(X)

y = string.(y)

iris = [X y]

function perclass_splits(y, percent)
    uniq_class = unique(y)
    keep_index = []
    for class in uniq_class
        class_index = findall(y .== class)
        row_index = randsubseq(class_index, percent)
        push!(keep_index, row_index...)
    end
    return keep_index
end

# split data between train and test

Random.seed!(1)

train_index = perclass_splits(y, 0.67)

test_index = setdiff(1:length(y), train_index)

# split features

